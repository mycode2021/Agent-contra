from time import strftime
from torch import multiprocessing as mp
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import argparse, numpy as np, os, torch, utils
os.environ['OMP_NUM_THREADS'] = '1'


def get_args(config):
    args = argparse.ArgumentParser(
        '''Directions: Deep-Reinforcement-Learning PPO methods to training contra.''')
    args.add_argument("--game", type=str, default="Contra-Nes")
    args.add_argument("--state", type=str, default="Level1")
    args.add_argument("--action_type", type=str, default="complex")
    args.add_argument("--processes", type=int, default=32)
    args.add_argument("--from_model", type=str, default="")
    args.add_argument("--render", action="store_true")
    [args.add_argument("--%s"%k, type=v["type"], default=v["default"]) for k, v in config.items()]
    return args.parse_args()

def run_train(opt):
    torch.cuda.manual_seed(123) if torch.cuda.is_available() else torch.manual_seed(123)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.path.isdir(opt.saved_path) or os.makedirs(opt.saved_path)
    os.path.isdir(opt.tensorboard_path) or os.makedirs(opt.tensorboard_path)
    os.path.isdir(opt.record_path) or os.makedirs(opt.record_path)
    os.path.isdir(opt.log_path) or os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.tensorboard_path)
    env, num_inputs, num_actions = utils.create_runtime_env(
        opt.game, opt.state, opt.action_type, opt.record_path)
    ppo_model = utils.PPO(num_inputs, num_actions).to(device)
    rnd_model = utils.RND(num_inputs, num_actions).to(device)
    if opt.from_model:
        memory = "%s/%s"%(opt.model_path, opt.from_model)
        assert os.path.isfile(memory), "The trained model does not exist."
        print("Loading trained from model %s."%memory)
        map_location = None if torch.cuda.is_available() else device
        ppo_model.load_state_dict(torch.load(memory, map_location=map_location))
    parameters = list(ppo_model.parameters()) + list(rnd_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=opt.lr)
    forward_mse = nn.MSELoss(reduction='none')
    envs = utils.MultiprocessAgent(opt)
    state = envs.reset()
    reward_rms = utils.RunningMeanStd()
    obs_rms = utils.RunningMeanStd(state.shape)
    obs_rms.update(state)
    discounted_reward = utils.RewardForwardFilter(opt.int_gamma)
    state = torch.from_numpy(np.concatenate(state, 0)).to(device)
    max_xscroll, curr_episode = 0, 0

    while True:
        old_log_policies, actions, values, states, ext_rewards, int_rewards, dones = \
            [], [], [], [], [], [], []

        for _ in range(opt.local_steps):
            states.append(state)
            logits, value = ppo_model(state)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            state, ext_reward, done, info = envs.step(action.cpu())
            next_obs = ((state-obs_rms.mean)/np.sqrt(obs_rms.var)).clip(-5, 5)
            next_obs = torch.from_numpy(np.concatenate(next_obs, 0)).to(device)
            state = torch.from_numpy(np.concatenate(state, 0)).to(device)
            int_reward = rnd_model.reward(next_obs.float())
            int_rewards.append(int_reward)
            ext_reward = torch.FloatTensor(ext_reward).to(device)
            ext_rewards.append(ext_reward)
            done = torch.FloatTensor(done).to(device)
            dones.append(done)

        _, next_value = ppo_model(state)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        per_reward = np.array([discounted_reward.update(r) for r in int_rewards])
        mean, std, count = np.mean(per_reward), np.std(per_reward), len(per_reward)
        reward_rms.update_from_moments(mean, std**2, count)
        int_rewards /= np.sqrt(reward_rms.var)
        int_rewards = torch.from_numpy(int_rewards).to(device)
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        ext_gae, int_gae, ext_R, int_R= 0, 0, [], []

        for value, ext_reward, done in list(zip(values, ext_rewards, dones))[::-1]:
            ext_gae = ext_gae * opt.ext_gamma * opt.tau
            ext_gae = ext_gae + ext_reward + opt.ext_gamma * next_value.detach() * (1-done) - value.detach()
            next_value = value
            ext_R.append(ext_gae+value)

        for value, int_reward, done in list(zip(values, int_rewards, dones))[::-1]:
            int_gae = int_gae * opt.int_gamma * opt.tau
            int_gae = int_gae + int_reward + opt.int_gamma * next_value.detach() * (1-done) - value.detach()
            next_value = value
            int_R.append(int_gae+value)

        ext_R = torch.cat(ext_R[::-1]).detach()
        int_R = torch.cat(int_R[::-1]).detach()
        advantages = (ext_R-values) * opt.ext_coef + (int_R-values) * opt.int_coef
        ppo_model.train()

        for _ in trange(opt.num_epochs):
            indice = torch.randperm(opt.local_steps*opt.processes)
            for n in range(opt.batch_size):
                indice_start = n * opt.local_steps * opt.processes // opt.batch_size
                indice_end = (n+1) * opt.local_steps * opt.processes // opt.batch_size
                batch_indices = indice[indice_start:indice_end]
                predict_next_state_feature, target_next_state_feature = rnd_model(states[batch_indices])
                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                mask = torch.rand(len(forward_loss)).to(device)
                mask = (mask<opt.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss*mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))
                logits, value = ppo_model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy-old_log_policies[batch_indices])
                clamp = torch.clamp(ratio, 1.0-opt.epsilon, 1.0+opt.epsilon)
                ratio_min = torch.min(ratio*advantages[batch_indices], clamp*advantages[batch_indices])
                actor_loss = -torch.mean(ratio_min)
                critic_loss = F.smooth_l1_loss(ext_R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + 0.5 * critic_loss - opt.beta * entropy_loss + forward_loss
                writer.add_scalar("PPO agent (loss/episode)", total_loss, curr_episode)
                parameters = list(ppo_model.parameters()) + list(rnd_model.parameters())
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 0.5)
                optimizer.step()

        max_xscroll, curr_step, curr_xscroll, finish = utils.runner(
            curr_episode, max_xscroll, env, opt, ppo_model)
        # with open("%s/evaluate.log"%opt.log_path, "a") as f:
        #     f.write("Episode: %d, Step: %d, Xscroll: %d, Finish: %s, Loss: %f.\n"%(
        #         curr_episode, curr_step, curr_xscroll, finish, total_loss))
        print("Episode: %d, Step: %d, Xscroll: %d, Finish: %s, Loss: %f."%(
            curr_episode, curr_step, curr_xscroll, finish, total_loss))
        curr_episode += 1


if __name__ == "__main__":
    opt = get_args(utils.configure)
    run_train(opt)
