from time import strftime
from torch import multiprocessing as mp
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse, numpy as np, os, torch, utils
os.environ['OMP_NUM_THREADS'] = '1'


def get_args(config):
    args = argparse.ArgumentParser(
        '''Directions: Deep-Reinforcement-Learning PPO methods to training contra.''')
    args.add_argument("--game", type=str, default="Contra-Nes")
    args.add_argument("--state", type=str, default="Level1")
    args.add_argument("--action_type", type=str, default="complex")
    args.add_argument("--processes", type=int, default=3)
    args.add_argument("--from_model", type=str, default="")
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
    env, num_inputs, num_actions = utils.create_runtime_env(opt.game, opt.state, opt.action_type)
    env.close()
    global_model = utils.PPO(num_inputs, num_actions)
    rnd_model = utils.RND(num_inputs, num_actions)
    if opt.from_model:
        memory = "%s/%s"%(opt.model_path, opt.from_model)
        assert os.path.isfile(memory), "The trained model does not exist."
        print("Loading trained from model %s."%memory)
        map_location = None if torch.cuda.is_available() else device
        global_model.load_state_dict(torch.load(memory, map_location=map_location))
    global_model = global_model.to(device)
    global_model.share_memory()
    rnd_model = rnd_model.to(device)
    optimizer = torch.optim.Adam(
        list(global_model.parameters())+list(rnd_model.parameters()), lr=opt.lr)
    forward_mse = nn.MSELoss(reduction='none')
    envs = utils.MultiprocessAgent(opt)
    run = mp.get_context("spawn")
    process = run.Process(target=utils.runner, args=(opt, global_model))
    process.start()
    state = envs.reset()
    reward_rms = utils.RunningMeanStd()
    obs_rms = utils.RunningMeanStd(state.shape)
    discounted_reward = utils.RewardForwardFilter(0.999)
    obs_rms.update(state)
    state = torch.from_numpy(np.concatenate(state, 0))
    state = state.to(device)
    curr_episode = 0

    while True:
        curr_episode += 1
        old_log_policies, actions, values, states, rewards, dones = [], [], [], [], [], []

        for _ in range(opt.local_steps):
            states.append(state)
            logits, value = global_model(state)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            state, reward, done, info = envs.step(action.cpu())
            next_obs = ((state-obs_rms.mean)/np.sqrt(obs_rms.var)).clip(-5, 5)
            next_obs = torch.from_numpy(np.concatenate(next_obs, 0))
            next_obs = next_obs.to(device)
            state = torch.from_numpy(np.concatenate(state, 0))
            state = state.to(device)
            intrinsic_reward = rnd_model.reward(next_obs.float())
            per_reward = np.array([discounted_reward.update(r) for r in intrinsic_reward])
            mean, std, count = np.mean(per_reward), np.std(per_reward), len(per_reward)
            reward_rms.update_from_moments(mean, std**2, count)
            reward += intrinsic_reward / np.sqrt(reward_rms.var)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)
            rewards.append(reward)
            dones.append(done)

        _, next_value, = global_model(state)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae, R = 0, []

        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1-done) - value.detach()
            next_value = value
            R.append(gae + value)

        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values

        for _ in range(opt.num_epochs):
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
                logits, value = global_model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy-old_log_policies[batch_indices])
                clamp = torch.clamp(ratio, 1.0-opt.epsilon, 1.0+opt.epsilon)
                ratio_min = torch.min(ratio*advantages[batch_indices], clamp*advantages[batch_indices])
                actor_loss = -torch.mean(ratio_min)
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                writer.add_scalar("PPO agent (loss/episode)", total_loss, curr_episode)
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(global_model.parameters())+list(rnd_model.parameters()), 0.5)
                optimizer.step()

        with open("%s/train.log"%opt.log_path, "a") as f:
            f.write("Time: %s, Episode: %d, Loss: %f.\n"%(strftime("%F %T"), curr_episode, total_loss))


if __name__ == "__main__":
    opt = get_args(utils.configure)
    run_train(opt)
