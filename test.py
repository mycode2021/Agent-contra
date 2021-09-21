from time import sleep
from torch.nn import functional as F
import argparse, os, torch, utils
os.environ["OMP_NUM_THREADS"] = "1"


def get_args(config):
    args = argparse.ArgumentParser(
        '''Directions: Testing tool to play contra from the trained model.''')
    args.add_argument("--game", type=str, default="Contra-Nes")
    args.add_argument("--state", type=str, default="Level1")
    args.add_argument("--action_type", type=str, default="complex")
    args.add_argument("--model_path", type=config["model_path"]["type"],
                                      default=config["model_path"]["default"])
    args.add_argument("--from_model", type=str, default="")
    return args.parse_args()

def run_test(opt):
    torch.manual_seed(123)
    memory = "%s/%s"%(opt.model_path, opt.from_model)
    assert os.path.isfile(memory), "The trained model does not exist."
    score = 0

    try:
        env, num_inputs, num_actions = utils.create_runtime_env(opt.game, opt.state, opt.action_type)
        model = utils.PPO(num_inputs, num_actions)
        model.eval()
        model.load_state_dict(torch.load(memory, map_location=torch.device("cpu")))
        state = torch.from_numpy(env.reset())

        while True:
            logits, value = model(state)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()
            state, reward, done, info = env.step(action)
            score += reward
            env.render()
            sleep(0.03)
            if done:
                print("Play %s level %s:\nscore %.2f, evaluation %s."%(
                      opt.game, opt.state, score, "success" if info["finish"] else "failure"))
                break
            state = torch.from_numpy(state)
    finally:
        env.close()


if __name__ == "__main__":
    opt = get_args(utils.configure)
    run_test(opt)
