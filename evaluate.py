from torch.nn import functional as F
from torch.multiprocessing import Pool, cpu_count
import argparse, os, torch, utils
os.environ["OMP_NUM_THREADS"] = "1"


def get_args(config):
    args = argparse.ArgumentParser(
        '''Directions: Evaluating tool to play contra from the trained directroy.''')
    args.add_argument("--game", type=str, default="Contra-Nes")
    args.add_argument("--state", type=str, default="Level1")
    args.add_argument("--action_type", type=str, default="complex")
    args.add_argument("--processes", type=int, default=cpu_count())
    args.add_argument("--model_path", type=config["model_path"]["type"],
                                      default=config["model_path"]["default"])
    args.add_argument("--from_dir", type=str, default="")
    return args.parse_args()

def run_test(data, opt):
    torch.manual_seed(123)
    memory = "%s/%s/%s"%(opt.model_path, opt.from_dir, data)
    assert os.path.isfile(memory), "The trained model does not exist."

    try:
        env, num_inputs, num_actions = utils.create_runtime_env(opt.game, opt.state, opt.action_type)
        model = utils.PPO(num_inputs, num_actions)
        model.eval()
        model.load_state_dict(torch.load(memory, map_location=torch.device("cpu")))
        xscroll, state = 0, torch.from_numpy(env.reset())

        while True:
            logits, value = model(state)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()
            state, reward, done, info = env.step(action)
            xscroll = max(xscroll, info["xscroll"])
            if done:
                print("model %s, xscroll %d, finish %s."%(
                    data, xscroll, info["finish"]))
                break
            state = torch.from_numpy(state)
    finally:
        env.close()

def run_evaluate(opt):
    print("Play %s level %s:"%(opt.game, opt.state))
    models = os.listdir("%s/%s"%(opt.model_path, opt.from_dir))
    pool = Pool(opt.processes)

    for data in models:
        pool.apply_async(run_test, args=(data, opt))
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    opt = get_args(utils.configure)
    run_evaluate(opt)
