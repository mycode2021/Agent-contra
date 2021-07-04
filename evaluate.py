import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import shutil
from time import sleep

import torch
from torch.nn import functional as F
from torch.multiprocessing import Pool

import utils


def get_args(config):
    args = argparse.ArgumentParser(
        '''Directions: Evaluating tool to play contra from the trained directroy.''')
    args.add_argument("--game", type=str, default="Contra-Nes")
    args.add_argument("--state", type=str, default="Level1")
    args.add_argument("--action_type", type=str, default="complex")
    args.add_argument("--processes", type=int, default=10)
    args.add_argument("--loading_path", type=config["loading_path"]["type"],
                                        default=config["loading_path"]["default"])
    args.add_argument("--from_dir", type=str, default="")
    return args.parse_args()


def run_test(data, opt):
    torch.manual_seed(123)
    memory = "%s/%s/%s"%(opt.loading_path, opt.from_dir, data)
    assert os.path.isfile(memory), "The trained model does not exist."
    score, time = 0, 0

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
            if done:
                print("model %s, score %.2f, evaluation %s."%(
                      data, score, "success" if info["finish"] else "failure"))
                break
            state = torch.from_numpy(state)
    finally:
        env.close()


def run_evaluate(opt):
    print("Play %s level %s:"%(opt.game, opt.state))
    models = os.listdir("%s/%s"%(opt.loading_path, opt.from_dir))
    pool = Pool(opt.processes)

    for data in models:
        pool.apply_async(run_test, args=(data, opt))
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    opt = get_args(utils.configure)
    run_evaluate(opt)
