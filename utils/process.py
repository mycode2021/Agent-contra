import os
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
from torch import multiprocessing as mp
from torch.nn import functional as F

import utils


def runner(opt, global_model):
    torch.cuda.manual_seed(123) if torch.cuda.is_available() else torch.manual_seed(123)
    action_space = utils.Actions.get(opt.action_type)
    done, curr_step, score, max_score, count = True, 0, 0, 0, 0

    try:
        env, num_inputs, num_actions = utils.create_runtime_env(opt.game, opt.state, opt.action_type,
                                                                record=opt.record_path)
        local_model = utils.PPO(num_inputs, num_actions)
        torch.cuda.is_available() and local_model.cuda()
        local_model.eval()
        state = torch.from_numpy(env.reset())

        while True:
            curr_step += 1
            if torch.cuda.is_available():
                state = state.cuda()
            done and local_model.load_state_dict(global_model.state_dict())
            logits, value = local_model(state)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()
            state, reward, done, info = env.step(action)
            score += reward
            env.render()
            time.sleep(0.03)
            with open("%s/runner.log"%opt.log_path, "a") as f:
                now = datetime.now()
                f.write("Time: %s Status: %s Reward: %+.2f Action: %-s\n"%(str(datetime.date(now))+" "+str(
                        datetime.time(now)), "end" if done else "run", reward, " + ".join(action_space[action])))
            if curr_step > opt.global_steps:
                done = True
            if done:
                with open("%s/evaluate.log"%opt.log_path, "a") as f:
                    f.write("Episode: %d, Step: %d, Score: %+.2f, Evaluation: %s.\n"%(
                            count, curr_step, score, "success" if info["finish"] else "failure"))
                if info["finish"]:
                    torch.save(local_model.state_dict(), "%s/%d.pass"%(opt.saved_path, count))
                elif score > max_score:
                    torch.save(local_model.state_dict(), "%s/%d.save"%(opt.saved_path, count))
                    max_score = score
                curr_step, score, state = 0, 0, env.reset()
                count += 1
            state = torch.from_numpy(state)
    except KeyboardInterrupt:
        print("The keyboard caused the agent process to exit.")
    finally:
        env.close()


def worker(worker_conn, agent_conn, kwargs):
    def run(env, action):
        state, reward, done, info = env.step(action)
        return env.reset() if done else state, reward, done, info

    try:
        agent_conn.close()
        envs = [utils.create_runtime_env(**args)[0] for args in kwargs]
        while True:
            request, actions = worker_conn.recv()
            if request == "step":
                worker_conn.send([run(env, action) for env, action in zip(envs, actions)])
            elif request == "reset":
                worker_conn.send([env.reset() for env in envs])
            elif request == "render":
                worker_conn.send([env.render(mode="rgb_array") for env in envs])
            elif request == "close":
                worker_conn.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print("The keyboard caused the agent process to exit.")
    finally:
        for env in envs:
            env.close()


class MultiprocessAgent:
    def __init__(self, opt, context="spawn"):
        self.waiting, self.closed, self.viewer, = False, False, None
        self.num_workers = opt.processes
        games = [{"game": opt.game, "state": opt.state, "action_type": opt.action_type}] * opt.processes
        games = np.array_split(games, self.num_workers)
        work = mp.get_context(context)
        self.agent_conns, self.worker_conns = zip(*[work.Pipe() for _ in range(self.num_workers)])
        self.processes = [work.Process(target=worker, args=(worker_conn, agent_conn, game))
                          for worker_conn, agent_conn, game in zip(self.worker_conns, self.agent_conns, games)]
        for process in self.processes:
            process.daemon = True
            with self._clear_mpi_env_vars():
                process.start()
        for worker_conn in self.worker_conns:
            worker_conn.close()

    def reset(self):
        self._assert_not_closed()
        for agent_conn in self.agent_conns:
            agent_conn.send(("reset", None))
        states = [agent_conn.recv() for agent_conn in self.agent_conns]
        states = self._flatten_list(states)
        return self._flatten_states(states)

    def step(self, actions):
        self._step_async(actions)
        return self._step_wait()

    def render(self, mode="human"):
        images = self._get_images()
        big_image = self._tile_images(images)
        if mode == "human":
            self._get_viewer().imshow(big_image)
            return self._get_viewer().isopen
        elif mode == "rgb_array":
            return big_image
        else:
            raise NotImplementedError

    def close(self):
        if self.closed:
            return
        elif self.viewer is not None:
            self.viewer.close()
        self._close_extras()
        self.closed = True

    def _step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.num_workers)
        for agent_conn, action in zip(self.agent_conns, actions):
            agent_conn.send(("step", action))
        self.waiting = True

    def _step_wait(self):
        self._assert_not_closed()
        results = self._flatten_list([agent_conn.recv() for agent_conn in self.agent_conns])
        self.waiting = False
        states, rewards, dones, infos = zip(*results)
        return self._flatten_states(states), np.stack(rewards), np.stack(dones), infos

    def _get_images(self):
        self._assert_not_closed()
        for agent_conn in self.agent_conns:
            agent_conn.send(("render", None))
        images = [agent_conn.recv() for agent_conn in self.agent_conns]
        return self._flatten_list(images)

    def _tile_images(self, image_n_h_w_c):
        image_n_h_w_c = np.asarray(image_n_h_w_c)
        n, h, w, c = image_n_h_w_c.shape
        H = int(np.ceil(np.sqrt(n)))
        W = int(np.ceil(float(n)/H))
        image_n_h_w_c = np.array(list(image_n_h_w_c)+[image_n_h_w_c[0]*0 for _ in range(n, H*W)])
        image_H_W_h_w_c = image_n_h_w_c.reshape(H, W, h, w, c)
        image_Hxh_Wxw_c = image_H_W_h_w_c.transpose(0, 2, 1, 3, 4)
        return image_Hxh_Wxw_c.reshape(H*h, W*w, c)

    def _get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

    def _close_extras(self):
        self.closed = True
        if self.waiting:
            for agent_conn in self.agent_conns:
                agent_conn.recv()
        for agent_conn in self.agent_conns:
            agent_conn.send(("close", None))
        for process in self.processes:
            process.join()

    @contextmanager
    def _clear_mpi_env_vars(self):
        removed_environment = {}
        for key, value in list(os.environ.items()):
            for prefix in ["OMPI_", "PMI_"]:
                if key.startswith(prefix):
                    removed_environment[key] = value
                    del os.environ[key]
        try:
            yield
        finally:
            os.environ.update(removed_environment)

    def _flatten_states(self, states):
        assert isinstance(states, (list, tuple)) and len(states) > 0
        if isinstance(states[0], dict):
            keys = states[0].keys()
            return {key: np.stack([state[key] for state in states]) for key in keys}
        else:
            return np.stack(states)

    def _flatten_list(self, sequences):
        assert isinstance(sequences, (list, tuple)) and all([len(sequence) for sequence in sequences])
        return [seq for sequence in sequences for seq in sequence]

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate the process after calling closed."

    def __del__(self):
        if not self.closed:
            self.close()
