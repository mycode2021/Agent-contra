from collections import deque
from gym import spaces
from retro import make
import cv2, gym, numpy as np, utils


class ActionsDiscretizer(gym.ActionWrapper):
    def __init__(self, env, actions):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = env.buttons
        self._actions = []
        for action in actions:
            arr = np.array([False]*len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action].copy()

class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, width, height))
        self.shape = width, height

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.shape, interpolation=cv2.INTER_AREA)
        frame = frame[None, :, :]
        return frame

class AllowBacktracking(gym.Wrapper):
    def __init__(self, env, width=84, height=84, skip=4):
        super(AllowBacktracking, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, width, height))
        self.states = deque(np.zeros((4, width, height), dtype=np.float32), maxlen=4)
        self.score, self.skip = 0, skip

    def step(self, action):
        total_reward, state_buffer = 0, deque(maxlen=2)
        for _ in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward + (info["score"]-self.score) * 0.5
            self.score = info["score"]
            state_buffer.append(state)
            if done: break
        else:
            _, _, done, _ = self.env.step(0)
        self.states.append(np.max(np.concatenate(state_buffer, 0), 0))
        return np.array(self.states)[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self, **kwargs):
        self.score, state = 0, self.env.reset(**kwargs)
        self.states.extend(np.concatenate([state for _ in range(4)], 0))
        return np.array(self.states)[None, :, :, :].astype(np.float32)

class RewardScaler(gym.RewardWrapper):
    def __init__(self, env, scale=0.25):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale

class ContraWinner(gym.Wrapper):
    def __init__(self, env):
        super(ContraWinner, self).__init__(env)
        self.zeros, self.level, self.lives, self.finish = 0, 0, 0, None
        self.correct = deque(maxlen=50)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.finish == None:
            self.lives, self.level = info["lives"], info["level"]
            self.finish = lambda level: level != self.level
        self.lives = max(self.lives, info["lives"])
        info["finish"] = self.finish(info["level"])
        self.correct.append((action, max(0, reward)))
        self.zeros += not reward
        done |= info["finish"] or info["lives"] < self.lives or self.zeros > 500 or  \
            self.correct.count((action, 0)) == self.correct.maxlen
        if done:
            if info["finish"]:
                reward += 1000
            else:
                reward += -10
        return state, reward, done, info

    def reset(self, **kwargs):
        self.zeros, self.level, self.lives, self.finish = 0, 0, 0, None
        self.correct.clear()
        return self.env.reset(**kwargs)

def create_runtime_env(game, state, action_type, record=False):
    actions = utils.Actions.get(action_type)
    assert actions, "Invalid action type."
    env = make(game, state, record=record)
    env = ActionsDiscretizer(env, actions)
    env = ProcessFrame(env)
    env = AllowBacktracking(env)
    env = RewardScaler(env)
    env = ContraWinner(env)
    return env, env.observation_space.shape[0], len(actions)
