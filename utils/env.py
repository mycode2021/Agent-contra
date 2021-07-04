from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces
from retro import make

import utils


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
    def __init__(self, env, width=96, height=96):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, width, height))
        self.shape = width, height

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.shape, interpolation=cv2.INTER_AREA)
        frame = frame[None, :, :]
        return frame


class RewardScaler(gym.Wrapper):
    def __init__(self, env, scale=0.08, width=96, height=96, skip=4):
        super(RewardScaler, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(skip, width, height))
        self.states = deque(np.zeros((skip, width, height), dtype=np.float32), maxlen=4)
        self.xscroll, self.scale, self.skip = 0, scale, skip

    def step(self, action):
        total_reward, state_buffer = 0, deque(maxlen=2)
        for _ in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward * self.scale + (info["xscroll"]-self.xscroll) * (0.2-self.scale)
            self.xscroll = info["xscroll"]
            state_buffer.append(state)
            if done: break
        else:
            _, _, done, _ = self.env.step(0)
        total_reward = np.clip(total_reward, -1, 1)
        self.states.append(np.max(np.concatenate(state_buffer, 0), 0))
        return np.array(self.states)[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self, **kwargs):
        self.xscroll, state = 0, self.env.reset(**kwargs)
        self.states.extend(np.concatenate([state for _ in range(self.skip)], 0))
        return np.array(self.states)[None, :, :, :].astype(np.float32)


class ContraWinner(gym.Wrapper):
    def __init__(self, env):
        super(ContraWinner, self).__init__(env)
        self.zeros, self.level, self.lives, self.finish = 0, 0, 0, None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.finish == None:
            self.lives, self.level = info["lives"], info["level"]
            self.finish = lambda level: level != self.level
        self.lives = max(self.lives, info["lives"])
        info["finish"] = self.finish(info["level"])
        self.zeros += not reward
        done |= info["finish"] or info["lives"] < self.lives or self.zeros > 500
        if done:
            reward += 100 if info["finish"] else -10
        return state, reward, done, info

    def reset(self, **kwargs):
        self.zeros, self.level, self.lives, self.finish = 0, 0, 0, None
        return self.env.reset(**kwargs)


def create_runtime_env(game, state, action_type, record=False):
    actions = utils.Actions.get(action_type)
    assert actions, "Invalid action type."
    env = make(game, state, record=record)
    env = ActionsDiscretizer(env, actions)
    env = ProcessFrame(env)
    env = RewardScaler(env)
    env = ContraWinner(env)
    return env, env.observation_space.shape[0], len(actions)
