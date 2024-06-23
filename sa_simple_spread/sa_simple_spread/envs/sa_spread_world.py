import numpy as np
import pygame

import gymnasium as gym
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.spaces.utils import flatten
from pettingzoo.mpe import simple_spread_v3


class SaSimpleSpreadWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, n_agents=3, seed=None):
        # init multi agent environment
        self.ma_env = simple_spread_v3.env(N=n_agents, render_mode=render_mode)
        self.ma_env.reset(seed=seed)

        # idea: having an observation space with all agent observations
        self.observation_space = Box(low=-np.infty, high=np.infty, shape=(n_agents, 18), dtype=np.float_)

        # same idea for action space
        self.action_space = MultiDiscrete([5 for i in range(n_agents)])
        # self.action_space = MultiDiscrete([5,5,5])

        self.n_agents = n_agents

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.ma_env.reset(seed=seed)
        observations = []   # we gather the observations of each agent and save it to return it in the end
        infos = {}          # we ignore infos

        # idea: init with every agent doing nothing and return all observations
        for i in range(len(self.ma_env.agents)):
            self.ma_env.step(0)
            observation, _, _, _, _ = self.ma_env.last()
            observations.append(flatten(Box(low=-200, high=200, shape=(18,)), observation))

        observations = np.array(observations)

        return observations, infos

    def step(self, action):
        _, _, terminated, truncation, info = self.ma_env.last()

        # idea: shared space for all obs, rewards and infos
        # -> outside we will predict all actions based on all observations, and return the reward of the first agent
        # and we ignore infos
        observations = []
        observation = None
        rewards = []
        reward = 0
        infos = {}

        # perform all actions at one time step
        for agent in range(self.n_agents):
            _, _, terminated, truncation, _ = self.ma_env.last()
            if terminated or truncation:
                self.ma_env.step(None)
            else:
                self.ma_env.step(action[agent])
                observation, reward, terminated, truncation, info = self.ma_env.last()

            observations.append(observation)
            rewards.append(reward)

        np_observations = np.array(observations)
        return np_observations, rewards[0], terminated or truncation, False, infos

    def render(self):
        self.ma_env.render()

    def close(self):
        self.ma_env.close()
