import time

from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces.utils import flatten
import gymnasium as gym

# register our custom gym
register(
     id="sa_simple_spread/SASimpleSpread-v0",
     entry_point="sa_simple_spread.envs:SaSimpleSpreadWorld",
     max_episode_steps=300,
)

SEED = 42
n_agents = 3

# env=gym.make('sa_simple_spread/SASimpleSpread-v0', render_mode='human', n_agents=n_agents)
env = simple_spread_v3.env(render_mode="human")

# Lade das trainierte Modell
model = PPO.load("sa_simple_spread_policies/joint_policy_700k.zip")
# model = PPO.load("sa_simple_spread_policies/gail_generator_1100k.zip")

# Simuliere Episoden und rendere sie
env.reset(seed=0)

# start with a all zero observation for all three agents
obs = []
for i in range(18): obs.append(0)

observations = []
for i in range(n_agents):
    observations.append(flatten(Box(low=-200, high=200, shape=(18,)), obs))

# monitor the action distribution for each agent


# Unendliche Schleife, um die Simulation fortzusetzen
while True:
    curr_obs = []
    for i in range(n_agents):
        if isinstance(obs, tuple):
            obs = obs[0]
        _, _, terminated, truncated, _ = env.last()

        # differ between random and trained policy
        # print("predict: " + str(model.predict(np.array(observations))[0]))
        action = model.predict(np.array(observations))[0][i]

        if terminated or truncated:
            env.step(None)
        else:
            env.step(action)
        obs, reward, terminated, truncated, info = env.last()
        curr_obs.append(flatten(Box(low=-200, high=200, shape=(18,)), obs))

        # check end condition
        if terminated or truncated:
            env.reset()

    observations = curr_obs
    time.sleep(0.01)
