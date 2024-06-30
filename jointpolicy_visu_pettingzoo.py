import time

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.mpe import simple_spread_v3
from gymnasium.envs.registration import register
from gymnasium.spaces import Box
from gymnasium.spaces.utils import flatten
import numpy as np

# register our custom gym
register(
     id="sa_simple_spread/SASimpleSpread-v0",
     entry_point="sa_simple_spread.envs:SaSimpleSpreadWorld",
     max_episode_steps=300,
)

SEED = 42
n_agents = 3

# Load normal simple spread environment
env = simple_spread_v3.env(render_mode="human")

# Load policy you want to visualise
# model = PPO.load("sa_simple_spread_policies/joint_policy_700k.zip")         # expert
model = PPO.load("sa_simple_spread_policies/gail_generator_300k.zip")    # gail trained

# Simulate episodes and render them
env.reset(seed=SEED)

# start with an all zero observation for all three agents
obs = []
for i in range(18):  # (18 = n_agents * 6)
    obs.append(0)

observations = []
for i in range(n_agents):
    observations.append(flatten(Box(low=-200, high=200, shape=(18,)), obs))

# Infinite loop to keep the simulation ongoing
while True:
    curr_obs = []
    for i in range(n_agents):
        if isinstance(obs, tuple):
            obs = obs[0]
        _, _, terminated, truncated, _ = env.last()

        # differ between random and trained policy
        action = model.predict(np.array(observations))[0][i]
        # action = env.action_space("agent_"+str(i)).sample()  # uncomment to see random behavior

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
    time.sleep(0.2)
