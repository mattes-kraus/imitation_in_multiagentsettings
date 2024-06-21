from gymnasium.envs.registration import register

register(
     id="sa_simple_spread/SASimpleSpread-v0",
     entry_point="sa_simple_spread.envs:SASimpleSpreadEnv",
     max_episode_steps=300,
)