import csv

from gymnasium.envs.registration import register
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env


# specify the amount of agents you want to use
N_AGENTS = 4

# register our custom environment
register(
     id="sa_simple_spread/SASimpleSpread-v0",
     entry_point="sa_simple_spread.envs:SaSimpleSpreadWorld",
     max_episode_steps=300,
)

SEED = 42
rng = np.random.default_rng(SEED)

env = make_vec_env(
    'sa_simple_spread/SASimpleSpread-v0',
    rng=rng,
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    env_make_kwargs={"n_agents": N_AGENTS}
)


def train_expert():
    # note: use `download_best_expert` instead to download a pretrained, competent expert
    print("Training an expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    # train multiple agents, so we can evaluate later which one is the best
    # for j in range(10):
    for j in range(1):
        expert.learn(100_000)
        expert.save(f"sa_simple_spread_policies/joint_policy_{N_AGENTS}_agents_{j}00k.zip")
    return expert


def download_best_expert():
    # from evaluations we know 700k expert is best expert
    return PPO.load(f"sa_simple_spread_policies/joint_policy_{N_AGENTS}_700k.zip")


def sample_expert_transitions():
    # get expert
    # expert = download_best_expert()
    expert = train_expert()  # uncomment to train your own experts

    # print for evaluation
    expert_rewards_after_training, _ = evaluate_policy(
        expert, env, 100, return_episode_rewards=False
    )
    print("expert reward", expert_rewards_after_training)

    # sampling trajectories we can later imitate
    print("Sampling expert transitions.")
    samples = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
    )

    flattened = rollout.flatten_trajectories(samples)
    return flattened


rollouts = sample_expert_transitions()

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True
)

# train and evaluate the expert
# -----------------------------
print("evaluating experts")
with open('expert_policies_joint_spaces.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['training steps', 'trained avg reward', 'np mean'])

    for i in range(1): #10
        expert = PPO.load(f"sa_simple_spread_policies/joint_policy_{N_AGENTS}_agents_{i}00k.zip")
        expert_reward, _ = evaluate_policy(
            expert, env, 100, return_episode_rewards=False
        )
        writer.writerow([i * 100_000, expert_reward, np.mean(expert_reward)])

# train GAIL and evaluate the generator
# -------------------------------------
#   if you want to retrain the expert, change in sample_expert_transitions()
#   line
#       expert = download_best_expert()
#   to
#       expert = train_expert()

print("evaluating gail policies")
with open('gail_generator_rewards.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['training steps', 'trained avg reward'])

    for i in range(1): #10
        try:
            learner = PPO.load(f"sa_simple_spread_policies/gail_generator_{N_AGENTS}_agents_{i}00k.zip")
        except:
            print(f"/sa_simple_spread_policies/gail_generator_{N_AGENTS}_agents_{i}00k.zip not found, therefore trained")
            gail_trainer.train(100_000)
            learner.save(f"sa_simple_spread_policies/gail_generator_{N_AGENTS}_agents_{i}00k")

        learner_rewards, _ = evaluate_policy(
            learner, env, 100, return_episode_rewards=False
        )
        print(str(i) + " :" + str(learner_rewards))
        writer.writerow([i * 100_000, learner_rewards])
