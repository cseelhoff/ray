"""
Example of running an RLlib Trainer against a locally running Unity3D editor
instance (available as Unity3DEnv inside RLlib).
For a distributed cloud setup example with Unity,
see `examples/serving/unity3d_[server|client].py`

To run this script against a local Unity3D engine:
1) Install Unity3D and `pip install mlagents`.

2) Open the Unity3D Editor and load an example scene from the following
   ml-agents pip package location:
   `.../ml-agents/Project/Assets/ML-Agents/Examples/`
   This script supports the `3DBall`, `3DBallHard`, `SoccerStrikersVsGoalie`,
    `Tennis`, and `Walker` examples.
   Specify the game you chose on your command line via e.g. `--env 3DBall`.
   Feel free to add more supported examples here.

3) Then run this script (you will have to press Play in your Unity editor
   at some point to start the game and the learning process):
$ python unity3d_env_local.py --env 3DBall --stop-reward [..]
  [--framework=torch]?
"""

import argparse
import os

import ray
from ray import tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.policy.policy import PolicySpec
from gym.spaces import Box

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="Walker",
    help="The name of the Env to run in the Unity3D editor:"
         "`Walker` (feel free to add more and PR!)")
parser.add_argument(
    "--file-name",
    type=str,
    default=None,
    help="The Unity3d binary (compiled) game, e.g. "
         "'/home/ubuntu/soccer_strikers_vs_goalie_linux.x86_64'. Use `None` for "
         "a currently running Unity3D editor.")
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
         "Trainer state.")
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
         "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=999999,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999.0,
    help="Reward at which we stop training.")
parser.add_argument(
    "--horizon",
    type=int,
    default=6000,
    help="The max. number of `step()`s for any episode (per agent) before "
         "it'll be reset again automatically.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")

if __name__ == "__main__":

    ray.init(local_mode=False)

    args = parser.parse_args()


    # Get policies (different agent types; "behaviors" in MLAgents) and
    # the mappings from individual agents to Policies.
    # policies, policy_mapping_fn = get_policy_configs_for_game(args.env)
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return args.env


    config = {
        "env": "unity3d",
        "env_config": {
            "file_name": args.file_name,
            "episode_horizon": args.horizon,
        },
        # For running in editor, force to use just one Worker (we only have
        # one Unity running)!
        "num_workers": args.num_workers if args.file_name else 0,
        # Other settings.
        "lr": 0.00005,
        "lambda": 0.95,
        "gamma": 0.995,
        "sgd_minibatch_size": 512,
        "train_batch_size": 5120,  # buffer_size
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,  # int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "simple_optimizer": True,
        "num_sgd_iter": 1,  # num_epoch
        "rollout_fragment_length": 512,  # time_horizon
        "clip_param": 0.2,  # epsilon
        "kl_coeff": 0.05,  # default is not present. 0 will likely worsen stability
        "kl_target": 0.05,  # default 0.01
        # Multi-agent setup for the particular env.
        "multiagent": {
            "policies": {
                args.env: PolicySpec(
                    observation_space=Box(float("-inf"), float("inf"), (307,)),
                    action_space=Box(float("-inf"), float("inf"), (46,))),
            },
            "policy_mapping_fn": policy_mapping_fn,
        },
        "model": {
            "fcnet_hiddens": [1024, 1024, 1024],
        },
        "framework": "torch",
        "no_done_at_end": True,
        "entropy_coeff": 0.0000,  # beta - entropy is about 40 times larger for RLLIB - 0 will limit exploration likely
        "exploration_config": {
            "type": "StochasticSampling",
        },
        "explore": True,
        "eager_tracing": False, # eager helps debugging and requires tf2
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    from rllib.agents import ppo

    tune.register_env(
        "unity3d",
        lambda c: Unity3DEnv(
            file_name=c["file_name"],
            no_graphics=(c["file_name"] is not None),
            episode_horizon=c["episode_horizon"],
        ))

    trainer = ppo.PPOTrainer(env="unity3d", config=config)

    while True:
        print(trainer.train())
