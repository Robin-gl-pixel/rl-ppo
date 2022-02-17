import os

from pprint import pprint
import argparse

import seaborn as sns


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env", type=str, help="environment name", default="MountainCarContinuous-v0"
    )
    parser.add_argument(
        "--max_ep_len", type=int, help="max timesteps in one episode", default=1000
    )

    parser.add_argument(
        "--max_training_timesteps",
        type=int,
        help="break training loop if timeteps > max_training_timesteps",
        default=int(3e6),
    )

    parser.add_argument(
        "--action_std",
        type=float,
        help="starting std for action distribution (Multivariate Normal)",
        default=0.6,
    )
    parser.add_argument(
        "--action_std_decay_rate",
        type=float,
        help="linearly decay action_std (action_std = action_std - action_std_decay_rate)",
        default=0.05,
    )
    parser.add_argument(
        "--min_action_std",
        type=float,
        help="minimum action_std (stop decay after action_std <= min_action_std)",
        default=0.1,
    )
    parser.add_argument(
        "--action_std_decay_freq",
        type=int,
        help="action_std decay frequency (in num timesteps)",
        default=int(2.5e5),
    )

    parser.add_argument(
        "--update_timestep",
        type=int,
        help="update policy every n timesteps",
        default=4000,
    )

    parser.add_argument(
        "--K_epochs",
        type=int,
        help="update policy for K epochs in one PPO update",
        default=80,
    )

    parser.add_argument(
        "--eps_clip",
        type=float,
        help="clip parameter for PPO",
        default=0.2,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="discount factor",
        default=0.99,
    )

    parser.add_argument(
        "--lr_actor",
        type=float,
        help="learning rate for actor network",
        default=0.0003,
    )
    parser.add_argument(
        "--lr_critic",
        type=float,
        help="learning rate for critic network",
        default=0.001,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="set random seed if required (0 = no random seed)",
        default=0,
    )

    parser.add_argument(
        "--all_loss",
        type=list,
        help="all loss to test with",
        default=[
            "clipped_loss",
            "adaptative_KL_loss",
            "A2C_loss",
        ],
    )

    parser.add_argument(
        "--beta_kl",
        type=int,
        help="initialisation ppo KL loss",
        default=3,
    )
    parser.add_argument("--d_targ", type=int, help="target ppo KL loss", default=1)

    parser.add_argument(
        "--total_test_episodes", type=int, help="nb test episode to make gif", default=1
    )

    parser.add_argument(
        "--run_num_pretrained",
        type=float,
        help="set this to load a particular checkpoint num to make gif",
        default=0,
    )

    return parser


def reset_config(opt, print_=False):

    config = {}
    assert opt.env in [
        "MountainCarContinuous-v0",
    ]

    config["env_name"] = opt.env
    config["max_ep_len"] = opt.max_ep_len
    config["max_training_timesteps"] = opt.max_training_timesteps
    config["print_freq"] = opt.max_ep_len * 10
    config["log_freq"] = opt.max_ep_len * 2
    config["save_model_freq"] = int(1e5)

    config["action_std"] = opt.action_std
    config["action_std_decay_rate"] = opt.action_std_decay_rate
    config["min_action_std"] = opt.min_action_std
    config["action_std_decay_freq"] = opt.action_std_decay_freq
    config["update_timestep"] = opt.update_timestep
    config["K_epochs"] = opt.K_epochs
    config["eps_clip"] = opt.eps_clip
    config["gamma"] = opt.gamma
    config["lr_actor"] = opt.lr_actor
    config["lr_critic"] = opt.lr_critic

    config["random_seed"] = opt.random_seed

    config["beta_kl"] = opt.beta_kl
    config["d_targ"] = opt.d_targ

    config["run_num_pretrained"] = opt.d_targ

    # TODO: adapt main test code to plot and compare the loss
    config["color"] = {
        "A2C_loss": sns.color_palette("Set2")[0],
        "adaptative_KL_loss": sns.color_palette("Set2")[1],
        "clipped_loss": sns.color_palette("Set2")[2],
    }
    config["solved_reward"] = {
        "LunarLander-v2": 230,
        "MountainCarContinuous-v0": 300,
        "CartPole-v1": 300,
        "MountainCar-v0": 300,
    }

    if print_:
        print("Training config : \n")
        pprint(config)
    return config


def make_exp_dir(environment_name, log_dir="experiences", log_dir2="PPO_logs"):
    log_dir = "experiences"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### log files for multiple runs are NOT overwritten
    log_dir = os.path.join(log_dir, log_dir2)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = os.path.join(log_dir, environment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
