import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
from PIL import Image

import gym

from ppo import PPO
from config_sys_argv import get_arguments, reset_config, make_exp_dir


def save_gif_images(environment_name, loss_name, opt, gif_images_dir, gif_dir):
    print(
        "============================================================================================"
    )

    env = gym.make(environment_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    config = reset_config(opt, print_=False)

    ppo_agent = PPO(
        state_dim,
        action_dim,
        loss_name,
        config["lr_actor"],
        config["lr_critic"],
        config["gamma"],
        config["K_epochs"],
        config["eps_clip"],
        config["beta_kl"],
        config["d_targ"],
        config["action_std"],
    )

    checkpoint_path = os.path.join(
        "experiences",
        "PPO_preTrained",
        environment_name,
        "PPO_{}_{}_{}.pth".format(
            environment_name, config["random_seed"], config["run_num_pretrained"]
        ),
    )
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print(
        "--------------------------------------------------------------------------------------------"
    )
    test_running_reward = 0

    for ep in range(1, config["total_test_episodes"] + 1):

        ep_reward = 0
        state = env.reset()

        for t in range(1, config["max_ep_len"] + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            img = env.render(mode="rgb_array")

            img = Image.fromarray(img)
            img.save(gif_images_dir + "/" + str(t).zfill(6) + ".jpg")

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print("Episode: {} \t\t Reward: {}".format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print(
        "============================================================================================"
    )
    print("total number of frames / timesteps / images saved : ", t)
    avg_test_reward = test_running_reward / config["total_test_episodes"]
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print(
        "============================================================================================"
    )


######################## generate gif from saved images ########################


def save_gif(env_name, gif_images_dir, gif_dir):

    print(
        "============================================================================================"
    )

    gif_num = 0  #### change this to prevent overwriting gifs in same env_name folder

    # adjust following parameters to get desired duration, size (bytes) and smoothness of gif
    total_timesteps = 300
    step = 10
    frame_duration = 150

    # input images
    gif_images_dir_ = gif_images_dir + "/*.jpg"

    gif_path = gif_dir + "/PPO_" + env_name + "_gif_" + str(gif_num) + ".gif"

    img_paths = sorted(glob.glob(gif_images_dir_))
    img_paths = img_paths[:total_timesteps]
    img_paths = img_paths[::step]

    print("total frames in gif : ", len(img_paths))
    print(
        "total duration of gif : "
        + str(round(len(img_paths) * frame_duration / 1000, 2))
        + " seconds"
    )

    # save gif
    img, *imgs = [Image.open(f) for f in img_paths]
    img.save(
        fp=gif_path,
        format="GIF",
        append_images=imgs,
        save_all=True,
        optimize=True,
        duration=frame_duration,
        loop=0,
    )

    print("saved gif at : ", gif_path)

    print(
        "============================================================================================"
    )


if __name__ == "__main__":

    parser = get_arguments()

    opt = parser.parse_args()
    # make directory for saving gif images
    gif_images_dir = make_exp_dir(
        opt.env, log_dir="experiences", log_dir2="PPO_gif_images"
    )

    gif_dir = make_exp_dir(opt.env, log_dir="experiences", log_dir2="PPO_gifs")

    for loss in opt.all_loss:
        #### log files for multiple runs are NOT overwritten
        loss_gif_images_dir = os.path.join(gif_images_dir, loss)
        if not os.path.exists(loss_gif_images_dir):
            os.makedirs(loss_gif_images_dir)

        #### log files for multiple runs are NOT overwritten
        loss_gif_dir = os.path.join(gif_dir, loss)
        if not os.path.exists(loss_gif_dir):
            os.makedirs(loss_gif_dir)

        # TODO ad run_num_pretrained
        # save .jpg images in PPO_gif_images folder

        save_gif_images(opt.env, loss, opt, gif_images_dir, gif_dir)

        # save .gif in PPO_gifs folder using .jpg images
        save_gif(opt.env, gif_images_dir, gif_dir)
