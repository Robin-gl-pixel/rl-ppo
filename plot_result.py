import os
import sys

import pandas as pd

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import gym

from config_sys_argv import get_arguments, reset_config, make_exp_dir


def plot_result(
    all_df,
    label_list,
    saving_path,
    var=["timestep", "reward"],
):
    plt.figure(figsize=(10, 6))
    for loss_name in label_list:
        sns.lineplot(
            x=var[0],
            y=var[1],
            ci="sd",
            data=all_df[loss_name],
            label=loss_name,
        )

    plt.savefig(saving_path)


if __name__ == "__main__":

    parser = get_arguments()

    opt = parser.parse_args()

    environment_name = opt.env

    all_loss_df = {}
    for loss_name in opt.all_loss:
        checkpoint_path = os.path.join(
            "experiences",
            "PPO_logs",
            environment_name,
            loss_name,
            "PPO_{}_log_{}.csv".format(
                environment_name, opt.random_seed, opt.run_num_pretrained
            ),
        )
        all_loss_df[loss_name] = pd.read_csv(checkpoint_path)

    saving_path = os.path.join("experiences", "image", environment_name, "rewards.png")

    plot_result(all_loss_df, opt.all_loss, saving_path)
