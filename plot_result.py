import os
import sys

import pandas as pd

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import gym


def plot_result(all_df, label_list, saving_path="experiences/image/MountainCarContinuous-v0/rewards.png", var=["timestep", "reward"]):
    plt.figure(figsize=(10, 6))
    for i in range(len(label_list)):
        loss_name = label_list[i]
        sns.lineplot(
            x=var[0],
            y=var[1],
            ci="sd",
            data=all_df[i],
            label=loss_name,
        )

    plt.savefig(saving_path)


def plot_sensitivity(all_df, label_list, var=["timestep", "reward"], saving_path="experiences/image/MountainCarContinuous-v0/sensitivity.png"):
    plt.figure(figsize=(8, 4))
    for i in range(len(label_list)):
        r = all_df[i]
        col = list(sns.color_palette("Set1") + sns.color_palette("Set3"))[i]
        sns.lineplot(
            x=var[0], y=var[1], ci="sd", data=r, color=col, label=label_list[i]
        )
    plt.title("Sensitivity of the reward for each loss")
    plt.savefig(saving_path)
if __name__ == "__main__":


    esperience_name = "MountainCarContinuous-v0"
    a2c_loss_path ="experiences/PPO_logs/MountainCarContinuous-v0/A2C_loss/PPO_MountainCarContinuous-v0_log_0.csv"
    kl_loss_path = "experiences/PPO_logs/MountainCarContinuous-v0/adaptative_KL_loss/PPO_MountainCarContinuous-v0_log_0.csv"
    clipped_loss_path = "experiences/PPO_logs/MountainCarContinuous-v0/clipped_loss/PPO_MountainCarContinuous-v0_log_1.csv"
    
    a2c_loss_df = pd.read_csv(a2c_loss_path)
    kl_loss_df = pd.read_csv(kl_loss_path)
    clipped_loss_df = pd.read_csv(clipped_loss_path)

    label_list = ["a2c_loss", "kl_loss", "clipped_loss"]
    plot_sensitivity([a2c_loss_df, kl_loss_df, clipped_loss_df], label_list)
    plot_result([a2c_loss_df, kl_loss_df, clipped_loss_df], label_list)