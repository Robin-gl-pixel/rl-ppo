import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

from ppo import PPO
from config_sys_argv import get_arguments, reset_config, make_exp_dir

################################### Training ###################################


def train():
    print(
        "============================================================================================"
    )

    parser = get_arguments()

    opt = parser.parse_args()

    print("training environment name : " + opt.env)

    log_dir = make_exp_dir(opt, "experiences", "PPO_logs")

    for loss in opt.all_loss:
        print("-----------------" + loss + "-----------------")

        config = reset_config(opt, print_=False)
        config["loss_name"] = loss

        env = gym.make(config["env_name"])

        # state space dimension
        state_dim = env.observation_space.shape[0]

        # action space dimension
        action_dim = env.action_space.shape[0]

        #### log files for multiple runs are NOT overwritten
        log_loss_dir = os.path.join(log_dir, loss)
        if not os.path.exists(log_loss_dir):
            os.makedirs(log_loss_dir)

        #### get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_loss_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = (
            log_loss_dir
            + "/PPO_"
            + config["env_name"]
            + "_log_"
            + str(run_num)
            + ".csv"
        )

        print("current logging run number for " + config["env_name"] + " : ", run_num)
        print("logging at : " + log_f_name)
        #####################################################

        ################### checkpointing ###################
        run_num_pretrained = (
            0  #### change this to prevent overwriting weights in same env_name folder
        )
        pretrained_dir = make_exp_dir(opt, "experiences", "PPO_preTrained")
        #### log files for multiple runs are NOT overwritten
        loss_pretrained_dir = os.path.join(pretrained_dir, loss)
        if not os.path.exists(loss_pretrained_dir):
            os.makedirs(loss_pretrained_dir)

        checkpoint_path = loss_pretrained_dir + "PPO_{}_{}_{}.pth".format(
            config["env_name"], config["random_seed"], run_num_pretrained
        )
        print("save checkpoint path : " + checkpoint_path)

        print(
            "============================================================================================"
        )

        ################# training procedure ################

        # initialize a PPO agent
        ppo_agent = PPO(
            state_dim,
            action_dim,
            config["lr_actor"],
            config["lr_critic"],
            config["gamma"],
            config["K_epochs"],
            config["eps_clip"],
            config["action_std"],
        )

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print(
            "============================================================================================"
        )

        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write("episode,timestep,reward\n")

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        # training loop
        while time_step <= config["max_training_timesteps"]:

            state = env.reset()
            current_ep_reward = 0

            for t in range(1, config["max_ep_len"] + 1):

                # select action with policy
                action = ppo_agent.select_action(state)
                state, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % config["update_timestep"] == 0:
                    ppo_agent.update()

                # if continuous action space; then decay action std of ouput action distribution
                if time_step % config["action_std_decay_freq"] == 0:
                    ppo_agent.decay_action_std(
                        config["action_std_decay_rate"], config["min_action_std"]
                    )

                # log in logging file
                if time_step % config["log_freq"] == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write(
                        "{},{},{}\n".format(i_episode, time_step, log_avg_reward)
                    )
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % config["print_freq"] == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print(
                        "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                            i_episode, time_step, print_avg_reward
                        )
                    )

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % config["save_model_freq"] == 0:
                    print(
                        "--------------------------------------------------------------------------------------------"
                    )
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                    print(
                        "Elapsed Time  : ",
                        datetime.now().replace(microsecond=0) - start_time,
                    )
                    print(
                        "--------------------------------------------------------------------------------------------"
                    )

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        env.close()

        # print total training time
        print(
            "============================================================================================"
        )
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print(
            "============================================================================================"
        )


if __name__ == "__main__":

    train()
