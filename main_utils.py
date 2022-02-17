from datetime import datetime


def print_average_reward(time_step, i_episode, print_avg_reward):
    # print average reward till last episode
    print_avg_reward = round(print_avg_reward, 2)

    print(
        "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
            i_episode, time_step, print_avg_reward
        )
    )


def save_model_weights(time_step, ppo_agent, checkpoint_path, start_time):
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


def log_in_logging_file(log_avg_reward, log_f, i_episode, time_step):
    # log average reward till last episode
    log_avg_reward = round(log_avg_reward, 4)

    log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
    log_f.flush()
