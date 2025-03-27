"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import json
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
from RL_Algorithm.Algorithm.SARSA import SARSA
from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
from RL_Algorithm.Algorithm.MC import MC
from tqdm import tqdm
from collections import defaultdict



# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 5
    action_range = [-12.0, 12.0]  # [min, max]
    # discretize_state_weight = [10,20, 2, 2] 
    discretize_state_weight = [10,20, 2, 2]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.3
    # learning_rate = 0.3
    n_episodes = 12000
    start_epsilon = 1.0

    epsilon_decay = 0.9997

    final_epsilon = 0.01

    # discount = 0.99
    discount = 0.5

    # state_count = defaultdict(int)

    

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp

    Algorithm_name = "Q_learning"
    # Algorithm_name = "SARSA"
    # Algorithm_name = "Double_Q_learning"
    # Algorithm_name = "MC"

    agent = Q_Learning(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount
    )

    # agent = SARSA(
    #     num_of_action=num_of_action,
    #     action_range=action_range,
    #     discretize_state_weight=discretize_state_weight,
    #     learning_rate=learning_rate,
    #     initial_epsilon=start_epsilon,
    #     epsilon_decay=epsilon_decay,
    #     final_epsilon=final_epsilon,
    #     discount_factor=discount
    # )

    # agent = Double_Q_Learning(
    #     num_of_action=num_of_action,
    #     action_range=action_range,
    #     discretize_state_weight=discretize_state_weight,
    #     learning_rate=learning_rate,
    #     initial_epsilon=start_epsilon,
    #     epsilon_decay=epsilon_decay,
    #     final_epsilon=final_epsilon,
    #     discount_factor=discount
    # )

    # agent = MC(
    #     num_of_action=num_of_action,
    #     action_range=action_range,
    #     discretize_state_weight=discretize_state_weight,
    #     learning_rate=learning_rate,
    #     initial_epsilon=start_epsilon,
    #     epsilon_decay=epsilon_decay,
    #     final_epsilon=final_epsilon,
    #     discount_factor=discount
    # )
    # reset environment
    obs, _ = env.reset()
    timestep = 0
    sum_reward = 0
    sum_count = 0
    # List of dicts for every step (for plotting)
    train_logs = []
    name_plot = "test_new_final"



    full_path = os.path.join(f"q_value/{task_name}", Algorithm_name,name_plot)
    # Ensure the directory exists
    os.makedirs(full_path, exist_ok=True)  # FIX: Create directory before writing

    # Define config parameters
    config = {
        'num_of_action': num_of_action,
        'action_range': action_range,
        'discretize_state_weight': discretize_state_weight,
        'learning_rate': learning_rate,
        'epsilon_decay': epsilon_decay,
        'discount': discount,
        # 'reward' : " 3 reward (decrease reward for pole_pos)"
    }

    # Save JSON file inside the directory
    config_path = os.path.join(full_path, "config.json")  # FIX: Correct file path
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)  # FIX: Use indentation for readability


    wandb.init(project="DRL_HW2_NEW",name=name_plot)

    # wandb.init(project="DRL_HW2",name="SARSA_1")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in tqdm(range(n_episodes)):
            #  for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                cumulative_reward = 0
                selected_actions = []  
                count = 0
                while not done:
                    # agent stepping
                    action, action_idx = agent.get_action(obs)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    reward_value = reward.item()
                    terminated_value = terminated.item() 
                    cumulative_reward += reward_value
                    selected_actions.append(action_idx)

#  -------------------------------------------------------------------------------

                    # Q-learning 
                    agent.update(obs,action_idx,reward,next_obs,done)

#  -------------------------------------------------------------------------------

                    # SARSA
                    # if not (terminated or truncated):
                    #     next_action, next_action_idx = agent.get_action(next_obs)
                    # else:
                    #     next_action_idx = None  # No next action if episode ends

                    # agent.update(obs, action_idx, reward, next_obs, next_action_idx, done)

#  -------------------------------------------------------------------------------

                    # Double Q
                    # agent.update(obs, action_idx, reward, next_obs, done)

#  -------------------------------------------------------------------------------

                    # MC 
                    # agent.obs_hist.append(obs)
                    # agent.action_hist.append(action_idx)
                    # agent.reward_hist.append(reward_value)

#  -------------------------------------------------------------------------------


                    done = terminated or truncated
                    obs = next_obs
                    count += 1
                    

                # MC 
                
                sum_count += count
                sum_reward += cumulative_reward
                 # Store data at the end of each episode
                train_logs.append({
                    "episode": episode,
                    # "selected_actions": selected_actions,
                    "cumulative_reward": cumulative_reward,
                    "epsilon": agent.epsilon
                })

                wandb.log({
                    "episode": episode,
                    "cumulative_reward": cumulative_reward,
                    "epsilon": agent.epsilon
                })

                if episode % 100 == 0:
                    print("avg_score: ", sum_reward / 100.0)
                    
                    wandb.log({
                        "sum_reward" : sum_reward / 100.0,
                        "count" :sum_count /10000
                    })

                    # visit_counts = list(agent.state_count.values())
                    # wandb.log({"state_visit_frequency": wandb.Histogram(visit_counts)})

                    sum_reward = 0
                    sum_count = 0
                    print(agent.epsilon)

                    # Save Q-Learning agent
                    q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    # full_path = os.path.join(f"q_value/{task_name}", Algorithm_name,name_plot)
                    agent.save_q_value(full_path, q_value_file)

                    # n_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    # full_path_n = os.path.join(f"n_value/{task_name}", Algorithm_name,name_plot)
                    # agent.save_n_value(full_path_n,n_value_file)

                if episode % 1000 == 0 :
                    # Create 2D heatmap matrix for cart position vs pole angle (first 2 dimensions of state)
                    cart_bins, pole_bins = discretize_state_weight[0], discretize_state_weight[1]
                    position_visit_counts = np.zeros((cart_bins, pole_bins))

                    # Fill heatmap matrix using state_count
                    for state, count in agent.state_count.items():
                        cart_idx = state[0]
                        pole_idx = state[1]
                        position_visit_counts[cart_idx, pole_idx] += count

                    # Plot heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(position_visit_counts, cmap='viridis')
                    plt.title('State Visit Frequency (Cart Pos vs Pole Angle)')
                    plt.xlabel('Pole Angle Bins')
                    plt.ylabel('Cart Position Bins')

                    # Log to Weights & Biases
                    wandb.log({"State Visit Heatmap": wandb.Image(plt)})
                    plt.close()

                agent.decay_epsilon()
                # agent.update()
             
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        print("!!! Training is complete !!!")
        break
    # ==================================================================== #

    train_logs.append({
        "state_count" : {str(k): v for k, v in agent.state_count.items()}
    })

    log_json_path = os.path.join(log_dir, "step_logs.json")
    os.makedirs(os.path.dirname(log_json_path), exist_ok=True)

    print(f"Saving logs (every step) to {log_json_path}")
    with open(log_json_path, "w") as f:
        json.dump(train_logs, f, indent=2)
    
    # close the simulator
    env.close()
    wandb.finish()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()