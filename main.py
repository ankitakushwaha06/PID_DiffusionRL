## Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

# Fix for distutils.version.LooseVersion issue BEFORE any other imports
import sys
import distutils.util
if not hasattr(distutils, 'version'):
    try:
        from packaging.version import Version
        class LooseVersion:
            def __init__(self, vstring=None):
                self.vstring = str(vstring) if vstring else ''
            def __repr__(self):
                return f"LooseVersion ({self.vstring!r})"
            def __str__(self):
                return str(self.vstring)
            def __eq__(self, other):
                return str(self) == str(other)
            def __ne__(self, other):
                return not self.__eq__(other)
            def __lt__(self, other):
                return str(self) < str(other)
            def __le__(self, other):
                return str(self) <= str(other)
            def __gt__(self, other):
                return str(self) > str(other)
            def __ge__(self, other):
                return str(self) >= str(other)
        class _VersionModule:
            LooseVersion = LooseVersion
        distutils.version = _VersionModule()
    except ImportError:
        pass

import argparse
import gymnasium as gym
import numpy as np
import os
import torch
import json
import dsrl
import dsrl.offline_bullet_safety_gym
import dsrl.offline_safety_gymnasium 
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter



hyperparameters = {
    
    
    'OfflineAntRun-v0': {
        'lr': 3e-4, 'eta': 0.1, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005,
        'cost_limit': 10.0, 'cost_threshold': 10.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },
    
    'OfflineBallRun-v0': {
        'lr': 3e-4, 'eta': 0.1,  'kp': 0.01, 'ki': 0.001, 'kd': 0.005, 
        'cost_limit': 10.0, 'cost_threshold': 10.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },
    
    'OfflineCarCircle-v0': {
        'lr': 3e-4, 'eta': 2.0, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005,
        'cost_limit': 10.0, 'cost_threshold': 10.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },

    'OfflineDroneRun-v0': {
        'lr': 3e-4, 'eta':2.0, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005, 
        'cost_limit': 10.0, 'cost_threshold': 10.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'normalize', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1,
        'use_huber': True
    },
    
    'OfflineBallCircle-v0': {
        'lr': 3e-4, 'eta': 3.0, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005, 
        'cost_limit': 10.0, 'cost_threshold': 10.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },

    'OfflinePointCircle1Gymnasium-v0': {
        'lr': 3e-4, 'eta': 1.0, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005,
        'cost_limit': 20.0, 'cost_threshold': 20.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },
    'OfflinePointCircle2Gymnasium-v0': {
        'lr': 3e-4, 'eta': 0.1, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005,
        'cost_limit': 20.0, 'cost_threshold': 20.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },
    
    'OfflineAntVelocityGymnasium-v1': {
        'lr': 3e-4, 'eta': 1.0, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005,
        'cost_limit': 20.0, 'cost_threshold': 20.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },
    'OfflineHalfCheetahVelocityGymnasium-v1': {
        'lr': 3e-4, 'eta': 1.0, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005,
        'cost_limit': 20.0, 'cost_threshold': 20.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },
    'OfflineHopperVelocityGymnasium-v1': {
        'lr': 3e-4, 'eta': 1.0, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005,
        'cost_limit': 20.0, 'cost_threshold': 20.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },
    'OfflineSwimmerVelocityGymnasium-v1': {
        'lr': 3e-4, 'eta': 0.001, 'kp': 0.01, 'ki': 0.001, 'kd': 0.005,
        'cost_limit': 20.0, 'cost_threshold': 20.0, 'max_q_backup': False, 'gn': 1.0,
        'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'top_k': 1
    },
}


def train_agent(dataset, state_dim, action_dim, max_action, device, output_dir, args):
    
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner('Loaded buffer')

    if args.algo == 'ql':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn,
                      cost_limit=args.cost_limit, 
                      kp=args.kp, 
                      ki=args.ki, 
                      kd=args.kd,
                      pid_delay_steps=args.pid_delay_steps,
                      use_huber=args.use_huber)
    elif args.algo == 'gaussian-ql':
        from agents.ql_gaussian import Gaussian_QL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn,
                      cost_limit=args.cost_limit,
                      kp=args.kp,
                      ki=args.ki,
                      kd=args.kd,
                      use_huber=args.use_huber)
    elif args.algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr)

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)
    writer = None  

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    eval_reward, eval_cost, eval_norm_reward, eval_norm_cost = None, None, None, None
    while (training_iters < max_timesteps) and (not early_stop):
        
        iterations = int(args.num_steps_per_epoch)
        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=writer)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        
        is_eval_epoch = (curr_epoch % args.eval_freq == 0) or (curr_epoch == 1)
        
        if is_eval_epoch:
            eval_reward, eval_res_std, eval_norm_reward, eval_res_std, eval_cost, eval_norm_cost = eval_policy(
                agent, args.env_name, args.seed, args.cost_threshold, eval_episodes=args.eval_episodes
            )
            # Store evaluations for model selection
            evaluations.append([eval_reward, eval_res_std, eval_norm_reward, eval_res_std, eval_cost, eval_norm_cost,
                                np.mean(loss_metric['bc_loss']), np.mean(loss_metric['ql_loss']),
                                np.mean(loss_metric['actor_loss']), np.mean(loss_metric['critic_loss']),
                                np.mean(loss_metric['estimated_cost']), np.mean(loss_metric['pid_error']),
                                curr_epoch])
            np.save(os.path.join(output_dir, "eval"), evaluations)

        # 3. Log to CSV (this ensures a row for every epoch)
        logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.record_tabular('Cost Critic Loss', np.mean(loss_metric['cost_critic_loss']))
        logger.record_tabular('Estimated Cost Proxy', np.mean(loss_metric['estimated_cost']))
        logger.record_tabular('PID Error', np.mean(loss_metric['pid_error']))
        logger.record_tabular('PID Lambda', np.mean(loss_metric['pid_lambda']))
        
        # Only log reward/cost if we actually evaluated them this epoch
        if is_eval_epoch:
            logger.record_tabular('Avg Episodic Reward', eval_reward)
            logger.record_tabular('Avg Episodic Cost', eval_cost)
            logger.record_tabular('Norm Episodic Reward', eval_norm_reward)
            logger.record_tabular('Norm Episodic Cost', eval_norm_cost)
        
        logger.dump_tabular() # Writes to CSV and Terminal

        # 4. Optional: Save best model logic based on BCE loss for stability
        bc_loss = np.mean(loss_metric['bc_loss'])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)
        metric = bc_loss

        if args.save_best_model and is_eval_epoch:
            agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline
    scores = np.array(evaluations)
    
    # Check if user manually specified best epoch
    if args.best_epoch is not None:
        # Find the evaluation row corresponding to the specified epoch
        epoch_indices = np.where(scores[:, -1] == args.best_epoch)[0]
        if len(epoch_indices) > 0:
            best_id = epoch_indices[0]
            best_res = {'model selection': 'manual', 'epoch': scores[best_id, -1],
                        'best normalized score avg': scores[best_id, 2],
                        'best normalized score std': scores[best_id, 3],
                        'best raw score avg': scores[best_id, 0],
                        'best raw score std': scores[best_id, 1],
                        'cost': scores[best_id, 4],
                        'normalized_cost': scores[best_id, 5],
                        'estimated_cost': scores[best_id, 10],
                        'bc_loss': scores[best_id, 6]}
            with open(os.path.join(output_dir, f"best_score_manual.txt"), 'w') as f:
                f.write(json.dumps(best_res))
            print(f"\n*** Using manually selected epoch {args.best_epoch} as best model ***")
            print(f"Norm Reward: {scores[best_id, 2]:.2f}, Norm Cost: {scores[best_id, 5]:.2f}, BC Loss: {scores[best_id, 6]:.4f}\n")
        else:
            print(f"\nWarning: Epoch {args.best_epoch} not found in evaluations. Using automatic selection.\n")
            args.best_epoch = None  # Fall back to automatic
    
    if args.best_epoch is None:  # Automatic selection
        if args.ms == 'online':
            best_id = np.argmax(scores[:, 2])
            best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                        'best normalized score avg': scores[best_id, 2],
                        'best normalized score std': scores[best_id, 3],
                        'best raw score avg': scores[best_id, 0],
                        'best raw score std': scores[best_id, 1]}
            with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
                f.write(json.dumps(best_res))
        elif args.ms == 'offline':
            # Refined Offline Model Selection:
            # 1. Metric 1: Diffusion (BC) Loss (scores[:, 6])
            # 2. Metric 2: Estimated Cost Proxy (scores[:, 10])
            
            estimated_costs = scores[:, 10]
            bc_losses = scores[:, 6]
            
            # Identify "safe" epochs where estimated cost <= threshold
            safe_mask = estimated_costs <= args.cost_threshold
            
            if np.any(safe_mask):
                # Among safe epochs, pick the one with the lowest BC loss
                safe_indices = np.where(safe_mask)[0]
                best_id = safe_indices[np.argmin(bc_losses[safe_mask])]
            else:
                # Fallback: Pick the epoch with the lowest estimated cost
                best_id = np.argmin(estimated_costs)
            
            best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1],
                    'estimated_cost': scores[best_id, 10],
                    'bc_loss': scores[best_id, 6]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

    

NORMALIZATION_CONSTANTS = {
    'OfflineCarCircle-v0': {'r_min': 3.484419822692871, 'r_max': 526.0},
    'OfflineDroneRun-v0': {'r_min': 10.55, 'r_max': 682.83},
    'OfflineBallRun-v0': {'r_min': 26.33, 'r_max': 1327.44},
    'OfflineAntRun-v0': {'r_min': 0.00176, 'r_max': 955.4818},
    # Safety Gymnasium
    'OfflinePointCircle1Gymnasium-v0': {'r_min': 20.067626953125, 'r_max': 61.72819519042969},
    'OfflinePointCircle2Gymnasium-v0': {'r_min': 20.478347778320312, 'r_max': 54.02184295654297},
    'OfflineAntVelocityGymnasium-v1': {'r_min': 6.1519012451171875, 'r_max': 2976.276611328125},
    'OfflineHalfCheetahVelocityGymnasium-v1': {'r_min': 5.7509765625, 'r_max': 2806.93310546875},
    'OfflineHopperVelocityGymnasium-v1': {'r_min': 37.05154800415039, 'r_max': 1911.396728515625},
    'OfflineSwimmerVelocityGymnasium-v1': {'r_min': 0.07114458084106445, 'r_max': 238.95831298828125},
}


def eval_policy(policy, env_name, seed, cost_threshold, eval_episodes=10):
    eval_env = gym.make(env_name)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        traj_cost = 0.
        state, info = eval_env.reset(seed=seed + 100)
        done = False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            traj_return += reward
            traj_cost += info.get('cost', 0)
        scores.append((traj_return, traj_cost))
    
   # NEW: Cost Calculation Section ---
    epsilon = 1e-8

    avg_reward = np.mean([s for s, c in scores])
    std_reward = np.std([s for s, c in scores])
    avg_cost = np.mean([c for s, c in scores]) # <-- It's calculated here
    std_cost = np.std([c for s, c in scores])

    r_min = NORMALIZATION_CONSTANTS.get(env_name, {}).get('r_min', 0)
    r_max = NORMALIZATION_CONSTANTS.get(env_name, {}).get('r_max', 1)

    # Standard D4RL-style 0-100 scale
    avg_norm_score = 100 * (avg_reward - r_min) / (r_max - r_min + epsilon)
    std_norm_score = 100 * std_reward / (r_max - r_min + epsilon)

     # Calculate normalized cost using the formula
    avg_norm_cost = avg_cost / (cost_threshold + epsilon)
    # --- END OF NEW SECTION ---

    print(f"--- Evaluation over {eval_episodes} episodes: Reward: {avg_reward:.2f} (Norm: {avg_norm_score:.2f}), Cost: {avg_cost:.2f} (Norm: {avg_norm_cost:.2f}) ---")
    return avg_reward, std_reward, avg_norm_score, std_norm_score, avg_cost, avg_norm_cost




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="OfflineDroneRun-v0", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--eval_episodes", default=100, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql', 'gaussian-ql']
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
    
    parser.add_argument("--gn", default=-1.0, type=float)
    parser.add_argument("--kp", default=None, type=float)
    parser.add_argument("--ki", default=None, type=float)
    parser.add_argument("--kd", default=None, type=float)
    parser.add_argument("--pid_delay_steps", default=0, type=int,
                        help="Explicit delay applied to the PID error signal before the controller update.")
    parser.add_argument("--cost_threshold_override", default=None, type=float, help="Override cost threshold for all tasks")
    parser.add_argument("--best_epoch", default=None, type=int, help="Manually specify which epoch to use as best model (overrides automatic selection)")

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'
   


    args.num_epochs = hyperparameters.get(args.env_name, {}).get('num_epochs', 1000)
    # Allow CLI overrides
    if args.eval_freq == 50 and args.env_name in hyperparameters:
        args.eval_freq = hyperparameters[args.env_name].get('eval_freq', 50)
    
    if args.eval_episodes == 100:
        args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr = hyperparameters.get(args.env_name, {}).get('lr', 3e-4)
    args.eta = hyperparameters.get(args.env_name, {}).get('eta', 1.0)
    args.max_q_backup = hyperparameters.get(args.env_name, {}).get('max_q_backup', False)
    args.reward_tune = hyperparameters.get(args.env_name, {}).get('reward_tune', 'no')
    args.gn = hyperparameters.get(args.env_name, {}).get('gn', -1.0)
    args.top_k = hyperparameters.get(args.env_name, {}).get('top_k', 1)
    args.cost_limit = hyperparameters.get(args.env_name, {}).get('cost_limit', 1.0)
    args.kp = args.kp if args.kp is not None else hyperparameters.get(args.env_name, {}).get('kp', 0.1)
    args.ki = args.ki if args.ki is not None else hyperparameters.get(args.env_name, {}).get('ki', 0.01)
    args.kd = args.kd if args.kd is not None else hyperparameters.get(args.env_name, {}).get('kd', 0.001)
    args.use_huber = hyperparameters.get(args.env_name, {}).get('use_huber', False)
    args.cost_threshold = hyperparameters.get(args.env_name, {}).get('cost_threshold', 1.0)
    
    if args.cost_threshold_override is not None:
        args.cost_threshold = args.cost_threshold_override
        args.cost_limit = args.cost_threshold_override

    # Setup Logging
    policy_family = "gaussian" if args.algo == "gaussian-ql" else "diffusion"
    algo_tag = args.algo if args.algo == "gaussian-ql" else f"diffusion-{args.algo}"
    file_name = f"{args.env_name}|{args.exp}|{algo_tag}|T-{args.T}"
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(policy_family=policy_family)
    variant.update(version=f"Diffusion-Policies-RL")

    env = gym.make(args.env_name)
    dataset = env.get_dataset()

#    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(dataset,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
