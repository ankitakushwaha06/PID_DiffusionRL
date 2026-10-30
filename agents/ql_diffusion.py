# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def q_max(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.max(q1, q2)


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 cost_limit=1.0,
                 kp=0.1,
                 ki=0.01,
                 kd=0.001,
                 pid_delay_steps=0,
                 use_huber=False):
        
        self.kp, self.ki, self.kd = kp, ki, kd  
        self.use_huber = use_huber
        self.pid_delay_steps = int(pid_delay_steps)
        if self.pid_delay_steps < 0:
            raise ValueError("pid_delay_steps must be non-negative")
        self.integral = 0.0
        self.prev_error = 0.0
        # Explicitly delayed controller feedback for theory-alignment sweeps.
        # A delay of d means the PID update at step t uses e_{t-d}; before d
        # measurements exist, the controller receives a neutral zero signal.
        self.pid_error_history = deque([0.0] * self.pid_delay_steps)
        self.penalty_lambda = 0.0 # Initial penalty coefficient
        

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        #creating cost critic
        self.cost_critic = Critic(state_dim, action_dim).to(device)
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=1e-4)





        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)           
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=lr_maxt, eta_min=0.)
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup


        # --- Store cost hyperparameters ---
        self.cost_limit = cost_limit

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def _controller_error(self, current_error):
        """Return the violation signal visible to the PID controller."""
        current_error = float(current_error)
        if self.pid_delay_steps == 0:
            return current_error

        self.pid_error_history.append(current_error)
        return self.pid_error_history.popleft()

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):


        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': [],'cost_critic_loss': [], 'cost_penalty_loss': [], 'pid_lambda': [], 'estimated_cost': [], 'pid_error': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, cost, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
           
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            # Clamping reward target to prevent 'e power' explosions
            target_q = torch.clamp(reward + not_done * self.discount * target_q, -2000.0, 2000.0).detach()

            if self.use_huber:
                critic_loss = F.huber_loss(current_q1, target_q) + F.huber_loss(current_q2, target_q)
            else:
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()
            """ Cost-Critic Training """
            ## Cost-Critic Training (Now mirrors Q-Critic) ##
            current_qc1, current_qc2 = self.cost_critic(state, action)
            
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_qc1, target_qc2 = self.cost_critic_target(next_state_rpt, next_action_rpt)
                target_qc1 = target_qc1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_qc2 = target_qc2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_qc = torch.max(target_qc1, target_qc2)
            else:
                next_action = self.ema_model(next_state)
                target_qc1, target_qc2 = self.cost_critic_target(next_state, next_action)
                target_qc = torch.max(target_qc1, target_qc2)
            
            # Enable Undiscounted Cost Critics to align Proxy and Real hits
            # Max possible cost is now the episode length (approx 1000)
            max_p_cost = 1000.0
            target_qc = torch.clamp(cost + not_done * 1.0 * target_qc, 0.0, max_p_cost).detach()

            if self.use_huber:
                cost_critic_loss = F.huber_loss(current_qc1, target_qc) + F.huber_loss(current_qc2, target_qc)
            else:
                cost_critic_loss = F.mse_loss(current_qc1, target_qc) + F.mse_loss(current_qc2, target_qc)

            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.cost_critic_optimizer.step()

            ## PID Controller Update for Lambda ##
            # 1. Sample new actions from the current policy
            new_action = self.actor(state)
            
            # 2. Compute violation e_t = (E[Q_c^max(s, a)] - kappa) / (kappa + eps)
            # Normalized error makes PID gains (kp, ki, kd) more universal
            predicted_cost = self.cost_critic.q_max(state, new_action)
            current_error = (predicted_cost.mean().item() - self.cost_limit) / (self.cost_limit + 1e-6)
            error = self._controller_error(current_error)
            
            # 3. Update PID state with Windup Protection (relaxed for exploration)
            self.integral = np.clip(self.integral + error, -10.0, 10.0)
            derivative = error - self.prev_error
            self.prev_error = error
            
            # 4. Update lambda_t (Lagrangian multiplier)
            # Very relaxed cap since costs are already near zero - allow exploration
            # Slower update rate (0.05 instead of 0.1) for more gradual penalty adjustment
            pid_update = self.kp * error + self.ki * self.integral + self.kd * derivative
            self.penalty_lambda = np.clip(self.penalty_lambda + 0.05 * pid_update, 0.0, 10.0)

            #""" Policy Training """
            bc_loss = self.actor.loss(action, state)
            
            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            
            # Combined Actor Loss
            # Normalize cost penalty by the limit to align its scale with q_loss (~1.0)
            cost_penalty = torch.clamp((predicted_cost - self.cost_limit) / (self.cost_limit + 1e-6), min=0.0)
            actor_loss = bc_loss + self.eta * q_loss + self.penalty_lambda * cost_penalty.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                 nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())
            metric['cost_critic_loss'].append(cost_critic_loss.item())
            metric['pid_lambda'].append(self.penalty_lambda)
            metric['cost_penalty_loss'].append(cost_penalty.mean().item())
            metric['estimated_cost'].append(predicted_cost.mean().item())
            metric['pid_error'].append(error)



        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
        return metric

    
    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        
        with torch.no_grad():
            # 1. Sample a batch of candidate actions
            action = self.actor.sample(state_rpt)
            
            # 2. Predict the cost and reward (Q-value) for each candidate
            predicted_costs = self.cost_critic_target.q_max(state_rpt, action).flatten()
            predicted_q_values = self.critic_target.q_min(state_rpt, action).flatten()
            
            # 3. Filter for safety: create a mask of actions where predicted cost is below the limit
            safe_mask = predicted_costs < self.cost_limit
            
            # 4. Choose the best action
            if safe_mask.any():
                # If there are any safe actions, find the one with the best Q-value
                low_value = torch.tensor(-1e6, dtype=predicted_q_values.dtype, device=predicted_q_values.device)
                safe_q_values = torch.where(safe_mask, predicted_q_values, low_value) # Make unsafe actions undesirable
                best_action_idx = torch.argmax(safe_q_values)
            else:
                # Fallback: if no actions are safe, choose the one with the lowest cost
                best_action_idx = torch.argmin(predicted_costs)
                
        return action[best_action_idx].cpu().data.numpy().flatten()

     
    
    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
            torch.save(self.cost_critic.state_dict(), f'{dir}/cost_critic_{id}.pth') # <-- ADD THIS
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')
            torch.save(self.cost_critic.state_dict(), f'{dir}/cost_critic.pth') # <-- ADD THIS

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth', map_location=self.device))
            self.cost_critic.load_state_dict(torch.load(f'{dir}/cost_critic_{id}.pth', map_location=self.device)) # <-- ADD THIS
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth', map_location=self.device))
            self.cost_critic.load_state_dict(torch.load(f'{dir}/cost_critic.pth', map_location=self.device)) # <-- ADD THIS
