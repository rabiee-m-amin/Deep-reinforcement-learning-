# -*- coding: utf-8 -*-
"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) Implementation for Resource Allocation Under Attacks
Enhanced version with improved flexibility and attack modeling
Author: Rabiee.Mohammad
"""

# Standard library imports
import sys
import os
import random
from copy import deepcopy

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Data loading and preprocessing
data_path = r'C:\Users\rabie\Dropbox\Cyber Security\TD3\results\Analysis over top algorithm\flexibility\LargeScaleData1.xlsx'
coef = np.load(r'C:\Users\rabie\Dropbox\Cyber Security\TD3\results\Analysis over top algorithm\flexibility\data.npy')
coef = coef.T 

# Process coefficient matrix for QoS calculation
coef_dic = {}
for row in range(0, len(coef)):
    coef_dic[row] = []
    
for row in range(0, len(coef)):
    for j in coef[row][coef[row].nonzero()]:
        coef_dic[row].append(j)

# Parameter configuration for objective function weights
params_list = [0.4, 0.4, 0.1, 1]
alpha = (1/9) * params_list[0]         # Weight for first component
beta = (1/46) * params_list[1]         # Weight for second component
gamma = (1/46) * params_list[2]        # Weight for third component
omega = (1/14500) * params_list[3]     # Weight for fourth component

# Load data from Excel sheets
demand = pd.read_excel(data_path, sheet_name='demand').T.to_dict()[0]
demand2 = deepcopy(demand)
capacity = pd.read_excel(data_path, sheet_name='capacity').T.to_dict()[0]

# Process availability matrix
avs = pd.read_excel(data_path, sheet_name='av', header=None).values
av = avs.T
choices = {}
for row in range(0, len(av)):
    choices[row] = len(av[row].nonzero()[0])

num_choices = sum(choices.values())

# Calculate cumulative limits for action space partitioning
limits = []
memory = 0
for i in choices:
    length = choices[i]
    limits.append(length + memory)
    memory = length + memory
    
st_demand = list(demand.values())
num_demand = len(st_demand)

class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples for off-policy learning"""

    def __init__(self, max_size=1e6):
        """Initialize replay buffer with maximum capacity"""
        self.storage = []
        self.max_size = max_size
        self.ptr = 0  # Pointer for circular buffer implementation

    def add(self, transition):
        """Add a new experience tuple to memory"""
        if len(self.storage) == self.max_size:
            # Overwrite old experiences when buffer is full
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            # Add new experience when buffer is not full
            self.storage.append(transition)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory"""
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        
        for i in ind: 
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.asarray(state))
            batch_next_states.append(np.asarray(next_state))
            batch_actions.append(np.asarray(action))
            batch_rewards.append(np.asarray(reward))
            batch_dones.append(np.asarray(done))
            
        return (np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), 
                np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OUActionNoise():
    """Ornstein-Uhlenbeck process for action space exploration"""
    
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        """Initialize OU noise parameters"""
        self.theta = theta  # Mean reversion rate
        self.mu = mu       # Long-term mean
        self.sigma = sigma # Volatility
        self.dt = dt       # Time step
        self.x0 = x0       # Initial value
        self.reset()

    def __call__(self):
        """Generate next noise sample using OU process"""
        x = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + 
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x

    def reset(self):
        """Reset the noise process to initial state"""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class Actor(nn.Module):
    """Actor (Policy) Network for TD3"""
    
    def __init__(self, state_dim, action_dim, max_action):
        """Initialize Actor network with specified dimensions"""
        super(Actor, self).__init__()
        self.h1 = nn.Linear(state_dim, 400)
        self.h2 = nn.Linear(400, 300)
        self.h3 = nn.Linear(300, action_dim)
        self.max_action = max_action  # For action clipping

    def forward(self, x):
        """Forward pass through the Actor network"""
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.max_action * torch.tanh(self.h3(x))  # Scale to action range
        return x

class Critic(nn.Module):
    """Twin Critic Networks for TD3"""
    
    def __init__(self, state_dim, action_dim):
        """Initialize twin Critic networks"""
        super(Critic, self).__init__()
        # First critic network
        self.h1 = nn.Linear(state_dim + action_dim, 400)
        self.h2 = nn.Linear(400, 300)
        self.h3 = nn.Linear(300, 1)
        
        # Second critic network
        self.h4 = nn.Linear(state_dim + action_dim, 400)
        self.h5 = nn.Linear(400, 300)
        self.h6 = nn.Linear(300, 1)

    def forward(self, x, u):
        """Forward pass through both critic networks"""
        xu = torch.cat([x, u], 1)  # Concatenate state and action
        
        # First critic forward pass
        x1 = F.relu(self.h1(xu))
        x1 = F.relu(self.h2(x1))
        x1 = self.h3(x1)
        
        # Second critic forward pass
        x2 = F.relu(self.h4(xu))
        x2 = F.relu(self.h5(x2))
        x2 = self.h6(x2)
        
        return x1, x2

    def Q1(self, x, u):
        """Get Q-value from first critic only (used for actor updates)"""
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.h1(xu))
        x1 = F.relu(self.h2(x1))
        x1 = self.h3(x1)
        return x1

class TD3(object):
    """TD3 (Twin Delayed DDPG) Algorithm Implementation"""
    
    def __init__(self, state_dim, action_dim, max_action):
        """Initialize TD3 agent with actor and critic networks"""
        # Actor networks (current and target)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        
        # Critic networks (current and target)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        
        self.max_action = max_action

    def select_action(self, state):
        """Select action using the current policy"""
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=512, discount=0.99, tau=0.005, 
              policy_noise=0.2, noise_clip=0.3, policy_freq=2):
        """Train the TD3 agent using experience replay"""
        
        for it in range(iterations):
            # Sample batch of transitions from replay buffer
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            
            # Select next action using target actor with added noise
            next_action = self.actor_target(next_state)
            
            # Add clipped noise to next action (target policy smoothing)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # Compute target Q-values using twin critics (clipped double Q-learning)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            
            # Get current Q-values from both critics
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Compute critic loss (MSE)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Update critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates (every policy_freq iterations)
            if it % policy_freq == 0:
                # Compute actor loss (maximize Q1)
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Soft update target networks (Polyak averaging)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        """Save trained model"""
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        """Load pre-trained model"""
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

def clip_reward(x):
    """Clip reward to [-1, 1] range for stable learning"""
    if x < -1:
        return -1
    elif x > 1:
        return 1
    else:
        return x

# TD3 Hyperparameters
max_action = 1
action_size = 49  # Updated action size
state_size = 101  # Updated state size including attack vectors
min_action = -1

# Training parameters
replay_size = 50000
start_training_after = 10001
update_target_every = 1
tau = 0.0005
episodes = 5000
discount = 0.99
batch_size = 256
exploration_noise = 0.1
exploration_end = 0.01
exp_decay = 0.99999
hidden_size = 64
actor_lr = 0.0003
critic_lr = 0.0003
reward_scale = 0.01

# Attack probability scenarios
probbbbs = [0.1]  # Single attack probability for this run
itrrr = 2000      # Initial exploration steps
n_games = 4       # Number of training runs
poason = 0.1      # Base attack probability parameter

# Main training loop
for p in range(0, len(probbbbs)):
    for yt in range(1, n_games):
        # Initialize TD3 agent
        agent = TD3(101, 49, 1)
        replay_buffer = ReplayBuffer(max_size=replay_size)

        # Initialize performance tracking
        notnewttop = 0
        score_history = []
        dif_ar = []
        Qos_ar = []
        sc = []
        num_attacked_allocations_ar = []
        num_flaged_allocations_ar = []
        scmean = []
        difmean = []
        QOSmean = []
        attacked_allocations_mean = []
        flaged_allocations_mean = []
        stats_rewards_list = []
        stats_actor_loss, stats_critic_loss = [], []

        # Initialize random allocation state
        st = []
        for i in choices:
            num = choices[i]
            desired_sum = demand[i]
            random_values = np.random.rand(num)
            scaled_values = (random_values / np.sum(random_values)) * desired_sum
            st.append(scaled_values)

        flattened_st = [round(item, 3) for sublist in st for item in sublist]
        
        # Initialize attack vectors
        poisson_dist = [0] * 26
        flag = [0] * 26
        state = np.random.rand(49)  # Updated state size
        poisson_or_flag = np.random.randint(0, 2)
        attack = random.uniform(0, 1)
        
        # Apply initial attack based on probability
        if attack < probbbbs[p]:
            if poisson_or_flag == 0:
                poisson_dist = np.zeros(26)
                targetp = random.randint(0, 25)
                poisson_dist[targetp] = 1
            else:
                flag = np.zeros(26)
                targetf = random.randint(0, 25)
                flag[targetf] = 1

        # Track best performance
        top_rew = -9999
        top_dif = -9999
        
        # Create server-task mapping
        nonzind = {}
        for i in range(0, len(av)):
            nonzind[i] = list(np.nonzero(av[i])[0])

        # Combine state with attack vectors
        state = list(state) + list(poisson_dist) + list(flag)

        cnttr = 0
        done = False
        score = 0
        state = np.array(state)
        state = state.reshape((1, 101))
        state2 = deepcopy(state)

        # Training episode loop
        while not done:
            # Action selection with exploration
            if cnttr >= itrrr:
                action = agent.select_action(np.array(state2))
                action = (action + np.random.normal(0, exploration_noise, size=49)).clip(-1, 1)
                
                # Check for NaN values in actions
                for i in action:
                    if np.isnan(i):
                        print("NaN detected in action")
            else: 
                # Random exploration during initial steps
                action = (np.random.random(49) - 0.5)

            # Process action and update state
            new_state = []
            state2 = np.array(state2).reshape(-1).tolist()
            for i in range(0, len(state2[0:49])):
                new_state.append(state2[i] + action[i])

            # Apply attack effects to capacity
            capacity2 = deepcopy(capacity)
            if attack <= probbbbs[p]:
                if poisson_or_flag == 0:
                    # Poisson attack - complete server failure
                    for i in range(0, len(poisson_dist)):
                        if poisson_dist[i] == 1:
                            capacity2[i] = 0
                            break
                else:
                    # Flag attack - partial capacity reduction
                    for i in range(0, len(flag)):
                        if flag[i] == 1:
                            capacity2[i] = capacity[i] * 0.5
                            break

            # Initialize metrics
            Qos = 0
            new_state_dic = {}
            numbr = 0

            # Partition actions by task
            for ii in range(0, len(action)):
                if numbr not in list(new_state_dic.keys()):
                    new_state_dic[numbr] = [new_state[ii]]
                else:
                    new_state_dic[numbr].append(new_state[ii])
                    
                if ii == 49:
                    break
                
                if ii + 1 == limits[numbr]:
                    numbr += 1

            # Normalize allocations for each task (ensure valid probability distributions)
            for i in new_state_dic:
                new_state_dic[i] = [(value + 1e-9) for value in new_state_dic[i]]
                new_state_dic[i] = [value / sum(new_state_dic[i]) for value in new_state_dic[i]]
                new_state_dic[i] = np.clip(new_state_dic[i], 0, 1)
                new_state_dic[i] = [(value + 1e-9) for value in new_state_dic[i]]
                new_state_dic[i] = [value / sum(new_state_dic[i]) for value in new_state_dic[i]]
                new_state_dic[i] = np.clip(new_state_dic[i], 0, 1)
                
                # Check for NaN values
                for jj in new_state_dic[i]:
                    if np.isnan(jj):
                        print("NaN detected in new_state_dic")

            # Initialize capacity usage tracking
            cap_usage = {}
            for i in capacity:
                cap_usage[i] = 0

            # Find attacked/flagged servers
            poisson_dist = np.array(poisson_dist)
            attacked_servers = np.where(poisson_dist > 0)[0]
            
            flag = np.array(flag)
            flaged_servers = np.where(flag > 0)[0]

            # Count allocations to compromised servers
            num_attacked_allocations = 0
            num_flaged_allocations = 0
            
            if attack < probbbbs[p]:
                if poisson_or_flag == 0:
                    # Count attacked allocations (weighted by demand)
                    if len(attacked_servers) > 0:
                        for task_id, server_indices in nonzind.items():
                            total_demand = demand2.get(task_id, 0)
                            for alloc_fraction, server_idx in zip(new_state_dic[task_id], server_indices):
                                if alloc_fraction > 0.01 and server_idx in attacked_servers:
                                    num_attacked_allocations += alloc_fraction * total_demand
                else:
                    # Count flagged allocations (weighted by demand)
                    if len(flaged_servers) > 0:
                        for task_id, server_indices in nonzind.items():
                            total_demand = demand2.get(task_id, 0)
                            for alloc_fraction, server_idx in zip(new_state_dic[task_id], server_indices):
                                if alloc_fraction > 0.01 and server_idx in flaged_servers:
                                    num_flaged_allocations += alloc_fraction * total_demand

            # Calculate capacity usage
            for i in nonzind:
                for j in range(0, len(nonzind[i])):
                    cap_usage[nonzind[i][j]] += new_state_dic[i][j] * demand2[i]

            # Track server usage for each task
            resp_deman = {}
            for i in range(0, 14):
                resp_deman[i] = 0

            uss = {}
            for j in new_state_dic:
                uss[j] = []
                for z in range(0, len(new_state_dic[j])):
                    if new_state_dic[j][z] > 0.01:
                        uss[j].append(nonzind[j][z])

            # Calculate QoS and capacity violations
            for i in nonzind:
                for j in range(0, len(nonzind[i])):
                    resp_deman[i] += new_state_dic[i][j]
                    Qos += new_state_dic[i][j] * coef_dic[i][j] * demand2[i]

            # Calculate capacity violations
            dif = 0
            for i in cap_usage:
                if cap_usage[i] > capacity2[i]:
                    dif += abs(cap_usage[i] - capacity2[i])

            # Normalize metrics and calculate reward
            Qos = Qos / 5000  # Normalization
            DIF = dif / 500   # Normalization
            reward = (Qos) - (DIF)

            # Prepare next state vector
            all_values = []
            for task_id in sorted(new_state_dic.keys()):
                all_values.extend(new_state_dic[task_id])
            checkstate = np.array(all_values[:49])

            # Store metrics
            Qos_ar.append(Qos)
            dif_ar.append(DIF)
            sc.append(reward)
            num_attacked_allocations_ar.append(num_attacked_allocations)
            num_flaged_allocations_ar.append(num_flaged_allocations)
            notnewttop += 1

            # Generate new attack for next step
            attack = random.uniform(0, 1)
            poisson_or_flag = np.random.randint(0, 2)
            poisson_dist = np.zeros(26)
            flag = np.zeros(26)

            if attack <= probbbbs[p]:
                if poisson_or_flag == 0:
                    targetp = random.randint(0, 25)
                    poisson_dist[targetp] = 1
                else:
                    targetf = random.randint(0, 25)
                    flag[targetf] = 1

            # Prepare next state
            new_state = ([value for sublist in new_state_dic.values() for value in sublist] + 
                        list(poisson_dist) + list(flag))

            # Track best performance
            if cnttr > 10000:
                if scmean[-1] > top_rew:
                    top_rew = scmean[-1]
                    notnewttop = 0
                    
                if -(sum(dif_ar[-1000:]) / len(dif_ar[-1000:])) > top_dif:
                    top_dif = -(sum(dif_ar[-1000:]) / len(dif_ar[-1000:]))

            # Calculate running averages
            if cnttr > 500:
                scmean.append(sum(sc[-10000:]) / len(sc[-10000:]))
                difmean.append((sum(dif_ar[-10000:]) / len(dif_ar[-10000:])))
                QOSmean.append((sum(Qos_ar[-10000:]) / len(Qos_ar[-10000:])))
                attacked_allocations_mean.append(np.mean(num_attacked_allocations_ar[-10000:]))
                flaged_allocations_mean.append(np.mean(num_flaged_allocations_ar[-10000:]))

            # Progress reporting
            if cnttr % 10000 == 0 and cnttr > 1000:
                print(f"Games: {yt}, DIF: {difmean[-1]}, sc mean: {scmean[-1]}, "
                      f"Top reward: {top_rew}, Top Dif: {top_dif}")

            # Plotting progress
            if cnttr % 10000 == 0:
                # Plot average scenario cost
                plt.plot(np.linspace(0, len(scmean), len(scmean)), scmean)
                plt.title("Scenario Cost Mean")
                plt.xlabel("Episode")
                plt.ylabel("Cost")
                plt.show()
                
                # Plot average QoS
                plt.plot(np.linspace(0, len(QOSmean), len(QOSmean)), QOSmean)
                plt.title("QoS Mean")
                plt.xlabel("Episode")
                plt.ylabel("Quality of Service")
                plt.show()
                
                # Plot difference mean
                plt.plot(np.linspace(0, len(difmean), len(difmean)), difmean)
                plt.title("Difference Mean")
                plt.xlabel("Episode")
                plt.ylabel("Difference Value")
                plt.show()
                
                # Plot attacked allocations average
                plt.plot(np.linspace(0, len(attacked_allocations_mean), len(attacked_allocations_mean)),
                         attacked_allocations_mean)
                plt.title("Attacked Allocations")
                plt.xlabel("Episode")
                plt.ylabel("Count per Step")
                plt.show()
                
                # Plot flagged allocations average
                plt.plot(np.linspace(0, len(flaged_allocations_mean), len(flaged_allocations_mean)),
                         flaged_allocations_mean)
                plt.title("Flagged Allocations")
                plt.xlabel("Episode")
                plt.ylabel("Count per Step")
                plt.show()

            cnttr += 1
            done = False

            # Store experience and train
            r = clip_reward(reward)
            replay_buffer.add((state2, new_state, action, r, 1-float(done)))

            # Train agent periodically
            if cnttr > 1 and cnttr % 256 == 0:
                agent.train(replay_buffer, 20, batch_size, 0.99, 0.005, 0.2, 0.3, 2)

            score += reward
            state2 = deepcopy(new_state)
            uss2 = deepcopy(uss)

            # Save results and terminate after sufficient training
            if cnttr > 400000:
                np.save(fr"C:\Users\rabie\Desktop\New project\TD3_{yt}_difmean.npy", difmean)
                np.save(fr"C:\Users\rabie\Desktop\New project\TD3_{yt}_scmean.npy", scmean)
                np.save(fr"C:\Users\rabie\Desktop\New project\TD3_{yt}_QOSmean.npy", QOSmean)
                np.save(fr"C:\Users\rabie\Desktop\New project\TD3_{yt}_attacked_allocations.npy", 
                       attacked_allocations_mean)
                np.save(fr"C:\Users\rabie\Desktop\New project\TD3_{yt}_flaged_allocations.npy", 
                       flaged_allocations_mean)
                break
