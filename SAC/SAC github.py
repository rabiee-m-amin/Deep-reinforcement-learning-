# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:30:24 2025

@author: Rabiee.Mohammad
"""

# -*- coding: utf-8 -*-
"""
SAC (Soft Actor-Critic) Implementation for Resource Allocation Under Attacks
Author: Rabiee.Mohammad
"""

# Standard library imports
import os
import random
from collections import namedtuple, deque
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
from torch.distributions import Normal, MultivariateNormal

# Data loading and preprocessing
data_path = r'C:\Users\rabie\Dropbox\New project GPU\LargeScaleData1.xlsx'
coef = np.load(r'C:\Users\rabie\Dropbox\New project GPU\data.npy')
coef = coef.T 

# Process coefficient matrix for QoS calculation
coef_dic = {}
for row in range(0, len(coef)):
    coef_dic[row] = []
    
for row in range(0, len(coef)):
    for j in coef[row][coef[row].nonzero()]:
        coef_dic[row].append(j)

# Parameter configuration
params_list = [0.4, 0.4, 0.1, 1]
alpha = (1/9) * params_list[0]         
beta = (1/46) * params_list[1]        
gamma = (1/46) * params_list[2]        
omega = (1/14500) * params_list[3]     

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

num_state = 98

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def hidden_init(layer):
    """Initialize hidden layer weights using fan-in rule"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model for SAC with continuous actions"""

    def __init__(self, state_size, action_size, seed, hidden_size=32, init_w=3e-3, 
                 log_std_min=-20, log_std_max=2):
        """Initialize Actor network parameters and build model"""
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Separate heads for mean and log standard deviation
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def reset_parameters(self):
        """Reset network parameters with proper initialization"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_linear.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Forward pass to get action distribution parameters"""
        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        """Sample action and compute log probability"""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device) 
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob
        
    def get_action(self, state):
        """Returns the action based on a squashed Gaussian policy"""
        mu, log_std = self.forward(state)
        std = log_std.exp()    
        dist = Normal(0, 1)                   
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std).to(device)
        return action[0]

class Critic(nn.Module):
    """Critic (Value) Model for SAC"""

    def __init__(self, state_size, action_size, seed, hidden_size=32):
        """Initialize Critic network parameters and build model"""
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset network parameters with proper initialization"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs to Q-values"""
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples for off-policy learning"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object"""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)

class Agent():
    """SAC Agent that interacts with and learns from the environment"""
    
    def __init__(self, state_size, action_size, random_seed, hidden_size, action_prior="uniform"):
        """Initialize an SAC Agent object"""
        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)
        self.seed = random_seed
        
        # SAC temperature parameter (entropy regularization)
        self.target_entropy = -action_size
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR) 
        self._action_prior = action_prior
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)     
        
        # Twin Critic Networks (Q-functions)
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        
        # Target Networks for stable learning
        self.critic1_target = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC, weight_decay=0) 

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, step):
        """Save experience in replay memory and trigger learning"""
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(step, experiences, GAMMA)
            
    def act(self, state):
        """Returns actions for given state as per current policy"""
        state = torch.from_numpy(state).float().to(device)
        action = self.actor_local.get_action(state).detach()
        return action

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences
        
        # Update critics using twin Q-learning
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        
        # Calculate Q-targets with entropy regularization
        if FIXED_ALPHA == None:
            self.alpha = torch.tensor(self.alpha, device=device)
            Q_targets = rewards.to(device) + (gamma * (1 - dones.to(device)) * (Q_target_next.to(device) - self.alpha.to(device) * log_pis_next.squeeze(0).to(device)))
        else:
            Q_targets = rewards.to(device) + (gamma * (1 - dones.to(device)) * (Q_target_next.to(device) - FIXED_ALPHA * log_pis_next.squeeze(0).to(device)))
        
        # Compute current Q-values
        Q_1 = self.critic1(states, actions).to(device)
        Q_2 = self.critic2(states, actions).to(device)
        critic1_loss = 0.5 * F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5 * F.mse_loss(Q_2, Q_targets.detach())
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        if step % d == 0:
            # Update actor and temperature parameter
            if FIXED_ALPHA == None:
                alpha = torch.exp(self.log_alpha)
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = -(self.log_alpha.to(device) * (log_pis.to(device) + self.target_entropy).detach().to(device)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = alpha
                
                # Apply action prior
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size, device=device), 
                                                    scale_tril=torch.ones(self.action_size, device=device).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred).to(device)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = torch.tensor(0.0, device=device) 

                actor_loss = (alpha.to(device) * log_pis.squeeze(0).to(device) - self.critic1(states, actions_pred.squeeze(0)).to(device) - policy_prior_log_probs).mean()
            else:
                actions_pred, log_pis = self.actor_local.evaluate(states)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), 
                                                    scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred).to(device)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                actor_loss = (FIXED_ALPHA * log_pis.squeeze(0).to(device) - self.critic1(states, actions_pred.squeeze(0)).to(device) - policy_prior_log_probs).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self.soft_update(self.critic1, self.critic1_target, TAU)
            self.soft_update(self.critic2, self.critic2_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def clip_reward(x):
    """Clip reward to [-1, 1] range for stable learning"""
    if x < -1:
        return -1
    elif x > 1:
        return 1
    else:
        return x

# SAC Hyperparameters
max_action = 1
action_size = 46
state_size = 98
min_action = -1

# Training parameters
TAU = 5e-3                    # Soft update parameter
BATCH_SIZE = 2                # Batch size for learning
LR_ACTOR = 0.0001            # Actor learning rate
LR_CRITIC = 0.0001           # Critic learning rate
GAMMA = 0.99                 # Discount factor
FIXED_ALPHA = None           # Fixed temperature (None for learnable)
HIDDEN_SIZE = 256            # Hidden layer size
BUFFER_SIZE = int(1e6)       # Replay buffer size

# Training configuration
itrrr = 2000                 # Initial exploration steps
n_games = 6                  # Number of training runs
poason = 0.1                 # Attack probability

# Track best performance globally
top_rew = -9999
top_dif = -9999

# Main training loop
for yt in range(1, n_games):
    # Initialize performance tracking
    score_history = []
    dif_ar = []
    Qos_ar = []
    num_attacked_allocations_ar = []
    num_flaged_allocations_ar = []
    sc = []
    scmean = []
    difmean = []
    QOSmean = []
    attacked_allocations_mean = []
    flaged_allocations_mean = []

    # Initialize SAC agent
    agent = Agent(state_size=state_size, action_size=action_size, 
                  random_seed=random.randint(0, 999), hidden_size=HIDDEN_SIZE, 
                  action_prior="uniform")

    notnewttop = 0
    
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
    state = np.random.rand(46)
    poisson_or_flag = np.random.randint(0, 2)
    attack = random.uniform(0, 1)
    
    # Apply initial attack
    if attack <= 0.1:
        if poisson_or_flag == 0:
            poisson_dist = np.zeros(26)
            targetp = random.randint(0, 25)
            poisson_dist[targetp] = 1
        else:
            flag = np.zeros(26)
            targetf = random.randint(0, 25)
            flag[targetf] = 1

    state = list(state) + list(poisson_dist) + list(flag)
    
    # Create server-task mapping
    nonzind = {}
    for i in range(0, len(av)):
        nonzind[i] = list(np.nonzero(av[i])[0])
    
    cnttr = 0
    done = False
    score = 0
    state = np.array(state)
    state = state.reshape((1, 98))
    state2 = deepcopy(state)
    
    # Training episode loop
    while not done:
        # Action selection
        if cnttr >= 0:
            action_v = agent.act(state2)
            action = np.clip(action_v.cpu(), -1, 1)
            
            # Check for NaN values
            for i in action:
                if torch.isnan(i):
                    print("NaN detected in action")
        else: 
            action = (np.random.random(46) - 0.5)

        # Process action and update state
        new_state = []
        state2 = state2.reshape(-1).tolist()
        for i in range(0, len(state2[0:46])):
            new_state.append(state2[i] + action[i])
        
        # Apply attack effects to capacity
        capacity2 = deepcopy(capacity)
        if attack <= 0.1:
            if poisson_or_flag == 0:
                for i in range(0, len(poisson_dist)):
                    if poisson_dist[i] == 1:
                        capacity2[i] = 0
                        break
            else:
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
                
            if ii == 46:
                break
            
            if ii + 1 == limits[numbr]:
                numbr += 1
                
        # Normalize allocations for each task
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
        
        if attack <= 0.1:
            if poisson_or_flag == 0:
                # Count attacked allocations
                if len(attacked_servers) > 0:
                    for task_id, server_indices in nonzind.items():
                        total_demand = demand2.get(task_id, 0)
                        for alloc_fraction, server_idx in zip(new_state_dic[task_id], server_indices):
                            if alloc_fraction > 0.01 and server_idx in attacked_servers:
                                num_attacked_allocations += alloc_fraction * total_demand
            else:
                # Count flagged allocations
                if len(flaged_servers) > 0:
                    for task_id, server_indices in nonzind.items():
                        total_demand = demand2.get(task_id, 0)
                        for alloc_fraction, server_idx in zip(new_state_dic[task_id], server_indices):
                            if alloc_fraction > 0.01 and server_idx in flaged_servers:
                                num_flaged_allocations += alloc_fraction * total_demand
        
        # Calculate capacity usage and QoS
        for i in nonzind:
            for j in range(0, len(nonzind[i])):
                cap_usage[nonzind[i][j]] += new_state_dic[i][j] * demand2[i]
                
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
                
        dif = 0
        for i in cap_usage:
            if cap_usage[i] > capacity2[i]:
                dif += abs(cap_usage[i] - capacity2[i])
                
        # Normalize metrics and calculate reward
        Qos = Qos / 5000  # Normalization
        DIF = dif / 500   # Normalization
        reward = (Qos) - (DIF)
        
        # Prepare next state
        all_values = []
        for task_id in sorted(new_state_dic.keys()):
            all_values.extend(new_state_dic[task_id])
        checkstate = np.array(all_values[:46])
        
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
        
        if attack <= 0.1:
            if poisson_or_flag == 0:
                targetp = random.randint(0, 25)
                poisson_dist[targetp] = 1
            else:
                targetf = random.randint(0, 25)
                flag[targetf] = 1
                
        new_state = [value for sublist in new_state_dic.values() for value in sublist] + list(poisson_dist) + list(flag)
        
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
            print("Games: ", yt, "DIF: ", difmean[-1], "sc mean:", scmean[-1], "Top reward: ", top_rew, "Top Dif :", top_dif)
            
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
        state2 = np.array(state2)
        state2 = state2.reshape((1, 98))
        
        new_state = np.array(new_state)
        new_state = new_state.reshape((1, 98))
        
        agent.step(state2, action_v.cpu(), reward, new_state, done, cnttr)

        score += reward
        state2 = deepcopy(new_state)
        uss2 = deepcopy(uss)
        
        # Save results and terminate after sufficient training
        if cnttr > 400000:
            np.save(r"C:\Users\rabie\OneDrive\Desktop\new project\SAC_{}_difmean".format(yt), difmean)
            np.save(r"C:\Users\rabie\OneDrive\Desktop\new project\SAC_{}_scmean".format(yt), scmean)
            np.save(r"C:\Users\rabie\OneDrive\Desktop\new project\SAC_{}_QOSmean".format(yt), QOSmean)
            np.save(r"C:\Users\rabie\OneDrive\Desktop\new project\SAC_{}_attacked_allocations".format(yt), attacked_allocations_mean)
            np.save(r"C:\Users\rabie\OneDrive\Desktop\new project\SAC_{}_flaged_allocations".format(yt), flaged_allocations_mean)
            break
