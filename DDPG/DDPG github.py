# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:20:21 2025

@author: Rabiee.Mohammad
"""

# Core imports
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import random

# Data loading paths
data_path = r'C:\Users\rabie\Dropbox\New project\LargeScaleData1.xlsx'
coef = np.load(r'C:\Users\rabie\Dropbox\New project\data.npy')

# Process coefficient matrix
coef = coef.T 
coef_dic = {}

# Create coefficient dictionary for QoS calculation
for row in range(0, len(coef)):
    coef_dic[row] = []
    
for row in range(0, len(coef)):
    for j in coef[row][coef[row].nonzero()]:
        coef_dic[row].append(j)    # Use in calculating QOS

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

class ReplayBuffer():
    """Experience replay buffer for storing and sampling training data"""
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuple to buffer"""
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Sample batch of experiences for training"""
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(np.asarray(X))
            y.append(np.asarray(Y))
            u.append(np.asarray(U))
            r.append(np.asarray(R))
            d.append(np.asarray(D))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1,1), np.array(d).reshape(-1,1)

# GPU/CPU device selection
device = T.device("cuda" if T.cuda.is_available() else "cpu")

class OUActionNoise():
    """Ornstein-Uhlenbeck noise process for exploration"""
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """Generate next noise sample"""
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """Reset noise process"""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
class ActorNet(nn.Module):
    """Actor network for policy approximation"""
    def __init__(self, state_size, action_size, hidden_size, action_max):
        super(ActorNet, self).__init__()
        self.dense_layer_1 = nn.Linear(state_size, hidden_size)
        self.dense_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)
        self.action_max = action_max
    
    def forward(self, x):
        """Forward pass through actor network"""
        x = T.clamp(x, -1.1, 1.1)  # Clamp input to prevent extreme values
        x = F.relu(self.dense_layer_1(x))
        x = F.relu(self.dense_layer_2(x))
        return T.tanh(self.output(x)) * self.action_max
    
class CriticNet(nn.Module):
    """Critic network for value function approximation"""
    def __init__(self, state_size, action_size, hidden_size):
        super(CriticNet, self).__init__()
        self.dense_layer_1 = nn.Linear(state_size + action_size, hidden_size)
        self.dense_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x, a):
        """Forward pass through critic network"""
        x = T.clamp(x, -1.1, 1.1)  # Clamp state input
        x = F.relu(self.dense_layer_1(T.cat((x, a), dim=1)))  # Concatenate state and action
        x = F.relu(self.dense_layer_2(x))
        return self.output(x)
        
class DDPGAgent():
    """Deep Deterministic Policy Gradient Agent"""
    def __init__(self, state_size, action_size, hidden_size, actor_lr, critic_lr, discount,
                 min_action, max_action, exploration_noise):
        self.action_size = action_size
        
        # Initialize networks
        self.actor = ActorNet(state_size, action_size, hidden_size, max_action).to(device)
        self.actor_target = ActorNet(state_size, action_size, hidden_size, max_action).to(device)
        self.critic = CriticNet(state_size, action_size, hidden_size).to(device)
        self.critic_target = CriticNet(state_size, action_size, hidden_size).to(device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Store hyperparameters
        self.discount = discount
        self.min_action = min_action
        self.max_action = max_action
        self.exploration_noise = exploration_noise
        
    def select_action(self, state):
        """Select action with exploration noise (schedule based on training step)"""
        with T.no_grad():
            input_state = T.FloatTensor(state).to(device)
            action = self.actor(input_state)
            action = action.detach().cpu().numpy()
            
            # Noise scheduling based on training steps
            if cnttr < 30000: 
                action = (action + np.random.normal(0., 0.3, 
                       size=self.action_size)).clip(self.min_action, self.max_action)   
            elif cnttr >= 30000 and cnttr < 100000:
                action = (action + np.random.normal(0., 0.05, 
                       size=self.action_size)).clip(self.min_action, self.max_action)   
            elif cnttr >= 100000 and cnttr < 200000:
                action = (action + np.random.normal(0., 0.01, 
                       size=self.action_size)).clip(self.min_action, self.max_action)   
            else:
                action = (action + np.random.normal(0., 0.001, 
                       size=self.action_size)).clip(self.min_action, self.max_action)   
        return action

    def train(self, replay_buffer, batch_size):
        """Train both actor and critic networks"""
        # Sample batch from replay buffer
        x0, x1, a, r, d = replay_buffer.sample(batch_size)
        state_batch = T.FloatTensor(x0).to(device)
        next_state_batch = T.FloatTensor(x1).to(device)
        action_batch = T.FloatTensor(a).to(device)
        reward_batch = T.FloatTensor(r).to(device)
        flipped_done_batch = T.FloatTensor(d).to(device) 

        # Calculate target values using target networks
        with T.no_grad():
            target_action = self.actor_target(next_state_batch).view(batch_size, -1)
            target_v = reward_batch + flipped_done_batch * self.discount * self.critic_target(next_state_batch, 
                                                                           target_action).view(batch_size, -1)
        
        # Get current Q-values from critic
        critic_v = self.critic(state_batch, action_batch).view(batch_size, -1)
        
        # Train critic network
        critic_loss = F.smooth_l1_loss(critic_v, target_v)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 
        
        # Train actor network
        train_action = self.actor(state_batch)
        actor_loss = -T.mean(self.critic(state_batch, train_action))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 
        
        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
        
    def update_target_network_soft(self, num_iter, update_every, update_tau=0.005):
        """Soft update of target networks"""
        if num_iter % update_every == 0:
            # Update critic target network
            for target_var, var in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_var.data.copy_((1.-update_tau) * target_var.data + (update_tau) * var.data)
            # Update actor target network
            for target_var, var in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_var.data.copy_((1.-update_tau) * target_var.data + (update_tau) * var.data)

def clip_reward(x):
    """Clip reward to [-1, 1] range"""
    if x < -1:
        return -1
    elif x > 1:
        return 1
    else:
        return x

# Hyperparameters
replay_size = 50000
replay_buffer = ReplayBuffer(max_size=replay_size)  
update_target_every = 1
tau = 0.005
action_size = 46
state_size = 98
min_action = -1
max_action = 1
discount = 0.97
batch_size = 128
exploration_noise = 0.01
hidden_size = 128  
actor_lr = 0.0005
critic_lr = 0.0005

# Training parameters
itrrr = 5000
poason = 0.1
n_games = 10


# Main training loop
for yt in range(0, n_games):
    # Initialize agent for each game
    agent = DDPGAgent(state_size=state_size, action_size=action_size, hidden_size=hidden_size, 
                      actor_lr=actor_lr, critic_lr=critic_lr, discount=discount, min_action=min_action,
                      max_action=max_action, exploration_noise=exploration_noise)

    # Initialize tracking variables
    notnewttop = 0
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
    stats_rewards_list = []
    stats_actor_loss, stats_critic_loss = [], []
    
    # Initialize state with random allocation
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
    
    # Apply initial attack if probability met
    if attack <= 0.1:
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
    
    # Create nonzero indices mapping
    nonzind = {}
    for i in range(0, len(av)):
        nonzind[i] = list(np.nonzero(av[i])[0])
        
    state = list(state) + list(poisson_dist) + list(flag)
    
    cnttr = 0
    done = False
    score = 0
    state = np.array(state)
    state = state.reshape((1, 98))
    state2 = deepcopy(state)
    
    # Training episode loop
    while not done: 
        # Action selection (random exploration for first 20k steps)
        if cnttr >= 20000:
            action = agent.select_action(state2)
        else: 
            action = (np.random.random(46) - 0.5)

        new_action_dic = {}
        new_state = []
        state2 = np.array(state2).reshape(-1).tolist()
        
        # Apply action to current state
        for i in range(0, len(state2[0:46])):
            new_state.append(state2[i] + action[i])
        new2 = deepcopy(new_state)   
        
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
        state_dic = {}
        actt = {}
        numbr = 0
        new_state_dic = {}
        
        # Partition actions by task
        for ii in range(0, len(action)):
            if numbr not in list(actt.keys()):
                actt[numbr] = [action[ii]]
                new_state_dic[numbr] = [new_state[ii]]
                state_dic[numbr] = [state2[ii]]
            else:
                actt[numbr].append(action[ii])
                new_state_dic[numbr].append(new_state[ii])
                state_dic[numbr].append(state2[ii])
                
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
        unallocated_penalty = 0
        
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
        reward = (Qos) - (DIF)  # Reward function
        
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
        new2 = state
        
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
            print("Games: ", yt, "dif: ", difmean[-1], "sc mean:", scmean[-1], "Top reward: ", top_rew, "Top Dif :", top_dif, "notnewttop :", notnewttop)
        
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
        
        # Store experience in replay buffer
        r = clip_reward(reward)
        replay_buffer.add((state2, new_state, action, r, 1-float(done)))
        
        # Training step
        if cnttr > 1 and cnttr % 20 == 0:
            actor_loss, critic_loss = agent.train(replay_buffer, batch_size)
            stats_actor_loss.append(actor_loss) 
            stats_critic_loss.append(critic_loss) 
            agent.update_target_network_soft(cnttr, update_target_every)
            
        score += reward
        state2 = deepcopy(new_state)
        uss2 = deepcopy(uss)
    
        # Save results and terminate after sufficient training
        if cnttr > 400000:
            np.save(r"C:\Users\rabie\Desktop\New project\TD3_{}_difmean".format(yt), difmean)
            np.save(r"C:\Users\rabie\Desktop\New project\TD3_{}_scmean".format(yt), scmean)
            np.save(r"C:\Users\rabie\Desktop\New project\TD3_{}_QOSmean".format(yt), QOSmean)
            np.save(r"C:\Users\rabie\Desktop\New project\TD3_{}_attacked_allocations".format(yt), attacked_allocations_mean)
            np.save(r"C:\Users\rabie\Desktop\New project\TD3_{}_flaged_allocations".format(yt), flaged_allocations_mean)
            break
