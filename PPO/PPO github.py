# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:25:29 2025

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
import os
from torch.distributions import Normal

# Data loading paths
data_path = r'C:\Users\rabie\Dropbox\New project\LargeScaleData1.xlsx'
coef = np.load(r'C:\Users\rabie\Dropbox\New project\data.npy')

# Process coefficient matrix
coef = coef.T 
coef_dic = {}
for row in range(0, len(coef)):
    coef_dic[row] = []
    
for row in range(0, len(coef)):
    for j in coef[row][coef[row].nonzero()]:
        coef_dic[row].append(j)  # Use in calculating QoS

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

class PPOMemory:
    """Memory buffer for storing PPO training experiences"""
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []
        self.batch_size = batch_size

    def recall(self):
        """Return all stored experiences"""
        return np.array(self.states),\
            np.array(self.new_states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.rewards),\
            np.array(self.dones)

    def generate_batches(self):
        """Generate random batches for training"""
        n_states = len(self.states)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, state, state_, action, probs, reward, done):
        """Store single experience tuple"""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.new_states.append(state_)

    def clear_memory(self):
        """Clear all stored experiences"""
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []

class ContinuousActorNetwork(nn.Module):
    """Actor network for continuous action spaces using Normal distribution"""
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=512, fc2_dims=512, chkpt_dir='models/'):
        super(ContinuousActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_continuous_ppo')
        
        # Network layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.alpha = nn.Linear(fc2_dims, n_actions)  # Mean parameters
        self.beta = nn.Linear(fc2_dims, n_actions)   # Standard deviation parameters

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """Forward pass to generate action distribution"""
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        # Generate parameters for Normal distribution
        alpha = F.relu(self.alpha(x)) + 1.0  # Mean (ensure positive)
        beta = F.relu(self.beta(x)) + 1.0    # Std deviation (ensure positive)
        dist = Normal(alpha, beta)
        return dist

    def save_checkpoint(self):
        """Save network weights"""
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load network weights"""
        self.load_state_dict(T.load(self.checkpoint_file))

class ContinuousCriticNetwork(nn.Module):
    """Critic network for value function estimation"""
    def __init__(self, input_dims, alpha,
                 fc1_dims=512, fc2_dims=512, chkpt_dir='models/'):
        super(ContinuousCriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_continuous_ppo')
        
        # Network layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)  # Value output

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """Forward pass to estimate state value"""
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)
        return v

    def save_checkpoint(self):
        """Save network weights"""
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load network weights"""
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    """PPO Agent for continuous action spaces"""
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=3e-4,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        # Hyperparameters
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 1e-3
        
        # Initialize networks
        self.actor = ContinuousActorNetwork(n_actions, input_dims, alpha)
        self.critic = ContinuousCriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, state_, action, probs, reward, done):
        """Store experience in memory"""
        self.memory.store_memory(state, state_, action, probs, reward, done)

    def save_models(self):
        """Save both actor and critic networks"""
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        """Load both actor and critic networks"""
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """Select action using current policy"""
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)
        return action.cpu().numpy().flatten(), probs.cpu().numpy().flatten()

    def calc_adv_and_returns(self, memories):
        """Calculate advantages and returns using GAE"""
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states)
            values_ = self.critic(new_states)
            deltas = r + self.gamma * values_ - values
            deltas = deltas.cpu().flatten().numpy()
            
            # Calculate advantages using GAE
            adv = [0]
            for dlt, mask in zip(deltas[::-1], dones[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = T.tensor(adv).float().unsqueeze(1).to(self.critic.device)
            
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)  # Normalize advantages
        return adv, returns

    def learn(self):
        """Update policy and value function using PPO"""
        # Retrieve experiences from memory
        state_arr, new_state_arr, action_arr, old_prob_arr, reward_arr, dones_arr = self.memory.recall()
        
        # Convert to tensors
        state_arr = T.tensor(state_arr, dtype=T.float).to(self.critic.device)
        action_arr = T.tensor(action_arr, dtype=T.float).to(self.critic.device)
        old_prob_arr = T.tensor(old_prob_arr, dtype=T.float).to(self.critic.device)
        new_state_arr = T.tensor(new_state_arr, dtype=T.float).to(self.critic.device)
        r = T.tensor(reward_arr, dtype=T.float).unsqueeze(1).to(self.critic.device)
        
        # Calculate advantages and returns
        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr, r, dones_arr))
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]

                # Calculate new policy probabilities
                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(new_probs.sum(1, keepdim=True) - old_probs.sum(1, keepdim=True))
                
                # PPO clipped objective
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                
                # Entropy bonus for exploration
                entropy = dist.entropy().sum(1, keepdims=True)
                
                # Actor loss
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)
                actor_loss -= self.entropy_coefficient * entropy
                
                # Update actor
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                # Update critic
                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()  

def action_adapter(a, max_a):
    """Adapt action from [0,1] to [-max_a, max_a]"""
    return 2 * (a - 0.5) * max_a

def clip_reward(x):
    """Clip reward to [-1, 1] range"""
    if x < -1:
        return -1
    elif x > 1:
        return 1
    else:
        return x

# Training parameters
max_action = 1
itrrr = 0  # Initial random exploration steps
n_games = 10
poason = 0.1

# Track best performance globally
top_rew = -9999
top_dif = -9999

# Main training loop
for yt in range(0, n_games):
    # Initialize PPO agent for each game
    agent = Agent(n_actions=46, batch_size=64, 
                  alpha=0.0003, n_epochs=10, 
                  input_dims=98)
    
    # Initialize tracking variables
    notnewttop = 0
    score_history = []
    dif_ar = []
    Qos_ar = []
    num_attacked_allocations_ar = []
    num_flaged_allocations_ar = []  
    attacked_allocations_mean = []
    flaged_allocations_mean = []
    sc = []
    scmean = []
    difmean = []
    QOSmean = []
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
            
    state = list(state) + list(poisson_dist) + list(flag)
    
    # Reset best performance trackers
    top_rew = -9999
    top_dif = -9999
    
    # Create nonzero indices mapping
    nonzind = {}
    for i in range(0, len(av)):
        nonzind[i] = list(np.nonzero(av[i])[0])
        
    cnttr = 0
    done = False
    score = 0
    state2 = deepcopy(state)
    state = np.array(state)
    state = state.reshape((1, 98))
    
    # Training episode loop
    while not done:
        # Action selection (PPO policy after initial exploration)
        if cnttr >= itrrr:
            act, prob = agent.choose_action(state2)
            action = action_adapter(act, max_action)
        else: 
            action = (np.random.random(46) - 0.5)

        new_action_dic = {}
        new_state = []
        
        # Apply action to current state
        for i in range(0, len(state2[0:46])):
            new_state.append(state2[i] + action[i])
            
        new2 = deepcopy(new_state) 
        capacity2 = deepcopy(capacity)
        
        # Apply attack effects to capacity
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
        reward = (Qos) - DIF 
        
        # Prepare next state
        all_values = []
        for task_id in sorted(new_state_dic.keys()):
            all_values.extend(new_state_dic[task_id])
        checkstate = np.array(all_values[:46])
        
        # Store metrics
        Qos_ar.append(Qos)
        dif_ar.append(dif)
        num_attacked_allocations_ar.append(num_attacked_allocations)
        num_flaged_allocations_ar.append(num_flaged_allocations)
        sc.append(reward)
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
        
        # Store experience and train
        r = clip_reward(reward)
        agent.remember(state2, new_state, act, prob, r, done)
        
        # PPO learning step (every 2048 steps)
        if cnttr > 1 and cnttr % 2048 == 0:
            agent.learn()
            
        score += reward
        state2 = deepcopy(new_state)
        uss2 = deepcopy(uss)
        
        # Save results and terminate after sufficient training
        if cnttr > 400000:
            np.save(r"C:\Users\rabie\Desktop\New project\PPO_{}_difmean".format(yt), difmean)
            np.save(r"C:\Users\rabie\Desktop\New project\PPO_{}_scmean".format(yt), scmean)
            np.save(r"C:\Users\rabie\Desktop\New project\PPO_{}_QOSmean".format(yt), QOSmean)
            np.save(r"C:\Users\rabie\Desktop\New project\PPO_{}_attacked_allocations".format(yt), attacked_allocations_mean)
            np.save(r"C:\Users\rabie\Desktop\New project\PPO_{}_flaged_allocations".format(yt), flaged_allocations_mean)
            break
