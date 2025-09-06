# Deep-reinforcement-learning-
# Deep Reinforcement Learning for Resilient Task–Server Allocation

This project investigates the application of **Deep Reinforcement Learning (DRL)** to dynamic resource allocation under server disruptions.  
Multiple DRL algorithms were implemented, trained, and compared in a simulated environment with varying disruption scenarios.  
A new **flexible allocation strategy** was also introduced to improve the system’s robustness.

---

## 📌 Overview

In this simulation, tasks must be allocated to a set of servers that may experience **attacks** (capacity reduced to zero) or **warnings/flags** (capacity halved).  
The learning agent must maximize **Quality of Service (QoS)** while minimizing **over-allocation** (measured as **DIF**).

We evaluate how different DRL methods adapt to disruptions and whether a targeted prioritization strategy can improve resilience.

---

## 🖥 Environment Setup

- **Observation Space (98 elements)**:
  1. **46 elements** – Task allocation amounts (14 tasks → 26 servers)
  2. **26 elements** – Attack indicators for each server (1 if attacked)
  3. **26 elements** – Flag indicators for each server (1 if flagged)

- **Action Space**:
  - **46 continuous actions**: allocation decisions for each task–server pair.

- **Disruption Model**:
  - Only **one disruption per step** (attack or flag)
  - Occurs in **~10% of steps**
  - Affects **only one server** at a time
  - Attacks and flags are **mutually exclusive**

---

## 🏆 DRL Algorithms Used

All agents were trained for **400,000 steps**, repeated **3 times** to reduce randomness.

- **TD3** (Twin Delayed Deep Deterministic Policy Gradient)
- **DDPG** (Deep Deterministic Policy Gradient)
- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor–Critic)

Performance metrics:
- **SC** – Composite score from QoS and DIF
- **QoS** – Normalized service quality
- **DIF** – Normalized over-allocation
- **Attack Allocation** – Tasks sent to blocked servers
- **Flag Allocation** – Tasks sent to degraded servers

---

## 📊 Key Results

**Performance Ranking (Best → Worst):**
1. **TD3**
2. **DDPG**
3. **PPO**
4. **SAC**

TD3 showed:
- Fast learning convergence  
- High QoS  
- Low over-allocation (DIF)  
- Reduced reliance on attacked/flagged servers

---

## 🔍 Extended Analysis: Varying Attack Probabilities

Attack probability **p** was tested at **0%, 5%, 10%, 15%, 20%** over **200,000 steps**.  
Findings:
- Higher **p** → More allocations to compromised servers
- TD3 maintained adaptability, but performance drop was inevitable with severe disruptions

---

## 🆕 Flexible Allocation Strategy

To improve robustness, a **server–task prioritization** method was added:
1. Identify tasks with **exclusive server options**
2. Count how many alternative servers each task has
3. Sort by **task coverage count** (descending)
4. Secondary sort by **server capacity** (descending)
5. Free adaptable servers for unique-task assignments

**Outcome**:
- Lower allocations to attacked (↓ ~ 4–6 tasks) and flagged servers (↓ ~ 2–3 tasks on average)
- Increased resilience under high-disruption scenarios

---

## 📂 Repository Contents



