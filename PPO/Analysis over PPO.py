# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 22:05:26 2025

@author: rabie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Load a single file

scmean1 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_1_scmean.npy")
scmean2 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_2_scmean.npy")
scmean3 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_3_scmean.npy")
flag_allocation1 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_1_flaged_allocations.npy")
flag_allocation2 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_2_flaged_allocations.npy")
flag_allocation3 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_3_flaged_allocations.npy")
attack_allocation1 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_1_attacked_allocations.npy")
attack_allocation2 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_2_attacked_allocations.npy")
attack_allocation3 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_3_attacked_allocations.npy")
qos1 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_1_QOSmean.npy")
qos2 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_2_QOSmean.npy")
qos3 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_3_QOSmean.npy")
difmean1 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_1_difmean.npy")
difmean2 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_2_difmean.npy")
difmean3 = np.load("C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/PPO_3_difmean.npy")



scmean_avg            = np.mean([scmean1, scmean2, scmean3], axis=0)
flag_allocation_avg   = np.mean([flag_allocation1, flag_allocation2, flag_allocation3], axis=0)
attack_allocation_avg = np.mean([attack_allocation1, attack_allocation2, attack_allocation3], axis=0)
qos_avg               = np.mean([qos1, qos2, qos3], axis=0)
difmean_avg           = np.mean([difmean1, difmean2, difmean3], axis=0)

# Save results
np.save(r"C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/mean/scmean_avg.npy", scmean_avg)
np.save(r"C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/mean/flag_allocation_avg.npy", flag_allocation_avg)
np.save(r"C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/mean/attack_allocation_avg.npy", attack_allocation_avg)
np.save(r"C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/mean/qos_avg.npy", qos_avg)
np.save(r"C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/mean/difmean_avg.npy", difmean_avg)

base_path = r"C:/Users/rabiee.mohammad/Dropbox/New project/PPO/results/mean/"

scmean_avg            = np.load(f"{base_path}scmean_avg.npy")
flag_allocation_avg   = np.load(f"{base_path}flag_allocation_avg.npy")
attack_allocation_avg = np.load(f"{base_path}attack_allocation_avg.npy")
qos_avg               = np.load(f"{base_path}qos_avg.npy")
difmean_avg           = np.load(f"{base_path}difmean_avg.npy")

# (title, data, color)
plots = [
    ("Score Mean", scmean_avg, 'red'),
    ("Flagged Allocation", flag_allocation_avg, 'blue'),
    ("Attacked Allocation", attack_allocation_avg, 'green'),
    ("QoS Mean", qos_avg, 'orange'),
    ("Dif Mean", difmean_avg, 'purple')
]

# Create one figure per dataset
for title, data, color in plots:
    plt.figure(figsize=(12, 6))
    plt.plot(data, color=color)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"PPO - {title}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# === 1. Fit parameters ===
mu_attack, sigma_attack = np.mean(attack_allocation_avg), np.std(attack_allocation_avg)
mu_flag, sigma_flag     = np.mean(flag_allocation_avg), np.std(flag_allocation_avg)

print(f"Attack Allocation → mean: {mu_attack:.4f}, std dev: {sigma_attack:.4f}")
print(f"Flag Allocation   → mean: {mu_flag:.4f}, std dev: {sigma_flag:.4f}")

# === 2. Plot for Attack Allocation ===
plt.figure(figsize=(12, 5))
count, bins, _ = plt.hist(attack_allocation_avg, bins=50, density=True, alpha=0.6, color='green', label="Data histogram")

# Normal distribution curve
x = np.linspace(min(bins), max(bins), 500)
plt.plot(x, norm.pdf(x, mu_attack, sigma_attack), 'k', lw=2, label=f"N({mu_attack:.2f}, {sigma_attack:.2f}²)")

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("PPO - Attack Allocation - Normal Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 3. Plot for Flag Allocation ===
plt.figure(figsize=(12, 5))
count, bins, _ = plt.hist(flag_allocation_avg, bins=50, density=True, alpha=0.6, color='blue', label="Data histogram")

# Normal distribution curve
x = np.linspace(min(bins), max(bins), 500)
plt.plot(x, norm.pdf(x, mu_flag, sigma_flag), 'k', lw=2, label=f"N({mu_flag:.2f}, {sigma_flag:.2f}²)")

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("PPO - Flagged Allocation - Normal Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# --- Metrics dictionary ---
metrics = {
    "SC Mean": [scmean1, scmean2, scmean3],
    "Dif Mean": [difmean1, difmean2, difmean3],
    "QoS Mean": [qos1, qos2, qos3]
}

colors = {
    "SC Mean": 'red',
    "Dif Mean": 'purple',
    "QoS Mean": 'orange'
}

# --- Create one figure per metric ---
for name, arrays in metrics.items():
    data = np.vstack(arrays)                      # shape: (3, N)
    avg  = np.mean(data, axis=0)
    minv = np.min(data, axis=0)
    maxv = np.max(data, axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(avg, color=colors[name], label=name, linewidth=2)
    plt.fill_between(
        range(len(avg)),
        minv, maxv,
        color=colors[name],
        alpha=0.2
    )

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"PPO - {name} with Min–Max Shading")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()