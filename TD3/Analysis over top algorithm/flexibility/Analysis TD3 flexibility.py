# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:31:44 2025

@author: rabie
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Directories ===
plots_dir = r"C:/Users/rabie/Dropbox/Cyber Security/TD3/results/Analysis over top algorithm/flexibility/comparison with normal version/plots"
os.makedirs(plots_dir, exist_ok=True)

normal_dir = r"C:/Users/rabie/Dropbox/Cyber Security/TD3/results/mean"
flex_dir   = r"C:/Users/rabie/Dropbox/Cyber Security/TD3/results/Analysis over top algorithm/flexibility/mean"

# === Load arrays ===
metrics = [
    ("scmean_avg", "Score Mean", "Score"),
    ("flag_allocation_avg", "Flagged Allocation", "Allocation"),
    ("attack_allocation_avg", "Attacked Allocation", "Allocation"),
    ("qos_avg", "QoS Mean", "QoS"),
    ("difmean_avg", "Difference Mean", "Difference")
]

# Dictionary for storing arrays
data_normal = {}
data_flex = {}

for metric, _, _ in metrics:
    data_normal[metric] = np.load(os.path.join(normal_dir, f"{metric}.npy"))
    data_flex[metric]   = np.load(os.path.join(flex_dir, f"{metric}.npy"))

# === Line plots comparison ===
colors = ["blue", "red"]  # blue = normal, red = flexible
labels = ["Normal", "Flexible"]

for metric, title, y_label in metrics:
    plt.figure(figsize=(12, 6))
    plt.plot(data_normal[metric], label=labels[0], color=colors[0], linewidth=1.8)
    plt.plot(data_flex[metric], label=labels[1], color=colors[1], linewidth=1.8)
    
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.title(f"{title} - Flexible vs Normal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, f"{metric}_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()

# === Normal distribution comparison for Attack Allocation ===
plt.figure(figsize=(10, 6))
for data, label, color in zip([data_normal["attack_allocation_avg"], data_flex["attack_allocation_avg"]], labels, colors):
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(mu - 4*std, mu + 4*std, 500)
    plt.plot(x, norm.pdf(x, mu, std), color=color, linewidth=2,
             label=f'{label} (μ={mu:.2f}, σ={std:.2f})')

plt.title("Normal Distribution - Attack Allocation")
plt.xlabel("Allocation Value")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "attack_allocation_normal_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# === Normal distribution comparison for Flagged Allocation ===
plt.figure(figsize=(10, 6))
for data, label, color in zip([data_normal["flag_allocation_avg"], data_flex["flag_allocation_avg"]], labels, colors):
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(mu - 4*std, mu + 4*std, 500)
    plt.plot(x, norm.pdf(x, mu, std), color=color, linewidth=2,
             label=f'{label} (μ={mu:.2f}, σ={std:.2f})')

plt.title("Normal Distribution - Flagged Allocation")
plt.xlabel("Allocation Value")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "flag_allocation_normal_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()
