# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:17:15 2025

@author: rabie
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plots_dir = r"C:/Users/rabie/Dropbox/Cyber Security/TD3/results/Analysis over top algorithm/All in one/plots"
os.makedirs(plots_dir, exist_ok=True)

base_dir = r"C:/Users/rabie/Dropbox/Cyber Security/TD3/results/Analysis over top algorithm"
probabilities = [0, 5, 10, 15, 20]

for p in probabilities:
    mean_path = os.path.join(base_dir, str(p), "mean")
    
    globals()[f"scmean_avg_{p}"] = np.load(os.path.join(mean_path, "scmean_avg.npy"))
    globals()[f"flag_allocation_avg_{p}"] = np.load(os.path.join(mean_path, "flag_allocation_avg.npy"))
    globals()[f"attack_allocation_avg_{p}"] = np.load(os.path.join(mean_path, "attack_allocation_avg.npy"))
    globals()[f"qos_avg_{p}"] = np.load(os.path.join(mean_path, "qos_avg.npy"))
    globals()[f"difmean_avg_{p}"] = np.load(os.path.join(mean_path, "difmean_avg.npy"))

print("Loaded variables:")
print([name for name in globals() if name.endswith(("_0", "_5", "_10", "_15", "_20"))])

colors = ['red', 'blue', 'green', 'orange', 'purple']
metrics = [
    ("scmean_avg", "Score Mean", "Score"),
    ("flag_allocation_avg", "Flagged Allocation", "Allocation"),
    ("attack_allocation_avg", "Attacked Allocation", "Allocation"),
    ("qos_avg", "QoS Mean", "QoS"),
    ("difmean_avg", "Difference Mean", "Difference"),
]
# Loop through each metric
for metric_name, title, y_label in metrics:
    plt.figure(figsize=(12, 6))
    
    for i, p in enumerate(probabilities):
        var_name = f"{metric_name}_{p}"
        data = globals()[var_name]
        plt.plot(data, label=f"p={p}%", color=colors[i], linewidth=1.8)
    
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.title(f"TD3 - {title} for Different Attack Probabilities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to specified directory
    filename = f"{metric_name}_all_probabilities.png"
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    

plt.figure(figsize=(10, 6))
for i, p in enumerate(probabilities):
    attack_data = globals()[f"attack_allocation_avg_{p}"]
    mu, std = np.mean(attack_data), np.std(attack_data)
    x = np.linspace(mu - 4*std, mu + 4*std, 500)
    plt.plot(x, norm.pdf(x, mu, std), color=colors[i], linewidth=2,
             label=f'p={p}% (μ={mu:.2f}, σ={std:.2f})')

plt.title("Normal Distribution - Attack Allocation (All Probabilities)")
plt.xlabel("Allocation Value")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

# Save
plt.savefig(os.path.join(plots_dir, "attack_allocation_normal_all_probs.png"), dpi=300, bbox_inches='tight')
plt.show()

# === Flagged Allocation ===
plt.figure(figsize=(10, 6))
for i, p in enumerate(probabilities):
    flag_data = globals()[f"flag_allocation_avg_{p}"]
    mu, std = np.mean(flag_data), np.std(flag_data)
    x = np.linspace(mu - 4*std, mu + 4*std, 500)
    plt.plot(x, norm.pdf(x, mu, std), color=colors[i], linewidth=2,
             label=f'p={p}% (μ={mu:.2f}, σ={std:.2f})')

plt.title("Normal Distribution - Flagged Allocation (All Probabilities)")
plt.xlabel("Allocation Value")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

# Save
plt.savefig(os.path.join(plots_dir, "flag_allocation_normal_all_probs.png"), dpi=300, bbox_inches='tight')
plt.show()