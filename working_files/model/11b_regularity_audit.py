import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
DATA_PATH = "vectorized_dataset.pt"
OUTPUT_PATH = "model/phylo_weights.pt"
VIS_PATH = "audit_heatmap.png"
LANGS = ["French", "Italian", "Portuguese", "Romanian", "Spanish"]
KAPPA = 2.0 

# PanPhon labels for visualization
FEATURE_LABELS = [
    "syl", "son", "cons", "cont", "delrel", # Major Class
    "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor", "distr", # Consonants
    "hi", "lo", "back", "rnd", "vel", "lab", "labz", "low", "high", "ret" # Vowels/Manner
]

def visualize_matrix(matrix, title, filename):
    plt.figure(figsize=(16, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=FEATURE_LABELS, yticklabels=LANGS)
    plt.title(title)
    plt.xlabel("PanPhon Features")
    plt.ylabel("Languages")
    # Add bucket dividers
    plt.axvline(x=5, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=14, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Heatmap saved to {filename}")

def run_audit():
    print(f"--- Starting Statistical Regularity Audit ---")
    dataset = torch.load(DATA_PATH)
    feature_errors = np.zeros((5, 24))
    counts = 0

    for entry in dataset:
        x = entry["X"].numpy()
        y = entry["Y"].numpy()
        for l in range(5):
            for f in range(24):
                mask = (y[:, f] != 0) 
                if mask.any():
                    err = np.mean((x[l, mask, f] - y[mask, f])**2)
                    feature_errors[l, f] += err
        counts += 1

    avg_mse = feature_errors / counts
    raw_weights = np.exp(-KAPPA * avg_mse)
    norm_weights = (raw_weights / np.sum(raw_weights)) * (5 * 24)

    os.makedirs("model", exist_ok=True)
    torch.save(torch.tensor(norm_weights, dtype=torch.float32), OUTPUT_PATH)
    
    # NEW: Generate Visualization
    visualize_matrix(norm_weights, "Audit Reliability Matrix (Statistical Prior)", VIS_PATH)
    
    print(f"Audit Complete. Reliability Matrix saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_audit()