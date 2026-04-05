"""
11_audit.py
FIXED: Applies a boolean mask during accuracy computation so that alignment 
padding vectors (gaps) do not artificially inflate tier accuracy (Issue 22).
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import NeuroPhyloLSTM, RomanceDataset
import os

# --- Configuration ---
FILE_PATH = "model/vectorized_dataset.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.33 

def run_audit():
    dataset = RomanceDataset(FILE_PATH)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)

    modes = ["baseline", "phylo"]
    results = {}

    print(f"{'Tier':<15} | {'Metric':<10} | {'Baseline':<10} | {'Phylo':<10} | {'Gain'}")
    print("-" * 65)

    for mode in modes:
        model = NeuroPhyloLSTM().to(DEVICE)
        weight_path = f"{mode}_fold_1.pth"
        
        if not os.path.exists(weight_path):
            print(f"Skipping {mode}: {weight_path} not found. Run 10_train.py first.")
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.eval()
        
        with torch.no_grad():
            preds = model(x, use_phylo=(mode == 'phylo'))
            
            # CRITICAL FIX 22: Generate the valid phoneme mask
            # True if the absolute sum of the 24 features is > 0 (meaning it's not a gap)
            mask = (y.abs().sum(-1) > 0)
            
            # Compute MSE only on valid phonemes
            sq_error = (preds - y)**2
            mse = sq_error[mask].mean().item()
            
            # Categorical Metric: Snapping
            snapped = torch.where(preds > THRESHOLD, 1.0, 
                      torch.where(preds < -THRESHOLD, -1.0, 0.0))
            
            # Compute Tier Accuracy only on valid phonemes
            skel_acc = (snapped[:,:,0:5] == y[:,:,0:5])[mask].float().mean().item()
            art_acc  = (snapped[:,:,5:14] == y[:,:,5:14])[mask].float().mean().item()
            col_acc  = (snapped[:,:,14:24] == y[:,:,14:24])[mask].float().mean().item()
            
            results[mode] = {
                "mse": mse,
                "skeleton": skel_acc,
                "articulation": art_acc,
                "color": col_acc
            }

    if "baseline" in results and "phylo" in results:
        b, p = results["baseline"], results["phylo"]
        
        for tier in ["skeleton", "articulation", "color"]:
            b_val, p_val = b[tier], p[tier]
            gain = ((p_val - b_val) / b_val) * 100 if b_val > 0 else 0.0
            print(f"{tier.capitalize():<15} | Accuracy   | {b_val:<10.2%} | {p_val:<10.2%} | {gain:+.1f}%")
            
        mse_gain = ((p["mse"] - b["mse"]) / b["mse"]) * 100 if b["mse"] > 0 else 0.0
        # For MSE, negative gain is better (error went down), so we flip the sign for display
        print(f"{'Global':<15} | Tanh MSE   | {b['mse']:<10.4f} | {p['mse']:<10.4f} | {-mse_gain:+.1f}%")

if __name__ == "__main__":
    run_audit()