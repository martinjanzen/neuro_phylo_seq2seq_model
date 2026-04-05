"""
09_training.py
FIXED: Introduces standard deviation reporting and an automated 
paired t-test ('--mode compare') for rigorous statistical validation (Issue 10).
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from model import NeuroPhyloLSTM, PhonologicalLoss, RomanceDataset

# --- Config & Strategy 3 Parameters ---
FILE_PATH = "model/vectorized_dataset.pt"
BATCH_SIZE = 8
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.001
REG_LAMBDA = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANGS = ["French", "Italian", "Portuguese", "Romanian", "Spanish"]
FEATURE_LABELS = [
    "syl", "son", "cons", "cont", "delrel",
    "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor", "distr",
    "hi", "lo", "back", "rnd", "vel", "lab", "labz", "low", "high", "ret"
]

def save_learned_heatmap(model, filename):
    model.eval()
    with torch.no_grad():
        # Reflects the new Sigmoid bound (Issue 8A)
        w_effective = torch.sigmoid(model.phylo_weights) * 2.0
        weights = w_effective.cpu().numpy()

    plt.figure(figsize=(16, 6))
    sns.heatmap(weights, annot=True, fmt=".2f", cmap="magma", 
                xticklabels=FEATURE_LABELS, yticklabels=LANGS)
    plt.title("Effective Phylogenetic Trust Matrix (Sigmoid Bounded)")
    plt.axvline(x=5, color='cyan', linestyle='--', alpha=0.5)
    plt.axvline(x=14, color='cyan', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n[Logging] Effective heatmap saved to {filename}")

def train_fold(fold, train_loader, val_loader, mode='baseline', verbose=True):
    model = NeuroPhyloLSTM().to(DEVICE)
    use_phylo = (mode == 'phylo')

    weight_lr = 0.0005 if use_phylo else 0.001
    optimizer = optim.Adam([
        {'params': [p for n,p in model.named_parameters() if n != 'phylo_weights']},
        {'params': [model.phylo_weights], 'lr': weight_lr}
    ], lr=LEARNING_RATE)

    criterion = PhonologicalLoss(mode=mode)
    best_val_mse = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = f"{mode}_fold_{fold}.pth"

    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x, use_phylo=use_phylo)
            main_loss = criterion(preds, y)
            
            if use_phylo:
                reg_penalty = torch.pow(model.phylo_weights - model.phylo_prior, 2).sum()
                loss = main_loss + (REG_LAMBDA * reg_penalty)
            else:
                loss = main_loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        current_val_mse = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                v_preds = model(vx, use_phylo=use_phylo)
                current_val_mse += ((v_preds - vy)**2).mean().item()
        
        avg_mse = current_val_mse / len(val_loader)
        
        if avg_mse < best_val_mse:
            best_val_mse = avg_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            break
            
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        
    if verbose:
        print(f"    [{mode.capitalize()}] Fold {fold} Complete | Best MSE: {best_val_mse:.4f} (Epoch {best_epoch})")
        
    return model, best_val_mse

def run_compare(dataset):
    """CRITICAL FIX 10: Runs paired K-Fold to establish statistical significance."""
    print("--- Running Enhanced 5-Fold Comparison (Baseline vs Phylo) ---")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    baseline_mses = []
    phylo_mses = []
    best_phylo_model = None
    min_phylo_loss = float('inf')

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\nEvaluating Split {fold+1}/5...")
        train_loader = DataLoader(Subset(dataset, train_ids), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_ids), batch_size=BATCH_SIZE, shuffle=False)
        
        _, b_mse = train_fold(fold+1, train_loader, val_loader, mode='baseline')
        baseline_mses.append(b_mse)
        
        p_model, p_mse = train_fold(fold+1, train_loader, val_loader, mode='phylo')
        phylo_mses.append(p_mse)
        
        if p_mse < min_phylo_loss:
            min_phylo_loss = p_mse
            best_phylo_model = copy.deepcopy(p_model)

    b_mean, b_std = np.mean(baseline_mses), np.std(baseline_mses)
    p_mean, p_std = np.mean(phylo_mses), np.std(phylo_mses)
    
    # Paired T-Test
    t_stat, p_val = stats.ttest_rel(baseline_mses, phylo_mses)

    print("\n" + "="*50)
    print("STATISTICAL VALIDATION REPORT")
    print("="*50)
    print(f"Baseline MSE: {b_mean:.4f} ± {b_std:.4f}")
    print(f"Phylo MSE:    {p_mean:.4f} ± {p_std:.4f}")
    print("-" * 50)
    
    diff = b_mean - p_mean
    improvement = (diff / b_mean) * 100
    
    if p_mean < b_mean:
        print(f"Improvement:  {improvement:.2f}% better than baseline")
    else:
        print(f"Regression:   {abs(improvement):.2f}% worse than baseline")
        
    print(f"T-Statistic:  {t_stat:.3f}")
    print(f"P-Value:      {p_val:.4f}")
    print("-" * 50)
    
    if p_val < 0.05 and p_mean < b_mean:
        print("Verdict: The Phylo framework's improvement is STATISTICALLY SIGNIFICANT.")
    elif p_val < 0.05 and p_mean > b_mean:
        print("Verdict: The Phylo framework is significantly worse.")
    else:
        print("Verdict: The difference is NOT statistically significant (p >= 0.05).")
        
    if best_phylo_model:
        save_learned_heatmap(best_phylo_model, "strategy_3_heatmap.png")

def run_standard(dataset, mode):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_loader = DataLoader(Subset(dataset, train_ids), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_ids), batch_size=BATCH_SIZE, shuffle=False)
        _, mse = train_fold(fold+1, train_loader, val_loader, mode=mode)
        fold_results.append(mse)
        
    print(f"\nFinal {mode.capitalize()} MSE: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'phylo', 'compare'])
    args = parser.parse_args()

    dataset = RomanceDataset(FILE_PATH)
    
    if args.mode == 'compare':
        run_compare(dataset)
    else:
        run_standard(dataset, args.mode)