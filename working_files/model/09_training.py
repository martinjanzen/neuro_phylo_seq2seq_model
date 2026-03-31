import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from model import NeuroPhyloLSTM, PhonologicalLoss

# --- Config & Visualization Setup ---
FILE_PATH = "vectorized_dataset.pt"
BATCH_SIZE = 8
MAX_EPOCHS = 200
PATIENCE = 15  
LEARNING_RATE = 0.001
REG_LAMBDA = 0.05 # used to be 0.05 
K_FOLDS = 5    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANGS = ["French", "Italian", "Portuguese", "Romanian", "Spanish"]
FEATURE_LABELS = [
    "syl", "son", "cons", "cont", "delrel", 
    "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor", "distr", 
    "hi", "lo", "back", "rnd", "vel", "lab", "labz", "low", "high", "ret"
]

def save_learned_heatmap(model, filename):
    weights = model.phylo_weights.detach().cpu().numpy()
    plt.figure(figsize=(16, 6))
    sns.heatmap(weights, annot=True, fmt=".2f", cmap="magma", 
                xticklabels=FEATURE_LABELS, yticklabels=LANGS)
    plt.title("Post-Training Learned Trust Matrix (Hybrid)")
    plt.axvline(x=5, color='cyan', linestyle='--', alpha=0.5)
    plt.axvline(x=14, color='cyan', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Learned heatmap saved to {filename}")

class RomanceDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]["X"].transpose(0, 1) 
        y = self.data[idx]["Y"]                 
        return x, y

def train_one_fold(fold_idx, train_loader, val_loader, mode='baseline'):
    model = NeuroPhyloLSTM().to(DEVICE)
    use_phylo = (mode == 'phylo')
    
    # Anchor to the current weights (which should be your Audit weights)
    audit_prior = copy.deepcopy(model.phylo_weights.data).to(DEVICE)

    if use_phylo:
        # Optimization: Differential Learning Rates
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if n != 'phylo_weights']},
            {'params': [model.phylo_weights], 'lr': LEARNING_RATE * 0.1}
        ], lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    profile = 'extreme' if mode == 'phylo' else mode
    criterion = PhonologicalLoss(weight_profile=profile)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    best_epoch = 0
    
    checkpoint_path = f"{mode}_fold_{fold_idx}.pth"
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x, use_phylo=use_phylo)
            main_loss = criterion(preds, y)
            
            if use_phylo:
                # Tikhonov Regularization (Anchor to Audit)
                reg_penalty = torch.pow(model.phylo_weights - audit_prior, 2).sum()
                loss = main_loss + (REG_LAMBDA * reg_penalty)
            else:
                loss = main_loss
                
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x, use_phylo=use_phylo), y).item()
        
        avg_val = val_loss / len(val_loader)
        
        # --- Restored Logging Statements ---
        if avg_val < best_loss:
            best_loss = avg_val
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(best_model_wts, checkpoint_path)
            print(f"Fold {fold_idx} | Epoch {epoch+1:3} | New Best Val Loss: {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            print(f"Fold {fold_idx} | Early Stopping at Epoch {epoch+1} | Reverting to Best Epoch: {best_epoch}")
            break
            
    model.load_state_dict(best_model_wts)
    return model, best_loss

def run_k_fold():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', 
                choices=['baseline', 'constrained', 'extreme', 'phylo'])
    args = parser.parse_args()
    
    dataset = RomanceDataset(FILE_PATH)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    best_overall_model = None
    min_overall_loss = float('inf')
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"--- Starting Fold {fold+1}/{K_FOLDS} ---")
        train_loader = DataLoader(Subset(dataset, train_ids), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_ids), batch_size=BATCH_SIZE, shuffle=False)
        
        model, best_val_loss = train_one_fold(fold+1, train_loader, val_loader, mode=args.mode)
        fold_results.append(best_val_loss)

        if best_val_loss < min_overall_loss:
            min_overall_loss = best_val_loss
            best_overall_model = model

    print(f"\nMean Validation Loss: {np.mean(fold_results):.4f}")
    
    # NEW: Save learned visualization from the best performing fold
    if args.mode == 'phylo' and best_overall_model:
        save_learned_heatmap(best_overall_model, "learned_phylo_heatmap.png")

if __name__ == "__main__":
    run_k_fold()