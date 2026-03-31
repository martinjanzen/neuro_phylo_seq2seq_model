import torch
import numpy as np
import argparse
import os
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from model import NeuroPhyloLSTM
from torch.utils.data import Subset

# --- Configuration ---
DATA_PATH = "vectorized_dataset.pt"
MODES = ['baseline', 'extreme', 'phylo']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PanPhon 24-feature ordering
FEATURE_LABELS = [
    "syl", "son", "cons", "cont", "delrel", # Major Class (Aim 1 Focus)
    "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor", "distr", # Consonants (Aim 2 Focus)
    "hi", "lo", "back", "rnd", "vel", "lab", "labz", "low", "high", "ret" # Vowels/Manner
]

def quantize_prediction(tensor):
    """
    Maps continuous tanh outputs (-1 to 1) to discrete PanPhon classes (-1, 0, 1).
    Uses a 'Dead Zone' to force the model to commit to a feature bit.
    """
    val = tensor.detach().cpu().numpy()
    quantized = np.zeros_like(val)
    quantized[val > 0.33] = 1.0
    quantized[val < -0.33] = -1.0
    return quantized

def analyze_features():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    dataset = torch.load(DATA_PATH)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # mode -> feature -> list of scores per fold
    mode_results = {mode: {feat: [] for feat in FEATURE_LABELS} for mode in MODES}

    print(f"--- Feature-Level Sensitivity Analysis ---")
    print(f"Dataset: {len(dataset)} cognate sets | Device: {DEVICE}\n")

    for mode in MODES:
        use_phylo = (mode == 'phylo')

        for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            model = NeuroPhyloLSTM().to(DEVICE)
            weights_path = f"{mode}_fold_{fold_idx + 1}.pth"
            
            if not os.path.exists(weights_path):
                print(f"  [Skip] {weights_path} not found.")
                continue

            # map_location=DEVICE handles CPU/GPU cross-loading
            state_dict = torch.load(weights_path, map_location=DEVICE)
            
            # strict=False is the key here to handle the missing 'phylo_weights'
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)
            model.eval()
            
            print(f"  [Loaded] {weights_path} (Strict=False)")

            all_y_true = []
            all_y_pred = []

            with torch.no_grad():
                for i in val_ids:
                    # Prepare input: (1, seq, 120)
                    x = dataset[i]["X"].transpose(0, 1).unsqueeze(0).to(DEVICE)
                    y_true = dataset[i]["Y"].numpy()
                    y_pred_cont = model(x, use_phylo=use_phylo).squeeze(0)
                    y_pred = quantize_prediction(y_pred_cont)

                    # Masking: Only evaluate where target is not padding (all 0s)
                    mask = np.any(y_true != 0, axis=1)
                    if mask.any():
                        all_y_true.append(y_true[mask])
                        all_y_pred.append(y_pred[mask])

            # Concatenate all sequences in the fold validation set
            fold_true = np.concatenate(all_y_true, axis=0)
            fold_pred = np.concatenate(all_y_pred, axis=0)

            # Calculate Macro-F1 for each feature
            # Macro-F1 is used because features are sparse; it prevents 'Accuracy' 
            # from being inflated by the majority class.
            for f_idx, feat in enumerate(FEATURE_LABELS):
                score = f1_score(fold_true[:, f_idx], fold_pred[:, f_idx], 
                                 average='macro', zero_division=0)
                mode_results[mode][feat].append(score)

    # --- Comparative Feature Report ---
    header = f"{'Feature':<10} | {'Baseline':<10} | {'Extreme':<10} | {'Phylo':<10} | {'Winner'}"
    print(header)
    print("-" * len(header))

    for feat in FEATURE_LABELS:
        m_scores = {m: np.mean(mode_results[m][feat]) for m in MODES}
        winner = max(m_scores, key=m_scores.get).upper()
        
        # Color indicator: If Phylo or Extreme won, it validates an Aim.
        print(f"{feat:<10} | {m_scores['baseline']:<10.3f} | {m_scores['extreme']:<10.3f} | {m_scores['phylo']:<10.3f} | {winner}")

    print("\n--- Summary: Mean Global F1 ---")
    for mode in MODES:
        global_f1 = np.mean([np.mean(mode_results[mode][f]) for f in FEATURE_LABELS])
        print(f"{mode.upper():<10} | Mean F1: {global_f1:.4f}")

if __name__ == "__main__":
    analyze_features()