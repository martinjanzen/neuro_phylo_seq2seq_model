import argparse
import torch
import panphon
import codecs
import numpy as np
import csv
from pathlib import Path
from Levenshtein import distance
from model import NeuroPhyloLSTM
from sklearn.model_selection import KFold

# --- Setup ---
ft = panphon.FeatureTable()
DEVICE = torch.device("cpu")
char_map = {'+': 1.0, '-': -1.0, '0': 0.0}
EXCLUDE_CHARS = {'˧', '˦', '˨', '˩', '˥', '?', ' ', 'ː', 'ʲ', 'ʷ', 'ˤ'}
SUMMARY_FILE = "ablation_summary.txt"

print("Loading PanPhon lookup and dataset...")
seg_csv = Path(panphon.__file__).parent / 'data' / 'ipa_all.csv'
SEG_DATA = {}
with open(seg_csv, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    FEATURE_NAMES = [col for col in reader.fieldnames if col != 'ipa']
    for row in reader:
        ipa = row['ipa']
        vec = np.array([char_map.get(row[feat], 0.0) for feat in FEATURE_NAMES], dtype=np.float32)
        SEG_DATA[ipa] = vec

def vector_to_ipa(pred_vec, seg_data, weights, threshold=4.0):
    """
    Nearest-neighbor search using Weighted Euclidean Distance.
    """
    # 1. Feature Clamping to the articulatory space [-1, 1]
    pred_np = np.clip(pred_vec.detach().cpu().numpy(), -1.0, 1.0)
    
    # 2. Energy Gate: We use the unweighted absolute sum here.
    # This represents the total 'physical' signal regardless of feature importance.
    if np.abs(pred_np).sum() < threshold:
        return None
    
    best_seg = ""
    min_dist = float('inf')
    
    # Convert weights to numpy for vectorized distance calculation
    w_np = weights.cpu().numpy()

    for seg, seg_vec in seg_data.items():
        # 3. Weighted Squared Euclidean Distance: (p - q)^2 * w
        # This prioritizes Major Class and Place features over Manner/Source.
        dist = np.sum(w_np * (pred_np - seg_vec) ** 2)
        
        if dist < min_dist:
            min_dist = dist
            best_seg = seg
            
    return best_seg

def get_clean_ipa(vectors, seg_data, weights, is_gold=False):
    """
    Iterates through a sequence of vectors and converts to a filtered IPA string.
    """
    chars = []
    # Gold standard targets have no thresholding
    threshold = 0.0 if is_gold else 4.0
    
    for v in vectors:
        char = vector_to_ipa(v, seg_data, weights, threshold)
        
        # First-Gap Termination Strategy: stop if energy drops below threshold
        if char is None and not is_gold: 
            break 
            
        # Filter out non-phonemic markers (tones, etc.) to stabilize PED
        if char and char not in EXCLUDE_CHARS:
            chars.append(char)
            
    return "".join(chars)

def run_aggregate_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', 
                        choices=['baseline', 'constrained', 'extreme', 'phylo'],
                        help="Mode must match the weights used during training.")
    args = parser.parse_args()

    use_phylo = (args.mode == 'phylo')

    # 1. Define Inference Weights to match training profiles
    # This is critical for academic consistency between Loss and Eval
    eval_weights = torch.ones(24).to(DEVICE)
    if args.mode == 'constrained':
        eval_weights[0:5] = 10.0   # Major Class
        eval_weights[5:13] = 5.0   # Place
        eval_weights[13:] = 2.0    # Manner
    elif args.mode in ['extreme', 'phylo']:
        eval_weights[0:5] = 100.0
        eval_weights[5:13] = 10.0
        eval_weights[13:] = 1.0
    
    # 2. Setup Data and K-Fold
    data = torch.load("vectorized_dataset.pt")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_peds = []

    print(f"--- EVALUATING {args.mode.upper()} (Weighted Eval Strategy) ---")
    print(f"{'Fold':<10} | {'Mean PED'}")
    print("-" * 25)

    # 3. Aggregate Loop
    for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(data)):
        model = NeuroPhyloLSTM().to(DEVICE)
        weights_path = f"{args.mode}_fold_{fold_idx + 1}.pth"
        
        try:
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
        except FileNotFoundError:
            print(f"Fold {fold_idx+1:<5} | MISSING WEIGHTS")
            continue

        fold_total_ped = 0
        with torch.no_grad():
            for i in val_ids:
                # Prepare input: (batch, seq, num_langs, feat_dim) -> (batch, seq, 120)
                x = data[i]["X"].transpose(0, 1).unsqueeze(0).to(DEVICE)
                y_true = data[i]["Y"].to(DEVICE)
                y_pred = model(x, use_phylo=use_phylo).squeeze(0)

                # Use Weighted Euclidean Distance for decoding
                true_ipa = get_clean_ipa(y_true, SEG_DATA, eval_weights, is_gold=True)
                pred_ipa = get_clean_ipa(y_pred, SEG_DATA, eval_weights, is_gold=False)
                
                fold_total_ped += distance(true_ipa, pred_ipa)

        mean_val_ped = fold_total_ped / len(val_ids)
        fold_peds.append(mean_val_ped)
        print(f"Fold {fold_idx + 1:<5} | {mean_val_ped:.2f}")

    # 4. Final Aggregation & Logging
    if fold_peds:
        final_mean = np.mean(fold_peds)
        final_std = np.std(fold_peds)
        
        with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
            f.write(f"MODE: {args.mode.upper()} (Weighted Eval) | Mean PED: {final_mean:.4f} (±{final_std:.4f})\n")
        
        print("-" * 25)
        print(f"AGGREGATE {args.mode.upper()} PED: {final_mean:.2f} (±{final_std:.2f})")

if __name__ == "__main__":
    run_aggregate_evaluation()