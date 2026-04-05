"""
09_phylo_weights.py
FIXED: Derives phylogenetic priors directly from the self-calculated 
distance matrix rather than relying on Pei's (1949) ordinal estimates (Issue 16).
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration ---
MATRIX_PATH = Path("./distance_matrix.csv")
WEIGHTS_OUTPUT = Path("model/phylo_weights.pt")
# The neural network expects exactly this alphabetical order (Batch, Seq, 5, 24)
LANG_ORDER = ["French", "Italian", "Portuguese", "Romanian", "Spanish"]

def generate_phylo_weights(kappa=3.0):
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"Missing {MATRIX_PATH}. Run 06_phylogeny.py first.")
        
    print(f"Loading empirical distances from {MATRIX_PATH}...")
    df = pd.read_csv(MATRIX_PATH, index_col=0)
    
    # Extract the Latin column/row to get distances to the proto-language
    if "Latin" not in df.columns:
        raise ValueError("Latin not found in the distance matrix.")
        
    distances = df["Latin"].to_dict()
    
    # Filter for our 5 target languages and ensure correct ordering
    target_distances = {lang: distances[lang] for lang in LANG_ORDER}
    
    # Find the most conservative language (minimum distance) to anchor the decay
    min_dist = min(target_distances.values())
    conservative_lang = [k for k, v in target_distances.items() if v == min_dist][0]
    
    print(f"  Anchor language: {conservative_lang} (Distance: {min_dist:.4f})")
    
    matrix = np.ones((5, 24))
    
    for i, lang in enumerate(LANG_ORDER):
        d = target_distances[lang]
        
        # Laplacian Decay anchored to the most conservative witness
        # As distance increases from the baseline, trust decays exponentially.
        base_w = np.exp(-kappa * (d - min_dist))
        matrix[i, :] = base_w
        
        # Innovative Feature Gating: Dampen 'Color' (features 14-24) 
        # for branches with very high divergence scores (> 0.60)
        if d > 0.60:
            matrix[i, 14:24] *= 0.5 
            
        print(f"  {lang:<12} | Dist: {d:.4f} | Base Weight: {base_w:.4f}")

    return torch.tensor(matrix, dtype=torch.float32)

if __name__ == "__main__":
    WEIGHTS_OUTPUT.parent.mkdir(exist_ok=True)
    weights = generate_phylo_weights()
    torch.save(weights, WEIGHTS_OUTPUT)
    print(f"\nData-Driven Phylogenetic Weights Generated: {WEIGHTS_OUTPUT}")