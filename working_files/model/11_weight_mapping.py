import torch
import numpy as np

# --- STRATEGY A: Literature-Based Divergence ---
# Source: Pei, M. A. (1949). "A New Methodology for Determining Polyglot Classification."
dist_from_latin = {
    "Italian": 0.12, "Spanish": 0.20, "Romanian": 0.235, "Portuguese": 0.31, "French": 0.44
}

def generate_phylo_feature_matrix(kappa=2.0):
    """
    Generates a 5x24 Granular Reliability Matrix.
    Academically backed by Feature Geometry (Clements, 1985).
    """
    langs = ["French", "Italian", "Portuguese", "Romanian", "Spanish"]
    num_langs = len(langs)
    feat_dim = 24
    
    # Initialize Matrix
    matrix = np.ones((num_langs, feat_dim))
    
    # Feature Group Indices (Based on standard PanPhon 24-bit vector)
    # Group 1: Major Class (Structure/Skeleton)
    MAJOR_CLASS = slice(0, 5) 
    # Group 2: Consonantal Place/Manner 
    CONSONANTS = slice(5, 14)
    # Group 3: Vowels, Dorsal, and Laryngeal features
    VOWELS = slice(14, 24)

    for i, lang in enumerate(langs):
        d = dist_from_latin[lang]
        
        # Base trust based on Laplacian Decay (Felsenstein, 1985)
        base_w = np.exp(-kappa * (d - 0.12))
        
        # FEATURE-SPECIFIC TUNING
        if lang in ["French", "Portuguese"]:
            # French/Portuguese are Vowel Innovators. 
            # We preserve the structural skeleton but dampen the noisy vowel signal.
            matrix[i, MAJOR_CLASS] = base_w * 1.0  # Structure remains reliable
            matrix[i, CONSONANTS]  = base_w * 0.7  # Moderate trust for consonants
            matrix[i, VOWELS]      = base_w * 0.4  # High innovative noise; low trust
        else:
            # Conservative witnesses remain largely uniform
            matrix[i, :] = base_w 
            
    # Normalization: Ensure the sum of the matrix remains consistent (5 * 24)
    # This prevents 'Gradient Vanishing' by keeping input signal volume stable.
    matrix = (matrix / np.sum(matrix)) * (num_langs * feat_dim)
    
    return torch.tensor(matrix, dtype=torch.float32)

if __name__ == "__main__":
    print("--- Generating 5x24 Granular Reliability Matrix ---")
    weights = generate_phylo_feature_matrix(kappa=2.0)
    
    # Review first language (French) as a sanity check
    print(f"French Major Class Trust (Indices 0-4): {weights[0, 0:5].mean():.4f}")
    print(f"French Vowel/Manner Trust (Indices 14-23): {weights[0, 14:24].mean():.4f}")
    
    torch.save(weights, "phylo_weights.pt")
    print(f"\nSUCCESS: phylo_weights.pt updated with Strategy A Granular Matrix.")