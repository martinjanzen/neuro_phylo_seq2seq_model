"""
06_phylogeny.py
FIXED: Uses Neighbor-Joining (NJ). Phonetic distances are normalized 
by alignment width. Gap-to-phoneme distance is updated to 1.0 (Issue 14).
"""
import json
import panphon
import pandas as pd
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from pathlib import Path

# --- Config ---
INPUT_PATH = Path("./aligned_msa_data.json")
MATRIX_OUTPUT = Path("./distance_matrix.csv")
ft = panphon.FeatureTable()

def get_phonetic_distance(t1, t2):
    """Calculates articulatory distance between two tokens."""
    if t1 == t2: return 0.0
    
    # CRITICAL FIX 14: A gap vs a phoneme represents a full insertion/deletion.
    # We assign 1.0 (the max feature distance) instead of an arbitrary 0.5.
    if t1 == "-" or t2 == "-": return 1.0

    try:
        dist = ft.weighted_feature_distance(t1, t2)
    except Exception:
        dist = 1.0 # Fallback
    return dist

def run_phylogeny():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    languages = ["French", "Italian", "Portuguese", "Romanian", "Spanish", "Latin"]
    dist_sum = {l: {l2: 0.0 for l2 in languages} for l in languages}
    count_sum = 0

    print("Calculating normalized inter-language distances...")
    for concept, entry in data.items():
        alignment = entry["alignment"]
        width = entry["matrix_width"]
        if width == 0: continue
        
        for l1 in languages:
            for l2 in languages:
                if l1 in alignment and l2 in alignment:
                    raw_d = sum(get_phonetic_distance(alignment[l1][i], alignment[l2][i]) for i in range(width))
                    dist_sum[l1][l2] += (raw_d / width)
                    
        count_sum += 1

    # 1. Create a Full Square Matrix for the CSV
    full_matrix_data = []
    for l1 in languages:
        row = [dist_sum[l1][l2] / count_sum for l2 in languages]
        full_matrix_data.append(row)

    df = pd.DataFrame(full_matrix_data, index=languages, columns=languages)
    df.to_csv(MATRIX_OUTPUT)
    print(f"Success! Distance matrix saved to {MATRIX_OUTPUT}")

    # 2. Create the Lower Triangular Matrix for Bio.Phylo
    lower_triangular = []
    for i, l1 in enumerate(languages):
        row = []
        for j, l2 in enumerate(languages):
            if j <= i:
                row.append(dist_sum[l1][l2] / count_sum)
        lower_triangular.append(row)

    # 3. Build and Draw the Tree
    dm = DistanceMatrix(names=languages, matrix=lower_triangular)
    constructor = DistanceTreeConstructor()
    
    tree = constructor.nj(dm)

    print("\n--- Language Reconstruction Tree (Neighbor-Joining) ---")
    Phylo.draw_ascii(tree)

if __name__ == "__main__":
    run_phylogeny()