import json
import panphon
import pandas as pd # Added for easy matrix handling
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
    if t1 == "-" or t2 == "-": return 0.5 
    
    try:
        # Use weighted_feature_distance for better resolution on phoneme differences
        dist = ft.weighted_feature_distance(t1, t2)
    except Exception:
        dist = 1.0 # Fallback for unknown symbols
    return dist

def run_phylogeny():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    languages = ["French", "Italian", "Portuguese", "Romanian", "Spanish", "Latin"]
    dist_sum = {l: {l2: 0.0 for l2 in languages} for l in languages}
    count_sum = 0

    print("Calculating inter-language distances...")
    for concept, entry in data.items():
        alignment = entry["alignment"]
        width = entry["matrix_width"]
        
        for l1 in languages:
            for l2 in languages:
                if l1 in alignment and l2 in alignment:
                    # Calculate distance across all slots in the aligned matrix
                    d = sum(get_phonetic_distance(alignment[l1][i], alignment[l2][i]) 
                            for i in range(width))
                    dist_sum[l1][l2] += d
        count_sum += 1

    # 1. Create a Full Square Matrix for the CSV
    full_matrix_data = []
    for l1 in languages:
        row = [dist_sum[l1][l2] / count_sum for l2 in languages]
        full_matrix_data.append(row)

    # Save to CSV using Pandas for easy formatting
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

    # Build and Draw the Tree
    dm = DistanceMatrix(names=languages, matrix=lower_triangular)
    constructor = DistanceTreeConstructor()
    tree = constructor.upgma(dm)

    print("\n--- Language Reconstruction Tree ---")
    Phylo.draw_ascii(tree)

if __name__ == "__main__":
    run_phylogeny()