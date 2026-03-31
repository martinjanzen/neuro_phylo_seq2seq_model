import json
import unicodedata
import panphon
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
INPUT_PATH = Path("./aligned_msa_data.json")
OUTPUT_PATH = Path("./vectorized_dataset.pt")
LANGUAGES = ["French", "Italian", "Portuguese", "Romanian", "Spanish"]

ft = panphon.FeatureTable()
# Standardize the dimension based on a known good segment
TARGET_DIM = len(ft.word_to_vector_list('a')[0])

def clean_token(t):
    # 1. Normalize unicode (handles some diacritic issues)
    t = unicodedata.normalize('NFC', t)
    # 2. Handle geminates (e.g., 'ʀʀ' -> 'ʀ')
    if len(t) == 2 and t[0] == t[1]:
        return t[0]
    # 3. Strip problematic diacritics if panphon still fails
    # (Optional: can add more specific replacements here)
    return t

def get_feature_vector(token, is_gap=False):
    if is_gap:
        return [0.5] * TARGET_DIM
    
    char_map = {'+': 1.0, '-': -1.0, '0': 0.0}
    token = clean_token(token) # Pre-clean pass
    
    try:
        features = ft.fts(token)
        if features:
            vec = [char_map.get(x, 0.0) for x in features.strings()]
            if len(vec) < TARGET_DIM:
                vec += [0.0] * (TARGET_DIM - len(vec))
            return vec[:TARGET_DIM]
        
        # Secondary fallback: if clean_token didn't work, try stripping diacritics
        # This handles tokens like 'ə̂' -> 'ə'
        base_t = "".join([c for c in unicodedata.normalize('NFD', token) 
                         if unicodedata.category(c) != 'Mn'])
        features = ft.fts(base_t)
        if features:
            vec = [char_map.get(x, 0.0) for x in features.strings()]
            if len(vec) < TARGET_DIM: vec += [0.0] * (TARGET_DIM - len(vec))
            return vec[:TARGET_DIM]
            
    except Exception:
        pass
        
    return [0.0] * TARGET_DIM

def run_vectorization():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_width = max(entry["matrix_width"] for entry in data.values())
    print(f"Max matrix width: {max_width}")
    print(f"Enforcing Feature Dimension: {TARGET_DIM}")

    dataset = []

    for concept, entry in data.items():
        alignment = entry["alignment"]
        
        # 1. Build Input Matrix (5 Languages x Max_Width x TARGET_DIM)
        input_list = []
        for lang in LANGUAGES:
            lang_vectors = []
            # Default to gaps if language is missing for a concept
            tokens = alignment.get(lang, ["-"] * entry["matrix_width"])
            
            for i in range(max_width):
                if i < len(tokens):
                    t = tokens[i]
                    lang_vectors.append(get_feature_vector(t, is_gap=(t == "-")))
                else:
                    lang_vectors.append([0.0] * TARGET_DIM)
            input_list.append(lang_vectors)

        # 2. Build Target Sequence (Latin)
        target_vectors = []
        latin_tokens = alignment.get("Latin", ["-"] * entry["matrix_width"])
        for i in range(max_width):
            if i < len(latin_tokens):
                t = latin_tokens[i]
                target_vectors.append(get_feature_vector(t, is_gap=(t == "-")))
            else:
                target_vectors.append([0.0] * TARGET_DIM)

        dataset.append({
            "concept": concept,
            "X": torch.tensor(input_list, dtype=torch.float32),
            "Y": torch.tensor(target_vectors, dtype=torch.float32)
        })

    torch.save(dataset, OUTPUT_PATH)
    print(f"\nSuccess! Vectorized {len(dataset)} concepts.")
    print(f"Dataset saved to {OUTPUT_PATH}")

def verify_vec():
    dataset = torch.load("./vectorized_dataset.pt")

    # Pick a random sample (e.g., concept 'SALT')
    sample = dataset[0]
    X = sample['X']

    print(f"Tensor Shape: {X.shape}") # Should be (5, max_width, 22)
    print(f"Unique values in X: {torch.unique(X)}") 
    # Verify if 0.5 is present for your gaps!
    if 0.5 in X:
        print("✓ Gap handling (0.5) verified.")

    # Check for all-zero rows (simulated missing data or padding)
    if torch.all(X[0, -1, :] == 0):
        print("✓ Padding (0.0) verified.")

def heatmap():
    dataset = torch.load("./vectorized_dataset.pt")
    # Let's look at index 0 (e.g., ASH)
    # X shape is (Languages, Sequence, Features)
    # We'll plot the first language (French)
    french_ash = dataset[0]['X'][0].numpy() 

    plt.figure(figsize=(12, 6))
    sns.heatmap(french_ash, cmap="YlGnBu", annot=False)
    plt.title(f"Feature Matrix for {dataset[0]['concept']} (French)")
    plt.xlabel("22 PanPhon Features")
    plt.ylabel("Time / Phoneme Index")
    plt.show()

if __name__ == "__main__":
    run_vectorization()
    verify_vec()
    heatmap()