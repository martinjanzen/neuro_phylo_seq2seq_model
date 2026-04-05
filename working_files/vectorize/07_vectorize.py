"""
07_vectorize.py
FIXED: Gaps are now represented as [0.0] * 24 (Issue 14).
This aligns with Panphon's trinary semantics, where 0.0 means 'unspecified', 
preventing the LSTM from trying to interpret 0.5 as a partial articulatory feature.
"""
import json
import unicodedata
import panphon
import torch
from pathlib import Path

# --- Academic Configuration ---
INPUT_PATH = Path("aligned_msa_data.json")
OUTPUT_PATH = Path("model/vectorized_dataset.pt") 
LANGUAGES = ["French", "Italian", "Portuguese", "Romanian", "Spanish"]
TARGET_DIM = 24
ft = panphon.FeatureTable()

# Tracking API health
metrics = {"total_tokens": 0, "successful_vectors": 0, "errors": []}

def get_feature_vector(token, is_gap=False):
    # CRITICAL FIX 14: Gap is universally unspecified (0.0), not 0.5
    if is_gap: return [0.0] * TARGET_DIM
    
    char_map = {'+': 1.0, '-': -1.0, '0': 0.0}
    metrics["total_tokens"] += 1
    
    try:
        token = unicodedata.normalize('NFC', token)
        
        # STRATEGY 1: Modern Panphon API (>= 0.3)
        if hasattr(ft, 'word_fts'):
            segs = ft.word_fts(token)
            if segs:
                vecs = []
                for seg in segs:
                    if hasattr(seg, 'numeric'):
                        vecs.append(seg.numeric())
                    elif hasattr(seg, 'values'):
                        vecs.append(list(seg.values()))
                    else:
                        vecs.append([float(x) if isinstance(x, (int, float)) else char_map.get(str(x), 0.0) for x in seg])
                
                avg_vec = [sum(col)/len(col) for col in zip(*vecs)]
                metrics["successful_vectors"] += 1
                return (avg_vec + [0.0]*TARGET_DIM)[:TARGET_DIM]

        # STRATEGY 2: Legacy Panphon API
        if hasattr(ft, 'fts'):
            segs = ft.fts(token)
            if segs:
                if not isinstance(segs, list):
                    segs = [segs]
                    
                vecs = []
                for seg in segs:
                    if hasattr(seg, 'strings'):
                        v = [char_map.get(x, 0.0) for x in seg.strings()]
                    else:
                        v = [char_map.get(x, 0.0) for x in seg]
                    vecs.append(v)
                    
                avg_vec = [sum(col)/len(col) for col in zip(*vecs)]
                metrics["successful_vectors"] += 1
                return (avg_vec + [0.0]*TARGET_DIM)[:TARGET_DIM]

    except Exception as e:
        if len(metrics["errors"]) < 5:
            metrics["errors"].append(f"Token '{token}' -> {type(e).__name__}: {e}")
            
    return [0.0] * TARGET_DIM

def run_vectorization():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    GLOBAL_MAX_WIDTH = max(entry["matrix_width"] for entry in data.values())
    print(f"Global Alignment Width: {GLOBAL_MAX_WIDTH}")

    dataset = []

    for concept, entry in data.items():
        alignment = entry["alignment"]
        
        input_list = []
        for lang in LANGUAGES:
            tokens = alignment.get(lang, ["-"] * entry["matrix_width"])
            lang_vectors = []
            
            for i in range(GLOBAL_MAX_WIDTH):
                if i < len(tokens):
                    lang_vectors.append(get_feature_vector(tokens[i], tokens[i]=="-"))
                else:
                    lang_vectors.append([0.0] * TARGET_DIM)
            input_list.append(lang_vectors)

        target_list = []
        latin_tokens = alignment.get("Latin", ["-"] * entry["matrix_width"])
        for i in range(GLOBAL_MAX_WIDTH):
            if i < len(latin_tokens):
                target_list.append(get_feature_vector(latin_tokens[i], latin_tokens[i]=="-"))
            else:
                target_list.append([0.0] * TARGET_DIM)

        dataset.append({
            "concept": concept,
            "X": torch.tensor(input_list, dtype=torch.float32), 
            "Y": torch.tensor(target_list, dtype=torch.float32) 
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, OUTPUT_PATH)
    
    yield_rate = (metrics["successful_vectors"] / max(1, metrics["total_tokens"])) * 100
    print(f"Success: Vectorized {len(dataset)} concepts with global width {GLOBAL_MAX_WIDTH}")
    print(f"API Health Check: {yield_rate:.1f}% non-zero vector yield ({metrics['successful_vectors']}/{metrics['total_tokens']})")
    
    if metrics["errors"]:
        print("\n[Debug] Encountered errors during vectorization:")
        for err in metrics["errors"]:
            print(f"  - {err}")

if __name__ == "__main__":
    run_vectorization()