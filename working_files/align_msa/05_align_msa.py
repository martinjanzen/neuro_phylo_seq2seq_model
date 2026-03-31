import json
from lingpy.align.multiple import Multiple
from pathlib import Path

# --- Config ---
INPUT_PATH = Path("./tokenized_data.json")
OUTPUT_PATH = Path("./aligned_msa_data.json")

def run_msa():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    msa_results = {}

    print("Aligning cognate sets using Multiple Sequence Alignment...")
    for concept, entry in data.items():
        languages = entry["languages"]
        
        seqs = []
        labels = []
        
        for lang, lang_data in languages.items():
            # Check if tokens exist and take the first list of tokens
            if lang_data.get("tokens") and len(lang_data["tokens"]) > 0:
                current_tokens = lang_data["tokens"][0]
                
                # Ensure all tokens are strings (precaution against any non-string objects)
                current_tokens = [str(t) for t in current_tokens]
                
                seqs.append(current_tokens)
                labels.append(lang)

        # Triangulation requires at least two modern languages + Latin
        if len(seqs) < 2:
            continue

        try:
            # We use the 'Multiple' class which is more tolerant of pre-tokenized lists
            msa = Multiple(seqs)
            
            # This performs the progressive alignment
            # gap_penalty and other params can be tuned here if needed
            msa.prog_align()
            
            # .alm_matrix contains the final aligned sequences with '-' gaps
            aligned_matrix = msa.alm_matrix
            
            alignment_dict = {labels[i]: aligned_matrix[i] for i in range(len(labels))}
            
            msa_results[concept] = {
                "gloss": entry["gloss"],
                "alignment": alignment_dict,
                "matrix_width": len(aligned_matrix[0])
            }
            
        except Exception as e:
            print(f"  [Error] Alignment failed for {concept}: {e}")

    # --- Write Output ---
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(msa_results, f, ensure_ascii=False, indent=4)

    print(f"\nSuccess! Aligned {len(msa_results)} concepts.")
    print(f"Check your {OUTPUT_PATH} for the '-' gap markers.")

if __name__ == "__main__":
    run_msa()