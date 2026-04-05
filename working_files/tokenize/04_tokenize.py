"""
04_tokenize.py
FIXED: Implements a deterministic post-processing pass to ensure LingPy 
does not orphan length marks (ː), preserving geminate consonants (Issue 20).
"""
import json
import lingpy
from pathlib import Path
from collections import Counter

# --- Config ---
INPUT_PATH = Path("./ipa_output.json")
OUTPUT_PATH = Path("./tokenized_data.json")

def tokenize_ipa(ipa_str):
    """
    Robustly tokenize IPA, handling spaces, preventing illegal vowel-merges,
    and ensuring length marks stay attached to their consonants.
    """
    if not ipa_str:
        return []

    words = ipa_str.split()
    all_tokens = []

    for word in words:
        # Disable merge_vowels so strings like 'oaie' don't become a single token
        raw_tokens = lingpy.ipa2tokens(word, merge_vowels=False)
        
        # CRITICAL FIX 20: Deterministic geminate merging
        merged_tokens = []
        for t in raw_tokens:
            # If the token is just a length mark (or starts with one)
            if t.startswith('ː') and merged_tokens:
                # Append it to the previous phoneme
                merged_tokens[-1] += t
            else:
                merged_tokens.append(t)
                
        all_tokens.extend(merged_tokens)
        
    return all_tokens

def run_tokenization():
    print(f"Loading {INPUT_PATH}...")
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenized_output = {}
    symbol_counter = Counter()

    print("Tokenizing concepts...")
    for concept, entry in data.items():
        tokenized_entry = {
            "concept": entry["concept"],
            "gloss": entry["gloss"],
            "languages": {}
        }

        for lang, lang_data in entry["languages"].items():
            ipa_list = lang_data.get("ipa", [])
            if ipa_list is None:
                ipa_list = []

            tokens_list = []
            for ipa_str in ipa_list:
                try:
                    tokens = tokenize_ipa(ipa_str)
                    tokens_list.append(tokens)
                    symbol_counter.update(tokens)
                except Exception as e:
                    print(f"  [Warning] Tokenization failed for {lang} - {ipa_str}: {e}")
                    tokens_list.append([])

            tokenized_entry["languages"][lang] = {
                "ortho": lang_data["ortho"],
                "ipa": lang_data["ipa"],
                "tokens": tokens_list,
                "status": lang_data.get("status", "unknown")
            }

        tokenized_output[concept] = tokenized_entry

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tokenized_output, f, ensure_ascii=False, indent=4)

    print(f"\nSuccess! Tokenized data saved to {OUTPUT_PATH}")
    
    # Validation logging for geminates
    geminates_found = [sym for sym in symbol_counter.keys() if 'ː' in sym]
    print(f"Unique phonemes found: {len(symbol_counter)}")
    if geminates_found:
        print(f"Successfully preserved geminates (e.g., {geminates_found[:5]})")
    else:
        print("No geminates found in the dataset.")

if __name__ == "__main__":
    run_tokenization()