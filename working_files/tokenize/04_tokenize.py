import json
import lingpy
from pathlib import Path
from collections import Counter

# --- Config ---
INPUT_PATH = Path("./ipa_output.json")
OUTPUT_PATH = Path("./tokenized_data.json")

def tokenize_ipa(ipa_str):
    """
    Robustly tokenize IPA, handling spaces and avoiding illegal vowel-merges.
    """
    if not ipa_str:
        return []
    
    # Robustly handle spaces by tokenizing each word separately and joining
    words = ipa_str.split()
    all_tokens = []
    
    for word in words:
        # We disable merge_vowels for the modern languages to prevent 
        # strings like 'oaie' from becoming a single token, which PanPhon can't read.
        tokens = lingpy.ipa2tokens(word, merge_vowels=False)
        all_tokens.extend(tokens)
        
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
            # Robust check for None or empty
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

    # --- Write Output ---
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tokenized_output, f, ensure_ascii=False, indent=4)

    print(f"\nSuccess! Tokenized data saved to {OUTPUT_PATH}")
    print(f"Unique phonemes found: {len(symbol_counter)}")
    print(f"Sample: {list(symbol_counter.keys())[:10]}")

if __name__ == "__main__":
    run_tokenization()