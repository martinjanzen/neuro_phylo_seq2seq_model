"""
05_align_msa.py
FIXED: Implements Data Augmentation (Issue 6A).
Generates the Cartesian product of all available cognate variants per concept 
and aligns each combination separately, effectively inflating the dataset size.
"""
import json
import itertools
from lingpy.align.multiple import Multiple
from lingpy.align.pairwise import pw_align
from pathlib import Path

# --- Config ---
INPUT_PATH = Path("./tokenized_data.json")
OUTPUT_PATH = Path("./aligned_msa_data.json")

def align_latin_to_msa(romance_matrix, romance_labels, latin_tokens):
    """Aligns the Latin sequence to the already frozen Romance MSA."""
    if not latin_tokens:
        return romance_matrix, ["-"] * len(romance_matrix[0])
        
    consensus = []
    width = len(romance_matrix[0])
    for i in range(width):
        col = [row[i] for row in romance_matrix if row[i] != "-"]
        if not col:
            consensus.append("-")
        else:
            consensus.append(max(set(col), key=col.count))

    alm_latin, alm_cons, _ = pw_align(latin_tokens, consensus)
    
    final_romance_matrix = [[] for _ in romance_labels]
    final_latin = []
    
    orig_idx = 0
    for l_tok, c_tok in zip(alm_latin, alm_cons):
        final_latin.append(l_tok)
        if c_tok == "-":
            for i in range(len(romance_labels)):
                final_romance_matrix[i].append("-")
        else:
            for i in range(len(romance_labels)):
                final_romance_matrix[i].append(romance_matrix[i][orig_idx])
            orig_idx += 1
            
    return final_romance_matrix, final_latin

def run_msa():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    msa_results = {}
    total_combinations = 0

    print("Aligning cognate sets using Profile Alignment & Data Augmentation...")
    for concept, entry in data.items():
        languages = entry["languages"]
        
        romance_langs = []
        romance_tokens_lists = []
        latin_tokens_list = [["-"]]
        
        for lang, lang_data in languages.items():
            toks = lang_data.get("tokens", [])
            # Filter out empty token lists
            valid_toks = [[str(t) for t in variant] for variant in toks if variant]
            if not valid_toks: continue
            
            if lang == "Latin":
                latin_tokens_list = valid_toks
            else:
                romance_langs.append(lang)
                romance_tokens_lists.append(valid_toks)

        if len(romance_langs) < 2:
            continue

        # CRITICAL FIX 6A: Generate all combinations of Romance & Latin variants
        romance_combinations = list(itertools.product(*romance_tokens_lists))
        latin_combinations = latin_tokens_list
        
        combo_idx = 0
        for r_combo in romance_combinations:
            for l_seq in latin_combinations:
                try:
                    msa = Multiple(list(r_combo))
                    msa.prog_align()
                    romance_matrix = msa.alm_matrix
                    
                    final_romance_matrix, final_latin = align_latin_to_msa(romance_matrix, romance_langs, l_seq)
                    
                    alignment_dict = {romance_langs[i]: final_romance_matrix[i] for i in range(len(romance_langs))}
                    alignment_dict["Latin"] = final_latin
                    
                    # Suffix the concept name if there are multiple variants (e.g., BIRD_0, BIRD_1)
                    is_multiple = (len(romance_combinations) * len(latin_combinations)) > 1
                    concept_key = f"{concept}_{combo_idx}" if is_multiple else concept
                    
                    msa_results[concept_key] = {
                        "gloss": entry["gloss"],
                        "alignment": alignment_dict,
                        "matrix_width": len(final_latin)
                    }
                    combo_idx += 1
                    total_combinations += 1
                    
                except Exception as e:
                    print(f"  [Error] Alignment failed for {concept} combo {combo_idx}: {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(msa_results, f, ensure_ascii=False, indent=4)

    base_concepts = len(data)
    print(f"\nSuccess! Expanded {base_concepts} base concepts into {total_combinations} unique alignments via augmentation.")

if __name__ == "__main__":
    run_msa()