"""
03b_latin_ipa.py
Converts Classical Latin orthographic forms to Vulgar Latin / Proto-Romance IPA.

Phonological Basis:
1. József Herman (2000) "Vulgar Latin"
2. Alkire & Rosen (2010) "Romance Languages"
"""
import json
import re
from pathlib import Path

INPUT_PATH      = Path("./ipa_output.json")
INSPECTION_PATH = Path("./latin_ipa_inspection.txt")

def latin_to_ipa(form: str) -> str:
    s = form.strip().lower()

    # --- BLOCK 1: Compensatory Lengthening (Herman 2000) ---
    s = s.replace("ans", "ās").replace("ens", "ēs").replace("ins", "īs")
    s = s.replace("ons", "ōs").replace("uns", "ūs")

    # --- BLOCK 2: Early Consonant Weakening (Herman 2000: 39-41) ---
    s = re.sub(r"m$", "", s) 
    s = re.sub(r"^h", "", s) 
    s = s.replace("ns", "s") 

    # --- BLOCK 3: Loss of Greek Aspirates (Herman 2000: 44) ---
    s = s.replace("ph", "p").replace("th", "t").replace("ch", "k")

    # --- BLOCK 4: Consonantal glides (w -> β, i -> j) (Alkire & Rosen 2010: 59) ---
    s = re.sub(r"^u(?=[aeiouāēīōūy])", "β", s)
    s = re.sub(r"(?<=[aeiouāēīōūy])u(?=[aeiouāēīōūy])", "β", s)
    s = s.replace("v", "β")
    
    s = re.sub(r"^i(?=[aeiouāēīōūy])", "j", s)
    # CRITICAL FIX 11: Medial consonantal i
    s = re.sub(r"(?<=[aeiouāēīōūy])i(?=[aeiouāēīōūy])", "j", s)

    # --- BLOCK 5: Yodization (Alkire & Rosen 2010) ---
    s = re.sub(r"[eiīē](?=[aeiouāēīōūy])", "j", s)

    # --- BLOCK 6: Labiovelars and Geminates ---
    s = re.sub(r"qu(?=[aeiouāēīōūy])", "kʷ", s)
    s = re.sub(r"gu(?=[aeiouāēīōūy])", "ɡʷ", s)
    s = re.sub(r"([bcdfgklmnpqrstvw])\1", lambda m: m.group(1) + "ː", s)
    s = re.sub(r"g(?=n)", "ŋ", s)

    # --- BLOCK 7: The Pan-Romance Vowel Collapse (Alkire & Rosen 2010: 24-25) ---
    s = s.replace("ae", "##EPSILON##")
    s = s.replace("oe", "##E##")
    s = s.replace("au", "aw")
    
    VOWEL_MAP = {
        "ī": "##I##", "i": "##E##", "ē": "##E##", "e": "##EPSILON##",
        "ā": "##A##", "a": "##A##", "o": "##OPEN_O##", "ō": "##O##",
        "u": "##O##", "ū": "##U##", "y": "##I##", "ȳ": "##I##"
    }
    for cl_vowel, vg_placeholder in VOWEL_MAP.items():
        s = s.replace(cl_vowel, vg_placeholder)
        
    s = s.replace("##I##", "i").replace("##E##", "e").replace("##EPSILON##", "ɛ")
    s = s.replace("##A##", "a").replace("##OPEN_O##", "ɔ").replace("##O##", "o").replace("##U##", "u")

    # --- BLOCK 8: Hiatus Contraction ---
    s = re.sub(r"([aeiouɔɛ])\1", r"\1", s)

    # --- BLOCK 9: Remaining Consonants ---
    s = s.replace("c", "k")
    s = s.replace("x", "ks").replace("z", "dz")
    s = s.replace("g", "ɡ")
    s = re.sub(r"n(?=ɡ|k)", "ŋ", s)

    return s

def process_dataset():
    print("Loading ipa_output.json...")
    if not INPUT_PATH.exists():
        print(f"File {INPUT_PATH} not found. Skipping execution.")
        return
        
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    
    updated = 0
    for concept, entry in data.items():
        latin = entry["languages"].get("Latin")
        if not latin or latin.get("status") not in ["manual", "rules-latin", "rules-vulgar-latin"]:
            continue
        ipa_forms = [latin_to_ipa(f) for f in latin["ortho"]]
        data[concept]["languages"]["Latin"]["ipa"] = ipa_forms
        data[concept]["languages"]["Latin"]["status"] = "rules-vulgar-latin"
        updated += 1

    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Updated {updated} Latin entries to Vulgar Latin targets.")

if __name__ == "__main__":
    process_dataset()