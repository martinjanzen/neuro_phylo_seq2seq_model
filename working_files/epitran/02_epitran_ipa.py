"""
02_epitran_ipa.py
-----------------
Converts orthographic forms in swadesh_ielex.json to IPA using epitran
for the 5 supported Romance languages. Latin and Sardinian are flagged
separately for manual/cltk handling in the next step.

Install dependencies first:
    pip install epitran lingpy panphon

Epitran also requires language-specific flite voice data for some languages.
If you hit a flite error, run:
    pip install flite  (Linux)
    or see: https://github.com/dmort27/epitran#installation

Outputs:
    ipa_output.json         — full dataset with IPA added alongside ortho forms
    ipa_inspection.txt      — human-readable side-by-side for manual review
    ipa_failures.txt        — any forms epitran could not convert
    ipa_flagged.txt         — Latin and Sardinian forms needing separate treatment
"""

import json
import epitran
from pathlib import Path
from collections import defaultdict

INPUT_PATH      = Path("./swadesh_ielex.json")
OUTPUT_PATH     = Path("./ipa_output.json")
INSPECTION_PATH = Path("./ipa_inspection.txt")
FAILURES_PATH   = Path("./ipa_failures.txt")
FLAGGED_PATH    = Path("./ipa_flagged.txt")

# ── Epitran language codes ────────────────────────────────────────────────────
# Only languages epitran supports reliably
EPITRAN_LANGS = {
    "French":     "fra-Latn",
    "Spanish":    "spa-Latn",
    "Italian":    "ita-Latn",
    "Portuguese": "por-Latn",
    "Romanian":   "ron-Latn",
}

# These need separate handling — do NOT pass to epitran
MANUAL_LANGS = {"Latin", 
                # "Sardinian: Logudoro"
                }

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading swadesh_ielex.json...")
with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)
print(f"  {len(data)} concepts loaded.\n")

# ── Initialize epitran transliterators ───────────────────────────────────────
print("Initializing epitran transliterators...")
transliterators = {}
for lang, code in EPITRAN_LANGS.items():
    try:
        transliterators[lang] = epitran.Epitran(code)
        print(f"  ✓ {lang} ({code})")
    except Exception as e:
        print(f"  ✗ {lang} ({code}) FAILED: {e}")
print()

# ── Process forms ─────────────────────────────────────────────────────────────
output      = {}
failures    = []   # (concept, lang, ortho_form, error)
flagged     = []   # (concept, lang, ortho_form)  — Latin/Sardinian

# For inspection: collect side-by-side
inspection_lines = []

for concept, entry in data.items():
    out_entry = {
        "concept":   entry["concept"],
        "gloss":     entry["gloss"],
        "languages": {}
    }

    inspection_lines.append(f"\n{'='*60}")
    inspection_lines.append(f"  {concept}")
    inspection_lines.append(f"{'='*60}")

    for lang, forms in entry["languages"].items():

        # Flag manual languages without attempting conversion
        if lang in MANUAL_LANGS:
            for form in forms:
                flagged.append((concept, lang, form))
            out_entry["languages"][lang] = {
                "ortho": forms,
                "ipa":   None,          # to be filled by 04_latin_ipa.py
                "status": "manual"
            }
            inspection_lines.append(
                f"  {lang:<25} {'[MANUAL]':<15} {', '.join(forms)}"
            )
            continue

        epi = transliterators.get(lang)
        if not epi:
            continue

        ipa_forms  = []
        lang_lines = []

        for form in forms:
            # 1. Robust Orthographic Cleaning
            clean_form = form.replace("≠", "").replace("[", "").replace("]", "").strip()
            clean_form = clean_form.split("(")[0].strip() 

            # 2. Phrasal Trimming: For Swadesh concepts, we usually need the main word.
            # If there's a space, we take the last word for adjectives/participles 
            # (e.g., 'essere seduto' -> 'seduto') or first word for others.
            if " " in clean_form:
                parts = clean_form.split()
                # If concept is SIT or STAND, the 2nd word is usually the core participle
                if concept in ["SIT", "STAND"] and len(parts) > 1:
                    clean_form = parts[-1] 
                else:
                    clean_form = parts[0]

            # 3. Portuguese 'ch' hallucination fix
            if lang == "Portuguese":
                clean_form = clean_form.replace("ch", "sh").replace("CH", "SH") # Force Epitran to see the fricative

            try:
                # 4. Transliterate
                ipa = epi.transliterate(clean_form)
                
                # 5. French "Eye" Special Case Fix
                if lang == "French" and clean_form == "oeil":
                    ipa = "œj"

                if lang == "Portuguese" and ipa.startswith("suv"):
                    ipa = ipa.replace("suv", "ʃuv", 1)
                
                if not ipa or ipa.strip() == "":
                    raise ValueError("Empty IPA output")
                
                ipa_forms.append(ipa)
                lang_lines.append(
                    f"  {lang:<25} {clean_form:<20} → {ipa}"
                )
            except Exception as e:
                failures.append((concept, lang, clean_form, str(e)))
                ipa_forms.append(None)
                lang_lines.append(
                    f"  {lang:<25} {clean_form:<20} → [FAILED: {e}]"
                )

        out_entry["languages"][lang] = {
            "ortho":  forms,
            "ipa":    ipa_forms,
            "status": "auto"
        }
        inspection_lines.extend(lang_lines)

    output[concept] = out_entry

# ── Write outputs ─────────────────────────────────────────────────────────────

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
print(f"Written: {OUTPUT_PATH}")

with open(INSPECTION_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(inspection_lines))
print(f"Written: {INSPECTION_PATH}")

with open(FAILURES_PATH, "w", encoding="utf-8") as f:
    if failures:
        f.write(f"{'CONCEPT':<15} {'LANGUAGE':<25} {'FORM':<20} ERROR\n")
        f.write("-" * 80 + "\n")
        for concept, lang, form, err in failures:
            f.write(f"{concept:<15} {lang:<25} {form:<20} {err}\n")
    else:
        f.write("No failures.\n")
print(f"Written: {FAILURES_PATH}")

with open(FLAGGED_PATH, "w", encoding="utf-8") as f:
    f.write("These forms need IPA conversion via cltk (Latin) or manual rules (Sardinian).\n")
    f.write(f"{'CONCEPT':<15} {'LANGUAGE':<25} FORM\n")
    f.write("-" * 60 + "\n")
    for concept, lang, form in flagged:
        f.write(f"{concept:<15} {lang:<25} {form}\n")
print(f"Written: {FLAGGED_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_forms   = sum(
    len(forms)
    for entry in data.values()
    for lang, forms in entry["languages"].items()
    if lang not in MANUAL_LANGS
)
total_flagged = len(flagged)
total_failed  = len(failures)
total_success = total_forms - total_failed

print(f"""
{'='*60}
  SUMMARY
{'='*60}
  Auto-converted (epitran):  {total_success}/{total_forms} forms
  Failures:                  {total_failed}
  Flagged for manual (Latin/Sardinian): {total_flagged}

  Next steps:
    1. Open ipa_inspection.txt and review IPA quality
       — spot-check at least 10 concepts across all 5 languages
       — pay attention to French (epitran is weakest here)
    2. Open ipa_failures.txt — if non-empty, fix ortho forms and re-run
    3. Run 04_latin_ipa.py to handle Latin via cltk
    4. Handle Sardinian (see ipa_flagged.txt) — manual or rule-based
{'='*60}
""")