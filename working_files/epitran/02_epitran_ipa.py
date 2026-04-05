"""
02_epitran_ipa.py
FIXED: Applies NFC Unicode Normalization before text manipulation to 
ensure Epitran accurately parses complex Romance diacritics (Issue 21).
"""
import json
import re
import epitran
import unicodedata
from pathlib import Path

INPUT_PATH      = Path("./swadesh_ielex.json")
OUTPUT_PATH     = Path("./ipa_output.json")
INSPECTION_PATH = Path("./ipa_inspection.txt")
FAILURES_PATH   = Path("./ipa_failures.txt")
FLAGGED_PATH    = Path("./ipa_flagged.txt")

EPITRAN_LANGS = {
    "French":     "fra-Latn",
    "Spanish":    "spa-Latn",
    "Italian":    "ita-Latn",
    "Portuguese": "por-Latn",
    "Romanian":   "ron-Latn",
}
MANUAL_LANGS = {"Latin"}

ORTHO_OVERRIDES = {
    "Italian": {
        "essere seduto": "seduto",
        "stare in piedi": "stare"
    }
}

POST_EPITRAN_EXCEPTIONS = {
    "French": {
        "oeil": "œj", "sang": "sɑ̃", "oreille": "ɔʀɛj", "voler": "vɔle",
        "os": "ɔs", "terre": "tɛʀ", "oeuf": "œf", "donner": "dɔne",
        "feuille": "fœj", "long": "lɔ̃", "racine": "ʀasin", "être": "ɛtʀ",
        "soleil": "sɔlɛj", "nager": "naʒe", "dent": "dɑ̃", "deux": "dø", "queue": "kø"
    },
    "Portuguese": {
        "chuva": "ʃuvɐ", "sangue": "sɐ̃ɡe", "osso": "oso", "orelha": "oɾeʎɐ",
        "olho": "oʎo", "peixe": "pejʃe", "cinza": "sĩzɐ", "pé": "pɛ",
        "ouvir": "oviɾ", "folha": "foʎɐ", "piolho": "pioʎo", "montanha": "mõtɐɲɐ",
        "noite": "nojte", "semente": "semẽte", "lingua": "lĩɡwɐ", "dois": "dojʃ",
        "unha": "uɲɐ", "cheio": "ʃejo", "joelho": "ʒueʎo"
    }
}

print("Loading swadesh_ielex.json...")
with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

print("Initializing epitran transliterators...")
transliterators = {}
for lang, code in EPITRAN_LANGS.items():
    try:
        transliterators[lang] = epitran.Epitran(code)
    except Exception as e:
        print(f"  ✗ {lang} ({code}) FAILED: {e}")

output = {}
failures = []   
flagged = []   
inspection_lines = []

for concept, entry in data.items():
    out_entry = {
        "concept":   entry["concept"],
        "gloss":     entry["gloss"],
        "languages": {}
    }

    inspection_lines.append(f"\n{'='*60}\n  {concept}\n{'='*60}")

    for lang, forms in entry["languages"].items():
        if lang in MANUAL_LANGS:
            for form in forms:
                flagged.append((concept, lang, form))
            out_entry["languages"][lang] = {"ortho": forms, "ipa": None, "status": "manual"}
            continue

        epi = transliterators.get(lang)
        if not epi: continue

        ipa_forms = []
        lang_lines = []

        for form in forms:
            # CRITICAL FIX 21: NFC Normalization
            clean_form = unicodedata.normalize('NFC', form)
            
            clean_form = clean_form.replace("≠", "").replace("[", "").replace("]", "").strip()
            clean_form = re.sub(r"\(.*?\)", "", clean_form).strip()

            if lang == "Romanian" and clean_form.startswith("a "):
                clean_form = clean_form[2:].strip()

            if lang in ORTHO_OVERRIDES and clean_form in ORTHO_OVERRIDES[lang]:
                clean_form = ORTHO_OVERRIDES[lang][clean_form]
            else:
                if " " in clean_form:
                    clean_form = clean_form.split()[0]

            try:
                if lang in POST_EPITRAN_EXCEPTIONS and clean_form in POST_EPITRAN_EXCEPTIONS[lang]:
                    ipa = POST_EPITRAN_EXCEPTIONS[lang][clean_form]
                else:
                    ipa = epi.transliterate(clean_form)
                
                if not ipa or ipa.strip() == "":
                    raise ValueError("Empty IPA output")
                
                ipa_forms.append(ipa)
                lang_lines.append(f"  {lang:<25} {clean_form:<20} → {ipa}")
                
            except Exception as e:
                failures.append((concept, lang, clean_form, str(e)))
                ipa_forms.append(None)
                lang_lines.append(f"  {lang:<25} {clean_form:<20} → [FAILED: {e}]")

        out_entry["languages"][lang] = {"ortho": forms, "ipa": ipa_forms, "status": "auto"}
        inspection_lines.extend(lang_lines)

    output[concept] = out_entry

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)