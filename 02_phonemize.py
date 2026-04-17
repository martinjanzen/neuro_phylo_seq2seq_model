import json, re, epitran, unicodedata
from pathlib import Path

_DIR = Path(__file__).parent
INPUT_PATH = _DIR / "swadesh_expanded.json"
OUTPUT_PATH = _DIR / "ipa_output.json"

EPITRAN_LANGS = {"French": "fra-Latn", "Spanish": "spa-Latn", "Italian": "ita-Latn", "Portuguese": "por-Latn", "Romanian": "ron-Latn"}

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
        "soleil": "sɔlɛj", "nager": "naʒe", "dent": "dɑ̃", "deux": "dø", "queue": "kø",
        "aile": "ɛl",
    },
    "Portuguese": {
        "chuva": "ʃuvɐ", "sangue": "sɐ̃ɡe", "osso": "oso", "orelha": "oɾeʎɐ",
        "olho": "oʎo", "peixe": "pejʃe", "cinza": "sĩzɐ", "pé": "pɛ",
        "ouvir": "oviɾ", "folha": "foʎɐ", "piolho": "pioʎo", "montanha": "mõtɐɲɐ",
        "noite": "nojte", "semente": "semẽte", "lingua": "lĩɡwɐ", "dois": "dojʃ",
        "unha": "uɲɐ", "cheio": "ʃejo", "joelho": "ʒueʎo"
    }
}

def latin_to_ipa(form: str) -> str:
    s = form.strip().lower().replace("c", "k").replace("x", "ks")
    s = re.sub(r"([aeiouyāēīōūȳ])ns", lambda m: {'a':'ā','e':'ē','i':'ī','o':'ō','u':'ū','y':'ȳ'}.get(m.group(1), m.group(1)) + 's', s)
    s = re.sub(r"m$", "", s)
    s = re.sub(r"^h", "", s)
    s = s.replace("ns", "s").replace("ph", "p").replace("th", "t").replace("kh", "k").replace("ch", "k")
    s = re.sub(r"^u(?=[aeiouāēīōūy])", "β", s)
    s = re.sub(r"^i(?=[aeiouāēīōūy])", "j", s)
    s = re.sub(r"(?<=[aeiouāēīōūy])u(?=[aeiouāēīōūy])", "β", s)
    s = re.sub(r"(?<=[aeiouāēīōūy])i(?=[aeiouāēīōūy])", "j", s)
    s = s.replace("v", "β")
    s = re.sub(r"(?<=[^aeiouāēīōūy])[eiīē](?=[aeiouāēīōūy])", "j", s)
    s = re.sub(r"qu(?=[aeiouāēīōūy])", "kʷ", s)
    s = re.sub(r"gu(?=[aeiouāēīōūy])", "ɡʷ", s)
    s = re.sub(r"([bcdfgklmnpqrstvw])\1", lambda m: m.group(1) + "ː", s)
    s = re.sub(r"g(?=n)", "ŋ", s).replace("ae", "##EPSILON##").replace("oe", "##E##").replace("au", "aw")
    s = re.sub(r"oɔ|ɔo", "ɔ", s)
    for _uo in ("ūō", "ūo", "uō", "uo"): s = s.replace(_uo, "##O##")
    for cl, vg in {"ī":"##I##", "i":"##E##", "ē":"##E##", "e":"##EPSILON##", "ā":"##A##", "a":"##A##", "o":"##OPEN_O##", "ō":"##O##", "u":"##O##", "ū":"##U##", "y":"##I##", "ȳ":"##I##"}.items(): s = s.replace(cl, vg)
    s = s.replace("##I##", "i").replace("##E##", "e").replace("##EPSILON##", "ɛ").replace("##A##", "a").replace("##OPEN_O##", "ɔ").replace("##O##", "o").replace("##U##", "u")
    s = re.sub(r"([aeiouɔɛ])\1", r"\1", s)
    s = s.replace("x", "ks").replace("z", "dz").replace("g", "ɡ")
    return re.sub(r"n(?=ɡ|k)", "ŋ", s)

def run_phonemize():
    with open(INPUT_PATH, encoding="utf-8") as f: data = json.load(f)
    transliterators = {lang: epitran.Epitran(code) for lang, code in EPITRAN_LANGS.items()}
    output = {}

    for concept, entry in data.items():
        out_entry = {"concept": entry["concept"], "gloss": entry["gloss"], "languages": {}}
        for lang, forms in entry["languages"].items():
            if lang == "Latin":
                clean_forms = [f for f in forms if f and str(f).strip()]
                out_entry["languages"][lang] = {"ortho": clean_forms, "ipa": [latin_to_ipa(f) for f in clean_forms] if clean_forms else None, "status": "rules-vulgar-latin" if clean_forms else "manual-no-forms"}
                continue
            
            epi, ipa_forms = transliterators.get(lang), []
            if not epi: continue
            
            for form in forms:
                clean_form = unicodedata.normalize('NFC', str(form)).replace("≠", "").replace("[", "").replace("]", "").strip()
                clean_form = re.sub(r"\(.*?\)", "", clean_form).strip()
                if lang == "Romanian": clean_form = re.sub(r"^[aA]\s+", "", clean_form).strip()
                if lang in ORTHO_OVERRIDES and clean_form in ORTHO_OVERRIDES[lang]: clean_form = ORTHO_OVERRIDES[lang][clean_form]
                elif " " in clean_form: clean_form = clean_form.split()[0]
                
                try:
                    ipa = POST_EPITRAN_EXCEPTIONS[lang][clean_form] if lang in POST_EPITRAN_EXCEPTIONS and clean_form in POST_EPITRAN_EXCEPTIONS[lang] else epi.transliterate(clean_form)
                    ipa_forms.append(ipa if ipa.strip() else None)
                except:
                    ipa_forms.append(None)
                    
            out_entry["languages"][lang] = {"ortho": forms, "ipa": ipa_forms, "status": "auto"}
        output[concept] = out_entry

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f: json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"{len(output)} phonemized concepts added to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_phonemize()