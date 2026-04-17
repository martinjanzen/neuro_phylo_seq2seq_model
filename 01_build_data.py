import json, csv
from pathlib import Path
from collections import defaultdict
from pycldf import Dataset

DIR = Path(__file__).parent
OUTPUT_PATH = DIR / "swadesh_expanded.json"
IELEX_CLDF_PATH = DIR / "iecor-master/cldf/cldf-metadata.json"
CONCEPTS_CSV = DIR / "evotext-ielex-data-and-tree-686ce09/data/concepts.csv"
TARGET_LANGUAGES = {"French", "Spanish", "Italian", "Portuguese", "Romanian", "Latin"}

SWADESH_100 = {
    "I":            {"I", "1SG", "FIRST PERSON SINGULAR"},
    "THOU":         {"THOU", "YOU (SINGULAR)", "YOU.SG", "2SG"},
    "WE":           {"WE", "1PL", "FIRST PERSON PLURAL"},
    "THIS":         {"THIS"},
    "THAT":         {"THAT"},
    "WHO":          {"WHO", "WHO/WHAT"},
    "WHAT":         {"WHAT", "WHO/WHAT"},
    "NOT":          {"NOT", "NEGATIVE"},
    "ALL":          {"ALL", "EVERY"},
    "MANY":         {"MANY", "MUCH"},
    "ONE":          {"ONE"},
    "TWO":          {"TWO"},
    "BIG":          {"BIG", "LARGE"},
    "LONG":         {"LONG"},
    "SMALL":        {"SMALL", "LITTLE"},
    "WOMAN":        {"WOMAN", "FEMALE"},
    "MAN":          {"MAN", "MALE PERSON", "MALE"},
    "PERSON":       {"PERSON", "HUMAN", "HUMAN BEING"},
    "FISH":         {"FISH"},
    "BIRD":         {"BIRD"},
    "DOG":          {"DOG"},
    "LOUSE":        {"LOUSE"},
    "TREE":         {"TREE"},
    "SEED":         {"SEED"},
    "LEAF":         {"LEAF"},
    "ROOT":         {"ROOT"},
    "BARK":         {"BARK (OF TREE)", "BARK"},
    "SKIN":         {"SKIN"},
    "FLESH":        {"FLESH", "MEAT", "FLESH/MEAT"},
    "BLOOD":        {"BLOOD"},
    "BONE":         {"BONE"},
    "GREASE":       {"GREASE", "FAT", "FAT (ORGANIC SUBSTANCE)"},
    "EGG":          {"EGG"},
    "HORN":         {"HORN"},
    "TAIL":         {"TAIL"},
    "FEATHER":      {"FEATHER"},
    "HAIR":         {"HAIR", "HAIR (ON HEAD)"},
    "HEAD":         {"HEAD"},
    "EAR":          {"EAR"},
    "EYE":          {"EYE"},
    "NOSE":         {"NOSE"},
    "MOUTH":        {"MOUTH"},
    "TOOTH":        {"TOOTH"},
    "TONGUE":       {"TONGUE"},
    "CLAW":         {"CLAW", "FINGERNAIL", "NAIL"},
    "FOOT":         {"FOOT"},
    "KNEE":         {"KNEE"},
    "HAND":         {"HAND"},
    "BELLY":        {"BELLY", "ABDOMEN", "STOMACH"},
    "NECK":         {"NECK"},
    "BREAST":       {"BREAST"},
    "HEART":        {"HEART"},
    "LIVER":        {"LIVER"},
    "DRINK":        {"DRINK"},
    "EAT":          {"EAT"},
    "BITE":         {"BITE"},
    "SEE":          {"SEE"},
    "HEAR":         {"HEAR"},
    "KNOW":         {"KNOW", "KNOW (A FACT)"},
    "SLEEP":        {"SLEEP"},
    "DIE":          {"DIE"},
    "KILL":         {"KILL"},
    "SWIM":         {"SWIM"},
    "FLY":          {"FLY (MOVE THROUGH AIR)"},
    "WALK":         {"WALK", "GO"},
    "COME":         {"COME"},
    "LIE":          {"LIE (RECLINE)", "LIE DOWN", "LIE"},
    "SIT":          {"SIT"},
    "STAND":        {"STAND"},
    "GIVE":         {"GIVE"},
    "SAY":          {"SAY", "SPEAK"},
    "SUN":          {"SUN"},
    "MOON":         {"MOON"},
    "STAR":         {"STAR"},
    "WATER":        {"WATER"},
    "RAIN":         {"RAIN", "RAIN (PRECIPITATION)"},
    "STONE":        {"STONE", "ROCK"},
    "SAND":         {"SAND"},
    "EARTH":        {"EARTH", "SOIL", "GROUND"},
    "CLOUD":        {"CLOUD"},
    "SMOKE":        {"SMOKE"},
    "FIRE":         {"FIRE"},
    "ASH":          {"ASH", "ASHES"},
    "BURN":         {"BURN"},
    "PATH":         {"PATH", "ROAD", "WAY"},
    "MOUNTAIN":     {"MOUNTAIN"},
    "RED":          {"RED"},
    "GREEN":        {"GREEN"},
    "YELLOW":       {"YELLOW"},
    "WHITE":        {"WHITE"},
    "BLACK":        {"BLACK"},
    "NIGHT":        {"NIGHT"},
    "HOT":          {"HOT", "WARM"},
    "COLD":         {"COLD"},
    "FULL":         {"FULL"},
    "NEW":          {"NEW"},
    "GOOD":         {"GOOD"},
    "ROUND":        {"ROUND"},
    "DRY":          {"DRY"},
    "NAME":         {"NAME"},
}
GLOSS_REMAP = {
    "BLUNT":               "DULL",
    "FAT (ORGANIC SUBSTANCE)": "GREASE",
    "FEAR (TO BE AFRAID)": "FEAR (BE AFRAID)",
    "LIE (REST)":          "LIE",
    "RAINING OR RAIN":     "RAIN",
    "ROAD":                "PATH",
    "BE ALIVE":            "LIVE",
    "BLOW (OF WIND)":      "BLOW",
    "EARTH (SOIL)":            "EARTH",
    "FINGERNAIL OR TOENAIL":   "CLAW", 
    "FLY (MOVE THROUGH AIR)":  "FLY",
    "HORN (ANATOMY)":          "HORN",
    "SMOKE (EXHAUST)":         "SMOKE"
}
LATIN_SUPPLEMENT = {
    "BACK":              ["tergum"],
    "BAD":               ["malus"],
    "BREATHE":           ["spīrāre"],
    "COUNT":             ["numerāre"],
    "DAY (NOT NIGHT)":   ["diēs"],
    "DIG":               ["fodere"],
    "DIRTY":             ["sordidus"],
    "DUST":              ["puluis"],
    "EARTH":      ["terra"],
    "FALL":              ["cadere"],
    "FAR":               ["longe"],
    "FEAR (BE AFRAID)":  ["timēre"],
    "FIGHT":             ["pugnāre"],
    "FIVE":              ["quīnque"],
    "FLOWER":            ["flōs"],
    "FLY": ["uolāre"],
    "FOG":               ["nebula"],
    "FOREST":            ["silua"],
    "FOUR":              ["quattuor"],
    "FREEZE":            ["gelāre"],
    "FRUIT":             ["frūctus"],
    "GRASS":             ["herba"],
    "HEAVY":             ["grauis"],
    "HIT":               ["ferīre"],
    "HORN":    ["cornū"],
    "HUNT":              ["uenārī"],
    "ICE":               ["glaciēs"],
    "LAKE":              ["lacus"],
    "LAUGH":             ["rīdēre"],
    "LEFT":              ["sinister"],
    "LEG":               ["crūs"],
    "LIVE":              ["uīuere"],
    "MEAT":              ["carō"],
    "NARROW":            ["angustus"],
    "NEAR":              ["prope"],
    "OLD":               ["uetus"],
    "PLAY":              ["lūdere"],
    "PULL":              ["trahere"],
    "PUSH":              ["pellere"],
    "RIGHT":             ["dexter"],
    "RIVER":             ["flūmen"],
    "ROTTEN":            ["putridus"],
    "SALT":              ["sāl"],
    "SEA":               ["mare"],
    "SEW":               ["suere"],
    "SHARP":             ["acūtus"],
    "SHORT":             ["breuis"],
    "SING":              ["cantāre"],
    "SKY":               ["caelum"],
    "SMOKE":   ["fūmus"],
    "SMOOTH":            ["lēuis"],
    "SNAKE":             ["serpēns"],
    "SNOW":              ["nix"],
    "STICK":             ["baculum"],
    "STRAIGHT":          ["rēctus"],
    "SWELL":             ["tumēre"],
    "THICK":             ["crassus"],
    "THREE":             ["trēs"],
    "THROW":             ["iacere"],
    "TIE":               ["ligāre"],
    "TURN":              ["uertere"],
    "VOMIT":             ["uomere"],
    "WASH":              ["lauāre"],
    "WET":               ["ūmidus"],
    "WIDE":              ["lātus"],
    "WIND":              ["uentus"],
    "WING":              ["āla"],
    "WORM":              ["uermis"],
    "YEAR":              ["annus"],
    "BELLY":             ["uenter"],
    "BIG":               ["magnus"],
    "EAT":               ["edere"],
    "FIRE":              ["ignis"],
    "KILL":              ["occīdere"],
    "KNOW":              ["scīre"],
    "LIVER":             ["iecur"],
    "MAN":               ["homō"],
    "MOUTH":             ["ōs"],
    "SAND":              ["harēna"],
    "SKIN":              ["pellis"],
    "SMALL":             ["paruus"],
    "SMELL":             ["sentīre"],
    "STAND":             ["stāre"],
    "STONE":             ["lapis"],
    "THINK":             ["pēnsāre"],
    "WHITE":             ["albus"],
    "WOMAN":             ["fēmina"],
    "YELLOW":            ["flāuus"],
    "WALK":              ["ambulāre"],
}
ROMANCE_SUPPLEMENT = {
    "French": {
        "BACK": ["dos"], "BELLY": ["ventre"], "BURN": ["brûler"], "DIG": ["creuser"],
        "DIRTY": ["sale"], "FALL": ["tomber"], "FOG": ["brouillard"], "HEAD": ["tête"],
        "HEAR": ["entendre"], "HEAVY": ["lourd"], "HIT": ["frapper"], "KILL": ["tuer"],
        "LEFT": ["gauche"], "MEAT": ["viande"], "RIVER": ["rivière"], "SEED": ["graine"],
        "SHARP": ["aigu"], "SMALL": ["petit"], "STAND": ["se tenir"], "THICK": ["épais"],
        "TIE": ["lier"], "TURN": ["tourner"]
    },
    "Italian": {
        "BACK": ["schiena"], "BAD": ["cattivo"], "BELLY": ["pancia"], "BURN": ["bruciare"],
        "DIRTY": ["sporco"], "DRY": ["secco"], "HEAD": ["testa"], "HEAR": ["sentire"],
        "LEFT": ["sinistra"], "NEAR": ["vicino"], "PUSH": ["spingere"], "RIGHT": ["destra"],
        "RIVER": ["fiume"], "ROTTEN": ["marcio"], "SMALL": ["piccolo"], "STONE": ["pietra"],
        "THROW": ["gettare"], "WET": ["bagnato"], "WOMAN": ["donna"], "YELLOW": ["giallo"]
    },
    "Portuguese": {
        "BACK": ["costas"], "BLACK": ["preto"], "EAT": ["comer"], "FEAR (BE AFRAID)": ["temer"],
        "FEATHER": ["pena"], "FIGHT": ["lutar"], "HIT": ["bater"], "ICE": ["gelo"],
        "KILL": ["matar"], "LEG": ["perna"], "LONG": ["longo"], "NEAR": ["perto"],
        "NECK": ["pescoço"], "PULL": ["puxar"], "PUSH": ["empurrar"], "RED": ["vermelho"],
        "SAND": ["areia"], "SMELL": ["cheirar"], "STICK": ["pau"], "TAIL": ["cauda"],
        "TIE": ["amarrar"], "WOMAN": ["mulher"], "FOREST": ["floresta"], "YELLOW": ["amarelo"],
        "HORN": ["chifre"]
    },
    "Romanian": {
        "BAD": ["rău"], "BELLY": ["burtă"], "BIG": ["mare"], "BIRD": ["pasăre"],
        "BREATHE": ["respira"], "COLD": ["rece"], "COUNT": ["număra"], "DIG": ["săpa"],
        "DIRTY": ["murdar"], "DRY": ["uscat"], "DUST": ["praf"], "EARTH": ["pământ"],
        "FAR": ["departe"], "FEAR (BE AFRAID)": ["teme"], "FEATHER": ["pană"], "FOG": ["ceață"],
        "FREEZE": ["îngheța"], "HAIR": ["păr"], "HEART": ["inimă"], "HEAVY": ["greu"],
        "HIT": ["lovi"], "HUNT": ["vâna"], "KNOW": ["ști"], "LEAF": ["frunză"],
        "LEFT": ["stânga"], "LEG": ["picior"], "MAN": ["bărbat"], "MOUTH": ["gură"],
        "NEAR": ["aproape"], "NECK": ["gât"], "PULL": ["trage"], "PUSH": ["împinge"],
        "SAND": ["nisip"], "SHARP": ["ascuțit"], "SMALL": ["mic"], "SMELL": ["mirosi"],
        "SMOOTH": ["neted"], "SNOW": ["zăpadă"], "STICK": ["băț"], "THINK": ["gândi"],
        "THROW": ["arunca"], "TREE": ["copac"], "TURN": ["întoarce"], "WET": ["ud"],
        "WHITE": ["alb"], "WIDE": ["larg"], "FOREST": ["pădure"], "WALK": ["merge"]
    },
    "Spanish": {
        "BURN": ["quemar"], "DOG": ["perro"], "EAT": ["comer"], "FAR": ["lejos"],
        "FEAR (BE AFRAID)": ["temer"], "FIGHT": ["pelear"], "HAIR": ["pelo"], "ICE": ["hielo"],
        "KILL": ["matar"], "LEG": ["pierna"], "LONG": ["largo"], "NEAR": ["cerca"],
        "SAND": ["arena"], "SMELL": ["oler"], "STICK": ["palo"], "THICK": ["grueso"],
        "THROW": ["tirar"], "TIE": ["atar"], "WIDE": ["ancho"], "WOMAN": ["mujer"],
        "FOREST": ["bosque"], "WORM": ["gusano"], "YELLOW": ["amarillo"], "KNEE": ["rodilla"]
    }
}

def build_and_expand():    
    # build unified target list (Swadesh 100 + Dunn 207)
    target_glosses = {}
    
    # swadesh 100
    for canonical, variants in SWADESH_100.items():
        for v in variants:
            target_glosses[v.upper()] = canonical
            
    # dunn 207
    dunn_cids = {}
    if CONCEPTS_CSV.exists():
        with open(CONCEPTS_CSV, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                gloss = row["CONCEPTICON_GLOSS"].strip().upper()
                cid = row["CONCEPTICON_ID"].strip()
                if gloss and cid:
                    canonical = GLOSS_REMAP.get(gloss, gloss)
                    target_glosses[gloss] = canonical
                    dunn_cids[cid] = canonical
    else:
        print(f"{CONCEPTS_CSV} not found")

    # query CLDF
    ds = Dataset.from_metadata(str(IELEX_CLDF_PATH))
    lang_lookup = {r["ID"]: r["Name"] for r in ds.iter_rows("LanguageTable") if r["Name"] in TARGET_LANGUAGES}
    
    param_lookup = {}
    for r in ds.iter_rows("ParameterTable"):
        cid = str(r.get("Concepticon_ID", "")).strip()
        if cid in dunn_cids:
            param_lookup[r["ID"]] = dunn_cids[cid]
            continue
            
        candidates = set(val.upper().strip() for f in ("Name", "Concepticon_Gloss", "Gloss") if (val := r.get(f)))
        for gloss in candidates:
            if gloss in target_glosses:
                param_lookup[r["ID"]] = target_glosses[gloss]
                break

    cognate_lookup = {r["Form_ID"]: r.get("Cognateset_ID", "UNKNOWN") for r in ds.iter_rows("CognateTable")} if "CognateTable" in ds else {}

    # collect forms
    raw_result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in ds.iter_rows("FormTable"):
        lang_name, canonical = lang_lookup.get(r["Language_ID"]), param_lookup.get(r["Parameter_ID"])
        if not lang_name or not canonical: continue
        form = r.get("Form") or r.get("Value", "")
        cog_id = cognate_lookup.get(r["ID"], f"UNLINKED_{r['ID']}")
        if form and form not in raw_result[canonical][cog_id][lang_name]:
            raw_result[canonical][cog_id][lang_name].append(form)

    # select cognate sets based on romance coverage
    dataset = {}
    for canonical, cognate_groups in raw_result.items():
        best_cog_id, best_romance_count, best_total_forms = None, -1, -1
        for cog_id, langs in cognate_groups.items():
            romance_langs = set(langs.keys()) - {"Latin"}
            total_forms = sum(len(v) for k, v in langs.items() if k != "Latin")
            
            if len(romance_langs) > best_romance_count or (len(romance_langs) == best_romance_count and total_forms > best_total_forms):
                best_romance_count, best_total_forms, best_cog_id = len(romance_langs), total_forms, cog_id
                
        if best_cog_id:
            dataset[canonical] = {"concept": canonical.lower(), "gloss": canonical, "languages": dict(cognate_groups[best_cog_id])}

    # supplements
    for gloss, entry in dataset.items():
        
        # latin
        existing_latin = entry["languages"].get("Latin", [])
        if not [f for f in existing_latin if f and str(f).strip()] and gloss in LATIN_SUPPLEMENT:
            entry["languages"]["Latin"] = LATIN_SUPPLEMENT[gloss]
            
        # romance
        for lang in TARGET_LANGUAGES - {"Latin"}:
            existing_romance = entry["languages"].get(lang, [])
            if not [f for f in existing_romance if f and str(f).strip()]:
                if lang in ROMANCE_SUPPLEMENT and gloss in ROMANCE_SUPPLEMENT[lang]:
                    entry["languages"][lang] = ROMANCE_SUPPLEMENT[lang][gloss]
            
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        
    print(f"{len(dataset)} concepts added to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_and_expand()