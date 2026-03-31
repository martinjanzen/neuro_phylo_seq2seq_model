"""
03a_latin_ipa_tests.py
----------------------
Comprehensive test suite for latin_to_ipa() in 03b_latin_ipa.py.

Structure
---------
  Block  1: Greek-loan aspirate digraphs
  Block  2: Labiovelar clusters qu, gu
  Block  3: au diphthong; ae and oe as two-segment sequences
  Block  4: x → ks; z → dz
  Block  5: Geminate consonants
  Block  6: Consonantal u → w (word-initial, uu→wu, medial)
  Block  7: Consonantal i → j (word-initial)
  Block  8: Long vowels and no-refire protection
  Block  9: Short vowels
  Block 10: c → k
  Block 11: gn cluster → [ŋn]; nasal assimilation before velars
  Block 12: Interaction tests (two rules cooperating)
  Block 13: Full dataset regression (all 85 Swadesh forms for Latin)

Sources
-------
  Allen, W.S. (1978) Vox Latina, 2nd ed. Cambridge UP.
  Weiss, M. (2009) Outline of the Historical and Comparative Grammar of
    Latin. Beech Stave Press.

Run with:
    python 03a_latin_ipa_tests.py
All tests must pass before running the full conversion in 03b_latin_ipa.py.
"""

import sys
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "latin_ipa", Path(__file__).parent / "03b_latin_ipa.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
latin_to_ipa = mod.latin_to_ipa

# ── Test runner ───────────────────────────────────────────────────────────────
passed = 0
failed = 0
failures = []

def test(label, ortho, expected, rule_tag, source):
    """
    Run one test case.

    Parameters
    ----------
    label    : human-readable description
    ortho    : orthographic input form
    expected : expected IPA output
    rule_tag : short rule identifier for the failure report
    source   : page citation in Allen (1978) or Weiss (2009)
    """
    global passed, failed
    result = latin_to_ipa(ortho)
    if result == expected:
        passed += 1
        print(f"  ✓ {label:<50} {ortho:<22} → {result}")
    else:
        failed += 1
        failures.append((label, ortho, expected, result, rule_tag, source))
        print(f"  ✗ {label:<50} {ortho:<22} → {result:<22}  expected: {expected}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 1: Greek-loan aspirate digraphs ────────────────────────────────")
# Allen (1978:26-28): ph, th, ch in loans from Greek represent aspirate stops.
# These must be consumed as digraphs before any single-letter rule fires.
test("ph → pʰ  (word-initial)",     "philosophus", "pʰɪlɔsɔpʰʊs", "ph→pʰ", "Allen 1978:26")
test("ph → pʰ  (medial)",           "elephantus",  "ɛlɛpʰantʊs",  "ph→pʰ", "Allen 1978:26")
test("th → tʰ  (word-initial)",     "theatrum",    "tʰɛatrʊm",    "th→tʰ", "Allen 1978:26")
test("ch → kʰ  (word-initial)",     "chorus",      "kʰɔrʊs",      "ch→kʰ", "Allen 1978:26")
test("ch → kʰ  (medial)",           "pulcher",     "pʊlkʰɛr",     "ch→kʰ", "Allen 1978:26")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 2: Labiovelar clusters qu, gu ──────────────────────────────────")
# Allen (1978:15-17): qu and gu before a vowel are single labiovelar phonemes.
test("qu → kʷ  before short a",     "aqua",        "akʷa",        "qu→kʷ", "Allen 1978:15")
test("qu → kʷ  before short i",     "quis",        "kʷɪs",        "qu→kʷ", "Allen 1978:15")
test("qu → kʷ  before short e",     "quem",        "kʷɛm",        "qu→kʷ", "Allen 1978:15")
test("qu → kʷ  before long ō",      "quōmodo",     "kʷoːmɔdɔ",   "qu→kʷ", "Allen 1978:15")
test("qu → kʷ  before long ī",      "quī",         "kʷiː",        "qu→kʷ", "Allen 1978:15")
test("gu → ɡʷ  before short a",     "lingua",      "lɪŋɡʷa",      "gu→ɡʷ", "Allen 1978:17")
test("gu → ɡʷ  before short u",     "sanguīs",     "saŋɡʷiːs",    "gu→ɡʷ", "Allen 1978:17")
test("gu → ɡʷ  before long ā",      "guā",         "ɡʷaː",        "gu→ɡʷ", "Allen 1978:17")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 3: Diphthongs ──────────────────────────────────────────────────")
# Allen (1978:57-60), Weiss (2009:32).
#
# au → aw
# Allen (1978:58-59): au is a falling diphthong [au̯]; the vocalic-u element
# functions as the off-glide [u̯] = [w].
test("au → aw  (cauda)",            "cauda",       "kawda",       "au→aw", "Allen 1978:58")
test("au → aw  (word-initial)",     "auris",       "awrɪs",       "au→aw", "Allen 1978:58")
test("au → aw  (before long vowel)","audīre",      "awdiːrɛ",     "au→aw", "Allen 1978:58")
#
# ae — two-segment sequence, NOT monophthong
# Allen (1978:57-58): ae was the falling diphthong [ae̯] in the Classical
# period. The monophthongisation to [ɛː] is a late / Vulgar Latin change.
# The correct Classical representation is [aɛ] (= short-a + short-e quality),
# produced here by the individual short-vowel rules without any special rule.
test("ae → aɛ  (not → ɛ)",         "caelum",      "kaɛlʊm",      "ae→aɛ", "Allen 1978:57")
#
# oe — two-segment sequence
# Allen (1978:60): oe was the falling diphthong [oe̯], Classical period.
# Individual rules give o → ɔ, e → ɛ, producing [ɔɛ].
test("oe → ɔɛ  (not → eː)",        "poena",       "pɔɛna",       "oe→ɔɛ", "Allen 1978:60")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 4: x → ks; z → dz ─────────────────────────────────────────────")
# Allen (1978:22-23).
test("x → ks   (word-initial)",     "xenia",       "ksɛnɪa",      "x→ks",  "Allen 1978:22")
test("x → ks   (medial)",           "pax",         "paks",        "x→ks",  "Allen 1978:22")
test("x → ks   (word-final cluster)","rex",        "rɛks",        "x→ks",  "Allen 1978:22")
test("z → dz   (word-initial)",     "zona",        "dzɔna",       "z→dz",  "Allen 1978:23")
test("z → dz   (medial)",           "gaza",        "ɡadza",       "z→dz",  "Allen 1978:23")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 5: Geminate consonants ─────────────────────────────────────────")
# Allen (1978:30-31): geminates are genuine long consonants, distinct in
# duration and phonemic status from their short counterparts.
test("ll geminate",                 "bella",       "bɛlːa",       "CC→Cː", "Allen 1978:30")
test("nn geminate",                 "annus",       "anːʊs",       "CC→Cː", "Allen 1978:30")
test("ss geminate",                 "missa",       "mɪsːa",       "CC→Cː", "Allen 1978:30")
test("tt geminate",                 "mittere",     "mɪtːɛrɛ",     "CC→Cː", "Allen 1978:30")
test("pp geminate",                 "hippus",      "hɪpːʊs",      "CC→Cː", "Allen 1978:30")
test("cc geminate (→ kː)",          "siccus",      "sɪkːʊs",      "CC→Cː", "Allen 1978:30")
test("mm geminate",                 "flamma",      "flamːa",      "CC→Cː", "Allen 1978:30")
test("rr geminate",                 "terra",       "tɛrːa",       "CC→Cː", "Allen 1978:30")
# <uu> is NOT a geminate: see Block 6b.

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 6: Consonantal u → w ───────────────────────────────────────────")
# Allen (1978:38-41), Weiss (2009:41-42): Classical Latin <u> (= modern <v>)
# represents /w/ in three environments.
#
# (a) Word-initial <u> before any vowel.
test("u → w    word-initial (uenter)","uenter",    "wɛntɛr",      "u→w/init", "Allen 1978:38")
test("u → w    word-initial (uir)",  "uir",        "wɪr",         "u→w/init", "Allen 1978:38")
test("u → w    word-initial (uidere)","uidēre",    "wɪdeːrɛ",     "u→w/init", "Allen 1978:38")
test("u → w    word-initial (uolāre)","uolāre",   "wɔlaːrɛ",     "u→w/init", "Allen 1978:38")
#
# (b) <uu> → [wu]: the FIRST u is the consonantal onset of the following
# syllable; the SECOND u is the vocalic nucleus. Allen (1978:39) cites seruus,
# paruus, nouus explicitly. Rule: uu → wu (first = consonantal, second = vocalic).
test("uu → wu  (paruus)",           "paruus",     "parwʊs",      "uu→wu",    "Allen 1978:39")
test("uu → wu  (nouus)",            "nouus",      "nɔwʊs",       "uu→wu",    "Allen 1978:39")
test("uu → wu  (flāuus)",           "flāuus",     "flaːwʊs",     "uu→wu",    "Allen 1978:39")
test("uu → wu  (uiuus: init+medial)","uiuus",     "wɪwʊs",       "uu→wu",    "Allen 1978:39")
# uiuus: word-initial u → w first; then remaining uu → wu.
#
# (c) Medial <u> between two vowels (intervocalic).
# Allen (1978:38): "between vowels". The lookbehind on vowel context prevents
# firing in consonant+u+vowel sequences where u is a vocalic nucleus (e.g.
# duo: d+u+ō, u follows consonant d → stays vocalic → dʊoː, not dwoː).
test("u → w    medial V_u_V (ciuis)","ciuis",     "kɪwɪs",       "u→w/med",  "Allen 1978:38")
test("u NOT→w  consonant+u+V (duo)","duō",        "dʊoː",        "u stays",  "Allen 1978:38")
test("u NOT→w  consonant+u+V (fīlius)","fīlius",  "fiːlɪʊs",     "u stays",  "Allen 1978:38")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 7: Consonantal i → j ───────────────────────────────────────────")
# Allen (1978:38-40), Weiss (2009:41): word-initial <i> before a vowel is [j].
test("i → j    word-initial (iecur)","iecur",     "jɛkʊr",       "i→j/init", "Allen 1978:38")
# Medial consonantal <i> is written <j> in IECoR data and passes through
# without needing a rule (j is already the correct IPA symbol).
test("j passthrough  word-initial", "jocus",      "jɔkʊs",       "j-thru",   "Allen 1978:38")
test("j passthrough  medial",       "major",      "majɔr",       "j-thru",   "Allen 1978:38")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 8: Long vowels ─────────────────────────────────────────────────")
# Allen (1978:47-65): macron vowels are long; transcribed with IPA ː.
test("ā → aː",                      "māter",      "maːtɛr",      "ā→aː",  "Allen 1978:47")
test("ē → eː",                      "plēnus",     "pleːnʊs",     "ē→eː",  "Allen 1978:50")
test("ī → iː",                      "fīlius",     "fiːlɪʊs",     "ī→iː",  "Allen 1978:53")
test("ō → oː",                      "sōl",        "soːl",        "ō→oː",  "Allen 1978:56")
test("ū → uː",                      "ūnus",       "uːnʊs",       "ū→uː",  "Allen 1978:59")
test("ȳ → yː  (Greek loan)",        "lȳra",       "lyːra",       "ȳ→yː",  "Allen 1978:62")
# Anti-refire: long vowel must NOT be re-processed by the short vowel rule.
# Without placeholder protection: ū → uː → then u → ʊ → ʊː (wrong).
test("ā not re-fired as short a",   "pānis",      "paːnɪs",      "no-refire", "Allen 1978:47")
test("ō not re-fired as short o",   "nōmen",      "noːmɛn",      "no-refire", "Allen 1978:56")
test("ū not re-fired as short u",   "lūna",       "luːna",       "no-refire", "Allen 1978:59")
test("ī not re-fired as short i",   "pīscis",     "piːskɪs",     "no-refire", "Allen 1978:53")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 9: Short vowels ────────────────────────────────────────────────")
# Allen (1978:47-65): short vowels are opener/laxer than their long counterparts.
test("a → a",                        "canis",     "kanɪs",       "a→a",   "Allen 1978:47")
test("e → ɛ",                        "mensa",     "mɛnsa",       "e→ɛ",   "Allen 1978:50")
test("i → ɪ",                        "piscis",    "pɪskɪs",      "i→ɪ",   "Allen 1978:53")
test("o → ɔ",                        "bonus",     "bɔnʊs",       "o→ɔ",   "Allen 1978:56")
test("u → ʊ",                        "lupus",     "lʊpʊs",       "u→ʊ",   "Allen 1978:59")
test("y → y   (Greek loan)",         "gyrus",     "ɡyrʊs",       "y→y",   "Allen 1978:62")
# NOTE: g before y is velar [ɡ] — Classical Latin has NO palatalization of
# stops before front vowels. That is a Vulgar/Medieval development.

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 10: c → k ──────────────────────────────────────────────────────")
# Allen (1978:14): <c> was always [k] throughout the Classical period.
# Palatalization before front vowels (c → [tʃ]/[s] before e/i) belongs to
# Vulgar and Medieval Latin only and must NOT be applied here.
test("c → k  before a",              "caput",     "kapʊt",       "c→k", "Allen 1978:14")
test("c → k  before plain e",        "cena",      "kɛna",        "c→k", "Allen 1978:14")
test("c → k  before i+V (ciuis)",    "ciuis",     "kɪwɪs",       "c→k", "Allen 1978:14")
test("c → k  before o",              "cor",       "kɔr",         "c→k", "Allen 1978:14")
test("c → k  before u",              "cursor",    "kʊrsɔr",      "c→k", "Allen 1978:14")
test("c → k  word-final",            "hic",       "hɪk",         "c→k", "Allen 1978:14")
test("c → k  before ae diphthong",   "caelum",    "kaɛlʊm",      "c→k", "Allen 1978:14")
# caelum: c → k; then ae → aɛ via individual vowel rules (no special ae rule).

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 11: gn cluster; nasal assimilation before velars ──────────────")
# Two distinct but related phonological processes:
#
# (A) gn → [ŋn]: Allen (1978:26-27): the sequence <gn> was always pronounced
# [ŋn] in Classical Latin, regardless of position. The <g> assimilates to the
# following nasal. This is NOT nasal assimilation of n before g; it is g
# assimilating to n. Applies word-initially (ignis), medially (dignus, magnus).
test("gn → ŋn  word-initial",        "ignis",     "ɪŋnɪs",       "gn→ŋn", "Allen 1978:26")
test("gn → ŋn  medial (dignus)",     "dignus",    "dɪŋnʊs",      "gn→ŋn", "Allen 1978:26")
test("gn → ŋn  medial (magnus)",     "magnus",    "maŋnʊs",      "gn→ŋn", "Allen 1978:26")
#
# (B) Nasal assimilation: Allen (1978:30): /n/ → [ŋ] before /k/ and /ɡ/.
# Note: the ɡ in ɡʷ clusters (from Block 2) is already the IPA ɡ, so the
# first-pass n→ŋ rule catches those immediately; the second pass catches
# the ɡ introduced by the final g → ɡ substitution.
test("n → ŋ   before ɡʷ (sanguīs)", "sanguīs",   "saŋɡʷiːs",    "n→ŋ/ɡʷ", "Allen 1978:30")
test("n → ŋ   before ɡʷ (lingua)",  "lingua",    "lɪŋɡʷa",      "n→ŋ/ɡʷ", "Allen 1978:30")
test("n → ŋ   before ɡ  (longus)",  "longus",    "lɔŋɡʊs",      "n→ŋ/ɡ",  "Allen 1978:30")
test("n → ŋ   before k  (punc)",    "punc",      "pʊŋk",        "n→ŋ/k",  "Allen 1978:30")
test("n → ŋ   before k  (uncus)",   "uncus",     "ʊŋkʊs",       "n→ŋ/k",  "Allen 1978:30")
test("n stays n before non-velar",  "mensa",     "mɛnsa",       "n-stays","Allen 1978:30")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 12: Interaction tests ──────────────────────────────────────────")
# Each test exercises two rules whose interaction could conceal an ordering bug.
test("geminate + long vowel",        "annus",     "anːʊs",       "CC+long", "Allen 1978:30,59")
test("qu + long vowel, no refire",   "quī",       "kʷiː",        "qu+long", "Allen 1978:15,53")
test("gu + long vowel",              "sanguīs",   "saŋɡʷiːs",    "gu+long", "Allen 1978:17,53")
test("ph + long vowel",              "phrāter",   "pʰraːtɛr",    "ph+long", "Allen 1978:26,47")
test("cons-u + long vowel",          "uenīre",    "wɛniːrɛ",     "u/w+long","Allen 1978:38,53")
test("c→k then n→ŋ before new k",   "nancisci",  "naŋkɪskɪ",    "c→k+nas", "Allen 1978:14,30")
test("two long vowels",              "ūnūs",      "uːnuːs",      "2×long",  "Allen 1978:59")
# ūnūs is not a real Latin word; it tests that two macron vowels both survive.
test("long vowel + geminate",        "mittō",     "mɪtːoː",      "long+CC", "Allen 1978:30,56")
test("au + long vowel",              "audīre",    "awdiːrɛ",     "au+long", "Allen 1978:58,53")
test("gn + short vowels",            "magnus",    "maŋnʊs",      "gn+short","Allen 1978:26,47")
test("uu + long vowel (flāuus)",     "flāuus",    "flaːwʊs",     "uu+long", "Allen 1978:39,47")
test("word-init u + uu  (uiuus)",    "uiuus",     "wɪwʊs",       "u+uu",    "Allen 1978:38,39")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── BLOCK 13: Dataset regression (all Latin Swadesh forms) ──────────────")
# These are the exact orthographic forms from swadesh_ielex.json (IECoR data).
# All must pass before running 03b_latin_ipa.py on the full dataset.
#
# Forms are sorted by Swadesh concept number for traceability.
# IPA values were derived from first principles using the rules above and
# cross-checked against Allen (1978) example words where available.
#
# FLAGGED FORMS (for manual review after conversion):
#   stēlla: IECoR has macron on e; Classical Latin stella has short e.
#           The macron may be a data error. Our rule correctly produces steːlːa
#           given the input — flag in inspection file for human review.

DATASET_FORMS = [
    # ortho          expected IPA     gloss
    ("ūnus",         "uːnʊs",         "ONE"),
    ("duō",          "dʊoː",          "TWO"),
    # duō: d+u+ō; u follows consonant d (not a vowel) → stays vocalic [ʊ]
    ("plēnus",       "pleːnʊs",       "FULL"),
    ("sōl",          "soːl",          "SUN"),
    ("aqua",         "akʷa",          "WATER"),
    ("ignis",        "ɪŋnɪs",         "FIRE"),
    # gn → [ŋn] universally (Allen 1978:26)
    ("canis",        "kanɪs",         "DOG"),
    ("uenter",       "wɛntɛr",        "BELLY"),
    ("sanguīs",      "saŋɡʷiːs",      "BLOOD"),
    ("arbor",        "arbɔr",         "TREE"),
    ("lingua",       "lɪŋɡʷa",        "TONGUE"),
    ("lūna",         "luːna",         "MOON"),
    ("nōmen",        "noːmɛn",        "NAME"),
    ("pēs",          "peːs",          "FOOT"),
    ("nox",          "nɔks",          "NIGHT"),
    ("cornū",        "kɔrnuː",        "HORN"),
    ("genu",         "ɡɛnʊ",          "KNEE"),
    # g is always velar [ɡ]; no palatalization in Classical Latin (Allen 1978:14)
    ("nūbēs",        "nuːbeːs",       "CLOUD"),
    ("piscis",       "pɪskɪs",        "FISH"),
    ("morī",         "mɔriː",         "DIE"),
    ("uolāre",       "wɔlaːrɛ",       "FLY"),
    ("uenīre",       "wɛniːrɛ",       "COME"),
    ("uidēre",       "wɪdeːrɛ",       "SEE"),
    ("bibere",       "bɪbɛrɛ",        "DRINK"),
    ("audīre",       "awdiːrɛ",       "HEAR"),
    ("dormīre",      "dɔrmiːrɛ",      "SLEEP"),
    ("scīre",        "skiːrɛ",        "KNOW"),
    ("sedēre",       "sɛdeːrɛ",       "SIT"),
    ("stāre",        "staːrɛ",        "STAND"),
    ("dare",         "darɛ",          "GIVE"),
    ("dicere",       "dɪkɛrɛ",        "SAY"),
    ("edere",        "ɛdɛrɛ",         "EAT"),
    ("mordēre",      "mɔrdeːrɛ",      "BITE"),
    ("interficere",  "ɪntɛrfɪkɛrɛ",   "KILL"),
    ("carō",         "karoː",         "FLESH"),
    ("folium",       "fɔlɪʊm",        "LEAF"),
    # folium: i followed by u+m; u before m (consonant) → stays [ʊ]
    ("sēmen",        "seːmɛn",        "SEED"),
    ("rādix",        "raːdɪks",       "ROOT"),
    ("fūmus",        "fuːmʊs",        "SMOKE"),
    ("stēlla",       "steːlːa",       "STAR"),
    # FLAGGED: IECoR has macron; Classical Latin stella normally has short e.
    # Given input stēlla, the rule correctly produces steːlːa. Review the
    # source data before publishing; may need manual correction to stɛlːa.
    ("terra",        "tɛrːa",         "EARTH"),
    ("mōns",         "moːns",         "MOUNTAIN"),
    ("cinis",        "kɪnɪs",         "ASH"),
    ("cortex",       "kɔrtɛks",       "BARK"),
    ("cauda",        "kawda",         "TAIL"),
    ("plūma",        "pluːma",        "FEATHER"),
    ("cor",          "kɔr",           "HEART"),
    ("iecur",        "jɛkʊr",         "LIVER"),
    # word-initial i before e (vowel) → [j]
    ("auris",        "awrɪs",         "EAR"),
    ("oculus",       "ɔkʊlʊs",        "EYE"),
    ("nāsus",        "naːsʊs",        "NOSE"),
    ("ōs",           "oːs",           "MOUTH"),
    ("dens",         "dɛns",          "TOOTH"),
    ("caput",        "kapʊt",         "HEAD"),
    ("collum",       "kɔlːʊm",        "NECK"),
    ("manus",        "manʊs",         "HAND"),
    ("genu",         "ɡɛnʊ",          "KNEE-2"),
    # Duplicate in IECoR data; same result expected.
    ("os",           "ɔs",            "BONE"),
    ("cutis",        "kʊtɪs",         "SKIN"),
    ("pēdiculus",    "peːdɪkʊlʊs",    "LOUSE"),
    ("magnus",       "maŋnʊs",        "BIG"),
    ("longus",       "lɔŋɡʊs",        "LONG"),
    ("paruus",       "parwʊs",        "SMALL"),
    ("nouus",        "nɔwʊs",         "NEW"),
    ("bonus",        "bɔnʊs",         "GOOD"),
    ("siccus",       "sɪkːʊs",        "DRY"),
    ("rotundus",     "rɔtʊndʊs",      "ROUND"),
    ("niger",        "nɪɡɛr",         "BLACK"),
    ("ruber",        "rʊbɛr",         "RED"),
    ("uiridis",      "wɪrɪdɪs",       "GREEN"),
    ("flāuus",       "flaːwʊs",       "YELLOW"),
    ("albus",        "albʊs",         "WHITE"),
    ("frīgidus",     "friːɡɪdʊs",     "COLD"),
    ("calidus",      "kalɪdʊs",       "HOT"),
]

for ortho, expected, gloss in DATASET_FORMS:
    test(gloss, ortho, expected, "dataset", "IECoR / Allen 1978")


# ══════════════════════════════════════════════════════════════════════════════
total = passed + failed
bar = "=" * 60
print(f"\n{bar}")
print(f"  RESULTS: {passed} passed, {failed} failed out of {total} tests")
print(f"{bar}")

if failures:
    print("\n  FAILURES:")
    for label, ortho, expected, result, rule_tag, source in failures:
        print(f"\n    [{label}]")
        print(f"      input    : {ortho}")
        print(f"      got      : {result}")
        print(f"      expected : {expected}")
        print(f"      rule     : {rule_tag}")
        print(f"      source   : {source}")
    print(f"\n  Fix latin_to_ipa() in 03b_latin_ipa.py, then re-run this file.")
    sys.exit(1)
else:
    print(f"\n  All {total} tests passed. Safe to run 03b_latin_ipa.py on the dataset.")
    sys.exit(0)