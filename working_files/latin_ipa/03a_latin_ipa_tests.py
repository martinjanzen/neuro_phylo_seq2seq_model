"""
03a_latin_ipa_tests.py
Comprehensive test suite for Vulgar Latin / Proto-Romance sound rules.
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

passed = 0
failed = 0
failures = []

def test(label, ortho, expected, source):
    global passed, failed
    result = latin_to_ipa(ortho)
    if result == expected:
        passed += 1
        print(f"  ✓ {label:<45} {ortho:<12} → {result}")
    else:
        failed += 1
        failures.append((label, ortho, expected, result, source))
        print(f"  ✗ {label:<45} {ortho:<12} → {result:<12} expected: {expected}")

print("\n── BLOCK 1: Vulgar Consonant Weakening (Herman 2000) ─────────────────")
test("Final -m drop (short u lowers)",   "lupum",      "lopo",      "Herman 2000:39") 
test("Initial h- drop",                "homō",       "ɔmo",       "Herman 2000:40")
test("ns -> s cluster reduction",      "mensis",     "meses",     "Herman 2000:41") 
test("Greek aspirates merge to plain", "philosophus","pelɔsɔpos", "Herman 2000:44")

print("\n── BLOCK 2: Pan-Romance Vowel Collapse (Alkire & Rosen 2010) ──────────")
test("Yodization (i in hiatus)",       "fīlius",     "filjos",    "Alkire 2010:24")
test("i -> e (short i lowers)",        "piscis",     "peskes",    "Alkire 2010:24")
test("ē -> e",                         "plēnus",     "plenos",    "Alkire 2010:24")
test("e -> ɛ (short e)",               "terra",      "tɛrːa",     "Alkire 2010:24")
test("ā -> a (short unstressed e->ɛ)", "māter",      "matɛr",     "Alkire 2010:24")
test("a -> a",                         "canis",      "kanes",     "Alkire 2010:24")
test("o -> ɔ (short o)",               "cor",        "kɔr",       "Alkire 2010:24")
test("ō -> o",                         "sōl",        "sol",       "Alkire 2010:24")
test("u -> o (short u lowers)",        "lupus",      "lopos",     "Alkire 2010:24")
test("ū -> u",                         "lūna",       "luna",      "Alkire 2010:24")
test("ae -> ɛ",                        "caelum",     "kɛlo",      "Alkire 2010:25")
test("oe -> e",                        "poena",      "pena",      "Alkire 2010:25")

print("\n── BLOCK 3: Glides and Lenition (Issue 11 Fixes) ────────────────────")
test("Word-initial u -> β (lenition)", "uenter",     "βɛntɛr",    "Alkire 2010:59")
test("Intervocalic u -> β",            "ciuis",      "keβes",     "Alkire 2010:59")
test("v printed as v -> β",            "vivo",       "βeβɔ",      "Alkire 2010:59")
test("Word-initial i -> j",            "iecur",      "jɛkor",     "Allen 1978:38")
test("Medial i -> j (Issue 11)",       "maior",      "majɔr",     "Allen 1978:38")
test("Medial i -> j (peior)",          "peior",      "pɛjɔr",     "Allen 1978:38")

print("\n── BLOCK 4: Regression (Swadesh Concept Verification) ────────────────")
test("ONE",    "ūnus",    "unos",    "Swadesh")
test("TWO",    "duō",     "do",      "Swadesh") 
test("FIRE",   "ignis",   "eŋnes",   "Swadesh")
test("BLOOD",  "sanguīs", "saŋɡʷis", "Swadesh")
test("TONGUE", "lingua",  "leŋɡʷa",  "Swadesh")
test("MOON",   "lūna",    "luna",    "Swadesh")
test("NIGHT",  "nox",     "nɔks",    "Swadesh")
test("FISH",   "piscis",  "peskes",  "Swadesh")
test("NEW",    "nouus",   "nɔβos",   "Swadesh") 

print("\n" + "=" * 60)
total = passed + failed
print(f"  RESULTS: {passed} passed, {failed} failed out of {total} tests")
print("=" * 60)

if failures:
    print("\n  FAILURES:")
    for label, ortho, expected, result, source in failures:
        print(f"    [{label}] input: {ortho} | expected: {expected} | got: {result} | source: {source}")
    sys.exit(1)
else:
    print("\n  All tests passed. Safe to run 03b_latin_ipa.py on the dataset.")
    sys.exit(0)