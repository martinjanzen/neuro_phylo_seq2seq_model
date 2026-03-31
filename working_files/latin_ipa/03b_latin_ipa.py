"""
03b_latin_ipa.py
----------------
Converts Classical Latin orthographic forms (with macrons) to broad IPA
using a deterministic rules-based approach.

Phonological basis
------------------
This function targets the phonemic inventory of the Classical Latin of the
late Republic / early Empire (~100 BCE – 100 CE), the period described by
the two primary sources:

  Allen, W.S. (1978) Vox Latina: A Guide to the Pronunciation of Classical
    Latin, 2nd ed. Cambridge University Press.
  Weiss, M. (2009) Outline of the Historical and Comparative Grammar of
    Latin. Beech Stave Press.

Input conventions (matching IECoR / CLDF data)
------------------------------------------------
  - Long vowels are marked with macrons: ā ē ī ō ū (ȳ for Greek loans)
  - The letter <u> represents BOTH vocalic /ʊ/ and consonantal /w/ (= Classical v)
  - The letter <i> represents BOTH vocalic /ɪ/ and consonantal /j/
  - The letter <v> does not appear in IECoR data; handled for robustness

Output
------
  Broad (phonemic-level) IPA. Allophonic detail (e.g. voicing assimilation
  of final -m, velarisation of l, tap vs. trill distinction for r) is
  intentionally omitted; it is irrelevant for the downstream LingPy
  tokenisation and PanPhon vectorisation steps.

Run 03a_latin_ipa_tests.py FIRST. All tests must pass before converting
the full dataset.

Input:  ipa_output.json  (from 02_epitran_ipa.py)
Output: ipa_output.json  (updated in-place, Latin entries filled)
        latin_ipa_inspection.txt  (side-by-side ortho → IPA for review)
"""

import json
import re
from pathlib import Path

INPUT_PATH      = Path("./ipa_output.json")
INSPECTION_PATH = Path("./latin_ipa_inspection.txt")


def latin_to_ipa(form: str) -> str:
    """
    Convert one Classical Latin orthographic form to broad IPA.

    Rule ordering is load-bearing: each block assumes previous blocks have
    already run. Do not reorder without re-running the full test suite.

    Parameters
    ----------
    form : str
        Orthographic Latin word, possibly with macrons for long vowels.
        Case-insensitive; stripped of leading/trailing whitespace.

    Returns
    -------
    str
        Broad IPA transcription.
    """
    s = form.strip().lower()

    # ── BLOCK 1: Greek-loan aspirate digraphs ─────────────────────────────────
    # Allen (1978:26-28): ph, th, ch in Greek loans represent aspirate stops,
    # transcribed pʰ, tʰ, kʰ. Must precede all single-character rules so that
    # <ph> is consumed as a unit before <p> or <h> are processed individually.
    s = s.replace("ph", "pʰ")
    s = s.replace("th", "tʰ")
    s = s.replace("ch", "kʰ")

    # ── BLOCK 2: Labiovelar clusters ──────────────────────────────────────────
    # Allen (1978:15-17): <qu> and <gu> before a vowel represent the single
    # labiovelar phonemes /kʷ/ and /ɡʷ/. The <u> here is not a vowel nucleus
    # but the labial off-glide component of the consonant.
    # Must precede the consonantal-u rules in Block 8.
    s = re.sub(r"qu(?=[aeiouāēīōūy])", "kʷ", s)
    s = re.sub(r"gu(?=[aeiouāēīōūy])", "ɡʷ", s)

    # ── BLOCK 3: au diphthong ─────────────────────────────────────────────────
    # Allen (1978:58-59): Classical Latin <au> is a falling diphthong [au̯].
    # The vocalic-u element functions as the off-glide, which we transcribe as
    # [w] (= [u̯]). Must precede the short-vowel rules in Block 10 (which would
    # otherwise convert <u> to [ʊ]) and precede the consonantal-u rules in
    # Block 8 (which would otherwise convert the off-glide <u> to [w] by a
    # different mechanism — same result, but rule ordering must be explicit).
    #
    # NOTE on <ae> and <oe>:
    # Allen (1978:57-58): <ae> was still a genuine falling diphthong [ae̯] in
    # the Classical period. The monophthongisation to [ɛː] is a late / Vulgar
    # Latin development and must NOT be applied here.
    # Allen (1978:60): <oe> was similarly the diphthong [oe̯].
    # Both are correctly represented as two-segment sequences by the individual
    # short vowel rules in Block 10 (ae → aɛ; oe → ɔɛ). No special rule needed.
    s = s.replace("au", "aw")

    # ── BLOCK 4: x, z ─────────────────────────────────────────────────────────
    # Allen (1978:22): <x> represents the cluster /ks/.
    # Allen (1978:23): <z> in Greek loans represents /dz/.
    s = s.replace("x", "ks")
    s = s.replace("z", "dz")

    # ── BLOCK 5: Geminate consonants ──────────────────────────────────────────
    # Allen (1978:30-31): Latin geminates are genuine long consonants, distinct
    # from their short counterparts. Transcribed with the IPA length mark ː on
    # the first of the pair.
    # <u> is intentionally excluded from the character class: <uu> is never a
    # geminate vowel but always consonantal-u + vocalic-u (handled in Block 8).
    s = re.sub(r"([bcdfgklmnpqrstvw])\1", lambda m: m.group(1) + "ː", s)

    # ── BLOCK 6: gn cluster ───────────────────────────────────────────────────
    # Allen (1978:26-27): the sequence <gn> was pronounced [ŋn] throughout the
    # Classical period, regardless of word position or syllable boundary.
    # The <g> in <gn> assimilates to the following nasal, becoming [ŋ].
    # Must precede the g → ɡ substitution in Block 11; if g → ɡ ran first,
    # this rule could not match the ASCII <g>.
    # Examples: ignis [ɪŋnɪs], dignus [dɪŋnʊs], magnus [maŋnʊs].
    s = re.sub(r"g(?=n)", "ŋ", s)

    # ── BLOCK 7: Consonantal i → j ────────────────────────────────────────────
    # Allen (1978:38-40), Weiss (2009:41): <i> before a vowel at word-initial
    # position is a palatal glide [j], not the vowel [ɪ].
    # Examples: iecur [jɛkʊr], iam, iungere.
    # NOTE: medial consonantal <i> (e.g. maior, peior) is written <j> in the
    # IECoR data and needs no conversion; <j> passes through untouched.
    s = re.sub(r"^i(?=[aeiouāēīōūy])", "j", s)

    # ── BLOCK 8: Consonantal u → w ────────────────────────────────────────────
    # Allen (1978:38-41), Weiss (2009:41-42): Classical Latin <u> (= modern
    # printed <v>) represents the labio-velar approximant [w] in three
    # environments. All three sub-rules must run before the short-vowel rule
    # u → ʊ in Block 10.
    #
    # (a) Word-initial <u> before any vowel is always consonantal [w].
    #     Examples: uenter → [wɛntɛr], uir → [wɪr], uidere → [wɪdeːrɛ].
    s = re.sub(r"^u(?=[aeiouāēīōūy])", "w", s)
    #
    # (b) The sequence <uu> always represents consonantal-u + vocalic-u, i.e.
    #     [w][ʊ]. Allen (1978:39) explicitly cites seruus, paruus, nouus as
    #     examples. The FIRST u is the consonantal onset of the following
    #     syllable; the SECOND u is the vocalic nucleus. Rule: uu → wu.
    #     Must precede sub-rule (c) to avoid (c) incorrectly consuming one of
    #     the two u's first.
    s = s.replace("uu", "wu")
    #
    # (c) Medial <u> between two vowels is consonantal [w].
    #     Allen (1978:38): "between vowels". The lookbehind restricts the rule
    #     to the vowel+u+vowel context, preventing it from firing on
    #     consonant+u+vowel sequences where u is a vocalic syllable nucleus
    #     (e.g. duo [dʊoː]: <u> follows <d>, a consonant, so rule does not fire).
    s = re.sub(r"(?<=[aeiouāēīōūy])u(?=[aeiouāēīōūy])", "w", s)
    #
    # Robustness: some editions print <v>; IECoR uses <u> throughout.
    s = s.replace("v", "w")

    # ── BLOCK 9: Long vowels → placeholders ───────────────────────────────────
    # Allen (1978:47-65): macron vowels ā ē ī ō ū represent long vowels /aː eː
    # iː oː uː/. Greek loan ȳ represents /yː/.
    # The placeholder mechanism prevents the short-vowel rules in Block 10 from
    # re-processing the base-letter component of a long vowel (e.g., without
    # this protection: ū → uː, then u → ʊ incorrectly yields ʊː).
    LONG_TO_PLACEHOLDER = {
        "ā": "##AA##", "ē": "##EE##", "ī": "##II##",
        "ō": "##OO##", "ū": "##UU##", "ȳ": "##YY##",
    }
    PLACEHOLDER_TO_IPA = {
        "##AA##": "aː", "##EE##": "eː", "##II##": "iː",
        "##OO##": "oː", "##UU##": "uː", "##YY##": "yː",
    }
    for ch, ph in LONG_TO_PLACEHOLDER.items():
        s = s.replace(ch, ph)

    # ── BLOCK 10: Short vowels ────────────────────────────────────────────────
    # Allen (1978:47-65): short vowels transcribed as follows at phonemic level.
    # a → [a]  (low central; Allen 1978:47)
    # e → [ɛ]  (mid front; Allen 1978:50 — short e was opener than long ē)
    # i → [ɪ]  (near-high front; Allen 1978:53)
    # o → [ɔ]  (mid back; Allen 1978:56 — short o was opener than long ō)
    # u → [ʊ]  (near-high back; Allen 1978:59)
    # y → [y]  (front rounded; Greek loans only; Allen 1978:62)
    #
    # <ae> (= a + e) and <oe> (= o + e) are processed correctly here as two
    # separate segments, giving [aɛ] and [ɔɛ] respectively. This is the
    # phonemically accurate representation of the Classical diphthongs [ae̯]
    # and [oe̯] (Allen 1978:57-60; Weiss 2009:32). No special pre-rule needed.
    for ch, ipa in {"a": "a", "e": "ɛ", "i": "ɪ", "o": "ɔ", "u": "ʊ", "y": "y"}.items():
        s = s.replace(ch, ipa)

    # Restore long vowels from placeholders
    for ph, ipa in PLACEHOLDER_TO_IPA.items():
        s = s.replace(ph, ipa)

    # ── BLOCK 11: Remaining consonants ────────────────────────────────────────
    # c → k: Allen (1978:14): <c> was always velar [k] in Classical Latin,
    # with no palatalization before front vowels. Palatalization is a later
    # Vulgar / Medieval Latin development.
    s = s.replace("c", "k")

    # Nasal assimilation: Allen (1978:30): /n/ assimilates to [ŋ] before velar
    # consonants /k/ and /ɡ/. Two passes are needed because the g → ɡ
    # substitution (below) introduces new [ɡ] that were not present when the
    # first pass ran.
    s = re.sub(r"n(?=ɡ)", "ŋ", s)    # pass 1: before ɡʷ (Block 2) and any ɡ already present
    s = re.sub(r"n(?=k)", "ŋ", s)    # pass 1: before k (from c → k above)
    s = s.replace("g", "ɡ")          # ASCII g → IPA ɡ (hook-g)
    s = re.sub(r"n(?=ɡ)", "ŋ", s)    # pass 2: before ɡ introduced by the line above

    # All other consonants (b d f h j k l m n p r s t w) have IPA symbols
    # identical to their Latin orthographic letters at the phonemic level and
    # require no further substitution.

    return s


# ── Spot-check table (quick sanity gate before full conversion) ───────────────
SPOT_CHECKS = [
    # (orthographic form,  expected IPA,   gloss label)
    ("ūnus",    "uːnʊs",    "ONE"),
    ("plēnus",  "pleːnʊs",  "FULL"),
    ("sōl",     "soːl",     "SUN"),
    ("aqua",    "akʷa",     "WATER"),
    ("ignis",   "ɪŋnɪs",    "FIRE"),     # gn → [ŋn] (Allen 1978:26-27)
    ("canis",   "kanɪs",    "DOG"),
    ("uenter",  "wɛntɛr",   "BELLY"),
    ("sanguīs", "saŋɡʷiːs", "BLOOD"),
    ("arbor",   "arbɔr",    "TREE"),
    ("lingua",  "lɪŋɡʷa",   "TONGUE"),
]

print("=== SPOT CHECKS (verify before full run) ===")
all_passed = True
for ortho, expected, gloss in SPOT_CHECKS:
    result = latin_to_ipa(ortho)
    ok = result == expected
    if not ok:
        all_passed = False
    mark = "✓" if ok else "✗"
    print(f"  {mark} [{gloss}] {ortho:<15} → {result:<22} (expected: {expected})")

if not all_passed:
    print("\n  Spot checks FAILED. Run 03a_latin_ipa_tests.py for diagnostics.")
    print("  Do not convert the dataset until all spot checks pass.")
    exit(1)
else:
    print("\n  All spot checks passed.\n")


# ── Load ipa_output.json and fill Latin entries ───────────────────────────────
print("Loading ipa_output.json...")
with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

inspection_lines = [
    "LATIN IPA CONVERSION — FULL INSPECTION",
    "Generated by 03b_latin_ipa.py",
    "=" * 60,
]
updated = 0

for concept, entry in data.items():
    latin = entry["languages"].get("Latin")
    if not latin or latin.get("status") != "manual":
        continue
    ipa_forms = []
    inspection_lines.append(f"\n[{concept}]")
    for form in latin["ortho"]:
        ipa = latin_to_ipa(form)
        ipa_forms.append(ipa)
        inspection_lines.append(f"  {form:<22} → {ipa}")
    data[concept]["languages"]["Latin"]["ipa"]    = ipa_forms
    data[concept]["languages"]["Latin"]["status"] = "rules-latin"
    updated += 1

with open(INPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
print(f"  Updated {updated} Latin entries in {INPUT_PATH}")

with open(INSPECTION_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(inspection_lines))
print(f"  Inspection file written to {INSPECTION_PATH}")
print(f"\n  Next steps:")
print(f"    1. Open latin_ipa_inspection.txt and review all {updated} entries.")
print(f"       Flag any form that looks wrong for manual correction.")