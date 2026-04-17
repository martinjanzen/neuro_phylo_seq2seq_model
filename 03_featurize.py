import json, re, lingpy, panphon, torch, unicodedata
import pandas as pd, numpy as np
from pathlib import Path
from lingpy.align.multiple import Multiple
from lingpy.align.pairwise import pw_align

_DIR = Path(__file__).parent
INPUT_PATH = _DIR / "ipa_output.json"
MATRIX_OUTPUT = _DIR / "distance_matrix.csv"
VECTOR_OUTPUT = _DIR / "vectorized_dataset.pt"

LANGUAGES = ["French", "Italian", "Portuguese", "Romanian", "Spanish"]
TARGET_DIM = 24
ft = panphon.FeatureTable()

def tokenize_ipa(ipa_str):
    if not ipa_str: return []
    clean = re.sub(r'\s+', '', ipa_str.strip())
    if not clean: return []
    raw_tokens, merged_tokens = lingpy.ipa2tokens(clean, merge_vowels=False), []
    for t in raw_tokens:
        if t.startswith('ː') and merged_tokens: merged_tokens[-1] += t
        elif t in ['\u0361', '\u035C'] and merged_tokens: merged_tokens[-1] += t
        elif (t.startswith('\u0361') or t.startswith('\u035C')) and merged_tokens: merged_tokens[-1] += t
        elif merged_tokens and (merged_tokens[-1].endswith('\u0361') or merged_tokens[-1].endswith('\u035C')): merged_tokens[-1] += t
        else: merged_tokens.append(t)
    return merged_tokens

def get_phonetic_distance(t1, t2):
    if t1 == t2: return 0.0
    if t1 == "-" or t2 == "-": return 1.0
    try:
        def _to_vec(token):
            segs = ft.fts(token)
            if not segs: return None
            vecs = [np.array(s.numeric()) for s in (segs if isinstance(segs, list) else [segs])
                    if hasattr(s, 'numeric')]
            return np.mean(vecs, axis=0) if vecs else None
        v1, v2 = _to_vec(t1), _to_vec(t2)
        return 1.0 if v1 is None or v2 is None else float(np.mean(np.abs(v1 - v2)))
    except: return 1.0

def get_feature_vector(token: str, is_gap: bool = False) -> list:
    if is_gap: return [0.0] * TARGET_DIM
    try:
        token = unicodedata.normalize('NFC', token)
        segs = ft.word_fts(token) if hasattr(ft, 'word_fts') else ft.fts(token)
        if segs:
            vecs = []
            for seg in (segs if isinstance(segs, list) else [segs]):
                if hasattr(seg, 'numeric'): vecs.append(seg.numeric())
                elif hasattr(seg, 'values'): vecs.append(list(seg.values()))
                else: vecs.append([float(x) if isinstance(x, (int, float))
                                   else {'+':1.0,'-':-1.0,'0':0.0}.get(str(x), 0.0) for x in seg])
            avg_vec = [sum(col) / len(col) for col in zip(*vecs)]
            return (avg_vec + [0.0] * TARGET_DIM)[:TARGET_DIM]
    except: pass
    return [0.0] * TARGET_DIM

def run_pipeline():
    # tokenize
    with open(INPUT_PATH, "r", encoding="utf-8") as f: data = json.load(f)
    tokenized = {}
    for concept, entry in data.items():
        tokenized[concept] = {
            "concept": entry["concept"], "gloss": entry["gloss"],
            "languages": {
                lang: {"tokens": [tokenize_ipa(i) for i in lang_data.get("ipa", []) if i]}
                for lang, lang_data in entry["languages"].items()
            }
        }

    # align MSA
    dataset_aligned = []
    dropped_total = 0

    for concept, entry in tokenized.items():
        romance_langs, romance_tokens, latin_tokens = [], [], [["-"]]

        for lang, lang_data in entry["languages"].items():
            toks = [[str(t) for t in v] for v in lang_data.get("tokens", []) if v]
            if not toks: continue
            if lang == "Latin": latin_tokens = toks
            else: romance_langs.append(lang); romance_tokens.append(toks)

        if len(romance_langs) < 2 or latin_tokens == [["-"]]: continue

        paired = sorted(zip(romance_langs, romance_tokens), key=lambda p: p[0])
        romance_langs = [p[0] for p in paired]
        romance_tokens = [p[1] for p in paired]

        try:
            msa = Multiple(list(tuple(v[0] for v in romance_tokens)))
            msa.prog_align()
            matrix = msa.alm_matrix
            width = len(matrix[0])

            # majority-vote latin alignment
            per_lang_cols = []
            for j in range(len(romance_langs)):
                alm_lat, alm_rom, _ = pw_align(latin_tokens[0], list(matrix[j]))
                col_map, rom_idx = {}, 0
                for lt, rt in zip(alm_lat, alm_rom):
                    if rt != "-":
                        if rom_idx < width: col_map[rom_idx] = lt
                        rom_idx += 1
                per_lang_cols.append(col_map)

            final_latin = []
            for idx in range(width):
                votes = [c[idx] for c in per_lang_cols if c.get(idx, "-") != "-"]
                final_latin.append(max(set(votes), key=votes.count) if votes else "-")

            dropped_total += max(0,
                sum(1 for t in latin_tokens[0] if t != "-") -
                sum(1 for t in final_latin    if t != "-"))

            align_dict = {romance_langs[i]: list(matrix[i]) for i in range(len(romance_langs))}
            align_dict["Latin"] = final_latin
            dataset_aligned.append({"concept": concept, "width": width, "alignment": align_dict})
        except: pass

    # distance matrix
    phylo_langs = LANGUAGES + ["Latin"]
    dist_sum = {l: {l2: 0.0 for l2 in phylo_langs} for l in phylo_langs}
    count_sum = 0

    for entry in dataset_aligned:
        align = entry["alignment"]
        if not all(l in align for l in phylo_langs): continue
        width = entry["width"]

        for l1 in phylo_langs:
            for l2 in phylo_langs:
                if l1 == l2: continue
                seq1, seq2 = align[l1], align[l2]
                dist_sum[l1][l2] += sum(
                    get_phonetic_distance(seq1[i], seq2[i]) for i in range(width)
                ) / width
        count_sum += 1

    # symmetrize + normalize
    for l1 in phylo_langs:
        for l2 in phylo_langs:
            avg = (dist_sum[l1][l2] + dist_sum[l2][l1]) / 2
            dist_sum[l1][l2] = dist_sum[l2][l1] = avg
        dist_sum[l1][l1] = 0.0

    matrix_vals = [[dist_sum[l1][l2] / count_sum if count_sum > 0 else 0.0
                    for l2 in phylo_langs] for l1 in phylo_langs]
    pd.DataFrame(matrix_vals, index=phylo_langs, columns=phylo_langs).to_csv(MATRIX_OUTPUT)

    # vectorize
    global_max = max(d["width"] for d in dataset_aligned)
    final_tensors = []

    for d in dataset_aligned:
        align = d["alignment"]

        # build X: shape (L, S, 24)
        x_rows = []
        for lang in LANGUAGES:
            t = align.get(lang, ["-"] * d["width"])
            x_rows.append([
                get_feature_vector(t[i], t[i] == "-") if i < len(t) else [0.0] * 24
                for i in range(global_max)
            ])
        x_t = torch.tensor(x_rows, dtype=torch.float32)

        assert x_t.shape == (len(LANGUAGES), global_max, 24), (
            f"X tensor shape {x_t.shape} != ({len(LANGUAGES)}, {global_max}, 24) "
            f"for concept '{d['concept']}'"
        )

        # build Y: shape (S, 25)
        y_seq = align.get("Latin", ["-"] * d["width"])
        y_rows = [
            get_feature_vector(
                y_seq[i] if i < len(y_seq) else "-",
                i >= len(y_seq) or y_seq[i] == "-"
            ) + [1.0 if (i >= len(y_seq) or y_seq[i] == "-") else 0.0]
            for i in range(global_max)
        ]
        y_t = torch.tensor(y_rows, dtype=torch.float32)

        assert y_t.shape == (global_max, 25), (
            f"Y tensor shape {y_t.shape} != ({global_max}, 25) "
            f"for concept '{d['concept']}'"
        )

        final_tensors.append({"concept": d["concept"], "X": x_t, "Y": y_t})

    torch.save(final_tensors, VECTOR_OUTPUT)
    print(f"{len(final_tensors)} concepts saved with global width {global_max}")

if __name__ == "__main__":
    run_pipeline()