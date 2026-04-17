# Phylogenetic Evaluation of Phonologically-Informed Vulgar Latin Reconstruction via Encoder-Decoder LSTM

## Setup

Install all items in `requirements.txt`.

Epitran requires additional language data packages:
```bash
python -m epitran.download fra-Latn spa-Latn ita-Latn por-Latn ron-Latn
```



## Data

Ensure the following are in the project root before running:
- `iecor-master/` — IECoR CLDF dataset (https://github.com/lexibank/iecor)
- `evotext-ielex-data-and-tree-686ce09/` — Dunn-207 concept list (https://zenodo.org/records/5556801)



## Pipeline

Run scripts in order. Each step produces an output file consumed by the next.

**1. Build dataset**
```bash
python 01_build_data.py
```
Output: `swadesh_expanded.json`

**2. Phonemize**
```bash
python 02_phonemize.py
```
Output: `ipa_output.json`

**3. Featurize and align**
```bash
python 03_featurize.py
```
Output: `vectorized_dataset.pt`, `distance_matrix.csv`

**4. Ablation study**
```bash
python run_ablation.py
```
Output: `results_ablation.txt`, `reconstruction_fold1.pth`

**5. Attention analysis**
```bash
python run_attention.py
```
Output: `results_attention.txt`

**6. Tier hierarchy evaluation**
```bash
python run_tiers.py
```
Output: `results_tiers.txt`, `reconstruction_fold1.pth`

**7. Generate figures**
```bash
python generate_figures.py
```
Output: `fig1_ablation_delta.pdf` through `fig6_feature_f1.pdf`

> Note: `generate_figures.py` was generated with Claude, I take no credit for the code used to generate figures.

> Note: Steps 4–6 are independent of each other and can be run in any order after step 3. Each trains its own models from scratch. Expect each to take several minutes on CPU depending on hardware.



## Expected Output Files

| File | Description |
|---|---|
| `swadesh_expanded.json` | Raw lexical forms per concept per language |
| `ipa_output.json` | IPA-transcribed forms |
| `vectorized_dataset.pt` | PanPhon feature tensors, MSA-aligned |
| `distance_matrix.csv` | 6×6 pairwise phonetic distance matrix |
| `results_ablation.txt` | Ablation statistics |
| `results_attention.txt` | Attention weight statistics |
| `results_tiers.txt` | Tier hierarchy statistics and baseline comparison |
| `reconstruction_fold1.pth` | Saved fold-0 model checkpoint |
| `fig1–fig6.pdf` | Paper figures |