import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── PASTE YOUR DATA HERE ──────────────────────────────────────────────────────

# ── ABLATION ─────────────────────────────────────────────────────────────────
BASELINE_PER_FOLD = [0.5000, 0.4649, 0.4924, 0.5717, 0.5431,
                     0.4766, 0.4736, 0.5046, 0.4297, 0.5112]

ABLATED_PER_FOLD = {
    "Italian":    [0.513098, 0.508979, 0.494494, 0.596516, 0.554547, 0.501514, 0.527668, 0.552584, 0.442497, 0.544656],
    "Spanish":    [0.494363, 0.492138, 0.482401, 0.567942, 0.536435, 0.472458, 0.504443, 0.517056, 0.464497, 0.520832],
    "Romanian":   [0.45915, 0.490992, 0.51129, 0.543035, 0.517908, 0.519044, 0.483239, 0.464854, 0.469803, 0.499774],
    "French":     [0.500608, 0.46382, 0.492775, 0.544569, 0.511076, 0.46782, 0.493201, 0.519502, 0.446001, 0.516383],
    "Portuguese": [0.454654, 0.470335, 0.486602, 0.534011, 0.52638, 0.475399, 0.500049, 0.521196, 0.437208, 0.495708]
}

# ── ATTENTION ─────────────────────────────────────────────────────────────────
# NOTE: the results file reports cross-fold means and stds but not raw per-fold
# weights. To generate Figures 3 and 4 you need to paste the fold-level weights
# from your run_attention.py console output (the "Fold X: weights=[...]" lines).
# Confirmed cross-fold means for reference:
#   Italian 0.3184, Spanish 0.2136, Portuguese 0.2383, Romanian 0.1701, French 0.0595
ATTENTION_PER_FOLD = [
    # [Italian, Spanish, Portuguese, Romanian, French]
    [0.41486, 0.308441, 0.061867, 0.087527, 0.127304],
    [0.402932, 0.207818, 0.159343, 0.13718, 0.092728],
    [0.296323, 0.215458, 0.192128, 0.181001, 0.11509],
    [0.364617, 0.321945, 0.112395, 0.162136, 0.038907],
    [0.134894, 0.493363, 0.313276, 0.053695, 0.004771],
    [0.181985, 0.202983, 0.246089, 0.332566, 0.036377],
    [0.500447, 0.192052, 0.052405, 0.180494, 0.074602],
    [0.216622, 0.091196, 0.672091, 2e-05, 0.020071],
    [0.328561, 0.09086, 0.2014, 0.331083, 0.048097],
    [0.154617, 0.223209, 0.235638, 0.379899, 0.006638],
]

# ── TIER HIERARCHY ────────────────────────────────────────────────────────────
# Model kappas are oracle-threshold (confirmed from results).
# Baseline kappas for majority/mean-of-inputs/copy-italian are NOT in the
# results file — only model advantage over best baseline is reported.
# Confirmed model advantages: Skeleton +0.0515, Articulation +0.0207, Color +0.0005
# You need to paste individual baseline kappas from your 06_eval_audit.py output.
TIER_KAPPAS = {
    #                  model    majority  mean-inp  copy-ita
    "Skeleton":     [0.4333,    0.387094,      0.320790,      0.389942],
    "Articulation": [0.6495,    0.602903,      0.406977,      0.490944],
    "Color":        [0.6936,    0.697517,      0.462625,      0.578147],
}

# Per-feature F1 — all 24 features confirmed from results
FEAT_NAMES = ["syl","son","cons","cont","delrel",
              "lat","nas","strid","voi","sg","cg","ant","cor","distr",
              "hi","lo","back","rnd","vel","lab","labz","low","high","ret"]
FEAT_TIERS = (["Skeleton"]*5 + ["Articulation"]*9 + ["Color"]*10)
FEAT_F1    = [
    # Skeleton
    0.338694, 0.418809, 0.355794, 0.428476, 0.473006,
    # Articulation
    0.487164, 0.478129, 0.474618, 0.445806, 1.000000, 1.000000,
    0.349292, 0.259075, 0.389047,
    # Color
    0.472034, 0.295528, 0.295528, 0.401811, 0.452168,
    1.000000, 0.229850, 0.495892, 1.000000, 1.000000,
]
DEGENERATE = {"sg", "cg", "lab", "high", "ret"}

# ── STYLE ─────────────────────────────────────────────────────────────────────
TIER_COLORS     = {"Skeleton": "#4C72B0", "Articulation": "#DD8452", "Color": "#55A868"}
LANG_ORDER      = ["Italian", "Spanish", "Portuguese", "Romanian", "French"]
LANG_COLORS     = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
BASELINE_LABELS = ["Model (oracle)", "Majority", "Mean-of-inputs", "Copy-Italian"]
BASELINE_COLORS = ["#333333", "#90CAF9", "#A5D6A7", "#FFCC80"]

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── FIGURE 1: ΔLoss bar chart ─────────────────────────────────────────────────
def fig_ablation_delta():
    baseline = np.array(BASELINE_PER_FOLD)
    deltas, stds = [], []
    for lang in LANG_ORDER:
        diffs = np.array(ABLATED_PER_FOLD[lang]) - baseline
        deltas.append(diffs.mean())
        stds.append(diffs.std())

    fig, ax = plt.subplots(figsize=(7, 4))
    xs   = np.arange(len(LANG_ORDER))
    bars = ax.bar(xs, deltas, yerr=stds, capsize=5,
                  color=LANG_COLORS, edgecolor="white", width=0.55)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(LANG_ORDER)
    ax.set_ylabel("ΔPhonologicalLoss (ablated − baseline)")
    ax.set_title("Figure 1: Per-Language Ablation Effect")
    for bar, d, s in zip(bars, deltas, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                d + (s + 0.002) * np.sign(d) if d != 0 else 0.002,
                f"{d:+.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig1_ablation_delta.png", dpi=300)
    print("Saved fig1_ablation_delta.png")

# ── FIGURE 2: Per-fold paired loss trajectory ─────────────────────────────────
def fig_ablation_folds():
    baseline = np.array(BASELINE_PER_FOLD)
    ablated  = np.array(ABLATED_PER_FOLD["Italian"])

    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(len(baseline)):
        increased = float(ablated[i]) > float(baseline[i])
        color = "#2196F3" if increased else "#F44336"
        ax.plot([0, 1], [baseline[i], ablated[i]],
                color=color, alpha=0.5, linewidth=1.2, zorder=3)
        ax.text(-0.04, baseline[i] + (i % 3 - 1) * 0.001,   # stagger by row mod 3
                f"F{i+1}", va="center", ha="right", fontsize=8, color="grey")

    ax.plot([0, 1],
            [baseline.mean(), ablated.mean()],
            color="black", linewidth=2.5, zorder=5, label="Fold mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Italian ablated"])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylabel("PhonologicalLoss")
    ax.set_title("Figure 2: Per-Fold Loss — Baseline vs. Italian Ablated")
    rise = mpatches.Patch(color="#2196F3", alpha=0.6, label="Loss increased (expected)")
    drop = mpatches.Patch(color="#F44336", alpha=0.6, label="Loss decreased (unexpected)")
    ax.legend(handles=[rise, drop], fontsize=9)
    fig.tight_layout()
    fig.savefig("fig2_ablation_folds.png", dpi=300)
    print("Saved fig2_ablation_folds.png")

# ── FIGURE 3: Mean attention weight bar chart ─────────────────────────────────
def fig_attention_bar():
    fw    = np.array(ATTENTION_PER_FOLD)   # (10, 5)
    means = fw.mean(axis=0)
    stds  = fw.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(LANG_ORDER))
    ax.bar(xs, means, yerr=stds, capsize=5,
           color=LANG_COLORS, edgecolor="white", width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(LANG_ORDER)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Figure 3: Mean Decoder Attention by Language\n"
                 "(ordered by conservatism, most → least)")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig3_attention_bar.png", dpi=300)
    print("Saved fig3_attention_bar.png")

# ── FIGURE 4: Fold-level attention heatmap ────────────────────────────────────
def fig_attention_heatmap():
    fw = np.array(ATTENTION_PER_FOLD)   # (10, 5)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(fw, aspect="auto", cmap="Blues", vmin=0, vmax=fw.max())
    ax.set_xticks(range(len(LANG_ORDER)))
    ax.set_xticklabels(LANG_ORDER)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(10)])
    ax.set_title("Figure 4: Fold-Level Attention Weights by Language")
    for i in range(10):
        for j in range(len(LANG_ORDER)):
            ax.text(j, i, f"{fw[i,j]:.3f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if fw[i,j] < fw.max()*0.6 else "white")
    plt.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig("fig4_attention_heatmap.png", dpi=300)
    print("Saved fig4_attention_heatmap.png")

# ── FIGURE 5: Tier kappa grouped bar chart ────────────────────────────────────
def fig_tier_kappas():
    tiers = list(TIER_KAPPAS.keys())
    vals  = np.array([TIER_KAPPAS[t] for t in tiers])   # (3, 4)
    n_bl  = vals.shape[1]
    width = 0.18
    xs    = np.arange(len(tiers))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, color) in enumerate(zip(BASELINE_LABELS, BASELINE_COLORS)):
        offset = (i - (n_bl-1)/2) * width
        bars   = ax.bar(xs + offset, vals[:, i], width,
                        label=label, color=color, edgecolor="white")
        for bar, v in zip(bars, vals[:, i]):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Cohen's κ")
    ax.set_ylim(0, 0.85)
    ax.set_title("Figure 5: Tier Reconstruction Accuracy — Model vs. Baselines\n"
                 "(model at oracle threshold; baselines at fixed threshold=0.5)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("fig5_tier_kappas.png", dpi=300)
    print("Saved fig5_tier_kappas.png")

# ── FIGURE 6: Per-feature F1 bar chart ───────────────────────────────────────
def fig_feature_f1():
    fig, ax = plt.subplots(figsize=(10, 5))
    xs     = np.arange(len(FEAT_NAMES))
    colors = [
        "#BBBBBB" if FEAT_NAMES[i] in DEGENERATE
        else TIER_COLORS[FEAT_TIERS[i]]
        for i in range(len(FEAT_NAMES))
    ]
    ax.bar(xs, FEAT_F1, color=colors, edgecolor="white", width=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Macro-F1 (threshold=0.5)")
    ax.set_title("Figure 6: Per-Feature Reconstruction F1\n"
                 "(grey = degenerate, excluded from substantive mean)")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    # Tier boundary markers
    for boundary in [4.5, 13.5]:
        ax.axvline(boundary, color="black", linewidth=0.8,
                   linestyle=":", alpha=0.5)

    # Substantive mean line
    substantive = [f for f, n in zip(FEAT_F1, FEAT_NAMES)
                   if n not in DEGENERATE]
    ax.axhline(np.mean(substantive), color="darkred", linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"Substantive mean = {np.mean(substantive):.3f}")

    legend_handles = [
        mpatches.Patch(color=TIER_COLORS["Skeleton"],     label="Skeleton"),
        mpatches.Patch(color=TIER_COLORS["Articulation"], label="Articulation"),
        mpatches.Patch(color=TIER_COLORS["Color"],        label="Color"),
        mpatches.Patch(color="#BBBBBB",                   label="Degenerate (excluded)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig6_feature_f1.png", dpi=300)
    print("Saved fig6_feature_f1.png")

# ── RUN ALL ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_ablation_delta()
    fig_ablation_folds()
    fig_attention_bar()
    fig_attention_heatmap()
    fig_tier_kappas()
    fig_feature_f1()