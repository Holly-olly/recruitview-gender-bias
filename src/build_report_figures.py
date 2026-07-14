#!/usr/bin/env python3
"""Build the report figures (Neural Haze palette) into outputs/figures/.

Generates the 10 figures referenced as placeholders in Report_human_vs_LLM.md.
All numbers come from data/recruitview_analysis.csv + outputs/*.csv (aggregate visuals,
no row-level score dumps). Run:  .venv/bin/python src/build_report_figures.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import gaussian_kde, pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Neural Haze palette (lib/palette.ts) ---
LIME, VIOLET, CYAN, MAGENTA = "#C8F135", "#9B30FF", "#00F5D4", "#FF2D78"
MIST, SLATE, INK, GHOST = "#C5C3D6", "#6E6A88", "#1C1928", "#F0EFF8"
HUMAN, LLM = MAGENTA, VIOLET          # method colours, consistent with the notebook
HAZE = LinearSegmentedColormap.from_list("haze", [GHOST, "#B79BE8", VIOLET, "#4E0FAD"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": SLATE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SLATE, "ytick.color": SLATE, "axes.titlecolor": INK,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})

SCALES = ["openness", "conscientiousness", "extraversion", "agreeableness",
          "neuroticism", "answer_score", "interview_score"]
LAB = {s: s.replace("_", " ") for s in SCALES}
d = pd.read_csv(ROOT / "data/recruitview_analysis.csv")


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


# --- Fig 1: research-flow diagram ---
def fig1_flow():
    fig, ax = plt.subplots(figsize=(9, 8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 12)

    def box(x, y, w, h, text, fc, tc="white", fs=9):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08",
                                    linewidth=0, facecolor=fc))
        ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color=SLATE, linewidth=1.4))

    box(5, 11.3, 6.4, 0.9, "RecruitView video interviews\n(76 questions, ~6 per person)", INK)
    box(5, 9.9, 6.4, 0.9, "Cleaning + gender coding\n2,011 to 1,929 to 1,885 / 328 persons", SLATE)
    box(5, 8.6, 6.4, 0.7, "7 transcript-scorable scales in scope", SLATE)
    arrow(5, 10.85, 5, 10.35); arrow(5, 9.45, 5, 8.95)
    box(2.4, 7.0, 4.0, 1.1, "HUMAN method\nvideo-perceived\nnuclear-norm MNL z", HUMAN, fs=8.5)
    box(7.6, 7.0, 4.0, 1.1, "LLM method (TEAS)\ngpt-4o-mini codes a\nGemini text-report to Likert", LLM, fs=8.5)
    arrow(5, 8.25, 2.6, 7.6); arrow(5, 8.25, 7.4, 7.6)
    box(5, 5.3, 8.2, 1.1, "SAME ENGINE\nICC  to  G-theory + D-study  to  Bayesian length/\nreadability ladder  to  EB/BLUP per-person score", INK, fs=8.5)
    arrow(2.4, 6.45, 4.0, 5.9); arrow(7.6, 6.45, 6.0, 5.9)
    box(2.4, 3.4, 3.6, 0.8, "human BLUP\nper person/scale", HUMAN, fs=8.5)
    box(7.6, 3.4, 3.6, 0.8, "LLM BLUP\nper person/scale", LLM, fs=8.5)
    arrow(4.0, 4.75, 2.6, 3.8); arrow(6.0, 4.75, 7.4, 3.8)
    box(5, 1.5, 7.4, 1.0, "Comparison (Section 6)\ndistributions + per-scale convergence:\nNEGATIVE on all 7 scales", MAGENTA, fs=9)
    arrow(2.4, 3.0, 4.2, 2.0); arrow(7.6, 3.0, 5.8, 2.0)
    save(fig, "fig1_flow.png")


# --- Fig 2: distributions of the 12 target scores ---
def fig2_distributions():
    scores12 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
                "overall_personality", "interview_score", "answer_score", "speaking_skills",
                "confidence_score", "facial_expression", "overall_performance"]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for ax, s in zip(axes.ravel(), scores12):
        x = d[s].dropna()
        ax.hist(x, bins=60, color=VIOLET, alpha=.85, edgecolor="white", linewidth=.2)
        ax.set_title(LAB.get(s, s.replace("_", " ")), fontsize=10)
        ax.set_xlabel("z-score"); ax.set_yticks([])
    fig.suptitle("Distributions of the 12 target scores (heavy leptokurtosis; compressed neuroticism)",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "fig2_distributions.png")


# --- Fig 3: OTE heatmap, question by trait ---
def fig3_ote():
    ote = pd.read_csv(ROOT / "outputs/stage1_overview/question_construct_ote.csv")
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    cols = [f"ote_{t}" for t in traits]
    M = ote[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 10))
    im = ax.imshow(M, aspect="auto", cmap=HAZE)
    ax.set_xticks(range(5)); ax.set_xticklabels([t[:5] for t in traits], rotation=30)
    ax.set_yticks([0, len(M) // 2, len(M) - 1]); ax.set_yticklabels(["Q1", f"Q{len(M)//2}", f"Q{len(M)}"])
    ax.set_ylabel("interview question (76)")
    ax.set_title("Opportunity To Express\n(question to trait cosine)", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="cosine")
    fig.tight_layout()
    save(fig, "fig3_ote_heatmap.png")


# --- Fig 4: within-person SD distribution per scale ---
def fig4_within_sd():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, s in zip(axes.ravel(), SCALES):
        sd = d.groupby("user_no")[s].std().dropna()
        ax.hist(sd, bins=30, color=CYAN, alpha=.85, edgecolor="white", linewidth=.3)
        ax.axvline(sd.median(), color=MAGENTA, ls="--", lw=1.2)
        ax.set_title(LAB[s], fontsize=10); ax.set_xlabel("within-person SD"); ax.set_ylabel("persons")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Within-person SD across persons, per scale (variability, not reliability)", fontsize=13)
    fig.tight_layout()
    save(fig, "fig4_within_person_sd.png")


# --- Fig 5 / Fig 7: length slopes ---
def _ladder(path):
    return pd.read_csv(path).set_index("scale")["logwc_b_wc"].reindex(SCALES)


def fig5_human_slope():
    h = _ladder(ROOT / "outputs/stage2_human/human_length_ladder.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(SCALES))[::-1]
    colors = [MAGENTA if v < 0 else SLATE for v in h.values]
    ax.barh(y, h.values, color=colors, height=.6)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([LAB[s] for s in SCALES])
    ax.set_xlabel("log word-count slope (human)")
    ax.set_title("Human: longer answers scored lower on 6 of 7 scales", fontsize=12)
    for yi, v in zip(y, h.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig5_human_length_slope.png")


def fig7_slope_compare():
    h = _ladder(ROOT / "outputs/stage2_human/human_length_ladder.csv")
    l = _ladder(ROOT / "outputs/stage3_teas/teas_length_ladder.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(SCALES))[::-1]
    ax.barh(y + .2, h.values, height=.38, color=HUMAN, label="human")
    ax.barh(y - .2, l.values, height=.38, color=LLM, label="LLM (TEAS)")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([LAB[s] for s in SCALES])
    ax.set_xlabel("log word-count slope")
    ax.set_title("Length-sign reversal: human penalises length, the LLM rewards it", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig7_length_slope_compare.png")


# --- Fig 6 / Fig 8: raw vs BLUP (mirrors notebook cells) ---
def _blup_grid(path, name, title):
    b = pd.read_csv(path)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, s in zip(axes.ravel(), SCALES):
        raw, bl = b[f"{s}_rawmean"].dropna(), b[f"{s}_blup"].dropna()
        lo, hi = min(raw.min(), bl.min()), max(raw.max(), bl.max())
        bins = np.linspace(lo, hi, 26)
        ax.hist(raw, bins=bins, color=MIST, alpha=.6, label="raw mean", edgecolor="none")
        ax.hist(bl, bins=bins, color=VIOLET, alpha=.75, label="BLUP", edgecolor="white", linewidth=.3)
        ax.axvline(bl.mean(), color=MAGENTA, ls="--", lw=1)
        ax.set_title(LAB[s], fontsize=10); ax.set_xlabel("per-person score"); ax.set_ylabel("persons")
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    axes.ravel()[-1].axis("off")
    fig.suptitle(title, fontsize=13); fig.tight_layout()
    save(fig, name)


def fig6_human_blup():
    _blup_grid(ROOT / "outputs/stage2_human/human_person_scores_blup.csv",
               "fig6_human_blup.png", "Raw mean vs BLUP per-person score, human (shrinkage)")


def fig8_llm_blup():
    _blup_grid(ROOT / "outputs/stage3_teas/teas_person_scores_blup.csv",
               "fig8_llm_blup.png", "Raw mean vs BLUP per-person score, LLM (on the Likert range)")


# --- Fig 9: KDE human vs LLM, within-method standardised ---
def fig9_distributions_compare():
    teas = pd.read_csv(ROOT / "experiments/evaluation_of_gemini_outputs/results/teas_run2_gpt-4o-mini.csv")
    m = d[["response_id"] + SCALES].merge(teas[["response_id"] + [f"E_{s}" for s in SCALES]],
                                          on="response_id", how="left")

    def z(x):
        x = x.dropna(); return (x - x.mean()) / x.std()

    grid = np.linspace(-3.5, 3.5, 300)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, s in zip(axes.ravel(), SCALES):
        for data, c, lab in [(z(m[s]), HUMAN, "human"), (z(m[f"E_{s}"]), LLM, "LLM")]:
            dens = gaussian_kde(data)(grid)
            ax.plot(grid, dens, color=c, lw=2, label=lab)
            ax.fill_between(grid, dens, color=c, alpha=.12)
        ax.set_title(LAB[s], fontsize=10); ax.set_xlabel("score (within-method z)"); ax.set_yticks([])
        ax.set_xlim(-3.5, 3.5)
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Score distributions by scale, human vs LLM (continuous z vs coarse Likert)", fontsize=13)
    fig.tight_layout()
    save(fig, "fig9_distributions_compare.png")


# --- Fig 10: convergence forest ---
def fig10_convergence():
    h = pd.read_csv(ROOT / "outputs/stage2_human/human_person_scores_blup.csv").set_index("user_no")
    t = pd.read_csv(ROOT / "outputs/stage3_teas/teas_person_scores_blup.csv").set_index("user_no")
    rs = []
    for s in SCALES:
        mm = pd.DataFrame({"h": h[f"{s}_blup"], "l": t[f"{s}_blup"]}).dropna()
        rs.append(pearsonr(mm.h, mm.l)[0])
    rs = pd.Series(rs, index=SCALES)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(SCALES))[::-1]
    ax.barh(y, rs.values, color=MAGENTA, height=.6)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(rs.mean(), color=VIOLET, ls="--", lw=1.4, label=f"mean r = {rs.mean():.2f}")
    ax.set_yticks(y); ax.set_yticklabels([LAB[s] for s in SCALES])
    ax.set_xlabel("Pearson r (human BLUP vs LLM BLUP)")
    ax.set_title("Human vs LLM convergence: negative on all seven scales", fontsize=12)
    for yi, v in zip(y, rs.values):
        ax.text(v - 0.008, yi, f"{v:.2f}", va="center", ha="right", fontsize=8.5, color=INK)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig10_convergence.png")


if __name__ == "__main__":
    fig1_flow(); fig2_distributions(); fig3_ote(); fig4_within_sd(); fig5_human_slope()
    fig6_human_blup(); fig7_slope_compare(); fig8_llm_blup(); fig9_distributions_compare()
    fig10_convergence()
    print("\nall figures written to", OUT.relative_to(ROOT))
