"""Stage 1 report figures — Neural Haze palette.

Builds the six figures referenced in docs/stage1_results_report.md directly
from the committed Stage 1 artefacts (no model refit required):

  1. dist_histograms.png          — small-multiple histograms of the 12 targets
                                     with a normal overlay + excess-kurtosis label
  2. vpc_stacked_bars.png         — stacked variance partition per target
  3. erho2_forest.png             — Eρ² per target with 95% CI, ref line at 0.70
  4. vpc_person_robust_range.png  — Gaussian VPC_person + Student-t [scale, marginal]
                                     range, ref line at 0.50
  5. duration_slope.png           — VPC_person with vs. without duration (m1 → m1d)
  6. gender_forest.png            — gender fixed effect ±95% CI, m1 and m1d overlaid

Inputs : data/recruitview_full.csv, outputs/stage1/{variance_components,
         target_classification,gender_effects}.csv, outputs/stage1/diagnostics.json
Outputs: outputs/stage1/*.png

Run from repo root:  python src/stage1_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────────
# Palette — mirrored from /Users/Olga/Documents/lib/palette.ts (Neural Haze).
# Source of truth is the .ts file; keep these in sync if it changes.
# ──────────────────────────────────────────────────────────────────────────
PAL = {
    "acidLime": "#C8F135",
    "limeMid": "#A3D420",
    "limeDark": "#6B9A0A",
    "electricViolet": "#9B30FF",
    "violetMid": "#7B1FE0",
    "violetDark": "#4E0FAD",
    "neuralCyan": "#00F5D4",
    "cyanMid": "#00C4A9",
    "hotMagenta": "#FF2D78",
    "magentaMid": "#CC1A5C",
    "ghostWhite": "#F0EFF8",
    "mist": "#C5C3D6",
    "slate": "#6E6A88",
    "deepSpace": "#1C1928",
    "void": "#0B0A14",
}

BG = PAL["deepSpace"]          # dark chart area
FG = PAL["ghostWhite"]         # labels on dark bg
GRID = PAL["slate"]            # grid / axes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage1"

TARGET_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
    "overall_personality": "Overall personality",
    "interview_score": "Interview score",
    "answer_score": "Answer score",
    "speaking_skills": "Speaking skills",
    "confidence_score": "Confidence",
    "facial_expression": "Facial expression",
    "overall_performance": "Overall performance",
}


def _style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.8)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def _new_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    return fig, ax


def _save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ──────────────────────────────────────────────────────────────────────────
# 1. Distribution small-multiples
# ──────────────────────────────────────────────────────────────────────────
def fig_histograms():
    df = pd.read_csv(ROOT / "data" / "recruitview_full.csv")
    diag = json.load(open(OUT / "diagnostics.json"))["diagnostics"]["distributions"]

    order = list(TARGET_LABELS)  # personality block first, then performance
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    fig.patch.set_facecolor(BG)

    for ax, t in zip(axes.ravel(), order):
        _style_ax(ax)
        x = df[t].dropna().values
        ek = diag[t]["excess_kurtosis"]
        near_normal = t == "neuroticism"
        bar_c = PAL["neuralCyan"] if near_normal else PAL["electricViolet"]

        ax.hist(x, bins=45, density=True, color=bar_c, alpha=0.55,
                edgecolor=BG, linewidth=0.3)
        # normal overlay using empirical mean/sd
        mu, sd = x.mean(), x.std()
        xs = np.linspace(x.min(), x.max(), 300)
        ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                color=PAL["acidLime"], lw=1.6)
        ax.set_title(TARGET_LABELS[t], fontsize=10)
        ax.annotate(f"excess kurt = {ek:.1f}", xy=(0.97, 0.92), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    color=PAL["acidLime"] if near_normal else PAL["hotMagenta"])
        ax.set_yticks([])

    legend = [
        Line2D([0], [0], color=PAL["acidLime"], lw=1.6, label="Normal reference"),
        Line2D([0], [0], marker="s", color=BG, markerfacecolor=PAL["electricViolet"],
               markersize=9, lw=0, label="Heavy-tailed (11 targets)"),
        Line2D([0], [0], marker="s", color=BG, markerfacecolor=PAL["neuralCyan"],
               markersize=9, lw=0, label="Neuroticism (near-normal)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               labelcolor=FG, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Distribution of each target score (density) with normal overlay",
                 color=FG, fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, "dist_histograms.png")


# ──────────────────────────────────────────────────────────────────────────
# helpers for ordering by VPC_person (descending = Table 1 order)
# ──────────────────────────────────────────────────────────────────────────
def _vc_headline():
    """m0, Gaussian, scale, n_q=6 — one row per target."""
    v = pd.read_csv(OUT / "variance_components.csv")
    return v[(v.model == "m0") & (v.estimator == "gaussian")
             & (v.residual_def == "scale") & (v.n_q == 6)].set_index("target")


# ──────────────────────────────────────────────────────────────────────────
# 2. Stacked variance partition
# ──────────────────────────────────────────────────────────────────────────
def fig_vpc_stacked():
    vc = _vc_headline()
    order = vc.sort_values("vpc_person").index.tolist()  # ascending → biggest on top
    labels = [TARGET_LABELS[t] for t in order]
    person = vc.loc[order, "vpc_person"].values
    sset = vc.loc[order, "vpc_set"].values
    ques = vc.loc[order, "vpc_question"].values
    resid = vc.loc[order, "vpc_resid"].values

    fig, ax = _new_fig((11, 7))
    y = np.arange(len(order))
    ax.barh(y, person, color=PAL["acidLime"], label="person (signal)")
    ax.barh(y, sset, left=person, color=PAL["neuralCyan"], label="set (cohort)")
    ax.barh(y, ques, left=person + sset, color=PAL["electricViolet"], label="question")
    ax.barh(y, resid, left=person + sset + ques, color=PAL["slate"], label="residual (noise)")
    ax.axvline(0.50, ls="--", lw=1.2, color=PAL["hotMagenta"], alpha=0.9)
    ax.text(0.50, len(order) - 0.3, " trust = 0.50", color=PAL["hotMagenta"],
            fontsize=9, va="top")
    for yi, p in zip(y, person):
        ax.text(p + 0.01, yi, f"{p:.2f}", va="center", ha="left",
                color=FG, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Variance partition coefficient")
    ax.set_title("Variance is concentrated in person and residual\n(Gaussian m0, n_q = 6)",
                 fontsize=12)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(FG)
    _save(fig, "vpc_stacked_bars.png")


# ──────────────────────────────────────────────────────────────────────────
# 3. Eρ² forest with 0.70 reference
# ──────────────────────────────────────────────────────────────────────────
def fig_erho2_forest():
    vc = _vc_headline()
    order = vc.sort_values("Erho2").index.tolist()  # ascending → best on top
    labels = [TARGET_LABELS[t] for t in order]
    mid = vc.loc[order, "Erho2"].values
    lo = vc.loc[order, "Erho2_lo"].values
    hi = vc.loc[order, "Erho2_hi"].values

    fig, ax = _new_fig((10, 7))
    y = np.arange(len(order))
    ax.hlines(y, lo, hi, color=PAL["mist"], lw=2.2, alpha=0.9)
    ax.plot(mid, y, "o", color=PAL["acidLime"], ms=7, zorder=3)
    ax.axvline(0.70, ls="--", lw=1.4, color=PAL["hotMagenta"])
    ax.text(0.70, -0.05, " 0.70 threshold", color=PAL["hotMagenta"],
            fontsize=9, va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Eρ²  (relative-decision G coefficient, n_q = 6)")
    ax.set_title("Six-question person-aggregate reliability falls at or below 0.70",
                 fontsize=12)
    _save(fig, "erho2_forest.png")


# ──────────────────────────────────────────────────────────────────────────
# 4. VPC_person Gaussian point + Student-t robust range, 0.50 reference
# ──────────────────────────────────────────────────────────────────────────
def fig_vpc_robust_range():
    tc = pd.read_csv(OUT / "target_classification.csv").set_index("target")
    order = tc.sort_values("VPC_person").index.tolist()
    labels = [TARGET_LABELS[t] for t in order]
    point = tc.loc[order, "VPC_person"].values
    rlo = tc.loc[order, "robust_range_lo"].values
    rhi = tc.loc[order, "robust_range_hi"].values

    fig, ax = _new_fig((10, 7))
    y = np.arange(len(order))
    for yi, a, b in zip(y, rlo, rhi):
        if np.isfinite(a) and np.isfinite(b):
            ax.hlines(yi, a, b, color=PAL["neuralCyan"], lw=2.2, alpha=0.8)
            ax.plot([a, b], [yi, yi], "|", color=PAL["neuralCyan"], ms=10)
    ax.plot(point, y, "o", color=PAL["acidLime"], ms=7, zorder=3,
            label="Gaussian headline")
    ax.axvline(0.50, ls="--", lw=1.4, color=PAL["hotMagenta"])
    ax.text(0.50, -0.05, " trust = 0.50", color=PAL["hotMagenta"],
            fontsize=9, va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("VPC_person  (Gaussian point + Student-t [scale, marginal] range)")
    ax.set_title("Even the optimistic tail treatment stays below the trust threshold",
                 fontsize=12)
    handles = [
        Line2D([0], [0], marker="o", color=BG, markerfacecolor=PAL["acidLime"],
               markersize=8, lw=0, label="Gaussian headline"),
        Line2D([0], [0], color=PAL["neuralCyan"], lw=2.2,
               label="Student-t range [scale → marginal]"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(FG)
    _save(fig, "vpc_person_robust_range.png")


# ──────────────────────────────────────────────────────────────────────────
# 5. Duration slope chart (m1 → m1d)
# ──────────────────────────────────────────────────────────────────────────
def fig_duration_slope():
    tc = pd.read_csv(OUT / "target_classification.csv").set_index("target")
    order = tc.sort_values("dur_sens_vpc_no_dur").index.tolist()  # ascending → biggest on top
    labels = [TARGET_LABELS[t] for t in order]
    no_dur = tc.loc[order, "dur_sens_vpc_no_dur"].values
    with_dur = tc.loc[order, "dur_sens_vpc_with_dur"].values
    delta = tc.loc[order, "dur_sens_delta"].values

    # Horizontal dumbbell: one row per target, m1 → m1d on the VPC_person axis.
    fig, ax = _new_fig((10, 7))
    y = np.arange(len(order))
    for yi, a, b in zip(y, no_dur, with_dur):
        ax.plot([b, a], [yi, yi], "-", color=PAL["slate"], lw=2.0, alpha=0.9, zorder=1)
    ax.plot(no_dur, y, "o", color=PAL["acidLime"], ms=8, zorder=3)
    ax.plot(with_dur, y, "o", color=PAL["hotMagenta"], ms=8, zorder=3)
    for yi, b, d in zip(y, with_dur, delta):
        ax.text(b - 0.004, yi + 0.28, f"Δ{d:+.03f}", va="bottom", ha="right",
                color=PAL["mist"], fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("VPC_person")
    ax.set_title("Partialling out duration removes person signal, not just noise",
                 fontsize=12)
    handles = [
        Line2D([0], [0], marker="o", color=BG, markerfacecolor=PAL["acidLime"],
               markersize=8, lw=0, label="without duration (m1)"),
        Line2D([0], [0], marker="o", color=BG, markerfacecolor=PAL["hotMagenta"],
               markersize=8, lw=0, label="with duration (m1d)"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(FG)
    _save(fig, "duration_slope.png")


# ──────────────────────────────────────────────────────────────────────────
# 6. Gender effect forest (m1 + m1d)
# ──────────────────────────────────────────────────────────────────────────
def fig_gender_forest():
    g = pd.read_csv(OUT / "gender_effects.csv")
    m1 = g[g.model == "m1"].set_index("target")
    m1d = g[g.model == "m1d"].set_index("target")
    order = m1.sort_values("gender_coef").index.tolist()
    labels = [TARGET_LABELS[t] for t in order]

    fig, ax = _new_fig((10, 7))
    y = np.arange(len(order))
    # m1 with CI
    ax.hlines(y, m1.loc[order, "ci_lo"], m1.loc[order, "ci_hi"],
              color=PAL["mist"], lw=2.0, alpha=0.85)
    ax.plot(m1.loc[order, "gender_coef"], y, "o", color=PAL["acidLime"], ms=7,
            zorder=3, label="m1 (gender)")
    # m1d point, slightly offset
    ax.plot(m1d.loc[order, "gender_coef"], y + 0.18, "D", color=PAL["neuralCyan"],
            ms=5, zorder=3, label="m1d (duration-adjusted)")
    ax.axvline(0.0, ls="--", lw=1.4, color=PAL["hotMagenta"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Gender fixed effect (SD; positive = male rated higher)")
    ax.set_title("Males rated significantly higher on 11 of 12 targets", fontsize=12)
    handles = [
        Line2D([0], [0], marker="o", color=BG, markerfacecolor=PAL["acidLime"],
               markersize=8, lw=0, label="m1 (gender) ±95% CI"),
        Line2D([0], [0], marker="D", color=BG, markerfacecolor=PAL["neuralCyan"],
               markersize=7, lw=0, label="m1d (duration-adjusted)"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(FG)
    _save(fig, "gender_forest.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building Stage 1 report figures (Neural Haze palette)…")
    fig_histograms()
    fig_vpc_stacked()
    fig_erho2_forest()
    fig_vpc_robust_range()
    fig_duration_slope()
    fig_gender_forest()
    print("Done →", OUT)


if __name__ == "__main__":
    main()
