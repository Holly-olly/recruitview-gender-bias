"""
Generator for notebooks/stage1_human_baseline.ipynb.

Builds the Stage 1 narrative notebook (one notebook per Stage) with nbformat, so
notebook JSON is never hand-written. Two parts:

  - Descriptive front-end (recomputed from data/recruitview_full.csv): gender
    distribution, target descriptives, distributions by gender, effect sizes,
    group-difference tests, correlation structure. Figures saved to outputs/stage1/.
  - Variance-decomposition / reliability back-end: LOADS the precomputed artifacts
    in outputs/stage1/ (produced by src/stage1_fit.py — the Bayesian crossed
    mixed-model ladder is heavy and is NOT re-run here) and renders the headline
    reliability table, gender effects, and the figures from src/stage1_plots.py.

Run:
    python src/build_stage1_notebook.py
Then execute:
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        notebooks/stage1_human_baseline.ipynb

Note: the descriptive part needs data/recruitview_full.csv (local, regenerate via
src/prepare_full_dataset.py). The reliability part needs only outputs/stage1/*.csv,
which are tracked in the repo.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "stage1_human_baseline.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- intro
md(r"""
# Stage 1 — Human Baseline: Reliability & Gender Signal

**Project:** recruitview-gender-bias
**Dataset:** RecruitView (2,011 responses · 331 participants · 12 z-scored targets)

This notebook walks through every step of Stage 1. The question Stage 1 answers is:
**are the human-derived RecruitView ratings reliable enough to act as a person-level
criterion, or should they be treated as one fallible measurement method among several?**

The narrative has two halves:

1. **Descriptive baseline** (recomputed here) — how psychologist-derived scores are
   distributed and how they differ by gender *before any LLM is involved*. Group
   differences here are sample properties, not model bias.
2. **Variance decomposition & reliability** — we reframe Generalizability Theory as a
   Bayesian crossed mixed-effects model ladder (fitted in `src/stage1_fit.py`) and ask
   how much variance is stable person signal vs. noise, then classify each target as
   trust / marginal / do-not-trust. The heavy model fitting is NOT re-run here; we load
   its outputs from `outputs/stage1/`.

**Gender encoding:** `0 = female`, `1 = male`.

> **Data note.** RecruitView is gated and licensed CC BY-NC 4.0; this repo does not
> redistribute its scores. The descriptive cells below need `data/recruitview_full.csv`
> (regenerate locally via `src/prepare_full_dataset.py`); the reliability cells need
> only the aggregate tables in `outputs/stage1/`, which are tracked here.
""")

code(r"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- locate repo root (works whether cwd is repo root or notebooks/) ---
ROOT = Path.cwd()
while not (ROOT / "outputs" / "stage1").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

OUT_DIR = ROOT / "outputs" / "stage1"      # all Stage 1 artifacts + figures live here
FIG_DIR = OUT_DIR
HAS_RAW = (ROOT / "data" / "recruitview_full.csv").exists()

sns.set_theme(style="whitegrid", context="notebook")
GENDER_LABELS = {0: "Female", 1: "Male"}          # 0 = female, 1 = male
GENDER_COLORS = {0: "#C45BAA", 1: "#4C72B0"}
PALETTE = [GENDER_COLORS[0], GENDER_COLORS[1]]

BIG_FIVE = ["openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism", "overall_personality"]
PERFORMANCE = ["interview_score", "answer_score", "speaking_skills",
               "confidence_score", "facial_expression", "overall_performance"]
TARGETS = BIG_FIVE + PERFORMANCE

print("repo root:", ROOT)
print("raw scores available (data/recruitview_full.csv):", HAS_RAW)
""")

md(r"""
## Part 1 — Descriptive baseline

The cells below recompute the descriptive baseline from the full score table. If the
local raw table is absent they are skipped, and the analysis resumes from the tracked
aggregates in Part 2.
""")

code(r"""
if HAS_RAW:
    df = pd.read_csv(ROOT / "data" / "recruitview_full.csv")
    df["gender_label"] = df["gender"].map(GENDER_LABELS)
    print(f"Responses: {len(df):,} | Participants: {df['user_no'].nunique()} "
          f"| Questions: {df['question_id'].nunique()}")
    # Schema only — raw RecruitView score rows are gated (CC BY-NC) and not displayed.
    print("Columns:", list(df.columns))
else:
    df = None
    print("data/recruitview_full.csv not found — skipping Part 1 (see README to regenerate).")
""")

md("### 1. Gender distribution")

code(r"""
if df is not None:
    resp = df["gender"].value_counts().sort_index()
    part = df.groupby("user_no")["gender"].first().value_counts().sort_index()
    summary = pd.DataFrame({
        "responses": resp,
        "responses_%": (resp / resp.sum() * 100).round(1),
        "participants": part,
        "participants_%": (part / part.sum() * 100).round(1),
    })
    summary.index = [GENDER_LABELS[i] for i in summary.index]
    display(summary)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, counts, title in [
        (axes[0], resp, f"By response (N = {resp.sum():,})"),
        (axes[1], part, f"By participant (N = {part.sum():,})"),
    ]:
        labels = [GENDER_LABELS[i] for i in counts.index]
        colors = [GENDER_COLORS[i] for i in counts.index]
        bars = ax.bar(labels, counts.values, color=colors)
        ax.set_title(title); ax.set_ylabel("Count")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    fig.suptitle("Gender distribution (0 = female, 1 = male)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_gender_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

md("### 2. Descriptive statistics of the 12 targets (overall)")

code(r"""
if df is not None:
    desc = df[TARGETS].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    display(desc.round(3))
""")

md(r"""
### 3. Distributions by gender

Violin plots of all 12 targets split by gender, y-axis clipped to ±4 SD for readability
(every variable has extreme outliers up to |z| ~10 that are kept in the data). The
sharp leptokurtosis visible here is what later forces the robust/Student-t branch.
""")

code(r"""
if df is not None:
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    for ax, col in zip(axes.ravel(), TARGETS):
        sns.violinplot(data=df, x="gender_label", y=col, hue="gender_label",
                       order=["Female", "Male"], palette=PALETTE, legend=False,
                       inner="quartile", cut=0, ax=ax)
        ax.set_ylim(-4, 4); ax.set_title(col); ax.set_xlabel(""); ax.set_ylabel("z-score")
    fig.suptitle("Target distributions by gender (clipped to ±4 SD)", y=1.001, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_distributions_by_gender.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

md(r"""
### 4. Means by gender and effect size (descriptive)

For each target: mean per gender, the male−female difference (z units), and **Cohen's d**
(pooled SD). `d > 0` means men score higher. These are *response-level* (n = 2,011) — a
descriptive baseline, not yet a reliability-aware estimate (see Part 2).
""")

code(r"""
def cohens_d(a, b):
    a, b = a.dropna(), b.dropna()
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled else np.nan

if df is not None:
    fem, mal = df[df["gender"] == 0], df[df["gender"] == 1]
    rows = [{
        "variable": col,
        "female_mean": fem[col].mean(), "male_mean": mal[col].mean(),
        "diff_M_minus_F": mal[col].mean() - fem[col].mean(),
        "cohens_d": cohens_d(mal[col], fem[col]),
    } for col in TARGETS]
    gmeans = pd.DataFrame(rows).set_index("variable").round(3)
    display(gmeans)

    d_sorted = gmeans["cohens_d"].sort_values()
    colors = ["#4C72B0" if v >= 0 else "#C45BAA" for v in d_sorted.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d_sorted.index, d_sorted.values, color=colors)
    for thr in (0.2, 0.5, -0.2, -0.5):
        ax.axvline(thr, color="grey", ls=":", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Cohen's d  (positive = men score higher)")
    ax.set_title("Gender gap in human ratings by target (descriptive effect size)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_cohens_d_by_target.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

md(r"""
### 5. Group-difference tests

Two levels: **response-level** (n = 2,011) treats every answer as independent (inflated
power — each participant gives ~6 answers), and **participant-level** (n = 331) averages
to one mean per participant first. The participant-level test is the honest descriptive
check; the model-based gender effect in Part 2 supersedes both.
""")

code(r"""
if df is not None:
    part_df = df.groupby(["user_no", "gender"])[TARGETS].mean().reset_index()
    pf, pm = part_df[part_df["gender"] == 0], part_df[part_df["gender"] == 1]
    rows = []
    for col in TARGETS:
        t_r, p_r = stats.ttest_ind(mal[col], fem[col], equal_var=False)
        t_p, p_p = stats.ttest_ind(pm[col], pf[col], equal_var=False)
        rows.append({"variable": col, "resp_t": round(t_r, 2), "resp_p": p_r,
                     "part_t": round(t_p, 2), "part_p": p_p,
                     "part_cohens_d": round(cohens_d(pm[col], pf[col]), 3)})
    tests = pd.DataFrame(rows).set_index("variable")
    m = len(tests); ranked = tests["part_p"].rank(method="first")
    tests["part_p_FDR_sig"] = tests["part_p"] <= (ranked / m * 0.05)
    display(tests.round(4))
""")

md("### 6. Correlation structure of the 12 targets")

code(r"""
if df is not None:
    corr = df[TARGETS].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix — 12 target variables")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ---------------------------------------------------------------- part 2
md(r"""
## Part 2 — Variance decomposition & reliability

A descriptive gender gap is not enough: before trusting any score we must know how much
of it is *stable person signal* vs. noise. We reframe Generalizability Theory as a ladder
of Bayesian crossed-random-effects models — `(1|user_no) + (1|set_id) + (1|set_id:question_id)`
— fitted per target in **`src/stage1_fit.py`** (59 fits; person crossed with set, only
question nested). Diagnostics (`src/stage1_diagnostics.py`) first established the design:
15 reconstructed question sets, a median of 6 questions per person, and pronounced
leptokurtosis (11/12 targets, excess kurtosis 8.8–13.4) that mandates a Student-t branch.

Those models are heavy; here we **load their outputs** and read off the conclusions.

> **Constraint that bounds every number below:** only the derived scores are available —
> no raw pairwise data, no rater identifiers. There is one observation per person×question
> cell, so person×question is confounded with the residual: VPC_person is a **lower bound**
> on signal, the residual an **upper bound** on noise, and inter-rater reliability is not
> estimable at this stage.
""")

code(r"""
# Leptokurtosis that motivates the robust branch (figure from src/stage1_plots.py)
tc = pd.read_csv(OUT_DIR / "target_classification.csv")
vc = pd.read_csv(OUT_DIR / "variance_components.csv")
ge = pd.read_csv(OUT_DIR / "gender_effects.csv")
print("loaded:", "target_classification", tc.shape, "| variance_components", vc.shape,
      "| gender_effects", ge.shape)
""")

md(r"""
### 7. Where the variance goes

Across all 12 targets variance splits almost entirely between **person** (~18–30%) and
**residual** (~70–82%); the set and question components are negligible. Question variance
is near-zero *by construction*: the source labels came from within-question pairwise
comparisons, so between-question means are already netted out.

![Variance partition](../outputs/stage1/vpc_stacked_bars.png)
""")

md(r"""
### 8. Reliability and the trust classification

Headline = the Gaussian variance partition at the median workload `n_q = 6`. We report the
Variance Partition Coefficient for person (VPC_person), the relative-decision G-coefficient
Eρ², the absolute-decision Φ, and a Student-t **robust range** bracketing VPC_person.
A target is **trust** only if VPC_person ≥ .50 and Eρ² ≥ .70.
""")

code(r"""
cols = ["target", "block", "VPC_person", "Erho2", "Phi",
        "robust_range_lo", "robust_range_hi", "class"]
tc_disp = (tc[[c for c in cols if c in tc.columns]]
           .sort_values("VPC_person", ascending=False)
           .round(3)
           .reset_index(drop=True))
tc_disp
""")

md(r"""
**No target reaches "trust."** Eleven of twelve are **do-not-trust**; only `answer_score`
is **marginal** (VPC_person ≈ .30, Eρ² ≈ .72). Single-clip person consistency is ~.18–.30,
and even the six-question person aggregate is only borderline reliable (Eρ² ≈ .57–.72).
The verdict is robust to how the heavy tails are treated — even the optimistic upper bound
never crosses .50.

![Eρ² forest](../outputs/stage1/erho2_forest.png)

![VPC_person robust range](../outputs/stage1/vpc_person_robust_range.png)
""")

md(r"""
### 9. Answer duration carries person signal (not just noise)

Partialling out answer duration drops VPC_person by ~.05–.094, almost entirely out of the
*person* component — verbose participants are consistently scored higher, so answer length
is partly a stable trait expression, not a pure nuisance.

![Duration](../outputs/stage1/duration_slope.png)
""")

md(r"""
### 10. Gender effect from the models (human baseline)

Holding the design constant, the model-based gender fixed effect (m1, male relative to
female) is the clean human-baseline gender signal the LLM phase will be benchmarked against.
""")

code(r"""
ge_m1 = (ge[ge["model"] == "m1"][["target", "gender_coef", "ci_lo", "ci_hi"]]
         .sort_values("gender_coef", ascending=False)
         .round(3)
         .reset_index(drop=True))
ge_m1
""")

md(r"""
Males are rated significantly higher on **11 of 12 targets** (0.17–0.39 SD, credible
intervals exclude zero); only neuroticism — the one undesirable trait — is null, which is
diagnostically useful: a blanket pro-male halo would have pushed neuroticism *lower* for men.

![Gender forest](../outputs/stage1/gender_forest.png)
""")

md(r"""
### 11. Stage 1 takeaways

- Human-derived ratings **do not meet a person-level reliability standard**: ~75% of
  single-clip variance is non-person (an upper bound on noise), and six-question aggregates
  are only borderline (Eρ² ≈ .57–.72). Robust to tail treatment.
- Answer **duration** carries part of the genuine person signal — not pure nuisance.
- **Gender baseline:** men rated higher on 11/12 targets; neuroticism null.
- **Decision:** treat human ratings **not as ground truth but as one measurement method** →
  carry forward to a multitrait–multimethod (MTMM) comparison against LLM-based scores in
  Stage 2. The male-higher pattern is retained as the human-baseline gender signal.

Full write-up: `docs/stage1_human_baseline.md`.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "recruitview (venv)",
    "language": "python",
    "name": "recruitview-venv",
}
nb["metadata"]["language_info"] = {"name": "python"}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
