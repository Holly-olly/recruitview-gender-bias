#!/usr/bin/env python3
"""Stage 2 (human) — Bayesian length+style ladder on the human ratings (clean analysis sample).

Per scale, bambi fits three nested models on the human z-scores (same rows for all three):
  m1 = gender + (1|user)+(1|question)                                # baseline
  wc = gender + log(word count) + (1|user)+(1|question)              # + length
  ms = gender + log(word count) + FK grade + (1|user)+(1|question)   # + readability (style block)

NB: an experimental 4th "content" rung (answer<->trait cosine + cosine x length, Olga's formula)
was built and then parked — the ladder stops at FK for now. See
docs/stage2_answer_trait_cosine_findings.md and src/stage2_answer_trait_cosine.py to revisit.

The style block follows Speer et al. (2025): "generic writing style" measured content-free.
Here it is Flesch-Kincaid grade (syntactic complexity), which is near-orthogonal to word count
(Spearman ~0.17 in this sample), so it adds a distinct axis length cannot capture. The strict
test: does the person variance component shrink FURTHER after readability, beyond length, or has
length already captured the stylistic channel?

For each model we report the raw variance components — sigma2_person, sigma2_question,
sigma2_residual — so it is visible *which* component moves (not only the collapsed person-VPC).
We also report the gender effect (M-F) and how much its coefficient attenuates after each
adjustment, the length slope, and the FK slope.

Descriptive framing (no causal claims):
  * d_sigma2_* = change in the variance component after ADJUSTMENT for length / readability,
    not "length absorbs person variance".
  * a negative log-wc / FK slope = "longer / more-complex responses were associated with lower
    ratings", not "humans penalise length".
  * gender_atten_pct = attenuation of the gender coefficient after adjustment,
    not "length explains the gender gap".

VPC definition note — this is NOT the same decomposition as the classical G-theory VPC:
  * In the crossed G-theory ANOVA, the person x question (p:q) interaction is folded into the
    residual, so VPC_person = sigma2_p / (sigma2_p + sigma2_pq_residual).
  * Here the mixed model estimates question as its own random effect, so
    VPC_person = sigma2_person / (sigma2_person + sigma2_question + sigma2_residual).
  The two numbers answer related but different questions; do not equate them one-to-one.

Writes outputs/stage2_human/human_length_ladder.csv.

Run:  .venv/bin/python src/stage2_human_ladder.py
"""
import logging
import warnings
import numpy as np
import pandas as pd
import bambi as bmb
from pathlib import Path

warnings.filterwarnings("ignore")
# silence PyMC sampler chatter ("Initializing NUTS...", "Sampling 4 chains...") and arviz
# convergence notes. Must be AFTER importing bambi/pymc — pymc resets its logger level on import.
for _lg in ("pymc", "arviz"):
    logging.getLogger(_lg).setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage2_human"
SCALES = ["openness", "conscientiousness", "extraversion", "agreeableness",
          "neuroticism", "answer_score", "interview_score"]
RE = "(1|user_no) + (1|question_id)"


def _var_components(idata):
    """Posterior-mean variance components (person, question, residual)."""
    p = idata.posterior
    sp = float((p["1|user_no_sigma"] ** 2).mean())
    sq = float((p["1|question_id_sigma"] ** 2).mean())
    sr = float((p["sigma"] ** 2).mean())
    return sp, sq, sr


def _vpc(sp, sq, sr):
    """Person VPC = sigma2_person / (sigma2_person + sigma2_question + sigma2_residual).
    NB: different decomposition from classical G-theory (see module docstring)."""
    return sp / (sp + sq + sr)


def _coef(idata, name):
    if name in idata.posterior:
        v = idata.posterior[name].values.reshape(-1)
        return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)
    return (np.nan, np.nan, np.nan)


def _atten(g_base, g_adj):
    """% attenuation of the gender coefficient; NaN when the baseline effect is ~0
    (95% CI crosses 0) — otherwise dividing by a near-zero gender effect is meaningless."""
    if not (g_base[1] < 0 < g_base[2]) and g_base[0]:
        return 100 * (1 - g_adj[0] / g_base[0])
    return np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data/recruitview_analysis.csv")
    st = pd.read_csv(ROOT / "experiments/length_confound/results/style_metrics.csv")
    d = d.merge(st[["id", "fk_grade"]], on="id", how="left")
    d["user_no"] = d["user_no"].astype(str)
    d["question_id"] = d["question_id"].astype(str)
    d["gender"] = pd.Categorical(d["gender"].map({0: "F", 1: "M"}), categories=["F", "M"])
    lg = np.log1p(d["n_tokens"])
    d["log_wc_z"] = (lg - lg.mean()) / lg.std()
    d["fk_z"] = (d["fk_grade"] - d["fk_grade"].mean()) / d["fk_grade"].std()

    rows = []
    for t in SCALES:
        # same rows for all three nested models (drop where FK is missing too)
        dd = d[["user_no", "question_id", "gender", "log_wc_z", "fk_z", t]].dropna().rename(columns={t: "y"})
        fit = lambda f: bmb.Model(f, dd, family="gaussian").fit(
            draws=800, tune=800, chains=4, cores=1, target_accept=0.9, random_seed=7, progressbar=False)
        i1 = fit(f"y ~ 1 + gender + {RE}")
        iw = fit(f"y ~ 1 + gender + log_wc_z + {RE}")
        im = fit(f"y ~ 1 + gender + log_wc_z + fk_z + {RE}")

        sp1, sq1, sr1 = _var_components(i1)
        spw, sqw, srw = _var_components(iw)
        spm, sqm, srm = _var_components(im)
        g1, gw, gm = _coef(i1, "gender"), _coef(iw, "gender"), _coef(im, "gender")
        lw, lm = _coef(iw, "log_wc_z"), _coef(im, "log_wc_z")
        fk = _coef(im, "fk_z")
        rows.append(dict(
            scale=t, n=len(dd),
            # variance components — person / question / residual, BEFORE (m1) and AFTER (wc) length
            s2_person_m1=round(sp1, 3), s2_person_wc=round(spw, 3), s2_person_ms=round(spm, 3),
            s2_question_m1=round(sq1, 3), s2_question_wc=round(sqw, 3),
            s2_resid_m1=round(sr1, 3), s2_resid_wc=round(srw, 3), s2_resid_ms=round(srm, 3),
            # change in each component after adjustment for response length (which component moves)
            d_person_len=round(spw - sp1, 3), d_question_len=round(sqw - sq1, 3),
            d_resid_len=round(srw - sr1, 3), d_person_fk=round(spm - spw, 3),
            # collapsed person-VPC across the ladder
            VPC_m1=round(_vpc(sp1, sq1, sr1), 3), VPC_wc=round(_vpc(spw, sqw, srw), 3),
            VPC_ms=round(_vpc(spm, sqm, srm), 3),
            # gender effect (M-F) and its attenuation after each adjustment
            gender_m1=round(g1[0], 3), gender_wc=round(gw[0], 3), gender_ms=round(gm[0], 3),
            gender_atten_wc=round(_atten(g1, gw), 1), gender_atten_ms=round(_atten(g1, gm), 1),
            # slopes (standardised); negative = longer / more-complex responses -> lower rating
            logwc_b_wc=round(lw[0], 3), logwc_b_ms=round(lm[0], 3),
            fk_b=round(fk[0], 3), fk_lo=round(fk[1], 3), fk_hi=round(fk[2], 3)))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "human_length_ladder.csv", index=False)
    _print_tables(tab)
    print(f"\nwrote {OUT/'human_length_ladder.csv'}")


def _print_tables(tab: pd.DataFrame) -> None:
    """English summary tables of the ladder (m1 -> +length -> +readability)."""
    print("Answer-length & readability ladder — human z-scores")
    print("  m1 = gender only  |  wc = + log word-count  |  ms = + Flesch-Kincaid readability\n")

    vc = tab[["scale", "s2_person_m1", "s2_person_wc", "s2_question_m1", "s2_question_wc",
              "s2_resid_m1", "s2_resid_wc"]].copy()
    vc.columns = ["scale", "person_before", "person_after", "question_before", "question_after",
                  "residual_before", "residual_after"]
    print("Variance components before vs after adjustment for response length:")
    print(vc.to_string(index=False))

    sl = tab[["scale", "VPC_m1", "VPC_wc", "VPC_ms", "logwc_b_wc", "fk_b",
              "gender_m1", "gender_wc", "gender_atten_wc"]].copy()
    sl.columns = ["scale", "VPC_m1", "VPC_wc", "VPC_ms", "length_b", "FK_b",
                  "gender_before", "gender_after", "gender_atten_%"]
    print("\nVPC across the ladder, length/readability slopes, and gender:")
    print("  length_b < 0 = longer responses were associated with lower ratings;")
    print("  gender_atten_% = gender-coefficient attenuation after adjustment for response length.")
    print(sl.to_string(index=False))


if __name__ == "__main__":
    main()
