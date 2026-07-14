#!/usr/bin/env python3
"""Stage 3 (TEAS/LLM) — the SAME length+style ladder as the human side, on TEAS scores.

Reuses the human ladder's engine (src/stage2_human_ladder.py) and the SAME predictors — style is
a property of the candidate's transcript, not of the scorer — so no new features are generated.
Only the dependent variable changes: TEAS gpt-4o-mini scores instead of human z-scores.

Per scale, three nested bambi models on the z-standardised TEAS score (same rows for all three):
  m1 = gender + (1|user)+(1|question)
  wc = gender + log(word count) + ...
  ms = gender + log(word count) + FK grade + ...

NB: the experimental content rung (answer<->trait cosine + cosine x length) was built and parked;
the ladder stops at FK. See docs/stage2_answer_trait_cosine_findings.md (holds the TEAS cosine
results too) and src/stage2_answer_trait_cosine.py to revisit.

Two TEAS-specific handling notes (see also the open critique in CLAUDE.md):
  * ABSTENTION (MNAR): E_<scale> is null when the coder abstained (openness ~37%, agreeableness
    ~44%; others low). Here they are LISTWISE-DELETED, exactly as the human ladder drops NaN — so
    openness/agreeableness estimates are on a biased ~55-63% subsample. Flagged, not fixed; a
    selection model is a separate task. The `n` and `abstain_pct` columns make the shrinkage explicit.
  * SCALE: TEAS scores are ordinal Likert (1-5; interview_score 0/1/2). They are z-standardised per
    scale so the Gaussian slopes are comparable to the human z-score ladder (ordinal sensitivity is
    a separate crosscheck, cf. src/stage3_teas_reml_crosscheck.py).

Output: outputs/stage3_teas/teas_length_ladder.csv  (mirror of human_length_ladder.csv).
Run:  .venv/bin/python src/stage3_teas_ladder.py
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import bambi as bmb

from stage2_human_ladder import _var_components, _vpc, _coef, _atten, RE, SCALES

warnings.filterwarnings("ignore")
# silence PyMC/arviz sampler chatter (must be AFTER importing bambi/pymc)
for _lg in ("pymc", "arviz"):
    logging.getLogger(_lg).setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage3_teas"
TEAS = ROOT / "experiments/evaluation_of_gemini_outputs/results/teas_run2_gpt-4o-mini.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    an = pd.read_csv(ROOT / "data/recruitview_analysis.csv")            # bridge: response_id <-> id
    teas = pd.read_csv(TEAS)
    st = pd.read_csv(ROOT / "experiments/length_confound/results/style_metrics.csv")

    keep = ["response_id", "id", "user_no", "question_id", "gender", "n_tokens"]
    d = an[keep].merge(teas[["response_id"] + [f"E_{s}" for s in SCALES]], on="response_id", how="left")
    d = d.merge(st[["id", "fk_grade"]], on="id", how="left")

    d["user_no"] = d["user_no"].astype(str)
    d["question_id"] = d["question_id"].astype(str)
    d["gender"] = pd.Categorical(d["gender"].map({0: "F", 1: "M"}), categories=["F", "M"])
    lg = np.log1p(d["n_tokens"])
    d["log_wc_z"] = (lg - lg.mean()) / lg.std()
    d["fk_z"] = (d["fk_grade"] - d["fk_grade"].mean()) / d["fk_grade"].std()

    rows = []
    for t in SCALES:
        ecol = f"E_{t}"
        # z-standardise the TEAS Likert score per scale (before dropping, on the merged sample)
        e = d[ecol]
        d["y_z"] = (e - e.mean()) / e.std()
        abstain_pct = round(100 * d[ecol].isna().mean(), 1)

        dd = d[["user_no", "question_id", "gender", "log_wc_z", "fk_z", "y_z"]].dropna().rename(
            columns={"y_z": "y"})
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
            scale=t, n=len(dd), abstain_pct=abstain_pct,
            s2_person_m1=round(sp1, 3), s2_person_wc=round(spw, 3), s2_person_ms=round(spm, 3),
            s2_question_m1=round(sq1, 3), s2_resid_m1=round(sr1, 3), s2_resid_ms=round(srm, 3),
            d_person_len=round(spw - sp1, 3), d_person_fk=round(spm - spw, 3),
            VPC_m1=round(_vpc(sp1, sq1, sr1), 3), VPC_wc=round(_vpc(spw, sqw, srw), 3),
            VPC_ms=round(_vpc(spm, sqm, srm), 3),
            gender_m1=round(g1[0], 3), gender_wc=round(gw[0], 3), gender_ms=round(gm[0], 3),
            gender_atten_wc=round(_atten(g1, gw), 1), gender_atten_ms=round(_atten(g1, gm), 1),
            logwc_b_wc=round(lw[0], 3), logwc_b_ms=round(lm[0], 3),
            fk_b=round(fk[0], 3), fk_lo=round(fk[1], 3), fk_hi=round(fk[2], 3)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "teas_length_ladder.csv", index=False)
    print(f"\nwrote {OUT/'teas_length_ladder.csv'}")


if __name__ == "__main__":
    main()
