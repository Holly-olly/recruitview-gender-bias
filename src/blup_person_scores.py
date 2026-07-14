#!/usr/bin/env python3
"""Transform per-question scores into ONE per-person score per scale, via the
empirical-Bayes / BLUP universe-score estimate (headline method from the
psychometric-aggregation review, 2026-07-11).

Model per scale (person x question crossed):  y_pq = mu + u_p + v_q + e_pq.
The person random-intercept posterior mean IS the G-theory universe score
(mu + u_p); shrinkage toward the grand mean is n-dependent, so persons with few
questions are pulled harder than persons with many. The raw arithmetic mean is
saved alongside for comparison (it equals the BLUP only in the balanced,
complete, perfectly-reliable case).

Engine: bambi (matches the Stage-1 Bayesian stack). Runs on the HUMAN ratings by
default; `--source teas` reuses it for the LLM scores.

Run:
    .venv/bin/python src/blup_person_scores.py
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data/recruitview_analysis.csv"          # clean sample (dup_keep & n_tokens>5)
TEAS_LONG = ROOT / "experiments/evaluation_of_gemini_outputs/results/teas_run2_long.csv"
OUT_DIR = {"human": ROOT / "outputs/stage2_human", "teas": ROOT / "outputs/stage3_teas"}
SCALES = ["openness", "conscientiousness", "extraversion", "agreeableness",
          "neuroticism", "answer_score", "interview_score"]


def load_scale(source: str, scale: str) -> pd.DataFrame:
    """Per-scale long frame on the CLEAN analysis sample: user_no, question_id, y (abstentions dropped)."""
    ana = pd.read_csv(ANALYSIS)
    if source == "human":
        d = ana[["user_no", "question_id", scale]].rename(columns={scale: "y"})
    else:  # teas — filter to the analysis response set
        long = pd.read_csv(TEAS_LONG)
        long = long[long.response_id.isin(set(ana.response_id))]
        d = (long[long.construct == scale][["user_no", "question_id", "score"]]
             .rename(columns={"score": "y"}))
    d = d.dropna(subset=["y"]).copy()
    d["user_no"] = d["user_no"].astype(str)
    d["question_id"] = d["question_id"].astype(str)
    return d


def person_blup(d: pd.DataFrame, draws: int, tune: int) -> pd.DataFrame:
    """Fit y ~ 1 + (1|user_no) + (1|question_id); return per-user BLUP + raw mean + n_q."""
    import bambi as bmb
    model = bmb.Model("y ~ 1 + (1|user_no) + (1|question_id)", d, family="gaussian")
    idata = model.fit(draws=draws, tune=tune, chains=4, cores=1,
                      target_accept=0.9, random_seed=7, progressbar=False)
    post = idata.posterior
    intercept = float(post["Intercept"].mean())
    person = post["1|user_no"].mean(("chain", "draw"))          # EB/BLUP deviations u_p
    users = [str(u) for u in person[person.dims[-1]].values]
    out = pd.DataFrame({"blup": intercept + person.values}, index=pd.Index(users, name="user_no"))
    out["rawmean"] = d.groupby("user_no")["y"].mean()
    out["nq"] = d.groupby("user_no")["y"].size()
    ndiv = int(idata.sample_stats["diverging"].sum())
    return out, ndiv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["human", "teas"], default="human")
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--tune", type=int, default=800)
    args = parser.parse_args()
    OUT_DIR[args.source].mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR[args.source] / f"{args.source}_person_scores_blup.csv"

    blup_tbl, raw_tbl, nq_series, summary = {}, {}, None, []
    for scale in SCALES:
        d = load_scale(args.source, scale)
        print(f"[{args.source}:{scale}] person x question BLUP  (n={len(d)}) ...", flush=True)
        res, ndiv = person_blup(d, args.draws, args.tune)
        blup_tbl[f"{scale}_blup"] = res["blup"]
        raw_tbl[f"{scale}_rawmean"] = res["rawmean"]
        nq_series = res["nq"] if nq_series is None else nq_series
        corr = float(np.corrcoef(res["blup"], res["rawmean"])[0, 1])
        shrink = 100 * (1 - res["blup"].std() / res["rawmean"].std())
        summary.append((scale, len(d), corr, res["rawmean"].std(), res["blup"].std(), shrink, ndiv))

    result = pd.concat([pd.DataFrame(blup_tbl), pd.DataFrame(raw_tbl)], axis=1)
    result["n_q"] = nq_series
    result.to_csv(out_path)

    print(f"\n{'scale':18s}{'n':>7}{'corr(BLUP,raw)':>16}{'sd_raw':>8}{'sd_blup':>9}{'shrink%':>9}{'ndiv':>6}")
    for s, n, c, sr, sb, sh, nd in summary:
        print(f"{s:18s}{n:>7}{c:>16.3f}{sr:>8.3f}{sb:>9.3f}{sh:>9.1f}{nd:>6}")
    print(f"\nwrote {out_path}  ({result.shape[0]} users x {result.shape[1]} columns)")
    print("BLUP = per-person universe score (EB-shrunk); rawmean kept for comparison; n_q = questions answered.")


if __name__ == "__main__":
    main()
