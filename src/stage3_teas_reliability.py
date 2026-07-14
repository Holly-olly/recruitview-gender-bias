#!/usr/bin/env python3
"""G-theory + Bayesian variance decomposition of the TEAS/Gemini LLM scores.

Estimates, per scale, the "share of true personality" ``VPC_person`` = the
proportion of a single LLM score's variance that is stable between-person (true)
signal versus noise -- plus reliability (E-rho^2, Phi, D-study) -- **mirroring the
Stage-1 human pipeline exactly** so the LLM and the human baseline are directly
comparable.

Engines (identical to Stage 1):
  * Classical G-theory via ``GeneralizIT`` (design ``Person x question_id``) -> the
    headline VPC_person = sigma2_person / (sigma2_person + sigma2_residual).
    Reference: ``src/stage2_human_gtheory.py``.
  * Bayesian ladder via ``bambi``/PyMC: ``score ~ 1 + gender + (1|user_no) +
    (1|question_id)`` and the length-controlled ``+ log_wc_z``. Reference:
    ``src/stage2_human_ladder.py``.

Classification uses the Stage-1 rule EXACTLY (reference ``src/stage2_human_tables.py``):
  trust if VPC_person >= .50; else marginal if E-rho^2(n=6) >= .70; else do-not-trust,
  evaluated at a FIXED n = 6 questions.

This script fixes three flaws found in ``src/stage3_teas_reml_crosscheck.py`` (kept for comparison):
  (a) it classified E-rho^2 at each scale's median n with a laxer trust>=.70 rule
      -> here: Stage-1 rule at fixed n=6;
  (b) it listwise-deleted abstentions silently -> here: abstention% is reported
      prominently and scales with >= 25% abstention are flagged PROVISIONAL
      (do-not-headline) with an MNAR caveat;
  (c) it never controlled answer length -> here: the length ladder is run and the
      drop in VPC_person when length is added (dVPC) is reported = the length-free
      share of true personality.
``interview_score`` is a 3-level (0-2) outcome; its Gaussian decomposition is coarse
and is flagged.

Run from the repository root:
    .venv/bin/python src/stage3_teas_reliability.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for _logger in ("bambi", "pymc", "pytensor", "arviz"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

# Scales in the same order Stage 1 reported: 5 Big Five, then the two performance
# scores. Big Five + answer_score are 1-5 Likert; interview_score is 0/1/2.
SCALES = ["openness", "conscientiousness", "extraversion", "agreeableness",
          "neuroticism", "answer_score", "interview_score"]
ABSTENTION_PROVISIONAL = 25.0   # >= this % missing -> provisional / do-not-headline
COARSE_ORDINAL = {"interview_score"}  # 3-level, Gaussian is coarse
N_FIXED = 6                     # Stage-1 classification is evaluated at this n

# A number token robust to plain decimals and scientific notation.
_NUM = r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)"


# --------------------------------------------------------------------------- #
# Pure functions: reliability formulas + the Stage-1 classification rule.
# These are the exact formulas from the task / Stage-1 references.
# --------------------------------------------------------------------------- #
def vpc_person(sigma2_person: float, sigma2_resid: float) -> float:
    """Share of true personality: sigma2_p / (sigma2_p + sigma2_r).

    Negative ANOVA components are clamped to 0 for the ratio (noted by the
    caller). Returns nan if the denominator is non-positive (not identifiable).
    """
    sp = max(sigma2_person, 0.0)
    sr = max(sigma2_resid, 0.0)
    denom = sp + sr
    if denom <= 0:
        return float("nan")
    return sp / denom


def erho2(sigma2_person: float, sigma2_resid: float, n: float) -> float:
    """Relative (rank-order) reliability of a mean over n crossed questions."""
    sp = max(sigma2_person, 0.0)
    sr = max(sigma2_resid, 0.0)
    if n <= 0 or (sp + sr / n) <= 0:
        return float("nan")
    return sp / (sp + sr / n)


def phi(sigma2_person: float, sigma2_question: float, sigma2_resid: float,
        n: float) -> float:
    """Absolute reliability: question difficulty counts as error too."""
    sp = max(sigma2_person, 0.0)
    sq = max(sigma2_question, 0.0)
    sr = max(sigma2_resid, 0.0)
    if n <= 0 or (sp + (sq + sr) / n) <= 0:
        return float("nan")
    return sp / (sp + (sq + sr) / n)


def n_for_rho(sigma2_person: float, sigma2_resid: float, rho: float) -> float:
    """Spearman-Brown D-study: questions needed for E-rho^2 >= rho."""
    sp = max(sigma2_person, 0.0)
    sr = max(sigma2_resid, 0.0)
    if sp <= 0:
        return float("inf")
    return (rho / (1 - rho)) * (sr / sp)


def classify_stage1(vpc: float, erho2_at_6: float) -> str:
    """Stage-1 rule EXACTLY: trust if VPC>=.50; elif E-rho^2(6)>=.70 marginal; else do-not-trust."""
    if not np.isfinite(vpc):
        return "unidentified"
    if vpc >= 0.50:
        return "trust"
    if np.isfinite(erho2_at_6) and erho2_at_6 >= 0.70:
        return "marginal"
    return "do-not-trust"


# --------------------------------------------------------------------------- #
# Classical engine: GeneralizIT (mirrors src/stage2_human_gtheory.py).
# --------------------------------------------------------------------------- #
def _capture(fn) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _capture_return(fn):
    """Run fn with stdout suppressed and return its value (GeneralizIT prints on init)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _grab(pattern: str, text: str, group: int = 1) -> float:
    match = re.search(pattern, text)
    return float(match.group(group)) if match else float("nan")


def fit_classical(scale_df: pd.DataFrame) -> dict:
    """Classical p x q variance components from GeneralizIT.

    ``scale_df`` has columns user_no, question_id, score (abstentions already
    dropped). Returns raw and clamped sigma2 for person / question / residual,
    the number of clamped components, and GeneralizIT's built-in person rho^2.
    """
    from generalizit import GeneralizIT

    data = (scale_df[["user_no", "question_id", "score"]].dropna()
            .rename(columns={"user_no": "Person"}))
    engine = _capture_return(
        lambda: GeneralizIT(data=data, design_str="Person x question_id", response="score"))
    _capture(engine.calculate_anova)
    _capture(engine.calculate_g_coefficients)
    var_txt = _capture(engine.variance_summary)
    g_txt = _capture(engine.g_coefficients_summary)

    sp = _grab(rf"(?m)^person\s+{_NUM}", var_txt)
    sq = _grab(rf"(?m)^question_id\s+{_NUM}", var_txt)
    sr = _grab(rf"(?m)^person x question_id\s+{_NUM}", var_txt)
    # GeneralizIT prints "Phi  rho^2"; person rho^2 is the 2nd number (at its
    # own harmonic-mean n) -- reported only as an extra corroboration.
    g_rho2 = _grab(rf"(?m)^person\s+{_NUM}\s+{_NUM}", g_txt, group=2)

    clamped = int(sum(1 for v in (sp, sq, sr) if np.isfinite(v) and v < 0))
    return {"sigma2_person_raw": sp, "sigma2_question_raw": sq, "sigma2_resid_raw": sr,
            "sigma2_person": max(sp, 0.0) if np.isfinite(sp) else float("nan"),
            "sigma2_question": max(sq, 0.0) if np.isfinite(sq) else float("nan"),
            "sigma2_resid": max(sr, 0.0) if np.isfinite(sr) else float("nan"),
            "n_clamped": clamped, "generalizit_person_rho2": g_rho2}


# --------------------------------------------------------------------------- #
# Bayesian engine: the Stage-1 length ladder (mirrors src/stage2_human_ladder.py).
# --------------------------------------------------------------------------- #
def _posterior_components(idata) -> tuple[float, float, float]:
    post = idata.posterior
    sp = float((post["1|user_no_sigma"] ** 2).mean())
    sq = float((post["1|question_id_sigma"] ** 2).mean())
    sr = float((post["sigma"] ** 2).mean())
    return sp, sq, sr


def _coef(idata, name: str) -> tuple[float, float, float]:
    """Posterior mean and 95% CI for a named fixed effect, else nans."""
    post = idata.posterior
    var = name
    if var not in post:
        # bambi may name a 2-level categorical coefficient "gender" with a coord,
        # or occasionally "gender[M]"; fall back to any matching variable.
        candidates = [v for v in post.data_vars if v == name or v.startswith(f"{name}[")]
        if not candidates:
            return float("nan"), float("nan"), float("nan")
        var = candidates[0]
    values = np.asarray(post[var]).reshape(-1)
    return float(values.mean()), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _rhat_max(idata, names: list[str]) -> float:
    import arviz as az
    try:
        present = [n for n in names if n in idata.posterior]
        summary = az.summary(idata, var_names=present, kind="diagnostics")
        return float(summary["r_hat"].max())
    except Exception:
        return float("nan")


def _fit_one(formula: str, data: pd.DataFrame, draws: int, tune: int,
             chains: int, cores: int, seed: int):
    import bambi as bmb
    model = bmb.Model(formula, data, family="gaussian")
    idata = model.fit(draws=draws, tune=tune, chains=chains, cores=cores,
                      target_accept=0.9, random_seed=seed, progressbar=False)
    return idata


def fit_bayes_ladder(scale_df: pd.DataFrame, draws: int, tune: int, chains: int,
                     cores: int, seed: int) -> dict:
    """Two-model Bayesian ladder: intercept+gender, then +log_wc_z.

    ``scale_df`` has user_no, question_id, gender (categorical F/M), log_wc_z, y.
    VPC is the 3-component share sigma2_p/(sigma2_p+sigma2_q+sigma2_r) for both
    models (mirrors Stage 1); a 2-component VPC from model 1 is also returned for
    the apples-to-apples cross-check against the classical estimate.
    """
    base = "y ~ 1 + gender + (1|user_no) + (1|question_id)"
    idata1 = _fit_one(base, scale_df, draws, tune, chains, cores, seed)
    idata2 = _fit_one(base + " + log_wc_z", scale_df, draws, tune, chains, cores, seed)

    sp1, sq1, sr1 = _posterior_components(idata1)
    sp2, sq2, sr2 = _posterior_components(idata2)
    vpc3_m1 = sp1 / (sp1 + sq1 + sr1)
    vpc3_m2 = sp2 / (sp2 + sq2 + sr2)
    vpc2_m1 = sp1 / (sp1 + sr1)   # matches the classical denominator

    gender1 = _coef(idata1, "gender")
    gender2 = _coef(idata2, "gender")
    length_beta = _coef(idata2, "log_wc_z")
    div1 = int(idata1.sample_stats["diverging"].sum())
    div2 = int(idata2.sample_stats["diverging"].sum())
    rhat = max(_rhat_max(idata1, ["1|user_no_sigma", "1|question_id_sigma", "sigma", "gender"]),
               _rhat_max(idata2, ["1|user_no_sigma", "1|question_id_sigma", "sigma", "gender", "log_wc_z"]))
    return {
        "bayes_sigma2_person": sp1, "bayes_sigma2_question": sq1, "bayes_sigma2_resid": sr1,
        "bayes_vpc_m1": vpc3_m1, "bayes_vpc_after_length": vpc3_m2, "dVPC": vpc3_m2 - vpc3_m1,
        "bayes_vpc2_m1": vpc2_m1,
        "gender_MminusF": gender1[0], "gender_lo": gender1[1], "gender_hi": gender1[2],
        "gender_MminusF_len": gender2[0],
        "length_beta": length_beta[0], "length_lo": length_beta[1], "length_hi": length_beta[2],
        "divergences_m1": div1, "divergences_m2": div2, "rhat_max": rhat,
    }


# --------------------------------------------------------------------------- #
# Data loading.
# --------------------------------------------------------------------------- #
def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (teas long on the CLEAN sample merged with length/gender, human reliability)."""
    ana = pd.read_csv(root / "data/recruitview_analysis.csv")   # clean sample (dup_keep & n_tokens>5)
    keep = set(ana["response_id"])
    teas = pd.read_csv(root / "experiments/evaluation_of_gemini_outputs/results/teas_run2_long.csv")
    teas = teas[teas["response_id"].isin(keep)].copy()          # filter TEAS to the analysis set

    rm = pd.read_csv(root / "experiments/length_confound/results/response_metrics.csv")
    rm = rm[rm["n_tokens"] > 5].copy()
    log_wc = np.log1p(rm["n_tokens"])
    rm["log_wc_z"] = (log_wc - log_wc.mean()) / log_wc.std()
    rm["gender"] = pd.Categorical(rm["gender"].round().astype(int).map({0: "F", 1: "M"}),
                                  categories=["F", "M"])
    length = rm[["user_no", "question_id", "n_tokens", "log_wc_z", "gender"]]

    merged = teas.merge(length, on=["user_no", "question_id"], how="left")
    human = pd.read_csv(root / "outputs/stage2_human/human_reliability.csv")  # clean-sample human baseline
    return merged, human


# --------------------------------------------------------------------------- #
# Per-scale analysis.
# --------------------------------------------------------------------------- #
def analyze_scale(scale: str, merged: pd.DataFrame, run_bayes: bool,
                  draws: int, tune: int, chains: int, cores: int, seed: int) -> dict:
    block = merged[merged["construct"] == scale].copy()
    n_total = len(block)
    scored = block.dropna(subset=["score"])
    n_scored = len(scored)
    abstention_pct = 100.0 * (1 - n_scored / n_total) if n_total else float("nan")
    obs_per_person = scored.groupby("user_no")["question_id"].nunique()
    obs_per_person_median = float(obs_per_person.median()) if len(obs_per_person) else float("nan")

    row = {"scale": scale, "n_scored": n_scored, "abstention_pct": abstention_pct,
           "obs_per_person_median": obs_per_person_median,
           "provisional_abstention": abstention_pct >= ABSTENTION_PROVISIONAL,
           "coarse_ordinal": scale in COARSE_ORDINAL, "notes": ""}

    notes = []
    if scale in COARSE_ORDINAL:
        notes.append("3-level outcome: Gaussian G-theory is coarse")

    # ---- Classical G-theory (headline VPC) ----
    cls = fit_classical(scored)
    row.update(cls)
    sp, sq, sr = cls["sigma2_person"], cls["sigma2_question"], cls["sigma2_resid"]
    # Guard: person variance must be identifiable (needs replication per person).
    if not np.isfinite(sp) or sp <= 0 or obs_per_person_median < 1.5:
        row["VPC_person"] = float("nan")
        notes.append("person variance not identifiable -> VPC blanked")
    else:
        row["VPC_person"] = vpc_person(sp, sr)
    row["Ero2_6"] = erho2(sp, sr, N_FIXED)
    row["Phi_6"] = phi(sp, sq, sr, N_FIXED)
    row["n_for_rho070"] = n_for_rho(sp, sr, 0.70)
    row["n_for_rho080"] = n_for_rho(sp, sr, 0.80)
    row["classification"] = classify_stage1(row["VPC_person"], row["Ero2_6"])
    if cls["n_clamped"]:
        notes.append(f"{cls['n_clamped']} negative variance component(s) clamped to 0")

    # ---- Bayesian ladder ----
    if run_bayes:
        dd = (scored[["user_no", "question_id", "gender", "log_wc_z", "score"]].dropna()
              .rename(columns={"score": "y"}))
        dd["user_no"] = dd["user_no"].astype(str)
        dd["question_id"] = dd["question_id"].astype(str)
        bayes = fit_bayes_ladder(dd, draws, tune, chains, cores, seed)
        row.update(bayes)
        row["crosscheck_abs_diff"] = (abs(row["VPC_person"] - bayes["bayes_vpc2_m1"])
                                      if np.isfinite(row["VPC_person"]) else float("nan"))
    row["notes"] = "; ".join(notes)
    return row


# --------------------------------------------------------------------------- #
# Printing.
# --------------------------------------------------------------------------- #
def _f(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "  -  "
    return f"{value:.{digits}f}"


def print_report(rows: pd.DataFrame, human: pd.DataFrame, run_bayes: bool) -> None:
    print("\n" + "=" * 108)
    print("TEAS / Gemini LLM scores -- G-theory 'share of true personality' (mirrors Stage-1 human pipeline)")
    print("=" * 108)

    print("\n[1] CLASSICAL G-THEORY (GeneralizIT, design Person x question_id) -- headline VPC_person\n")
    header = (f"{'scale':<18}{'n':>6}{'abst%':>7}{'s2_p':>7}{'s2_q':>7}{'s2_r':>7}"
              f"{'VPC_p':>7}{'Ero2_6':>8}{'Phi_6':>7}{'n>.70':>7}  {'class':<13}{'flag':<12}")
    print(header)
    print("-" * len(header))
    for _, r in rows.iterrows():
        flag = "PROVISIONAL" if r["provisional_abstention"] else ("coarse" if r["coarse_ordinal"] else "")
        print(f"{r['scale']:<18}{int(r['n_scored']):>6}{r['abstention_pct']:>7.1f}"
              f"{_f(r['sigma2_person']):>7}{_f(r['sigma2_question']):>7}{_f(r['sigma2_resid']):>7}"
              f"{_f(r['VPC_person']):>7}{_f(r['Ero2_6']):>8}{_f(r['Phi_6']):>7}"
              f"{_f(r['n_for_rho070'], 1):>7}  {r['classification']:<13}{flag:<12}")
    print("\nVPC_person = share of a single score's variance that is stable between-person (true) signal.")
    print("Rule: trust if VPC>=.50; elif Ero2(n=6)>=.70 marginal; else do-not-trust.  n>.70 = D-study questions.")

    if run_bayes:
        print("\n[2] BAYESIAN LADDER (bambi: score ~ 1 + gender + (1|user) + (1|question) [+ log_wc_z])\n")
        header2 = (f"{'scale':<18}{'VPC(B)':>8}{'VPC+len':>9}{'dVPC':>8}"
                   f"{'gender M-F':>12}{'len beta':>10}{'div':>6}{'rhat':>7}")
        print(header2)
        print("-" * len(header2))
        for _, r in rows.iterrows():
            gender = f"{r['gender_MminusF']:+.3f}"
            div = int(r.get("divergences_m1", 0)) + int(r.get("divergences_m2", 0))
            print(f"{r['scale']:<18}{_f(r['bayes_vpc_m1']):>8}{_f(r['bayes_vpc_after_length']):>9}"
                  f"{r['dVPC']:>+8.3f}{gender:>12}{r['length_beta']:>+10.3f}{div:>6}{_f(r['rhat_max'], 3):>7}")
        print("\ndVPC = VPC_person(after length) - VPC_person(before); < 0 = change in VPC after adjustment")
        print("for response length. VPC+len is the length-free share. gender M-F > 0 = males scored higher.")

        print("\n[3] CROSS-CHECK: classical vs Bayesian VPC, both 2-component (person/(person+resid))\n")
        header3 = (f"{'scale':<18}{'classical':>10}{'bayes(2c)':>10}{'|diff|':>8}  {'agree<=.02':<14}"
                   f"{'bayes(3c)':>10}{'s2_q(B)':>9}")
        print(header3)
        print("-" * len(header3))
        for _, r in rows.iterrows():
            diff = r.get("crosscheck_abs_diff", float("nan"))
            agree = "yes" if (np.isfinite(diff) and diff <= 0.02) else "NO (shrinkage)"
            print(f"{r['scale']:<18}{_f(r['VPC_person']):>10}{_f(r['bayes_vpc2_m1']):>10}"
                  f"{_f(diff):>8}  {agree:<14}{_f(r['bayes_vpc_m1']):>10}{_f(r['bayes_sigma2_question']):>9}")
        print("\nApples-to-apples check = classical vs bayes(2c). Gaps > .02 reflect Bayesian partial-pooling")
        print("shrinkage, largest where per-person replication is thin (high-abstention agreeableness). The")
        print("bayes(3c) VPC in [2] is additionally lower than classical when question variance s2_q is non-trivial.")

    # ---- Human side-by-side (recompute human Ero2(6) via the SAME formula) ----
    print("\n[4] TEAS vs HUMAN Stage-1 (human Ero2(6) recomputed through the same formula for parity)\n")
    header4 = (f"{'scale':<18}{'VPC human':>10}{'VPC TEAS':>10}{'dVPC':>8}   "
               f"{'Ero2h(6)':>9}{'Ero2T(6)':>9}{'dEro2':>8}")
    print(header4)
    print("-" * len(header4))
    hmap = human.set_index("scale")
    for _, r in rows.iterrows():
        scale = r["scale"]
        if scale in hmap.index:
            h = hmap.loc[scale]
            h_vpc = float(h["VPC_person"])
            h_erho6 = erho2(float(h["sigma2_person"]), float(h["sigma2_resid"]), N_FIXED)
        else:
            h_vpc = h_erho6 = float("nan")
        t_vpc, t_erho6 = r["VPC_person"], r["Ero2_6"]
        dv = t_vpc - h_vpc if np.isfinite(t_vpc) and np.isfinite(h_vpc) else float("nan")
        de = t_erho6 - h_erho6 if np.isfinite(t_erho6) and np.isfinite(h_erho6) else float("nan")
        print(f"{scale:<18}{_f(h_vpc):>10}{_f(t_vpc):>10}{dv:>+8.3f}   "
              f"{_f(h_erho6):>9}{_f(t_erho6):>9}{de:>+8.3f}")

    trust = ", ".join(rows.loc[rows["classification"].eq("trust"), "scale"]) or "(none)"
    marginal = ", ".join(rows.loc[rows["classification"].eq("marginal"), "scale"]) or "(none)"
    provisional = ", ".join(rows.loc[rows["provisional_abstention"], "scale"]) or "(none)"
    print("\n" + "=" * 108)
    print(f"TRUST (Stage-1 rule): {trust}")
    print(f"MARGINAL:             {marginal}")
    print(f"PROVISIONAL (abstention >= {ABSTENTION_PROVISIONAL:.0f}%, do-not-headline, MNAR): {provisional}")
    print("=" * 108)


# --------------------------------------------------------------------------- #
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/stage3_teas")
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--tune", type=int, default=800)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4,
                        help="Chains are seeded per-chain, so cores only affects speed, not results.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--classical-only", action="store_true",
                        help="Skip the Bayesian ladder (fast smoke test).")
    parser.add_argument("--scales", nargs="*", choices=SCALES, default=SCALES)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged, human = load_data(root)
    run_bayes = not args.classical_only
    rows = []
    for scale in args.scales:
        tag = "classical" if args.classical_only else "classical + Bayesian ladder"
        print(f"[{scale}] {tag} ...", flush=True)
        rows.append(analyze_scale(scale, merged, run_bayes, args.draws, args.tune,
                                  args.chains, args.cores, args.seed))
    table = pd.DataFrame(rows)

    # Sanity invariants before writing user-facing results.
    for col in ("VPC_person", "Ero2_6", "Phi_6"):
        finite = table[col].dropna()
        if not finite.between(-1e-9, 1 + 1e-9).all():
            raise RuntimeError(f"{col} out of [0,1]: {finite.tolist()}")

    print_report(table, human, run_bayes)

    out = args.output_dir / "teas_reliability.csv"
    table.to_csv(out, index=False, float_format="%.6f")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
