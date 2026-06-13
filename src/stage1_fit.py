"""
Stage 1 — Module 2: mixed-model ladder + variance decomposition (plan §4.2–4.4, §5).

Consumes the diagnostics hand-off (outputs/stage1/diagnostics.json +
recruitview_with_set.csv) and, per target, fits the generalizability-style mixed
models, extracts variance components, computes VPC + G/Φ with credible intervals,
and classifies each target trust / marginal / do-not-trust.

Random-effects structure (plan §2.1): (1|user_no) + (1|set_id) + (1|set_id:question_id).
Person is *crossed* with set (not nested) — 31 user_no touch >1 set (23 q27-anchor
artifacts + 8 genuine multi-session); only `question` is nested in `set`.

Residual-variance / headline convention (decision 2026-06-11, Olga):
  HEADLINE class = the conventional GAUSSIAN variance partition (residual = empirical
             residual variance). Most reviewer-defensible; leptokurtic spread counts as
             noise. Duration- and multi-session sensitivities are also Gaussian.
  ROBUST RANGE (reported as sensitivity, only when a target is leptokurtic) = a Student-t
             m0 fit summarised two ways: scale-based σ²_resid = σ² (down-weights tails →
             most optimistic VPC_person) and marginal σ²_resid = σ²·ν/(ν−2) (re-inflates
             tails → most pessimistic). The pair brackets how sensitive person-signal is
             to tail treatment.
NOTE: Gaussian and Student-t do NOT agree here — σ²_person is ~constant across all three,
but the residual denominator spans ~3× (e.g. overall_performance VPC_person 0.21–0.48).
All three numbers live in variance_components.csv; only `class` uses the Gaussian point.

Run (from the repository root):
    python src/stage1_fit.py              # full sweep (12 targets), background-friendly
    python src/stage1_fit.py --quick      # smoke: 1 target, reduced draws, writes all outputs
    python src/stage1_fit.py --targets overall_performance openness

Outputs (outputs/stage1/):
    variance_components.csv   — per target × model: σ², VPC[CI], Eρ²[CI], Φ[CI], rung, convergence
    gender_effects.csv        — gender fixed effect (m1 / m1d) per target
    target_classification.csv — class + rationale + duration / multi-session sensitivities
    stage1_handoff.json       — machine-readable summary for Stages 2–5
    fit_meta.json             — package versions, seed, sampler settings
    vpc_stacked_bars.png      — per-target variance-component partition (primary model)
    caterpillar_<target>.png  — person random effects for illustrative targets
    idata/<target>_<model>.nc — fitted InferenceData (gitignored, for reproducibility)

Dependencies: bambi, pymc, arviz (+ pandas, numpy, matplotlib).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for _name in ("pymc", "bambi", "pytensor"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import bambi as bmb        # noqa: E402  (after logger/warning setup)
import arviz as az         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "stage1"
DATA_FILE = OUT_DIR / "recruitview_with_set.csv"
DIAG_FILE = OUT_DIR / "diagnostics.json"
IDATA_DIR = OUT_DIR / "idata"

TARGETS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality", "interview_score", "answer_score",
    "speaking_skills", "confidence_score", "facial_expression",
    "overall_performance",
]
PERSONALITY_BLOCK = set(TARGETS[:6])
PERFORMANCE_BLOCK = set(TARGETS[6:])

GROUP_COLS = ["user_no", "set_id", "question_id"]
RE_TERMS = "(1|user_no) + (1|set_id) + (1|set_id:question_id)"

# Posterior variable names (verified against bambi 0.18 / pymc 6.0):
SIGMA_PERSON = "1|user_no_sigma"
SIGMA_SET = "1|set_id_sigma"
SIGMA_QUESTION = "1|set_id:question_id_sigma"
RE_PERSON = "1|user_no"  # the person effects array (dims include user_no__factor_dim)

# Classification thresholds (plan §4.4; overridable via config.thresholds in future).
TRUST_VPC, TRUST_ERHO2 = 0.50, 0.70
MARGINAL_VPC_LO = 0.25

CI_PROB = 0.95  # equal-tailed credible interval
_LO_Q, _HI_Q = (1 - CI_PROB) / 2 * 100, (1 + CI_PROB) / 2 * 100


# ──────────────────────────────────────────────────────────────────────────
# Data + diagnostics hand-off
# ──────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # grouping factors must be categorical (string) so bambi treats them as levels,
    # never as numeric covariates.
    for c in GROUP_COLS:
        df[c] = df[c].astype("Int64").astype(str)
    df["duration"] = df["duration"].astype("category")
    df["gender"] = df["gender"].astype("Int64").astype("category")  # 0=female, 1=male
    return df


def load_diagnostics() -> dict:
    return json.loads(DIAG_FILE.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# Model fitting (ladder with graceful degradation)
# ──────────────────────────────────────────────────────────────────────────
def _rhat_max(idata, var_names: list[str]) -> float:
    """Max R-hat over the listed scalar parameters (arviz 1.1 returns a DataTree)."""
    try:
        rh = az.rhat(idata, var_names=var_names)
    except Exception:
        return float("nan")
    if hasattr(rh, "groups"):  # DataTree
        vals = [float(np.nanmax(rh[g].to_dataset().to_array().values))
                for g in rh.groups if hasattr(rh[g], "to_dataset")]
        return max(vals) if vals else float("nan")
    return float(np.nanmax(rh.to_array().values))


def _n_divergences(idata) -> int:
    try:
        return int(idata.sample_stats["diverging"].values.sum())
    except Exception:
        return -1


# Ladder rungs: L0 = full crossed REs; L1 = drop set; L2 = person only (plan §4.2).
_LADDER = {
    "L0": "(1|user_no) + (1|set_id) + (1|set_id:question_id)",
    "L1": "(1|user_no) + (1|question_id)",
    "L2": "(1|user_no)",
}


def fit_model(df: pd.DataFrame, target: str, fixed: str, family: str,
              draws: int, tune: int, chains: int, seed: int,
              target_accept: float = 0.9) -> tuple[object, dict]:
    """
    Fit one model, descending the degradation ladder on failure / non-convergence.
    `fixed` is the fixed-effects part right of '~' (e.g. '1', '1 + gender').
    Returns (idata, info) where info records rung, family, convergence.
    """
    sigma_vars = {"L0": [SIGMA_PERSON, SIGMA_SET, SIGMA_QUESTION],
                  "L1": [SIGMA_PERSON, "1|question_id_sigma"],
                  "L2": [SIGMA_PERSON]}
    last_err = None
    for rung, re_terms in _LADDER.items():
        formula = f"{target} ~ {fixed} + {re_terms}"
        try:
            model = bmb.Model(formula, df, family=family)
            idata = model.fit(draws=draws, tune=tune, chains=chains, cores=chains,
                              target_accept=target_accept, random_seed=seed,
                              progressbar=False, inference_method="mcmc")
            check = sigma_vars[rung] + (["nu"] if family == "t" else [])
            rhat = _rhat_max(idata, check)
            ndiv = _n_divergences(idata)
            converged = (np.isfinite(rhat) and rhat < 1.05)
            info = {"rung": rung, "family": family, "formula": formula,
                    "rhat_max": round(float(rhat), 4), "divergences": ndiv,
                    "converged": bool(converged)}
            if converged:
                return idata, info
            last_err = f"{rung} non-converged (rhat={rhat:.3f})"
        except Exception as e:  # noqa: BLE001 — log reason, fall to next rung
            last_err = f"{rung} failed: {type(e).__name__}: {e}"
        print(f"    ↓ {target}/{fixed!r}: {last_err} → next rung", file=sys.stderr)
    raise RuntimeError(f"{target}: all rungs failed ({last_err})")


# ──────────────────────────────────────────────────────────────────────────
# Variance components → VPC / G-Φ (per posterior draw)
# ──────────────────────────────────────────────────────────────────────────
def extract_components(idata, rung: str) -> dict:
    """Per-draw variance components (flattened over chain×draw)."""
    post = idata.posterior
    def flat(name):
        return post[name].values.reshape(-1) if name in post else None
    s_person = flat(SIGMA_PERSON)
    s_set = flat(SIGMA_SET) if rung == "L0" else None
    s_q = flat(SIGMA_QUESTION) if rung == "L0" else flat("1|question_id_sigma")
    sigma = flat("sigma")
    nu = flat("nu")  # None for Gaussian
    return {
        "v_person": s_person ** 2,
        "v_set": (s_set ** 2) if s_set is not None else np.zeros_like(s_person),
        "v_question": (s_q ** 2) if s_q is not None else np.zeros_like(s_person),
        "sigma2": sigma ** 2,
        "nu": nu,
    }


def residual_variance(comp: dict, residual_def: str) -> np.ndarray:
    """scale → σ²; marginal → σ²·ν/(ν−2) (Student-t only; ν≤2 draws masked to nan)."""
    if residual_def == "marginal" and comp["nu"] is not None:
        nu = comp["nu"]
        return np.where(nu > 2, comp["sigma2"] * nu / (nu - 2), np.nan)
    return comp["sigma2"]


def metrics(comp: dict, residual_def: str, n_q: int) -> dict:
    """Per-draw VPCs + Eρ² + Φ for one residual definition and one n_q."""
    vp, vs, vq = comp["v_person"], comp["v_set"], comp["v_question"]
    vr = residual_variance(comp, residual_def)
    vtot = vp + vs + vq + vr
    return {
        "vpc_person": vp / vtot,
        "vpc_set": vs / vtot,
        "vpc_question": vq / vtot,
        "vpc_resid": vr / vtot,
        "erho2": vp / (vp + vq / n_q + vr / n_q),
        "phi": vp / (vp + (vs + vq) / n_q + vr / n_q),
    }


def summ(x: np.ndarray) -> tuple[float, float, float]:
    """mean, lo, hi over an array (nan-safe)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    return float(np.mean(x)), float(np.percentile(x, _LO_Q)), float(np.percentile(x, _HI_Q))


# ──────────────────────────────────────────────────────────────────────────
# Classification (plan §4.4)
# ──────────────────────────────────────────────────────────────────────────
def _base_class(vpc_person: float, erho2: float) -> str:
    if vpc_person >= TRUST_VPC and erho2 >= TRUST_ERHO2:
        return "trust"
    if vpc_person < MARGINAL_VPC_LO:
        return "do-not-trust"
    return "marginal"


def classify(vpc_person_draws: np.ndarray, erho2_mean: float) -> dict:
    vp_mean, vp_lo, vp_hi = summ(vpc_person_draws)
    cls = _base_class(vp_mean, erho2_mean)
    # uncertain if the VPC_person CI straddles a class boundary → prefer conservative
    crosses = (vp_lo < MARGINAL_VPC_LO <= vp_hi) or (vp_lo < TRUST_VPC <= vp_hi)
    if crosses:
        cls = _base_class(vp_lo, erho2_mean)  # lower bound = more conservative
    return {"class": cls, "uncertain": bool(crosses),
            "vpc_person_mean": vp_mean, "vpc_person_lo": vp_lo, "vpc_person_hi": vp_hi}


# ──────────────────────────────────────────────────────────────────────────
# Per-target driver
# ──────────────────────────────────────────────────────────────────────────
def gender_effect(idata) -> dict | None:
    """Posterior summary of the gender fixed effect (None if absent)."""
    post = idata.posterior
    name = next((v for v in post.data_vars if v == "gender" or v.startswith("gender")), None)
    if name is None:
        return None
    arr = post[name].values.reshape(-1)
    m, lo, hi = summ(arr)
    return {"coef": m, "lo": lo, "hi": hi}


def fit_target(df: pd.DataFrame, target: str, robust: bool, n_q_list: list[int],
               n_q_head: int, multisession_users: list[str],
               draws: int, tune: int, chains: int, seed: int,
               save_idata: bool) -> dict:
    """Run the model menu for one target; return rows + classification + idata to save."""
    est = "gaussian"  # headline estimator (plan decision 2026-06-11)
    t0 = time.time()
    print(f"  · {target}  (headline=gaussian{', +robust range' if robust else ''})", flush=True)

    fits: dict[str, tuple] = {}  # model_key -> (idata, info)
    # HEADLINE = Gaussian fits (decomposition, gender, duration, multi-session all Gaussian)
    fits["m0"] = fit_model(df, target, "1", "gaussian", draws, tune, chains, seed)
    fits["m1"] = fit_model(df, target, "1 + gender", "gaussian", draws, tune, chains, seed)
    fits["m1d"] = fit_model(df, target, "1 + gender + duration", "gaussian", draws, tune, chains, seed)
    # multi-session sensitivity (plan §8.C.3): drop the 8 genuine multi-session user_no
    df_f = df[~df["user_no"].isin(multisession_users)]
    fits["m0_filt"] = fit_model(df_f, target, "1", "gaussian", draws, tune, chains, seed)
    # ROBUSTNESS RANGE: Student-t unconditional, only for leptokurtic targets
    if robust:
        fits["m0_robust"] = fit_model(df, target, "1", "t", draws, tune, chains, seed)

    rows: list[dict] = []
    gender_rows: list[dict] = []
    for key, (idata, info) in fits.items():
        comp = extract_components(idata, info["rung"])
        # residual definitions to tabulate: primary = scale; add marginal for t-models
        defs = ["scale"] + (["marginal"] if info["family"] == "t" else [])
        for rdef in defs:
            for n_q in n_q_list:
                m = metrics(comp, rdef, n_q)
                vp = summ(m["vpc_person"]); er = summ(m["erho2"]); ph = summ(m["phi"])
                rows.append({
                    "target": target, "model": key, "estimator": info["family"],
                    "residual_def": rdef, "rung": info["rung"], "n_q": n_q,
                    "n_obs": int(idata.observed_data[list(idata.observed_data.data_vars)[0]].size),
                    "sigma2_person": float(np.mean(comp["v_person"])),
                    "sigma2_set": float(np.mean(comp["v_set"])),
                    "sigma2_question": float(np.mean(comp["v_question"])),
                    "sigma2_resid": float(np.nanmean(residual_variance(comp, rdef))),
                    "nu_mean": float(np.mean(comp["nu"])) if comp["nu"] is not None else None,
                    "vpc_person": vp[0], "vpc_person_lo": vp[1], "vpc_person_hi": vp[2],
                    "vpc_set": float(np.mean(m["vpc_set"])),
                    "vpc_question": float(np.mean(m["vpc_question"])),
                    "vpc_resid": float(np.nanmean(m["vpc_resid"])),
                    "Erho2": er[0], "Erho2_lo": er[1], "Erho2_hi": er[2],
                    "Phi": ph[0], "Phi_lo": ph[1], "Phi_hi": ph[2],
                    "rhat_max": info["rhat_max"], "divergences": info["divergences"],
                    "converged": info["converged"],
                })
        ge = gender_effect(idata)
        if ge is not None:
            gender_rows.append({"target": target, "model": key, "estimator": info["family"],
                                "gender_coef": ge["coef"], "ci_lo": ge["lo"], "ci_hi": ge["hi"]})

    # ── HEADLINE classification = Gaussian m0, headline n_q ──
    comp0 = extract_components(fits["m0"][0], fits["m0"][1]["rung"])
    m_head = metrics(comp0, "scale", n_q_head)  # Gaussian: scale == empirical residual variance
    cl = classify(m_head["vpc_person"], float(np.mean(m_head["erho2"])))
    erho2_head = summ(m_head["erho2"])
    phi_head = summ(m_head["phi"])

    # ── robustness range: Student-t m0 summarised scale (optimistic) & marginal (pessimistic) ──
    robust_scale_vpc = robust_marg_vpc = None
    if "m0_robust" in fits:
        comp_r = extract_components(fits["m0_robust"][0], fits["m0_robust"][1]["rung"])
        robust_scale_vpc = float(np.mean(metrics(comp_r, "scale", n_q_head)["vpc_person"]))
        robust_marg_vpc = float(np.nanmean(metrics(comp_r, "marginal", n_q_head)["vpc_person"]))
    robust_lo = min(v for v in (robust_scale_vpc, robust_marg_vpc) if v is not None) \
        if "m0_robust" in fits else None
    robust_hi = max(v for v in (robust_scale_vpc, robust_marg_vpc) if v is not None) \
        if "m0_robust" in fits else None

    # duration sensitivity (Gaussian): ΔVPC_person  m1 → m1d  (headline n_q)
    vpc_m1 = float(np.mean(metrics(
        extract_components(fits["m1"][0], fits["m1"][1]["rung"]), "scale", n_q_head)["vpc_person"]))
    vpc_m1d = float(np.mean(metrics(
        extract_components(fits["m1d"][0], fits["m1d"][1]["rung"]), "scale", n_q_head)["vpc_person"]))
    # multi-session sensitivity (Gaussian): ΔVPC_person  full → filtered
    vpc_filt = float(np.mean(metrics(
        extract_components(fits["m0_filt"][0], fits["m0_filt"][1]["rung"]),
        "scale", n_q_head)["vpc_person"]))

    block = "personality" if target in PERSONALITY_BLOCK else "performance"
    rng = f"; robust range [{robust_lo:.2f},{robust_hi:.2f}]" if robust_lo is not None else ""
    rationale = (f"VPC_person={cl['vpc_person_mean']:.2f} "
                 f"[{cl['vpc_person_lo']:.2f},{cl['vpc_person_hi']:.2f}], "
                 f"Eρ²={erho2_head[0]:.2f}; estimator=gaussian, rung={fits['m0'][1]['rung']}{rng}"
                 + ("; CI crosses a class boundary → conservative class" if cl["uncertain"] else "")
                 + ("; near-normal, interpret separately" if not robust else ""))

    classification = {
        "target": target, "class": cl["class"], "uncertain": cl["uncertain"],
        "estimator": est, "rung": fits["m0"][1]["rung"], "block": block,
        "neuroticism_flag": (target == "neuroticism"),
        "VPC_person": cl["vpc_person_mean"],
        "VPC_person_lo": cl["vpc_person_lo"], "VPC_person_hi": cl["vpc_person_hi"],
        "Erho2": erho2_head[0], "Erho2_lo": erho2_head[1], "Erho2_hi": erho2_head[2],
        "Phi": phi_head[0],
        "robust_scale_vpc": robust_scale_vpc, "robust_marginal_vpc": robust_marg_vpc,
        "robust_range_lo": robust_lo, "robust_range_hi": robust_hi,
        "dur_sens_vpc_no_dur": vpc_m1, "dur_sens_vpc_with_dur": vpc_m1d,
        "dur_sens_delta": vpc_m1d - vpc_m1,
        "multisession_vpc_full": cl["vpc_person_mean"], "multisession_vpc_filtered": vpc_filt,
        "multisession_delta": vpc_filt - cl["vpc_person_mean"],
        "rationale": rationale,
    }

    if save_idata:
        IDATA_DIR.mkdir(parents=True, exist_ok=True)
        for key, (idata, _info) in fits.items():
            idata.to_netcdf(str(IDATA_DIR / f"{target}_{key}.nc"))

    print(f"    ✓ {target}: {cl['class']}"
          f"{' (uncertain)' if cl['uncertain'] else ''}  "
          f"VPC_p={cl['vpc_person_mean']:.2f}  Eρ²={erho2_head[0]:.2f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return {"variance_rows": rows, "gender_rows": gender_rows,
            "classification": classification,
            "idata_m0": fits["m0"][0]}


# ──────────────────────────────────────────────────────────────────────────
# Plots (plan §5.5)
# ──────────────────────────────────────────────────────────────────────────
def plot_vpc_bars(class_rows: list[dict], var_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # primary m0, scale-based, headline n_q (median of the available n_q values)
    nq_vals = sorted({r["n_q"] for r in var_rows})
    nq_head = nq_vals[len(nq_vals) // 2]
    rows = [r for r in var_rows if r["model"] == "m0" and r["residual_def"] == "scale"
            and r["n_q"] == nq_head]
    rows = sorted(rows, key=lambda r: TARGETS.index(r["target"]))
    labels = [r["target"] for r in rows]
    person = np.array([r["vpc_person"] for r in rows])
    sset = np.array([r["vpc_set"] for r in rows])
    ques = np.array([r["vpc_question"] for r in rows])
    resid = np.array([r["vpc_resid"] for r in rows])

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    colors = {"person": "#2c7fb8", "set": "#7fcdbb", "question": "#c7e9b4", "resid": "#d9d9d9"}
    ax.bar(x, person, label="person (signal)", color=colors["person"])
    ax.bar(x, sset, bottom=person, label="set (cohort)", color=colors["set"])
    ax.bar(x, ques, bottom=person + sset, label="question", color=colors["question"])
    ax.bar(x, resid, bottom=person + sset + ques, label="residual (noise ceiling)",
           color=colors["resid"])
    ax.axhline(TRUST_VPC, ls="--", lw=1, color="k", alpha=.5)
    ax.text(len(labels) - .4, TRUST_VPC + .01, "VPC_person trust=.50", ha="right", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("variance partition (primary model, scale-based)")
    ax.set_ylim(0, 1)
    ax.set_title("Stage 1 — variance decomposition per target")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "vpc_stacked_bars.png", dpi=150)
    plt.close(fig)


def plot_caterpillar(target: str, idata) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if RE_PERSON not in idata.posterior:
        return
    eff = idata.posterior[RE_PERSON]
    dim = [d for d in eff.dims if d not in ("chain", "draw")][0]
    arr = eff.stack(s=("chain", "draw")).values  # (n_users, n_samples)
    mean = arr.mean(axis=1)
    lo = np.percentile(arr, _LO_Q, axis=1)
    hi = np.percentile(arr, _HI_Q, axis=1)
    order = np.argsort(mean)
    y = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.hlines(y, lo[order], hi[order], color="#2c7fb8", alpha=.35, lw=.6)
    ax.plot(mean[order], y, ".", ms=2, color="#08519c")
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("person random intercept (deviation)")
    ax.set_ylabel("participants (sorted)")
    ax.set_title(f"Caterpillar — person effects: {target}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"caterpillar_{target}.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Output assembly
# ──────────────────────────────────────────────────────────────────────────
CAVEATS = [
    "single-obs-per-cell: residual conflates person×question, rater, occasion, error "
    "→ noise is an UPPER bound, person-signal a LOWER bound (no rater facet observable).",
    "No inter-rater reliability, rater bias, or rater×gender interaction is estimable at Stage 1.",
    "Author-reported MNL stability (ρ=0.905±0.04) is label-solution stability, not IRR.",
    "gender may be confounded with set_id; interpret its fixed effect with the cohort flag.",
    "Centered z-like scores have no natural zero; Φ (absolute) and Eρ² (relative) answer "
    "different questions — both reported.",
    "HEADLINE class = Gaussian variance partition (leptokurtic spread counts as noise). "
    "A Student-t robustness range [scale σ², marginal σ²·ν/(ν−2)] brackets person-signal "
    "sensitivity to tail treatment; σ²_person is ~constant across all three, the residual "
    "denominator spans ~3× — so VPC_person is itself uncertain, not just its CI.",
]


def write_outputs(results: list[dict], diag: dict, meta: dict, n_q_head: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    var_rows = [r for res in results for r in res["variance_rows"]]
    gender_rows = [r for res in results for r in res["gender_rows"]]
    class_rows = [res["classification"] for res in results]

    pd.DataFrame(var_rows).to_csv(OUT_DIR / "variance_components.csv", index=False)
    pd.DataFrame(gender_rows).to_csv(OUT_DIR / "gender_effects.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "target_classification.csv", index=False)

    handoff = {
        "n_q_median": diag["diagnostics"]["n_q_median"],
        "n_q_headline": n_q_head,
        "gender_n": {"person": [diag["diagnostics"]["gender_n"]["person"]["male_1"],
                               diag["diagnostics"]["gender_n"]["person"]["female_0"]],
                     "clip": [diag["diagnostics"]["gender_n"]["clip"]["male_1"],
                              diag["diagnostics"]["gender_n"]["clip"]["female_0"]]},
        "gender_set_cramers_v":
            diag["diagnostics"]["cohort_confound"]["gender_x_set_person_level"]["cramers_v"],
        "robust_required": diag["diagnostics"]["robust_required"],
        "headline_convention": "Gaussian variance partition; robust [scale,marginal] range as sensitivity",
        "targets": {c["target"]: {"class": c["class"], "uncertain": c["uncertain"],
                                  "VPC_person": c["VPC_person"], "Erho2": c["Erho2"],
                                  "Phi": c["Phi"],
                                  "robust_range": [c["robust_range_lo"], c["robust_range_hi"]],
                                  "rung": c["rung"], "estimator": c["estimator"],
                                  "block": c["block"]}
                    for c in class_rows},
        "block_summary": _block_summary(class_rows),
        "caveats": CAVEATS,
    }
    (OUT_DIR / "stage1_handoff.json").write_text(
        json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "fit_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # plots
    plot_vpc_bars(class_rows, var_rows)
    for res in results:
        if res["classification"]["target"] in {"overall_performance", "openness", "neuroticism"}:
            plot_caterpillar(res["classification"]["target"], res["idata_m0"])


def _block_summary(class_rows: list[dict]) -> dict:
    out = {}
    for blk in ("personality", "performance"):
        rows = [c for c in class_rows if c["block"] == blk]
        out[blk] = {
            "n": len(rows),
            "n_trust": sum(c["class"] == "trust" for c in rows),
            "n_marginal": sum(c["class"] == "marginal" for c in rows),
            "n_do_not_trust": sum(c["class"] == "do-not-trust" for c in rows),
            "mean_VPC_person": float(np.mean([c["VPC_person"] for c in rows])) if rows else None,
            "mean_Erho2": float(np.mean([c["Erho2"] for c in rows])) if rows else None,
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 Module 2 — mixed-model variance decomposition")
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: 1 target, reduced draws, writes all outputs")
    ap.add_argument("--targets", nargs="+", default=None, help="subset of targets to fit")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-save-idata", action="store_true", help="skip persisting netCDF")
    args = ap.parse_args()

    if not DATA_FILE.exists() or not DIAG_FILE.exists():
        raise SystemExit("Run src/stage1_diagnostics.py first (need recruitview_with_set.csv "
                         "+ diagnostics.json).")

    df = load_data()
    diag = load_diagnostics()
    robust_by_target = {t: diag["diagnostics"]["distributions"][t]["robust"] for t in TARGETS}
    n_q_median = diag["diagnostics"]["n_q_median"]
    n_q_max = diag["diagnostics"]["clips_per_user"]["max"]
    n_q_list = sorted({4, n_q_median, n_q_max})
    n_q_head = n_q_median
    multisession = [str(u) for u in diag["set_reconstruction"]["genuine_multisession_users"]]

    targets = args.targets or TARGETS
    if args.quick:
        targets = targets[:1] if args.targets else ["overall_performance"]
        args.draws, args.tune, args.chains = 300, 300, 2

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed, "draws": args.draws, "tune": args.tune, "chains": args.chains,
        "n_q_list": n_q_list, "n_q_headline": n_q_head,
        "residual_convention": "scale-based primary; marginal t-variance sensitivity",
        "versions": {"bambi": bmb.__version__, "arviz": az.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "python": sys.version.split()[0]},
    }
    print(f"Stage 1 fit — {len(targets)} target(s), {args.chains}ch × "
          f"{args.draws}+{args.tune}, n_q={n_q_list} (headline {n_q_head})")
    print(f"  multi-session users excluded in sensitivity: {multisession}")

    t_start = time.time()
    results = []
    for t in targets:
        results.append(fit_target(
            df, t, robust_by_target[t], n_q_list, n_q_head, multisession,
            args.draws, args.tune, args.chains, args.seed,
            save_idata=not args.no_save_idata))

    write_outputs(results, diag, meta, n_q_head)
    print(f"\n✓ Stage 1 fit done in {time.time()-t_start:.0f}s → {OUT_DIR}")
    print("  classes:", {res["classification"]["target"]: res["classification"]["class"]
                         for res in results})


if __name__ == "__main__":
    main()
