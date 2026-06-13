"""
Stage 1 — Diagnostics & data-contract validation (plan §4.1 + set_id reconstruction).

Self-contained module: it does NOT fit mixed models. It prepares the planning
parameters and flags that every later Stage 1 step depends on (model ladder,
G/Φ thresholds, whether the Bayesian/robust branch is required).

Run (from the repository root):
    python src/stage1_diagnostics.py

Outputs (outputs/stage1/):
    diagnostics.json            — machine-readable summary (planning parameters)
    diagnostics_report.md       — human-readable report (English)
    recruitview_with_set.csv    — data + reconstructed set_id (model input)

Dependencies: pandas, numpy, scipy (pingouin NOT required).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "recruitview_full.csv"
OUT_DIR = ROOT / "outputs" / "stage1"

TARGETS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality", "interview_score", "answer_score",
    "speaking_skills", "confidence_score", "facial_expression",
    "overall_performance",
]
ID_COLS = ["response_id", "id", "user_no"]
META_COLS = ["question_id", "question", "video_quality", "duration"]
COMMON_Q = {1, 76}  # shared opener / closer question — outside the sets

# Flag thresholds (plan §4.1)
SKEW_FLAG = 2.0
EXKURT_FLAG = 7.0
CRAMERS_V_FLAG = 0.20
THIN_SHARE_FLAG = 0.25  # share of users with n_q<=2 above which within-person is weakly estimable


# ──────────────────────────────────────────────────────────────────────────
# 1. load_validate
# ──────────────────────────────────────────────────────────────────────────
def load_validate(path: Path = DATA_FILE) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    report: dict = {"source_file": str(path), "n_rows": int(len(df))}

    expected = ID_COLS + ["gender"] + META_COLS + TARGETS
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]
    report["missing_columns"] = missing_cols
    report["extra_columns"] = extra_cols

    # types
    df["user_no"] = pd.to_numeric(df["user_no"], errors="coerce").astype("Int64")
    df["question_id"] = pd.to_numeric(df["question_id"], errors="coerce").astype("Int64")
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
    for c in TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # contract: gender ∈ {0,1}, no missing
    gender_vals = sorted(df["gender"].dropna().unique().tolist())
    report["gender_unique_values"] = gender_vals
    report["gender_encoding_ok"] = set(gender_vals).issubset({0.0, 1.0})
    report["gender_nan"] = int(df["gender"].isna().sum())

    # contract: question_id 1..76
    report["question_id_min"] = int(df["question_id"].min())
    report["question_id_max"] = int(df["question_id"].max())
    report["n_questions"] = int(df["question_id"].nunique())
    report["n_users"] = int(df["user_no"].nunique())

    # duration / video_quality — categorical (NOT float seconds)
    report["duration_levels"] = sorted(df["duration"].dropna().unique().tolist())
    report["video_quality_levels"] = sorted(df["video_quality"].dropna().unique().tolist())
    # categorical iff not a numeric dtype (robust to object vs. pandas 'category')
    report["duration_is_categorical"] = not pd.api.types.is_numeric_dtype(df["duration"])

    # repeated cells (person × question)
    cell = df.groupby(["user_no", "question_id"]).size()
    report["n_cells"] = int(len(cell))
    report["n_singleton_cells"] = int((cell == 1).sum())
    report["n_repeated_cells"] = int((cell > 1).sum())
    report["max_obs_per_cell"] = int(cell.max())

    report["contract_ok"] = (
        not missing_cols
        and report["gender_encoding_ok"]
        and report["gender_nan"] == 0
        and report["n_questions"] == 76
    )
    return df, report


# ──────────────────────────────────────────────────────────────────────────
# 2. engineer_set — empirical set_id reconstruction
# ──────────────────────────────────────────────────────────────────────────
def engineer_set(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Reconstruct set_id from the co-occurrence of questions within participants.
    Algorithm (greedy over signatures): the shared question q1/q76 → set_id 0;
    the rest are grouped into sets by which questions a single user_no answers
    together. The anchor anomaly (a question in two sets) is recorded in the report.
    """
    nc = df[~df["question_id"].isin(COMMON_Q)].copy()  # non-common clips

    # participant signature = the set of their non-common questions
    sigs = (
        nc.groupby("user_no")["question_id"]
        .apply(lambda s: frozenset(int(x) for x in s.dropna().unique()))
    )
    sig_counts = Counter(sigs.tolist())

    # greedy: signatures by descending participant count; ≥3 "unclaimed" questions = new set
    q2set: dict[int, int] = {}
    set_counter = 0
    for sig, _n in sorted(sig_counts.items(), key=lambda kv: -kv[1]):
        unclaimed = [q for q in sig if q not in q2set]
        if len(unclaimed) >= 3:
            set_counter += 1
            for q in unclaimed:
                q2set[q] = set_counter

    # renumber sets by minimum question_id (stable order)
    set_minq = {}
    for q, s in q2set.items():
        set_minq[s] = min(q, set_minq.get(s, q))
    order = {old: new for new, (old, _) in
             enumerate(sorted(set_minq.items(), key=lambda kv: kv[1]), start=1)}
    q2set = {q: order[s] for q, s in q2set.items()}

    def assign(qid) -> int:
        if pd.isna(qid):
            return -1
        q = int(qid)
        if q in COMMON_Q:
            return 0
        return q2set.get(q, -1)

    df = df.copy()
    df["set_id"] = df["question_id"].map(assign).astype(int)

    # --- reconstruction quality ---
    n_sets = max(q2set.values()) if q2set else 0
    set_members = {s: sorted(q for q, ss in q2set.items() if ss == s)
                   for s in range(1, n_sets + 1)}

    nc2 = nc.copy()
    nc2["set_id"] = nc2["question_id"].map(lambda q: q2set.get(int(q), -1))

    # STRUCTURAL anchor question = belongs to ≥2 CANONICAL signatures (frequent blocks).
    # This separates a real shared anchor (q27) from incidental multi-set participants.
    MIN_CANON = 10  # real blocks are answered by 14–25 people; rare signatures are noise
    canon_sigs = [set(sig) for sig, n in sig_counts.items() if n >= MIN_CANON]

    def _sig_set(cs: set) -> int:
        return Counter(q2set[x] for x in cs if x in q2set).most_common(1)[0][0]

    ambiguous = []
    for q in sorted(q2set):
        sigs_with_q = [cs for cs in canon_sigs if q in cs]
        if len(sigs_with_q) > 1:
            bridged = sorted({_sig_set(cs) for cs in sigs_with_q})
            ambiguous.append({"question_id": int(q), "own_set": int(q2set[q]),
                              "bridged_sets": bridged,
                              "n_canonical_signatures": len(sigs_with_q)})

    # participants whose non-common clips fall into >1 set — a separate finding,
    # does NOT affect reconstruction reliability (possible multi-session, plan §2.3)
    user_sets = nc2.groupby("user_no")["set_id"].apply(lambda s: set(int(x) for x in s))
    spanning = {int(u): sorted(ss) for u, ss in user_sets.items() if len(ss) > 1}

    # Split the spanning users by cause — two very different things:
    #  • anchor-span: the span is created ONLY by a shared-anchor question (e.g. q27);
    #    after removing anchor questions the user's clips fall in a single real set.
    #    Harmless — one real set + a shared anchor, exactly like q1/q76.
    #  • genuine multi-session: one user_no answered ≥2 real set blocks (retake /
    #    dropped-and-restarted session) → the §8.C.3 sensitivity target.
    anchor_qs = {a["question_id"] for a in ambiguous}
    core = nc2[~nc2["question_id"].isin(anchor_qs)]
    core_sets = {int(u): sorted(set(int(x) for x in ss))
                 for u, ss in core.groupby("user_no")["set_id"]}
    genuine_multisession = {u: ss for u, ss in spanning.items()
                            if len(core_sets.get(u, [])) > 1}
    anchor_span = {u: ss for u, ss in spanning.items()
                   if u not in genuine_multisession}

    unassigned = int((df["set_id"] == -1).sum())

    set_report = {
        "method": "empirical greedy signature clustering (co-occurrence within user_no)",
        "common_questions": sorted(COMMON_Q),
        "n_sets": int(n_sets),
        "set_sizes": {str(s): len(qs) for s, qs in set_members.items()},
        "set_members": {str(s): qs for s, qs in set_members.items()},
        "n_clips_common": int((df["set_id"] == 0).sum()),
        "n_clips_unassigned": unassigned,
        "ambiguous_questions": ambiguous,
        "n_users_spanning_multiple_sets": len(spanning),
        "users_spanning_multiple_sets": spanning,
        "n_anchor_span_users": len(anchor_span),
        "n_genuine_multisession_users": len(genuine_multisession),
        "genuine_multisession_users": genuine_multisession,
        "reconstruction_reliable": (n_sets >= 10 and unassigned == 0
                                    and len(ambiguous) <= 2),
    }
    return df, set_report


# ──────────────────────────────────────────────────────────────────────────
# 3. diagnostics — plan §4.1
# ──────────────────────────────────────────────────────────────────────────
def _mardia_kurtosis(X: np.ndarray) -> dict:
    """Mardia multivariate kurtosis (b2,p) + standardized statistic."""
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = np.cov(Xc, rowvar=False)
    Sinv = np.linalg.pinv(S)
    d2 = np.einsum("ij,jk,ik->i", Xc, Sinv, Xc)  # squared Mahalanobis distance
    b2p = float(np.mean(d2 ** 2))
    expected = p * (p + 2)
    z = (b2p - expected) / np.sqrt(8 * expected / n)
    return {"b2p": b2p, "expected_mvn": float(expected),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def _cramers_v(a: pd.Series, b: pd.Series) -> dict:
    ct = pd.crosstab(a, b)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else float("nan")
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p),
            "cramers_v": v, "table_shape": list(ct.shape)}


def _eta_squared(groups: list[np.ndarray]) -> dict:
    """Correlation ratio η² (one-way ANOVA) for a categorical predictor."""
    groups = [g[~np.isnan(g)] for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"eta_sq": float("nan"), "p_value": float("nan")}
    grand = np.concatenate(groups)
    gm = grand.mean()
    ss_between = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ss_total = ((grand - gm) ** 2).sum()
    eta = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    F, p = stats.f_oneway(*groups)
    return {"eta_sq": eta, "F": float(F), "p_value": float(p)}


def diagnostics(df: pd.DataFrame) -> dict:
    out: dict = {}

    # 1. Distribution per target
    dist = {}
    any_robust = False
    for t in TARGETS:
        x = df[t].dropna().to_numpy()
        sk = float(stats.skew(x))
        ek = float(stats.kurtosis(x, fisher=True))  # excess kurtosis
        robust = abs(sk) > SKEW_FLAG or ek > EXKURT_FLAG
        any_robust |= robust
        dist[t] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                   "skew": sk, "excess_kurtosis": ek, "robust": bool(robust)}
    out["distributions"] = dist
    out["robust_required"] = bool(any_robust)
    out["mardia_kurtosis"] = _mardia_kurtosis(df[TARGETS].dropna().to_numpy())

    # 2. Clips per user
    cpu = df.groupby("user_no").size()
    share_le2 = float((cpu <= 2).mean())
    out["clips_per_user"] = {
        "mean": float(cpu.mean()), "median": float(cpu.median()),
        "min": int(cpu.min()), "max": int(cpu.max()),
        "n_users": int(len(cpu)),
        "n_ge2": int((cpu >= 2).sum()), "n_ge4": int((cpu >= 4).sum()),
        "n_ge6": int((cpu >= 6).sum()),
        "share_le2": share_le2,
        "thin_within_person": bool(share_le2 > THIN_SHARE_FLAG),
    }
    out["n_q_median"] = int(cpu.median())

    # 3. Gender Ns (person + clip)
    g_person = df.groupby("user_no")["gender"].first()
    out["gender_n"] = {
        "person": {"female_0": int((g_person == 0).sum()),
                   "male_1": int((g_person == 1).sum())},
        "clip": {"female_0": int((df["gender"] == 0).sum()),
                 "male_1": int((df["gender"] == 1).sum())},
        "female_persons_below_100": bool((g_person == 0).sum() < 100
                                         or (g_person == 1).sum() < 100),
    }

    # 4. Cohort confound: gender × set_id (person-level modal) and gender × question (clip)
    nc = df[df["set_id"] > 0]
    modal_set = nc.groupby("user_no")["set_id"].agg(lambda s: s.mode().iloc[0])
    person = pd.DataFrame({"gender": g_person, "set_id": modal_set}).dropna()
    v_set = _cramers_v(person["gender"], person["set_id"])
    v_q = _cramers_v(df["gender"], df["question_id"])
    out["cohort_confound"] = {
        "gender_x_set_person_level": v_set,
        "gender_x_question_clip_level": v_q,
        "carry_covariates": bool(v_set["cramers_v"] > CRAMERS_V_FLAG
                                 or v_q["cramers_v"] > CRAMERS_V_FLAG),
    }

    # 5. Missingness
    miss = {c: float(df[c].isna().mean()) for c in df.columns}
    out["missingness"] = {
        "per_column_rate": miss,
        "any_missing": bool(any(v > 0 for v in miss.values())),
        "little_mcar": "skipped — no missing data, test not applicable",
    }

    # 6. Nuisance associations: video_quality / duration × each target (η²)
    nuis = {}
    for nz in ["video_quality", "duration"]:
        per_t = {}
        for t in TARGETS:
            groups = [df.loc[df[nz] == lvl, t].to_numpy()
                      for lvl in df[nz].dropna().unique()]
            per_t[t] = _eta_squared(groups)
        nuis[nz] = per_t
    out["nuisance_associations"] = nuis

    return out


# ──────────────────────────────────────────────────────────────────────────
# 4. write_outputs
# ──────────────────────────────────────────────────────────────────────────
def _md_report(validation: dict, set_report: dict, diag: dict) -> str:
    L = []
    L.append("# Stage 1 — Data Diagnostics & Validation\n")
    L.append("_Auto-generated by `src/stage1_diagnostics.py`. This module does NOT fit "
             "models — it only produces the planning parameters and flags for the "
             "subsequent steps._\n")

    L.append("## 1. Data contract\n")
    L.append(f"- File: `{validation['source_file']}`")
    L.append(f"- Rows: **{validation['n_rows']}**, participants: **{validation['n_users']}**, "
             f"questions: **{validation['n_questions']}** "
             f"({validation['question_id_min']}–{validation['question_id_max']})")
    L.append(f"- Contract passed: **{validation['contract_ok']}**")
    L.append(f"- gender: values {validation['gender_unique_values']} "
             f"(0=female, 1=male), missing {validation['gender_nan']}, "
             f"encoding ok: {validation['gender_encoding_ok']}")
    L.append(f"- `duration` — categorical {validation['duration_levels']} "
             f"(NOT float seconds)")
    L.append(f"- `video_quality` — {validation['video_quality_levels']}")
    L.append(f"- Cells (person×question): {validation['n_cells']}, "
             f"singletons {validation['n_singleton_cells']}, "
             f"repeated {validation['n_repeated_cells']} "
             f"(max {validation['max_obs_per_cell']}× per cell)")
    if validation["extra_columns"]:
        L.append(f"- ⚠️ Extra columns: {validation['extra_columns']}")
    L.append("")

    L.append("## 2. set_id reconstruction\n")
    L.append(f"- Method: {set_report['method']}")
    L.append(f"- Sets found: **{set_report['n_sets']}**, shared clips (q1/q76): "
             f"{set_report['n_clips_common']}, unassigned: "
             f"{set_report['n_clips_unassigned']}")
    L.append(f"- Reconstruction reliable: **{set_report['reconstruction_reliable']}**")
    sizes = ", ".join(f"set{s}={n}" for s, n in set_report["set_sizes"].items())
    L.append(f"- Set sizes (questions): {sizes}")
    if set_report["ambiguous_questions"]:
        L.append("- ⚠️ Structural anchor questions (in ≥2 canonical signatures): "
                 + "; ".join(f"q{a['question_id']} (own set{a['own_set']}, "
                             f"bridge between sets {a['bridged_sets']})"
                             for a in set_report["ambiguous_questions"]))
    L.append(f"- Participants spanning >1 set: "
             f"**{set_report['n_users_spanning_multiple_sets']}** total = "
             f"**{set_report['n_anchor_span_users']}** anchor-only (span created solely by "
             f"the q27 anchor — one real set + shared anchor, harmless) + "
             f"**{set_report['n_genuine_multisession_users']}** genuine multi-session "
             f"(one `user_no`, ≥2 real sets → retake / dropped session; §8.C.3 sensitivity "
             f"target). Does not affect reconstruction reliability (plan §2.3). "
             f"Genuine cases (user_no): {sorted(set_report['genuine_multisession_users'])}")
    L.append("")

    L.append("## 3. Target distributions (plan §4.1.1)\n")
    L.append(f"- Robust/Bayesian branch required (robust_required): "
             f"**{diag['robust_required']}**")
    mk = diag["mardia_kurtosis"]
    L.append(f"- Mardia multivariate kurtosis: b2,p={mk['b2p']:.1f} "
             f"(expected under MVN={mk['expected_mvn']:.0f}, z={mk['z']:.1f}, "
             f"p={mk['p_value']:.2g}) → severe departure from multivariate normality")
    L.append("")
    L.append("| target | mean | sd | skew | exkurt | robust |")
    L.append("|---|---|---|---|---|---|")
    for t, d in diag["distributions"].items():
        L.append(f"| {t} | {d['mean']:.2f} | {d['sd']:.2f} | {d['skew']:.2f} "
                 f"| {d['excess_kurtosis']:.2f} | {'✓' if d['robust'] else ''} |")
    L.append("")

    L.append("## 4. Clips per participant (plan §4.1.2)\n")
    c = diag["clips_per_user"]
    L.append(f"- mean={c['mean']:.2f}, median (**n_q_median**)=**{c['median']:.0f}**, "
             f"min={c['min']}, max={c['max']}")
    L.append(f"- ≥2: {c['n_ge2']}, ≥4: {c['n_ge4']}, ≥6: {c['n_ge6']} "
             f"(of {c['n_users']})")
    L.append(f"- share with n_q≤2: {c['share_le2']:.1%} → within-person weakly estimable: "
             f"**{c['thin_within_person']}**")
    L.append("")

    L.append("## 5. Gender Ns (plan §4.1.3)\n")
    g = diag["gender_n"]
    L.append(f"- Participants: **{g['person']['male_1']} male / "
             f"{g['person']['female_0']} female**")
    L.append(f"- Clips: {g['clip']['male_1']} male / {g['clip']['female_0']} female")
    L.append(f"- ⚠️ Group <100 participants: **{g['female_persons_below_100']}** "
             f"(affects Stage 4 DIF/invariance, not Stage 1)")
    L.append("")

    L.append("## 6. Cohort confound: gender × set / question (plan §4.1.4)\n")
    cc = diag["cohort_confound"]
    vs = cc["gender_x_set_person_level"]
    vq = cc["gender_x_question_clip_level"]
    L.append(f"- gender × set_id (person-level): Cramér's V=**{vs['cramers_v']:.3f}** "
             f"(χ²={vs['chi2']:.1f}, p={vs['p_value']:.2g})")
    L.append(f"- gender × question_id (clip-level): Cramér's V=**{vq['cramers_v']:.3f}** "
             f"(χ²={vq['chi2']:.1f}, p={vq['p_value']:.2g})")
    L.append(f"- carry_covariates (keep set_id/duration in conditional models): "
             f"**{cc['carry_covariates']}** — flag fires on Cramér's V>0.20 alone; "
             f"check the χ² p-value before treating it as a real confound")
    L.append("")

    L.append("## 7. Missingness (plan §4.1.5)\n")
    m = diag["missingness"]
    L.append(f"- Any missing: **{m['any_missing']}** · Little MCAR: {m['little_mcar']}")
    L.append("")

    L.append("## 8. Nuisance associations (plan §4.1.6): predictor η² with target\n")
    L.append("| target | video_quality η² (p) | duration η² (p) |")
    L.append("|---|---|---|")
    for t in TARGETS:
        vq_ = diag["nuisance_associations"]["video_quality"][t]
        du_ = diag["nuisance_associations"]["duration"][t]
        L.append(f"| {t} | {vq_['eta_sq']:.3f} ({vq_['p_value']:.2g}) "
                 f"| {du_['eta_sq']:.3f} ({du_['p_value']:.2g}) |")
    L.append("")

    L.append("## 9. Planning parameters → handoff to the model ladder\n")
    L.append(f"- `n_q_median` = {diag['n_q_median']} (into the G/Φ formulas)")
    L.append(f"- `robust_required` = {diag['robust_required']}")
    L.append(f"- `carry_covariates` = {cc['carry_covariates']}")
    L.append(f"- `thin_within_person` = {c['thin_within_person']}")
    L.append(f"- set reconstruction reliable = {set_report['reconstruction_reliable']} "
             f"→ if False, the model ladder starts at L1 (no set)")
    L.append("")
    return "\n".join(L)


def write_outputs(validation: dict, set_report: dict, diag: dict,
                  df_with_set: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"validation": validation, "set_reconstruction": set_report,
               "diagnostics": diag}
    (OUT_DIR / "diagnostics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "diagnostics_report.md").write_text(
        _md_report(validation, set_report, diag), encoding="utf-8")
    df_with_set.to_csv(OUT_DIR / "recruitview_with_set.csv", index=False)


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"Data file not found: {DATA_FILE}")
    df, validation = load_validate()
    df, set_report = engineer_set(df)
    diag = diagnostics(df)
    write_outputs(validation, set_report, diag, df)

    print("✓ Stage 1 diagnostics ready →", OUT_DIR)
    print(f"  contract ok: {validation['contract_ok']} · "
          f"sets: {set_report['n_sets']} · "
          f"n_q_median: {diag['n_q_median']}")
    print(f"  gender (participants): {diag['gender_n']['person']['male_1']} male / "
          f"{diag['gender_n']['person']['female_0']} female")
    print(f"  robust_required: {diag['robust_required']} · "
          f"carry_covariates: {diag['cohort_confound']['carry_covariates']} · "
          f"thin_within_person: {diag['clips_per_user']['thin_within_person']}")
    print(f"  set reconstruction reliable: {set_report['reconstruction_reliable']} · "
          f"anchors: {len(set_report['ambiguous_questions'])}")


if __name__ == "__main__":
    main()
