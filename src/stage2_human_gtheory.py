#!/usr/bin/env python3
"""Stage 2 (human) — classical G-theory reliability of the human ratings.

GeneralizIT, design `Person x question_id`, per scale, on the clean analysis sample
(data/recruitview_analysis.csv). Reports the variance components (sigma2_person,
sigma2_question, sigma2_resid), the share of true personality (VPC_person), the relative
Eρ²(6) and absolute Φ(6) reliabilities, Φ-ceiling, D-study n→.70/.80, and the Stage-1
classification. Φ(6) counts question difficulty (sigma2_question) as error, so Φ ≤ Eρ².
Writes outputs/stage2_human/human_reliability.csv (schema matches the notebook 2.2 cell).

Run:  .venv/bin/python src/stage2_human_gtheory.py
"""
import warnings, io, re, math, contextlib
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage2_human"
SCALES = ["openness", "conscientiousness", "extraversion", "agreeableness",
          "neuroticism", "answer_score", "interview_score"]


def components(d: pd.DataFrame, scale: str):
    from generalizit import GeneralizIT
    dd = d[["user_no", "question_id", scale]].dropna().rename(columns={"user_no": "Person"})
    g = GeneralizIT(data=dd, design_str="Person x question_id", response=scale)
    g.calculate_anova(); g.calculate_g_coefficients()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g.variance_summary()
    v = buf.getvalue()
    sp = float(re.search(r"(?m)^person\s+(-?\d+\.\d+)", v).group(1))
    sq = float(re.search(r"(?m)^question_id\s+(-?\d+\.\d+)", v).group(1))
    sr = float(re.search(r"(?m)^person x question_id\s+(-?\d+\.\d+)", v).group(1))
    return max(sp, 0.0), max(sq, 0.0), max(sr, 0.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data/recruitview_analysis.csv")
    rows = []
    for s in SCALES:
        sp, sq, sr = components(d, s)
        vpc = sp / (sp + sr) if (sp + sr) > 0 else float("nan")
        ero6 = sp / (sp + sr / 6) if (sp + sr / 6) > 0 else float("nan")
        phi6 = sp / (sp + (sq + sr) / 6) if (sp + (sq + sr) / 6) > 0 else float("nan")
        cls = "trust" if vpc >= 0.50 else ("marginal" if ero6 >= 0.70 else "do-not-trust")
        def nfor(rho): return (rho / (1 - rho)) * (sr / sp) if sp > 0 else float("inf")
        rows.append(dict(scale=s, sigma2_person=round(sp, 3), sigma2_question=round(sq, 3),
                         sigma2_resid=round(sr, 3), VPC_person=round(vpc, 3), Ero2_6=round(ero6, 3),
                         Phi_6=round(phi6, 3), ceiling=round(math.sqrt(ero6), 3), classification=cls,
                         n_for_rho070=round(nfor(0.70), 1), n_for_rho080=round(nfor(0.80), 1)))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "human_reliability.csv", index=False)
    print(f"Human G-theory reliability (clean analysis sample n={len(d)}):\n")
    print(tab.to_string(index=False))
    print(f"\nwrote {OUT/'human_reliability.csv'}")


if __name__ == "__main__":
    main()
