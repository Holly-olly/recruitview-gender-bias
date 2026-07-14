#!/usr/bin/env python3
"""Stage 1 — build the canonical ANALYSIS SAMPLE that every stage reads.

recruitview_full.csv → drop retakes (dup_keep=False) AND answers ≤5 tokens
("no content to evaluate", 2026-07-12) → data/recruitview_analysis.csv.
Adds n_tokens / n_types (answer length). All downstream stages filter to this file.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/recruitview_analysis.csv"


def main() -> None:
    full = pd.read_csv(ROOT / "data/recruitview_full.csv")
    rm = pd.read_csv(ROOT / "experiments/length_confound/results/response_metrics.csv")[["id", "n_tokens", "n_types"]]
    dedup = full[full.dup_keep.astype(str).str.lower().isin(["true", "1"])]
    d = dedup.merge(rm, on="id")
    clean = d[d.n_tokens > 5].copy()
    clean.to_csv(OUT, index=False)
    print(f"raw {len(full)} → dedup {len(dedup)} (-{len(full)-len(dedup)} retakes) "
          f"→ analysis {len(clean)} (-{len(d)-len(clean)} ≤5-token)")
    print(f"analysis sample: {len(clean)} responses, {clean.user_no.nunique()} persons, "
          f"{clean.question_id.nunique()} questions")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
