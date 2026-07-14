#!/usr/bin/env python
"""
Build a review artefact for DUPLICATE responses: the same candidate (user_no) answering the
same question_id more than once. The user reviews these to decide which video to keep / exclude.

NOTE on dates: the dataset's mp4s have ZEROED creation_time (re-encoded), so there is NO reliable
recording timestamp. The only temporal hint is the `id` integer (lower id = assigned earlier in
the dataset's order) — a weak proxy, NOT a true capture date. Decision must rest on CONTENT
(transcript, duration, completeness, and whether the two were scored differently).

Outputs:
  outputs/stage1_overview/duplicate_responses_review.csv  — one row per video (group key + transcript + scores)
  outputs/stage1_overview/duplicate_responses_review.md   — grouped, transcripts side by side, similarity flag

Run: .venv/bin/python src/stage1_dedup_review.py
"""
from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "recruitview_full.csv"
ARROW = ROOT / "recruitview_train"
OUT_CSV = ROOT / "outputs" / "stage1_overview" / "duplicate_responses_review.csv"
OUT_MD = ROOT / "outputs" / "stage1_overview" / "duplicate_responses_review.md"

TS = re.compile(r"\[\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\]")
SCORES = ["openness", "conscientiousness", "extraversion", "agreeableness",
          "neuroticism", "answer_score", "interview_score", "overall_performance"]


def clean(t: str) -> str:
    """Strip [mm:ss - mm:ss] timestamps and collapse whitespace (for similarity / token count)."""
    return re.sub(r"\s+", " ", TS.sub(" ", str(t))).strip()


def main():
    df = pd.read_csv(FULL)
    g = df.groupby(["user_no", "question_id"])
    dup_keys = g.size()[g.size() > 1].index
    sub = df.set_index(["user_no", "question_id"]).loc[dup_keys].reset_index()
    # arrow `id` is zero-padded ('0724'); csv `id` is int (724) — normalise to int to match.
    dup_ids = set(sub["id"].astype(int))
    print(f"duplicate groups: {len(dup_keys)}   videos to fetch: {len(sub)}")

    # pull transcripts from arrow (text columns only — no video decode)
    from datasets import load_from_disk
    ds = load_from_disk(str(ARROW)).select_columns(["id", "transcript"])
    tmap = {int(r["id"]): r["transcript"] for r in ds if int(r["id"]) in dup_ids}
    sub["transcript"] = sub["id"].astype(int).map(tmap)
    miss = sub["transcript"].isna().sum()
    if miss:
        print(f"WARNING: {miss} transcripts still unmatched")

    # CSV: one row per video
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["user_no", "question_id", "question", "id", "duration"] + SCORES + ["transcript"]
    sub[cols].sort_values(["user_no", "question_id", "id"]).to_csv(OUT_CSV, index=False,
                                                                   quoting=csv.QUOTE_ALL)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")

    # MD: grouped, side by side, with similarity flag
    lines = ["# Duplicate responses — review (same candidate × same question)\n",
             f"*{len(dup_keys)} groups · {len(sub)} videos. **No recording date exists** "
             "(mp4 creation_time zeroed); `id` order is only a weak 'assigned-later' proxy. "
             "Judge by transcript / duration / completeness / score divergence.*\n",
             "**Similarity** = difflib ratio on timestamp-stripped text: "
             "**≥0.85** near-identical (likely retake/true dup → keep one); "
             "**≤0.50** divergent answers (review which is on-topic / complete).\n"]
    n_ident = n_div = n_mid = 0
    for (u, q), grp in sub.groupby(["user_no", "question_id"]):
        rows = grp.sort_values("id").to_dict("records")
        qtext = rows[0]["question"]
        lines.append(f"\n---\n\n## user_no {u} · q{q} — \"{qtext}\"  ({len(rows)} videos)\n")
        # pairwise similarity (first vs each other)
        base = clean(rows[0]["transcript"])
        sims = []
        for r in rows[1:]:
            s = SequenceMatcher(None, base, clean(r["transcript"])).ratio()
            sims.append(s)
        sim = max(sims) if sims else 0.0
        flag = ("🟢 near-identical (likely retake/true dup)" if sim >= 0.85
                else "🔴 divergent answers — review" if sim <= 0.50
                else "🟡 partial overlap — review")
        if sim >= 0.85: n_ident += 1
        elif sim <= 0.50: n_div += 1
        else: n_mid += 1
        lines.append(f"**Max similarity: {sim:.2f} → {flag}**\n")
        lines.append("| id | duration | tokens | O | C | E | A | N | answer | interview | overall_perf |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            toks = len(clean(r["transcript"]).split())
            lines.append("| **{id}** | {duration} | {tk} | {o:.2f} | {c:.2f} | {e:.2f} | "
                         "{a:.2f} | {n:.2f} | {ans:.2f} | {iv:.2f} | {op:.2f} |".format(
                             id=r["id"], duration=r["duration"], tk=toks,
                             o=r["openness"], c=r["conscientiousness"], e=r["extraversion"],
                             a=r["agreeableness"], n=r["neuroticism"], ans=r["answer_score"],
                             iv=r["interview_score"], op=r["overall_performance"]))
        lines.append("")
        for r in rows:
            lines.append(f"**id {r['id']}** ({r['duration']}):")
            lines.append(f"> {clean(r['transcript'])}\n")
    header_counts = (f"\n*Triage: 🟢 near-identical = {n_ident} · 🟡 partial = {n_mid} · "
                     f"🔴 divergent = {n_div}.*\n")
    lines.insert(3, header_counts)
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"triage: near-identical={n_ident}  partial={n_mid}  divergent={n_div}")


if __name__ == "__main__":
    main()
