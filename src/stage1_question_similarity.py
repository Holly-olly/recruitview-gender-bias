#!/usr/bin/env python3
"""Stage 1 — question structure via MiniLM embeddings.

(a) question ↔ question paraphrase similarity — which of the 76 interview questions are
    reformulations of each other (e.g. Q1 "Introduce yourself" ↔ Q76 "Tell me about yourself").
(b) question ↔ construct OTE (Opportunity To Express) — cosine of each question with a Big-Five
    trait descriptor: a proxy for whether the question gives the candidate a chance to express
    that construct (is the question in the same semantic field as the trait, or generic?).

Embeddings: sentence-transformers/all-MiniLM-L6-v2 via `transformers` (mean-pooled + normalized;
sentence_transformers avoided — torchcodec/ffmpeg conflict). Writes to outputs/stage1_overview/.
Run:  .venv/bin/python src/stage1_question_similarity.py
"""
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage1_overview"
NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Big-Five trait descriptors = NEO-PI-R facet lists (chosen 2026-07-13; the 4-set comparison in
# archive/scratch_nonessential_20260711/scratch/ote_compare.py showed extraversion coverage is descriptor-sensitive — NEO facets are the
# validated structure). Six facets per trait.
CONSTRUCTS = {
    "openness": "Openness: Fantasy, Aesthetics, Feelings, Actions, Ideas, Values.",
    "conscientiousness": "Conscientiousness: Competence, Order, Dutifulness, Achievement Striving, "
                         "Self-Discipline, Deliberation.",
    "extraversion": "Extraversion: Warmth, Gregariousness, Assertiveness, Activity, Excitement Seeking, "
                    "Positive Emotions.",
    "agreeableness": "Agreeableness: Trust, Straightforwardness, Altruism, Compliance, Modesty, "
                     "Tender-Mindedness.",
    "neuroticism": "Neuroticism: Anxiety, Angry Hostility, Depression, Self Consciousness, Impulsiveness, "
                   "Vulnerability.",
}


def encode(tok, mdl, texts):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)         # mean pooling
    return torch.nn.functional.normalize(emb, dim=1).numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/recruitview_full.csv")
    qmap = df.groupby("question_id")["question"].first()
    qids = sorted(qmap.index)

    tok = AutoTokenizer.from_pretrained(NAME)
    mdl = AutoModel.from_pretrained(NAME).eval()
    qemb = encode(tok, mdl, [qmap[q] for q in qids])

    # (a) question ↔ question
    S = qemb @ qemb.T
    np.fill_diagonal(S, -1.0)
    rows = []
    for i, q in enumerate(qids):
        j = int(np.argmax(S[i]))
        rows.append(dict(question_id=q, question=qmap[q], nearest_qid=qids[j],
                         nearest_sim=round(float(S[i, j]), 3), nearest_question=qmap[qids[j]]))
    qsim = pd.DataFrame(rows).sort_values("nearest_sim", ascending=False)
    qsim.to_csv(OUT / "question_similarity.csv", index=False)

    # (b) OTE: question ↔ construct
    cemb = encode(tok, mdl, list(CONSTRUCTS.values()))
    O = qemb @ cemb.T                                              # 76 × 5
    traits = list(CONSTRUCTS)
    ote = pd.DataFrame(O, columns=[f"ote_{k}" for k in traits], index=qids).round(3)
    ote.insert(0, "question", [qmap[q] for q in qids])
    ote["best_construct"] = [traits[k] for k in O.argmax(1)]
    ote["best_ote"] = O.max(1).round(3)
    # high-OTE flags: a question is a "high" elicitor of a trait if it sits in that trait's TOP
    # QUARTILE across the 76 questions (relative, per-trait — OTE cosines are low and each trait has
    # a different baseline, so an absolute cut would never flag neuroticism). A question can be high
    # on several traits. Only the 5 Big-Five constructs are OTE-classifiable: answer_score /
    # interview_score are answer-quality scales, not trait descriptors, so they get no OTE flag.
    for t in traits:
        ote[f"high_{t}"] = ote[f"ote_{t}"] >= ote[f"ote_{t}"].quantile(0.75)
    ote["n_high_traits"] = ote[[f"high_{t}" for t in traits]].sum(axis=1)
    ote["high_traits"] = [", ".join(t for t in traits if r[f"high_{t}"]) for _, r in ote.iterrows()]
    ote.index.name = "question_id"
    ote.to_csv(OUT / "question_construct_ote.csv")

    # dedup reciprocal pairs for the console
    seen, pairs = set(), []
    for _, r in qsim.iterrows():
        key = frozenset((r.question_id, r.nearest_qid))
        if key in seen:
            continue
        seen.add(key); pairs.append(r)
    print("=== (a) closest question pairs (paraphrases) ===")
    for r in pairs[:12]:
        print(f"  {r.nearest_sim:.2f}  Q{r.question_id:>2} ↔ Q{r.nearest_qid:>2}  |  "
              f"{r.question[:34]:34s}  <->  {r.nearest_question[:34]}")
    print(f"\n=== (b) OTE — best-fitting Big-Five construct per question ===")
    print(ote["best_construct"].value_counts().to_string())
    print(f"mean best-OTE {ote.best_ote.mean():.2f} (higher = questions more clearly express a trait)")
    print("high-OTE flags per trait (top quartile):",
          {t: int(ote[f"high_{t}"].sum()) for t in traits})
    print("questions by #traits flagged:", ote.n_high_traits.value_counts().sort_index().to_dict(),
          f"({int((ote.n_high_traits == 0).sum())} generic)")
    print(f"\nwrote {OUT}/question_similarity.csv + question_construct_ote.csv")


if __name__ == "__main__":
    main()
