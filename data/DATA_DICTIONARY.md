# Data Dictionary — recruitview-gender-bias

This repository does **not** redistribute the RecruitView dataset, which is gated and
licensed CC BY-NC 4.0 (https://huggingface.co/datasets/AI4A-lab/RecruitView, academic /
non-commercial use only). What is published here is the project's own **gender annotation
layer** plus aggregate analysis outputs. The full per-response score table is regenerated
locally from RecruitView (see *Data access* below).
---

## Published files

### `gender_by_user.csv` — canonical gender per participant

| Column | Type | Description |
|---|---|---|
| `user_no` | int | anonymised participant ID |
| `gender` | int | `0 = female`, `1 = male` |

One row per participant (331). This is the reproducible input to the pipeline:
`src/stage1_prepare_dataset.py` propagates each participant's label to all of their responses.

### `gender_by_response.csv` — per-response gender (keys only)

| Column | Type | Description |
|---|---|---|
| `response_id` | int | row key in RecruitView (1–2,011) |
| `user_no` | int | anonymised participant ID |
| `question_id` | int | question key (1–76) |
| `gender` | int | `0 = female`, `1 = male` |

A convenience join table at the response level. It carries only keys + gender (no
RecruitView scores or question text), so it can be published freely and joined to
RecruitView once that dataset is obtained through its gated access. 498 female / 1,513
male responses; 79 female / 252 male participants.

---

## The full score table 

20 columns × 2,011 rows. Not in the repo; 
(needs the raw RecruitView download). Documented here as the analysis schema.

### Identifiers (3)
| Column | Type | Description | Example |
|---|---|---|---|
| `response_id` | int | row number (1–2,011) | 1 |
| `id` | str | unique record ID | 0001 |
| `user_no` | str | anonymised participant ID | 147 |

### Demographics (1)
| Column | Type | Description | Values | Missing |
|---|---|---|---|---|
| `gender` | float | gender (manual coding) | 0 = female, 1 = male | 0% |

### Interview metadata (5)
| Column | Type | Description | Values | Note |
|---|---|---|---|---|
| `question_id` | str | question ID | 1–76 | 15 reconstructed question sets |
| `question` | str | question text | "Introduce yourself" | |
| `video_quality` | str | video quality flag | High / Low | Inert (η²≈0) — dropped from models |
| `duration` | str | answer length (categorical) | short / medium / long | Strongly associated with all 12 targets (η²=.03–.12); carries person signal |
| `dup_keep` | bool | dedup flag (added 2026-06-29) | True / False | `False` = retake duplicate (same candidate re-recorded same question; 82 rows / 75 groups). Keep latest `id` per (user_no, question_id). **Stage 2+ filters `df[df.dup_keep]`** → 1,929 rows. Stage 1 used all 2,011. |

### Big Five personality traits + overall (6) — normalised z-scores (M ≈ 0, SD ≈ 1)

All six are derived by the MNL + nuclear-norm model from pairwise comparisons (see *Score provenance*). The low-rank coupling entangles all 12 score columns.

| Column | Description | Range | Note |
|---|---|---|---|
| `openness` | Openness to experience | ≈ −7.8 to +9.3 | |
| `conscientiousness` | Conscientiousness | ≈ −6.9 to +6.6 | |
| `extraversion` | Extraversion | ≈ −7.2 to +8.9 | Most non-verbally expressed Big Five trait — modality-sensitive in LLM comparison |
| `agreeableness` | Agreeableness | ≈ −8.4 to +9.2 | |
| `neuroticism` | Neuroticism | ≈ −2.6 to +2.2 | Compressed variance (SD≈0.49 vs 0.88–1.28 for others); near-normal distribution — interpret separately |
| `overall_personality` | Overall personality index | ≈ −7.5 to +9.3 | **Independent holistic judgment**, not an arithmetic composite of the Big Five (R²=0.69 on block regression, ~31% unique variance) |

### Interview performance scores (6) — normalised z-scores (M ≈ 0, SD ≈ 1)

| Column | Description | Range | Note |
|---|---|---|---|
| `interview_score` | Overall interview score (hire preference) | ≈ −7.9 to +9.4 | Delivery-weighted: highest correlations with speaking and facial (r≈0.80), above answer (r≈0.74). Exclude from LLM outcomes |
| `answer_score` | Answer quality | ≈ −10.2 to +8.7 | Only marginal target in Stage 1 reliability (VPC_person=.30, Eρ²=.72); fully text-recoverable — primary LLM outcome |
| `speaking_skills` | Speaking skills | ≈ −9.4 to +7.9 | Vocal delivery; filler words may be stripped by Whisper transcription. Exclude from LLM outcomes |
| `confidence_score` | Confidence | ≈ −7.3 to +6.8 | Heavily non-verbal. Exclude from LLM outcomes |
| `facial_expression` | Facial expression | ≈ −7.5 to +9.3 | **Purely visual** — LLM cannot score this from transcripts. Exclude entirely |
| `overall_performance` | Overall performance | ≈ −9.3 to +8.8 | **Independent holistic judgment**, not a composite (R²=0.69); driven by confidence + answer, facial≈0. Exclude from primary LLM outcomes |

---

## Modality and LLM scope

Human ratings were produced from **video** (visual + vocal + verbal). The Stage-2 TEAS LLM score reads Gemini's **video-grounded report** (Gemini watched the video; OpenAI codes its verdict into numbers). This is a multi-trait multi-method design comparing the human video method against the Gemini-video-grounded LLM method. (The earlier transcript-only pairwise method was retired 2026-07-11.)

**LLM-scored set: 7 constructs 

| LLM role | Variables |
|---|---|
| **Primary (head-to-head)** | `answer_score`, `openness`, `conscientiousness`, `agreeableness`, `neuroticism`*, `extraversion`† |
| **Decision outcome** | `interview_score`‡ (the hire vote) |
| **Human-baseline only (NOT LLM-scored)** | `overall_personality`, `overall_performance`, `facial_expression`, `speaking_skills`, `confidence_score` |

\* neuroticism: near-normal, compressed variance — interpret separately  
† extraversion: most non-verbally expressed Big Five — flag modality-sensitive  
‡ delivery-weighted / weakly text-recoverable; scored for the **gender-gap in the hire decision**. Low human↔LLM convergence = modality artefact, not LLM failure; forced-choice (plain BT), not in the head-to-head.  
**Tie option (`None`) only for the Big Five** (Davidson); `answer_score` & `interview_score` are forced choice (plain BT).

Full rationale: [`docs/variable_inventory.md`](../docs/variable_inventory.md)

---

## Score provenance

The 12 target scores are **not** absolute ratings and should not be treated as ground truth. In the original RecruitView corpus, clinical psychologists made ≈27,310 **binary pairwise comparisons** between video clips answering the same question ("who appears more confident?", "which participant would you prefer to hire?"). These were converted to continuous, normalised z-scores (M ≈ 0, SD ≈ 1) via a **nuclear-norm-regularised multinomial logit (MNL) model** (Gupta et al., 2025, arXiv:2512.00450, §3.2.2–3.2.3).

---

## Dataset quirks

**Repeated answers (retakes).** Some participants answered a question more than once (e.g.
`user_no` 163 answered question 1 four times). These are **retakes** — the same answer
re-recorded (manual review confirmed same substance, reworded), not different content. **82
rows across 75 (user_no × question_id) groups.** No recording date exists (mp4 `creation_time`
zeroed). Flagged by **`dup_keep`** (keep latest `id` per group); **Stage 2+ excludes them**
(see column note above and `outputs/stage2/duplicate_*`).

**Opener/closer block (`response_id` 1–385).** `response_id` 1–193 = question 1
("Introduce yourself", 172 unique participants); 194–385 = question 76
("Tell me about yourself", 176 unique participants). These are the common opener/closer
present for most participants. Question 27 is a shared anchor bridging sets 6 and 9.

**Leptokurtosis.** Eleven of 12 targets have excess kurtosis 8.8–13.4; only `neuroticism` is near-normal (excess kurtosis 1.1). This mandates a robust / Student-t branch in any parametric modelling.


---

## Stage 2 — TEAS/Gemini LLM scores

The pairwise-LLM approach was retired 2026-07-11 → `archive/removed_pairwise_20260711/`. The
Stage-2 LLM method is now **TEAS**: OpenAI `gpt-4o-mini` acts as a content-analysis coder of each
Gemini video-report, emitting **absolute** scores on the 7 scales.

`experiments/evaluation_of_gemini_outputs/results/teas_run2_gpt-4o-mini.csv` — 1,929 dedup rows ×
7 scales. Per scale: `E_<scale>` = value (Big Five + `answer_score` 1–5; `interview_score` 0–2),
`c_<scale>` = confidence, `q_<scale>` = evidence quote. **Null `E_` = coder abstention, not 0.**
Reliability (G-theory + Bayesian): `results/gtheory_llm_reliability.csv` + `gtheory_llm_findings.md`.
Method index: `experiments/evaluation_of_gemini_outputs/docs/direction_summary.md`.

---

## Data access & license

- RecruitView is **gated** and licensed **CC BY-NC 4.0** — request access on Hugging Face,
  agree to the Responsible AI Usage Policy, then run `python src/stage1_prepare_dataset.py`
  to rebuild `recruitview_full.csv` locally from the raw download + `gender_by_user.csv`.
- The gender annotations and aggregate outputs in this repo are derivative works for
  **non-commercial academic research only**, with attribution to the RecruitView authors.
- This is bias **research**, not a hiring tool; do not use it for automated employment
  decisions or to re-identify participants.

