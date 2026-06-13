# Data Dictionary — recruitview-gender-bias

This repository does **not** redistribute the RecruitView dataset, which is gated and
licensed CC BY-NC 4.0 (https://huggingface.co/datasets/AI4A-lab/RecruitView, academic /
non-commercial use only). What is published here is the project's own **gender annotation
layer** plus aggregate analysis outputs. The full per-response score table is regenerated
locally from RecruitView (see *Data access* below).

| File | In repo? | Rows | Contents |
|---|---|---|---|
| `data/gender_by_user.csv` | **published** | 331 | one gender label per participant (pipeline input) |
| `data/gender_by_response.csv` | **published** | 2,011 | per-response gender, keys only (no scores) |
| `data/recruitview_full.csv` | local (regenerable) | 2,011 | full 20-column score table + gender |
| `data/recruitview_gender.csv` | local (protected) | — | raw manual gender-coding log (provenance) |

**Gender encoding (confirmed 2026-06-09): `0 = female`, `1 = male`.**

---

## Published files

### `gender_by_user.csv` — canonical gender per participant

| Column | Type | Description |
|---|---|---|
| `user_no` | int | anonymised participant ID |
| `gender` | int | `0 = female`, `1 = male` |

One row per participant (331). This is the reproducible input to the pipeline:
`prepare_full_dataset.py` propagates each participant's label to all of their responses.

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

## The full score table — `recruitview_full.csv` (local, regenerable)

20 columns × 2,011 rows. Not in the repo; regenerate with `python src/prepare_full_dataset.py`
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

### Interview metadata (4)
| Column | Type | Description | Values | Note |
|---|---|---|---|---|
| `question_id` | str | question ID | 1–76 | 15 reconstructed question sets |
| `question` | str | question text | "Introduce yourself" | |
| `video_quality` | str | video quality flag | High / Low | Inert (η²≈0) — dropped from models |
| `duration` | str | answer length (categorical) | short / medium / long | Strongly associated with all 12 targets (η²=.03–.12); carries person signal |

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

Human ratings were produced from **video** (visual + vocal + verbal). Stage 2+ scores the same responses from **transcripts only**. This creates a multi-trait multi-method design: the two measurement channels can only be compared on targets recoverable from text.

| LLM role | Variables |
|---|---|
| **Primary** | `answer_score`, `openness`, `conscientiousness`, `agreeableness`, `neuroticism`*, `extraversion`† |
| **Secondary** | `overall_personality` |
| **Human baseline only** | `interview_score`, `overall_performance` |
| **Exclude entirely** | `facial_expression`, `speaking_skills`, `confidence_score` |

\* neuroticism: near-normal, compressed variance — interpret separately  
† extraversion: most non-verbally expressed Big Five — flag modality-sensitive

Full rationale: [`docs/variable_inventory.md`](../docs/variable_inventory.md)

---

## Score provenance

The 12 target scores are **not** absolute ratings and should not be treated as ground truth. In the original RecruitView corpus, clinical psychologists made ≈27,310 **binary pairwise comparisons** between video clips answering the same question ("who appears more confident?", "which participant would you prefer to hire?"). These were converted to continuous, normalised z-scores (M ≈ 0, SD ≈ 1) via a **nuclear-norm-regularised multinomial logit (MNL) model** (Gupta et al., 2025, arXiv:2512.00450, §3.2.2–3.2.3).

Two consequences matter for analysis:

1. **Low-rank coupling.** The nuclear-norm regulariser couples all 12 score columns, inducing cross-trait correlation beyond what the pairwise responses contain. Some of the observed halo (general evaluative factor) is a model artefact.
2. **No raw pairwise data or rater identifiers are distributed.** This means inter-rater reliability is not estimable from the scores alone, and rater-level bias cannot be separated from genuine group differences.

**Stage 1 reliability finding:** No target reaches a person-level trust threshold (VPC_person ≥ .50). Only `answer_score` is marginal (VPC_person=.30, Eρ²=.72 at n_q=6). Human ratings are therefore treated as **one fallible measurement method** in an MTMM comparison, not as a criterion. Full report: [`docs/stage1_human_baseline.md`](../docs/stage1_human_baseline.md).

---

## Dataset quirks

**Repeated answers.** Some participants answered a question more than once (e.g. `user_no`
163 answered question 1 four times: `response_id` 9, 47, 53, 107).

**Opener/closer block (`response_id` 1–385).** `response_id` 1–193 = question 1
("Introduce yourself", 172 unique participants); 194–385 = question 76
("Tell me about yourself", 176 unique participants). These are the common opener/closer
present for most participants. Question 27 is a shared anchor bridging sets 6 and 9.

**Leptokurtosis.** Eleven of 12 targets have excess kurtosis 8.8–13.4; only `neuroticism` is near-normal (excess kurtosis 1.1). This mandates a robust / Student-t branch in any parametric modelling.


---

## Data access & license

- RecruitView is **gated** and licensed **CC BY-NC 4.0** — request access on Hugging Face,
  agree to the Responsible AI Usage Policy, then run `python src/prepare_full_dataset.py`
  to rebuild `recruitview_full.csv` locally from the raw download + `gender_by_user.csv`.
- The gender annotations and aggregate outputs in this repo are derivative works for
  **non-commercial academic research only**, with attribution to the RecruitView authors.
- This is bias **research**, not a hiring tool; do not use it for automated employment
  decisions or to re-identify participants.

