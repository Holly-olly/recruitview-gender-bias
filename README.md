# Auditing AI Scoring of Interview Transcripts 

This project audits an LLM-based video interview scoring system as a measurement instrument. Rather than asking whether the AI agrees with humans, it compares two independent scoring methods (human ratings and LLM-generated ratings) using the same psychometric framework to examine where they agree, where they diverge, and why.

### My approach

I develop and evaluate assessment systems that operate on human-generated data, whether AI-powered or not. My work focuses on auditing scoring and ranking systems, evaluating validity and reliability, and identifying structural sources of measurement error.

### Quick Results

Human ratings and an LLM-based scoring method were applied to the same RecruitView interviews (1,885 interview responses from 328 candidates across 76 interview questions). Both methods were evaluated using an identical psychometric pipeline:

- Intraclass Correlation (ICC)
- Generalizability Theory
- Bayesian multilevel variance decomposition
- BLUP person-level scores

### Key findings

* **Neither method produces person-level scores you can rely on.** In a single human score, only about 0.18 to 0.28 of the variance is stable between-person signal;
The strongest scale on each side is overall answer score, and even it is weak, with the LLM a little ahead of the human method. On most scales, for both, a candidate's score reflects more noise than a stable property of the person.
* **Both are dominated by answer length, with opposite sign.** Adjusting for word-count removes a
  large share of between-person variance on each side. Humans score longer answers **lower**, the
  LLM scores them **higher**. 
* **The two methods disagree systematically.** Per-person scores correlate **negatively on all seven scales** (mean Pearson about -0.43). Readability adds almost nothing beyond length.
* **Part of the observed gender differences is statistically associated with differences in response length.**


Full report: [Report_human_vs_LLM.md](Report_human_vs_LLM.md)

---

## Scope

```
                 RecruitView video interviews
                 (76 questions · ~6 per person)
                            │
                   Cleaning + gender coding
              (2,011 → 1,929 → 1,885 / 328 persons)
                            │
              7 transcript-scorable scales in scope
                            │
        ┌───────────────────┴───────────────────┐
        │                                         │
  HUMAN method                              LLM method (TEAS)
  video-perceived                     gpt-4o-mini codes pre-existing
  nuclear-norm MNL z                    Gemini text-report → Likert
        │                                        │
        └──────────── SAME ENGINE ───────────────┘
        ICC → G-theory + D-study → Bayesian length/readability
                ladder → EB/BLUP per-person score
        │                                         │
   human BLUP                                 LLM BLUP
   per person/scale                        per person/scale
        └───────────────┬────────────────────────┘
                        ▼
                   Comparison 
```

---

## Methodology

**Dataset.** RecruitView: asynchronous video interviews, 1,885 responses from 328 candidates across
76 questions (after removing retakes and no-content answers).

**Engine.** One-way ICC, generalizability theory with a D-study (GeneralizIT), a Bayesian
length-and-readability ladder (bambi / PyMC), and empirical-Bayes (BLUP) per-person aggregation, run
identically on both methods.

**Human scores.** Pairwise-derived within-question z-scores (nuclear-norm MNL), model-derived, not
absolute ratings.

**LLM scores.** OpenAI `gpt-4o-mini` codes each pre-existing Gemini text-report into a Likert value
per scale. 

**Gender.** Hand-coded per participant (0 = female, 1 = male); small female group, read cautiously.

---
## Repository Structure

```
recruitview-gender-bias/
├── README.md                        # this file
├── LICENSE                          # MIT (code); data: CC BY-NC 4.0
├── requirements.txt
├── data/                            # gender annotation layer + dictionary (no RecruitView scores)
├── src/                             # analysis pipeline (shared engine + per-part scripts)
├── notebooks/                       # unified reliability notebook
└── Report_human_vs_LLM.md           # detailed report
```

---
## Data access

This repository does **not** redistribute RecruitView, which is **gated** and licensed **CC BY-NC
4.0** (academic / non-commercial). Only aggregate tables and figures are shared here, never row-level scores. Request access at [huggingface.co/datasets/AI4A-lab/RecruitView](https://huggingface.co/datasets/AI4A-lab/RecruitView).

---

## My contacts
[olgamaslenkova@gmail.com](mailto:olgamaslenkova@gmail.com) 
[github.com/Holly-olly](https://github.com/Holly-olly)

