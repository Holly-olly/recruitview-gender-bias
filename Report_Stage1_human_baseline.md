# Reliability and Generalizability of Human-Derived Interview Ratings

*Stage 1 of a study on gender bias in automated (LLM-based) scoring of video interviews. This section reports the methods and results of a variance-decomposition analysis that asks whether human-derived ratings in the RecruitView corpus are reliable enough to serve as a person-level criterion, or should instead be treated as one fallible measurement method among several.*

---

## Methods

### Data

We analysed the RecruitView corpus: 2,011 single-question video-interview responses from 331 participants answering a bank of 76 questions, organised into 15 question sets (five questions plus a common opener/closer) to which participants were randomly assigned. Each response carries 12 continuous target scores — the Big Five traits (openness, conscientiousness, extraversion, agreeableness, neuroticism), an overall personality index, and six performance metrics (interview score, answer score, speaking skills, confidence, facial expression, overall performance). These scores were not absolute ratings: in the original corpus, clinical psychologists made ~27,000 pairwise comparisons that were converted to continuous, normalised scores (mean ≈ 0) via a nuclear-norm-regularised multinomial-logit model. Participant gender was hand-coded for all 331 participants (252 male, 79 female; 1,513 vs 498 clips; encoded 1 = male, 0 = female).

### The 12 target dimensions: provenance, modality, and forward use

The 12 continuous scores span two conceptual blocks. The **personality block** covers the Big Five (openness, conscientiousness, extraversion, agreeableness, neuroticism) plus an overall personality index. The **performance block** covers interview score, answer score, speaking skills, confidence score, facial expression, and overall performance.

**Provenance.** All 12 are *derived* quantities, not direct absolute ratings. Clinical psychologists made binary pairwise comparisons between video clips answering the same question ("Who appears more confident?", "Which participant would you prefer to hire?"), and a nuclear-norm-regularised multinomial logit model converted these into continuous z-scores. The low-rank coupling in the MNL model entangles all 12 columns, inducing some cross-trait correlation beyond what the individual pairwise judgments contain. The two "overall" variables (overall personality, overall performance) are *not* arithmetic composites of their blocks: regressing each on its block gives R² ≈ 0.69, leaving ~31% unique variance — they are independent holistic judgments that load heavily on a general evaluative factor rather than linear summaries of the other dimensions.

**Modality constraint.** Human scores were produced from *video* — psychologists observed facial expressions, vocal delivery, and speech content simultaneously. Later stages of this study score the same responses from *transcripts only*, creating a **multi-trait multi-method (MTMM)** structure in which human (video-perceived) and LLM (text-inferred) ratings are two fallible measurement channels. A construct is comparable across channels only if it is recoverable from the verbal record. Text-recoverability varies sharply:

| Variable | What it indexes | Text-recoverable? | Role in LLM comparison |
|---|---|---|---|
| `answer_score` | Content quality | **Fully** | **Primary** |
| `openness` | Intellectual curiosity, breadth | Mostly | **Primary** |
| `conscientiousness` | Organisation, diligence | Mostly | **Primary** |
| `agreeableness` | Warmth, cooperativeness | Mostly | **Primary** |
| `neuroticism` | Emotional stability (compressed variance) | Partial | **Primary** — interpret separately |
| `extraversion` | Sociability, expressiveness | Partial — most non-verbal of the Big Five | **Primary** — flag modality-sensitive |
| `overall_personality` | Holistic personality impression (R²=0.69) | Partial | **Secondary** |
| `interview_score` | Hire preference (delivery-weighted) | Weak — highest correlations are with speaking and facial | Human baseline only |
| `speaking_skills` | Vocal delivery; filler words often stripped by Whisper | Mostly no | Exclude as LLM outcome |
| `confidence_score` | Self-assurance; heavily non-verbal | Mostly no | Exclude as LLM outcome |
| `overall_performance` | Holistic performance (driven by confidence + answer) | Weak / mixed | Human baseline only |
| `facial_expression` | Facial expressiveness — purely visual | **No** | Exclude entirely |

In Stage 1 all 12 targets are included, because they were all produced from video and all enter the human-baseline gender analysis. From Stage 2 onward (LLM scoring), only the text-recoverable set is used: `answer_score` and the Big Five are the primary targets; `overall_personality` is secondary; `interview_score`, `speaking_skills`, `confidence_score`, `overall_performance`, and `facial_expression` are either excluded entirely or reported for the human baseline only, since any "gender gap" or "human–LLM divergence" on modality-dependent targets would be a measurement-channel artefact, not evidence of bias.

### Analytic framework and its constraint

We reframed Generalizability Theory as a set of Bayesian cross-classified mixed-effects models, decomposing each target's variance into a *person* component (the stable trait signal), *set* and *question* components, and a *residual*. Because only the derived scores are available — with **no raw pairwise data and no rater identifiers** — a rater facet is not observable. Two consequences follow and bound every claim below: (i) there is one observation per person×question cell, so the person×question interaction is **confounded with the residual**; the person component is therefore a **lower bound** on the true signal and the residual an **upper bound** on noise; and (ii) inter-rater reliability and rater bias are not estimable at this stage.

### Design structure and diagnostics

Question sets were reconstructed empirically from within-participant question co-occurrence (15 sets recovered, 0 unassigned). Because 31 participants touch more than one set (mostly an artefact of a shared anchor question; only 8 are genuine multi-session), person was modelled as **crossed with** set rather than nested.

Distributional diagnostics showed pronounced non-normality: 11 of 12 targets had excess kurtosis between 8.8 and 13.4, and Mardia's multivariate kurtosis was 692.9 (expected 168 under multivariate normality; z = 642, p < .001), mandating a robust/Student-t branch. Neuroticism was the exception (excess kurtosis 1.13; SD 0.49 vs ~0.88–1.28 elsewhere), with compressed variance, and is interpreted separately throughout. There were no missing data. Among nuisance covariates, **duration** (categorical: short/medium/long) was strongly associated with every target (η² = .030–.124, all p < 1e-13), whereas **video quality** was inert (η² ≈ 0, all p > .05) and was dropped. Gender was not confounded with set assignment (gender×set Cramér's V = .202, χ² p = .53), confirming successful randomisation; nuisance covariates were nonetheless retained as a precaution.

![Figure 1](../outputs/stage1/dist_histograms.png)

*Figure 1. Distribution of each of the 12 target scores (density) with an empirical-normal overlay and excess-kurtosis annotation. Eleven targets are sharply leptokurtic (excess kurtosis 8.8–13.4), motivating the robust/Student-t branch; neuroticism (cyan) is compressed and near-normal (excess kurtosis 1.1) and is interpreted separately.*

### Model specification

For each target we fitted a ladder of crossed-random-effects models in `bambi`/`PyMC` with the structure `(1 | user_no) + (1 | set_id) + (1 | set_id:question_id)`:

- **m0** — unconditional (pure variance decomposition);
- **m1** — adds gender as a between-person fixed effect;
- **m1d** — adds gender + duration (factor), to test over-control by answer length;
- **m0_filt** — m0 excluding the 8 genuine multi-session participants (sensitivity);
- **m0_robust** — Student-t residuals (heavy-tail sensitivity).

This produced 59 fits (11 heavy-tailed targets × 5 models; neuroticism × 4). All converged (r̂ ≤ 1.014; ≤ 1 divergence).

### Reliability indices and decision rule

From each model we derived Variance Partition Coefficients (VPCs), the relative-decision Generalizability coefficient Eρ² and the absolute-decision dependability coefficient Φ, computed for the median workload of `n_q = 6` questions per person (with sensitivity at n_q = 4 and 17). Each target was classified as **trust** (VPC_person ≥ .50 and Eρ² ≥ .70), **marginal**, or **do-not-trust** (VPC_person < .25 or σ²_person ≈ 0). Where a target's VPC_person credible interval straddled a class boundary, it was assigned the more conservative class (evaluated at the lower bound); this is why several targets with a point VPC_person just above .25 are nonetheless classified do-not-trust.

### Residual-variance convention

The leptokurtic spread can be counted as noise or partly down-weighted, which materially changes the residual denominator. We adopted the **Gaussian** variance partition as the reviewer-defensible headline (heavy-tailed spread treated as noise) and report a **Student-t robustness range** bracketing each VPC_person between a scale-based estimate (σ²; down-weights tails → optimistic) and a marginal estimate (σ²·ν/(ν−2); re-inflates tails → pessimistic). Because the residual denominator spans roughly threefold across these definitions while the person variance is comparatively stable, **VPC_person is itself uncertain, not merely its credible interval**; the range is reported alongside the headline.

---

## Results

### Variance is concentrated in person and residual; set and question are negligible

Across all 12 targets the variance partitioned almost entirely into the person component (~18–30%) and the residual (~70–82%), with set and question components negligible (each ≈ 0.1%). The near-zero question variance is expected: because the source labels were derived from *within-question* pairwise comparisons, between-question means are already netted out by construction. The "opportunity to display a trait" therefore does not appear as question-level variance — it surfaces instead through answer **duration** (see below).

![Figure 2](../outputs/stage1/vpc_stacked_bars.png)

*Figure 2. Variance partition per target (Gaussian m0, n_q = 6): person (signal), set, question and residual shares. Variance splits almost entirely between person (~18–30%) and residual (~70–82%); the set and question components are visually negligible. The dashed line marks the VPC_person = 0.50 trust threshold, which no target approaches.*

### No target reaches "trust"; person-level scores are noisy, person aggregates only marginal

No target met the trust criterion. Eleven of twelve were **do-not-trust**; only **answer score** reached **marginal** (VPC_person = .303, Eρ² = .722). Single-clip person consistency (VPC_person) ranged from .18 to .30, while the six-question person-aggregate reliability (Eρ²) ranged from .57 to .72 — moderate, but short of the conventional .70 for most targets. In other words, a single clip is dominated by non-person variance, and even averaging a person's six responses yields only borderline reliability.

**Table 1. Headline reliability and trust classification (Gaussian m0, n_q = 6).**

| Target | Block | VPC_person | Eρ² (rel.) | Φ (abs.) | Robust range (VPC_person) | Class |
|---|---|---|---|---|---|---|
| answer_score | performance | .303 | .722 | .722 | [.238, .492] | **marginal** |
| confidence_score | performance | .287 | .706 | .706 | [.224, .473] | do-not-trust |
| extraversion | personality | .274 | .692 | .692 | [.224, .478] | do-not-trust |
| overall_performance | performance | .273 | .692 | .692 | [.215, .485] | do-not-trust |
| overall_personality | personality | .273 | .691 | .691 | [.212, .474] | do-not-trust |
| facial_expression | performance | .267 | .686 | .685 | [.215, .445] | do-not-trust |
| openness | personality | .265 | .683 | .682 | [.175, .441] | do-not-trust |
| interview_score | performance | .265 | .682 | .682 | [.176, .468] | do-not-trust |
| agreeableness | personality | .249 | .664 | .664 | [.164, .466] | do-not-trust |
| speaking_skills | performance | .249 | .664 | .664 | [.141, .456] | do-not-trust |
| conscientiousness | personality | .227 | .637 | .637 | [.213, .374] | do-not-trust |
| neuroticism | personality | .183 | .572 | .572 | — (near-normal) | do-not-trust |

![Figure 3](../outputs/stage1/erho2_forest.png)

*Figure 3. Six-question person-aggregate reliability Eρ² per target with 95% credible interval, ordered descending; the dashed line marks the conventional 0.70 threshold. Every point estimate sits at or below 0.70, and only answer score's interval lies clearly to its right.*

### The conclusion is robust to tail treatment

Even under the most optimistic (scale-based Student-t) definition, the upper bound of VPC_person never crossed the .50 trust threshold (maxima: answer_score .492, overall_performance .485, extraversion .478). The "do-not-trust" verdict is therefore not an artefact of how the heavy tails are handled.

![Figure 4](../outputs/stage1/vpc_person_robust_range.png)

*Figure 4. Gaussian headline VPC_person (point) with the Student-t [scale, marginal] robustness range (whiskers), and the 0.50 trust threshold (dashed). The entire sensitivity band sits below 0.50 for every target; even the most optimistic upper bound (answer score, 0.49) does not cross it.*

### Answer duration carries person-level signal, not just noise

Partialling out duration reduced VPC_person by .05–.094 across the eleven heavy-tailed targets (.023 for near-normal neuroticism), and the reduction came almost entirely from the **person** component (e.g. openness σ²_person 0.327 → 0.189) rather than the residual. Verbose participants are consistently scored higher, so answer length is in part a stable trait expression rather than a pure measurement artefact — a finding that operationalises the "opportunity to display the trait" mechanism and cautions against treating duration as a nuisance to be removed.

![Figure 5](../outputs/stage1/duration_slope.png)

*Figure 5. VPC_person without duration (m1) vs. with duration (m1d) per target, Δ annotated. Adding duration lowers VPC_person by 0.05–0.094 (0.023 for near-normal neuroticism); the drop comes almost entirely from the person variance, indicating that answer length carries stable person signal rather than pure noise.*

### Multi-session participants and block differences

Excluding the eight genuine multi-session participants changed VPC_person by ≤ .007 for every target, so this dependence is immaterial. Performance metrics were slightly more person-consistent than personality traits (mean VPC_person .274 vs .245), and the only marginal target was a performance metric, weakly supporting a block difference; both blocks nonetheless remained do-not-trust.

### Gender differences in the human ratings

Holding the design constant, male participants received significantly higher scores on 11 of 12 targets (m1), with effects of 0.17–0.39 SD whose credible intervals excluded zero; only neuroticism was non-significant (−0.020, CI [−0.089, 0.051]). The largest gaps were for interview score (+0.393), overall personality (+0.381), agreeableness (+0.364) and facial expression (+0.362). The lone null is diagnostically useful: a blanket male-favouring response set would have pushed neuroticism — the one undesirable trait — *lower* for men, whereas the observed pattern (men higher on every desirable target, null on the single negative one) is consistent with an evaluative halo on desirability or with genuine pool differences, but not with a uniform pro-male bias; the compressed neuroticism variance, which lowers power, is a competing explanation. Adding duration (m1d) attenuated the coefficients only modestly — by ~12–16% (e.g. interview score 0.393 → 0.341) — and left them significant, indicating that answer length mediates only a small part of the gap, while the bulk (~85%) is independent of how long candidates spoke.

**Table 2. Gender fixed effect (m1, male relative to female; positive = male higher).**

| Target | Coef (SD) | 95% CI | m1d (dur-adjusted) |
|---|---|---|---|
| interview_score | +0.393 | [0.207, 0.578] | +0.341 |
| overall_personality | +0.381 | [0.195, 0.570] | +0.335 |
| agreeableness | +0.364 | [0.179, 0.549] | +0.322 |
| facial_expression | +0.362 | [0.185, 0.539] | +0.318 |
| speaking_skills | +0.344 | [0.152, 0.540] | +0.292 |
| confidence_score | +0.340 | [0.173, 0.512] | +0.302 |
| overall_performance | +0.335 | [0.145, 0.526] | +0.287 |
| extraversion | +0.315 | [0.143, 0.483] | +0.279 |
| openness | +0.306 | [0.129, 0.486] | +0.257 |
| answer_score | +0.305 | [0.112, 0.493] | +0.264 |
| conscientiousness | +0.170 | [0.041, 0.300] | +0.141 |
| neuroticism | −0.020 | [−0.089, 0.051] | −0.012 |

![Figure 6](../outputs/stage1/gender_forest.png)

*Figure 6. Gender fixed effect (male relative to female) ± 95% CI per target, ordered by effect size; m1 (gender only) with credible interval and m1d (duration-adjusted) point overlaid; the dashed line marks zero. The ratings favour men on 11 of 12 targets (neuroticism straddles zero) — opposite to the female-higher direction expected for agreeableness and neuroticism from self-report norms. Duration adjustment (m1d) attenuates each effect modestly without reversing it.*

**Comparison with self-report Big Five norms.** The gender pattern in the human ratings is best read against the established self-report literature, in which the two largest and most cross-culturally robust sex differences both favour women: women score higher on Neuroticism (d ≈ −0.40, the single most reliable effect) and on Agreeableness (d ≈ −0.15 to −0.50, inventory-dependent), with much smaller and less consistent differences on Extraversion, Conscientiousness and Openness (Costa et al., 2001; Schmitt et al., 2008, 55 nations, N = 17,637; Weisberg et al., 2011; Kajonius & Johnson, 2018, N ≈ 320,000). These female-higher effects are reliably attenuated in less wealthy, less gender-egalitarian contexts — the "gender-equality paradox" — so smaller differences would be expected in an Indian sample (Schmitt et al., 2008; Mac Giolla & Kajonius, 2019), and the small Indian self-report studies available are directionally consistent with the global pattern, women scoring higher on Agreeableness and Neuroticism, though typically underpowered (e.g., Gaikwad, 2021; Magan et al., 2014; see also Lodhi, Deo, & Belhekar, 2002, for five-factor measurement in the Indian context). Against this benchmark, our observer-based ratings agree only on the weak, culturally variable traits — men are rated slightly higher on Conscientiousness and Openness, a direction occasionally reported in Indian student samples — but diverge on exactly the two robust, female-favouring effects: Agreeableness is *reversed* (men rated higher), and Neuroticism shows *no difference* where self-report would predict the largest female-higher gap. A uniform male-higher pattern across 11 of 12 dimensions, including an Agreeableness reversal and a Neuroticism null, does not correspond to any documented Big Five personality profile. This is most consistent with the ratings indexing *perceived interview performance* and a general evaluative impression rather than latent trait standing — reinforcing the Stage 1 conclusion that the human scores are a fallible measurement method, not a trait criterion. Because rater identifiers are unavailable, this divergence cannot by itself be attributed to rater bias versus genuine performance differences; it is carried forward as a measurement property, not as established bias.

*(Effect sizes above are orientation values from the cited literature, not computed against the present data.)*

### What this analysis can and cannot establish

The design supports statements about the consistency and dependability of the *derived* scores and about score differences by gender. It cannot, with derived scores alone, separate genuine group differences from rater bias, nor estimate inter-rater reliability or rater×gender interactions — these require the raw pairwise data and rater identifiers, which are unavailable. The observed gender gap is thus a property of the human-derived scores, to be carried forward as a baseline rather than interpreted as established rater bias.

---

## Summary

Human-derived RecruitView ratings do not meet a person-level reliability standard: at the single-clip level they are dominated by non-person variance (~75%, an upper bound on noise given the unobservable person×question confound), and even six-question person aggregates are only borderline reliable (Eρ² ≈ .57–.72), with the verdict robust to tail treatment. Answer duration carries part of the genuine person signal. Human ratings are therefore best treated **not as ground truth but as one measurement method**, motivating a multitrait-multimethod comparison against the LLM-based scores in the next stage. The systematic male-higher rating pattern (11/12 targets) is retained as the human-baseline gender signal against which the LLM scoring will be benchmarked.

---

## References

Costa, P. T., Jr., Terracciano, A., & McCrae, R. R. (2001). Gender differences in personality traits across cultures: Robust and surprising findings. *Journal of Personality and Social Psychology, 81*(2), 322–331. https://doi.org/10.1037/0022-3514.81.2.322

Gaikwad, U. S. (2021). Gender difference between big five personality. *The International Journal of Indian Psychology, 9*(1), 652–658.

Kajonius, P. J., & Johnson, J. (2018). Sex differences in 30 facets of the five factor model of personality in the large public (N = 320,128). *Personality and Individual Differences, 129*, 126–130. https://doi.org/10.1016/j.paid.2018.03.026

Lodhi, P. H., Deo, S., & Belhekar, V. M. (2002). The five-factor model of personality: Measurement and correlates in the Indian context. In R. R. McCrae & J. Allik (Eds.), *The Five-Factor Model of Personality Across Cultures* (pp. 227–248). Kluwer Academic/Plenum.

Mac Giolla, E., & Kajonius, P. J. (2019). Sex differences in personality are larger in gender equal countries: Replicating and extending a surprising finding. *International Journal of Psychology, 54*(6), 705–711. https://doi.org/10.1002/ijop.12529

Magan, D., Mehta, M., Sarvottam, K., Yadav, R. K., & Pandey, R. M. (2014). Age and gender might influence big five factors of personality: A preliminary report in Indian population. *Indian Journal of Physiology and Pharmacology, 58*(4). PMID: 26215005.

Schmitt, D. P., Realo, A., Voracek, M., & Allik, J. (2008). Why can't a man be more like a woman? Sex differences in Big Five personality traits across 55 cultures. *Journal of Personality and Social Psychology, 94*(1), 168–182. https://doi.org/10.1037/0022-3514.94.1.168

Weisberg, Y. J., DeYoung, C. G., & Hirsh, J. B. (2011). Gender differences in personality across the ten aspects of the Big Five. *Frontiers in Psychology, 2*, 178. https://doi.org/10.3389/fpsyg.2011.00178
