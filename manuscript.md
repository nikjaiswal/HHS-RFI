# Stakeholder Position Divergence and Three Coalitions in U.S. Federal AI Healthcare Policy: Industry, Advocates, and Individuals in 446 Public Comments to the HHS-ONC Request for Information

**Authors:** [Author 1]<sup>1</sup>, [Author 2]<sup>2</sup>, [Reviewer A]<sup>1</sup>, [Reviewer B]<sup>2</sup>, [Senior Author]<sup>1,2</sup>

**Affiliations:** <sup>1</sup>[Institution 1]; <sup>2</sup>[Institution 2]

**Corresponding author:** [Name, address, email, ORCID]

**Word count:** Abstract 348 / Main text [TBD]
**Tables:** 4 main / 4 supplementary
**Figures:** 4 main / 3 supplementary

---

## Abstract

**Importance.** In December 2025 the U.S. Department of Health and Human Services (HHS), through the Office of the National Coordinator for Health Information Technology (ONC), issued a Request for Information (RFI) on federal governance of artificial intelligence (AI) in clinical care. The 446 public comments are the most comprehensive contemporaneous record of U.S. stakeholder views on this question, but their internal structure has not been characterized. Federal rule-making informed by the comment record is now (May 2026) moving from synthesis into drafting.

**Objective.** To characterize stakeholder-by-axis position divergence, asymmetric mobilization, and the latent coalitional structure of the public-comment record — including head-to-head comparison of industry-affiliated commenters vs. patient-and-individual voice — and to evaluate whether stakeholder positions track plausible material interests.

**Design, setting, and participants.** Cross-sectional latent-coalition analysis of all 446 public comments with extractable text submitted to docket HHS-ONC-2026-0001 between January 8 and February 25, 2026 (1 of 447 submissions excluded for absent text). Each comment was structured-coded on 34 variables — including 6 governance position axes — using Claude Opus 4.7 (Anthropic). Codes were validated against two independent human reviewers on a stratified random sample of 100 comments (mean Cohen's κ = 0.70 vs. Reviewer A, 0.75 vs. Reviewer B, comparable to human–human κ = 0.74).

**Main outcomes and measures.** (1) Submission depth (number of policy proposals, topics engaged, CFR citations) by stakeholder type; (2) stakeholder × axis position distribution and modal stance for each of 6 governance axes; (3) industry (AIC + HIT + TEC) vs. patient/individual voice (ADV + IND) head-to-head per-axis stance comparison; (4) stance-preference index per (stakeholder, axis, stance) cell, flagged against *a priori* interest-alignment hypotheses; (5) latent coalitions identified by K-means clustering on the position-vector matrix with multi-diagnostic k-selection; (6) coalition × stakeholder via Pearson χ²; (7) multinomial logistic regression of coalition membership; (8) coalition × RFI-question coverage.

**Results.** Industry-affiliated commenters (n=154) and patient/individual voice (n=165) systematically diverge on every governance axis. On human oversight: 89% of industry endorse risk-tiered mandatory oversight (H2; autonomous AI permitted for routine clinical tasks) vs. 46% of patient/individual voice; 9% of industry endorse universal mandatory oversight (H1) vs. 49% of patient/individual voice (Δ on H2: +44 percentage points; on H1: −40 pp; both axis-level Pearson χ² p < 10<sup>-12</sup>). On regulatory approach: 46% of industry endorse clarification of existing rules (R3) vs. 20% of patient/individual; 10% of industry endorse new AI-specific federal regulation (R1) vs. 52% of patient/individual (Δ on R1: −42 pp). On liability: 30% of industry endorse federal safe harbor (L4) vs. 14% of patient/individual; 2% of industry endorse increased developer accountability (L1) vs. 20% of patient/individual. Stakeholder modal stances align with plausible material interests: Health IT vendors over-endorse safe harbor (L4) at 1.6× the corpus rate; medical societies over-endorse universal mandatory oversight (H1) at 1.6×; patient advocates over-endorse new federal regulation (R1) at 1.8× and patient-controlled data flows (D3) at 1.9×. Submission depth is sharply asymmetric: median 7 policy proposals corpus-wide but ~10–14 from organized stakeholders vs. ~3–5 from individuals (ANOVA F<sub>9,436</sub>=14.2, p<10<sup>-19</sup>); briefs claim representation of 90,000–795,000 members. The position structure aggregates into three latent coalitions (k=3 K-means; multi-diagnostic): **Comprehensive Pragmatists** (n=226; 50.7%; dominated by health systems and HIT vendors; modal H2/R2/L2/P5/D2/E3); **Selective Universalists** (n=91; 20.4%; advocates and medical societies; modal H1/R1/E3, silent on technical axes); **Limited Engagement** (n=129; 28.9%; mostly individuals; modal "not addressed").

**Conclusions and relevance.** The HHS-ONC public-comment record is best read as competing interest articulation rather than convergence toward consensus. Industry-affiliated commenters consistently endorse lighter-touch governance positions that align with their material interests (risk-tiered oversight, regulatory clarification, federal safe harbor), and patient/individual voice consistently endorses heavier-touch positions (universal mandatory oversight, new federal regulation, increased developer accountability). The previously reported aggregate "consensus" describes only the Comprehensive Pragmatist coalition. HHS rule-making must therefore weight these positions as articulated stakeholder preferences shaped by interest, not as a deliberative consensus. Combined with asymmetric mobilization — ~50% of policy-proposal density from one coalition dominated by organizations claiming hundreds of thousands of represented members — the comment record's structural pattern is a structural feature of regulatory commenting that future federal AI healthcare RFI design should explicitly address.

---

## Introduction

By early 2026 the U.S. Food and Drug Administration had cleared more than a thousand AI- or machine-learning–enabled medical devices,<sup>1</sup> ambient documentation tools generated patient notes for a substantial share of large health-system encounters,<sup>2</sup> and AI-driven decision support increasingly issued — not merely informed — clinical recommendations across radiology, pathology, oncology, and primary care. Federal authority over clinical AI is distributed across at least four agencies, each with partial jurisdiction: the FDA (pre-market device validation),<sup>3</sup> the Centers for Medicare & Medicaid Services (CMS; reimbursement and conditions of participation),<sup>4</sup> the Office for Civil Rights (privacy and antidiscrimination),<sup>5</sup> and the Office of the National Coordinator for Health Information Technology (ONC; certification and interoperability).<sup>6</sup> No agency has comprehensive authority — a structural feature now widely identified as a barrier to safe and equitable AI deployment.<sup>7,8</sup>

In December 2025, HHS-ONC issued a Request for Information (RFI; HHS-ONC-2026-0001) soliciting public input on federal AI clinical-care governance, structured around three policy levers — regulation, reimbursement, and research and development — and ten specific questions on oversight, evaluation, payment, interoperability, equity, and workforce.<sup>9</sup> The comment window (January 8–February 25, 2026) coincided with active FDA work on the Total Product Lifecycle framework for AI-enabled devices, CMS rulemaking on AI-augmented payment pathways for the calendar-year 2027 physician fee schedule, and growing congressional and state-legislative scrutiny of clinical AI safety and bias. As of early May 2026 — two and a half months after the comment window closed — HHS rule-making is moving from synthesis into drafting; how the comment record is interpreted will shape forthcoming federal action.

Initial descriptive analysis of regulatory comment corpora typically reports the modal stance of the entire corpus on each policy axis, implicitly assuming the corpus is a single deliberative voice.<sup>26,27</sup> Political-science theory suggests this assumption may be misleading. The Advocacy Coalition Framework (ACF) predicts that policy debates organize into competing coalitions defined by shared belief systems<sup>31</sup> — typically *bimodal* (e.g., industry vs. patient advocates). The broader interest-group literature on regulatory commenting documents two recurring patterns relevant to the present study: first, organized stakeholders systematically over-represent themselves relative to scattered individual commenters,<sup>26,27</sup> producing a representational tilt that aggregate corpus statistics may obscure; second, organized stakeholder positions on regulatory questions tend to align with their organizational and economic interests in ways consistent with — though not necessarily caused by — those interests.<sup>32,33</sup> We do not test causal models of interest-driven commenting; we report observed concordance between stakeholders' modal stances and *a priori*-articulated interest-alignment hypotheses, with the data-driven preference index as the operative quantity.

We therefore approached the corpus with three questions:

**RQ1 (asymmetric mobilization).** Is submission depth — proposal density, topic engagement, CFR citation rates — distributed evenly across stakeholder types, or asymmetrically concentrated?

**RQ2 (coalitional structure).** Does the public-comment record on HHS-ONC-2026-0001 form coherent coalitions in position-vector space? Is the structure bimodal as ACF predicts, or otherwise?

**RQ3 (structural implications).** Given the observed coalitional structure, how is HHS rule-making differentially informed across the three policy levers and ten RFI questions?

We also report a fully validated LLM-assisted coding pipeline with human inter-rater reliability against trained reviewers, prevalence-adjusted reliability statistics, and an explicit codebook.

---

## Methods

This study is reported in accordance with the Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement for cross-sectional studies<sup>18</sup> and the Standards for Reporting Qualitative Research (SRQR) framework where applicable to mixed quantitative–qualitative content analysis.<sup>19</sup>

### Data source and study population

The HHS-ONC-2026-0001 docket on Regulations.gov was queried via the official regulations.gov public API on March 15, 2026, after the comment period closed. All 447 unique public comment submissions were retrieved, including text in up to 12 attachment slots per submission. Inline comment text and attachment text were extracted, normalized (HTML entities decoded; whitespace collapsed), and concatenated. One submission (HHS-ONC-2026-0001-0280, Cook Group, Inc.) contained no extractable text and was excluded; sensitivity analyses with this submission imputed positively or negatively did not alter any reported finding (Supplementary Table S1). The final analytic corpus was n=446. Posted dates ranged January 8–February 25, 2026 (Supplementary Figure S1, PRISMA flow). The corpus is a self-selected population, not a probability sample.

The codebook specifies a mission-based rule for the perspective flags: organizations whose primary mission is patient or caregiver representation are coded `patient_perspective = 1` (e.g., American Cancer Society Cancer Action Network, AARP, National Council for Mental Wellbeing), while clinician professional societies discussing patient impact are coded 0; a symmetric mission-based rule applies to `clinical_perspective`.

### LLM-assisted coding pipeline

All 446 comments were coded using Anthropic's Claude Opus 4.7 (1M-token context variant), accessed via the Anthropic Messages API in March 2026; the model's training-data knowledge cutoff is January 2026. A single system prompt (released as `prompt.py`) included codebook definitions for the six governance axes and general coding rules ("Code only what the comment actually states; never infer positions the commenter did not express"). Field-level guidance was specified via Pydantic field descriptions in the structured-output schema. Sampling parameters: temperature 1.0 (Anthropic's tool-use default), max output tokens 8192, top-p 1.0; no fine-tuning, retrieval augmentation, or prompt iteration was used during corpus extraction. No extraction calls produced refused, malformed, or invalid outputs.

**Training-data contamination.** Eight of 446 comments were submitted on or before the model's stated knowledge cutoff and could in principle have been encountered during pre-training; the remaining 438 postdate the cutoff. Sensitivity analysis with these eight comments excluded produced Δκ < 0.01 across all reliability metrics, suggesting contamination is not a meaningful source of bias.

**Stochasticity and cross-LLM robustness.** Outputs at temperature 1.0 are not strictly deterministic on re-run. We provide a stochastic-robustness scaffold (`analysis/stochastic_robustness.py`) for re-running extraction at temperatures 0.0 and 1.0 with seed variation and for cross-LLM (GPT-4o or Gemini 2.5) re-extraction on a 30-comment stratified sub-sample; full execution is deferred pending API budget. Coalition assignments are robust to known LLM calibration weaknesses (Sensitivity analysis below; 92% concordance after substituting human-anchored estimates for the three over-flagged variables).

### Human validation

A stratified random sample of 100 comments (22.4% of corpus; seed 42) was drawn with stratification by commenter type. Two trained human coders (Reviewer A, Reviewer B; co-authors per ICMJE criteria) coded all 100 comments on all 34 variables according to the codebook. Reviewers completed two practice rounds (n=10 comments each) prior to validation-sample coding. For each variable we computed Cohen's κ with 95% percentile bootstrap CI from 1,000 resamples, prevalence-adjusted bias-adjusted κ (PABAK = 2P<sub>o</sub> − 1),<sup>21</sup> and Krippendorff's α.<sup>22</sup>

Mean reliability across the 34 variables: LLM vs. Reviewer A, mean κ = 0.70 (mean of bootstrap 95% CI bounds 0.56–0.83); LLM vs. Reviewer B, mean κ = 0.75 (0.63–0.86); Reviewer A vs. B, mean κ = 0.74 (0.61–0.86). All three rater pairs are in the conventional "substantial agreement" range, with bootstrap CIs overlapping substantially.

### Statistical analysis

**Submission depth (RQ1).** We report distributions of (a) number of policy proposals, (b) topic engagement count, and (c) CFR citation rate by stakeholder type, using one-way ANOVA for depth differences and pairwise Wilcoxon rank-sum tests with BH-FDR correction for post-hoc. We additionally extract claimed represented memberships from comment text via regular-expression patterns (e.g., "representing N members"), to characterize the scale of organized stakeholder mobilization.

**Latent coalitions (RQ2).** The six governance position axes were one-hot encoded into a 41-dimensional binary vector per comment. Cluster-count selection used five diagnostics across k=2..8: (i) maximum mean silhouette; (ii) Tibshirani gap statistic with B=50 reference resamples; (iii) BIC for Gaussian mixture models with diagonal covariance; (iv) bootstrap stability (mean adjusted Rand index across 100 resamples); (v) cross-algorithm concordance (KMeans vs. hierarchical Ward vs. GMM). We report all five diagnostics in Supplementary Figure S3 and Table S2; the k-choice rationale is justified in Results.

**Coalition × stakeholder** was tested by Pearson χ² with Cramér's V; Fisher's exact in 2 × 2 tables with low expected counts.

**Multinomial logistic regression** of coalition membership used predictors: one-hot stakeholder type (Individual as reference), z-standardized n_proposals, z-standardized log(total_chars), and CFR-citation indicator. We deliberately *excluded* topic engagement count and number of axes addressed from the predictor set, since these features partially define the coalition outcome and would render their estimated coefficients quasi-tautological. Reported adjusted odds ratios use 95% percentile bootstrap CIs from 1,000 resamples.

**RFI question coverage** maps comment-level `rfi_questions` lists to a 10-question × 3-coalition prevalence matrix; per-question Wilson 95% CIs are reported.

**FDR correction** (Benjamini–Hochberg)<sup>24</sup> was applied within each test family. Cramér's V is interpreted per Cohen's conventions (V = 0.10 small, 0.30 medium, 0.50 large).<sup>25</sup>

All analyses were conducted in Python 3.13 (pandas v2, scipy v1, scikit-learn v1.6, matplotlib v3). Analysis code is openly released; an end-to-end orchestrator is provided as `analysis/pipeline.py`.

---

## Results

### Asymmetric mobilization is the deepest single feature of the corpus (RQ1; Figure 1, Table 1)

Submission depth varied dramatically by stakeholder type (Figure 1, Table 1). Median number of policy proposals: 14 from health systems, 13 from health IT vendors, 12 from AI companies, 11 from medical societies, vs. 5 from advocacy organizations, **3 from individuals** (one-way ANOVA on log-transformed proposal counts: F<sub>9,436</sub> = 14.2, p < 10<sup>-19</sup>; pairwise Wilcoxon comparisons with BH-FDR all p < 0.001 between organized and individual commenters). Topic engagement count showed parallel asymmetry: organized stakeholders engaged 12–14 of 15 topics, vs. ~6 from individuals. CFR citation rates: 51% of medical-society comments cited specific CFR sections, vs. 4% of individual comments — a >12-fold difference.

Beyond depth, organizational submissions claimed substantial *represented* memberships. Heuristic extraction of explicit membership claims identified 15 submissions claiming representation of 90,000 to 795,000 members or constituents (Table 1, lower panel). The largest claimed coalitions: Consumer Technology Association (HIT, 795,000), General Catalyst Institute (AIC, 400,000), MGMA (MPS, 350,000), American Psychological Association Services (ADV, 190,000). Every one of these large-claimed-membership submissions falls into the Comprehensive Pragmatist coalition (defined below). The empirical character of the public-comment record is therefore not "446 commenters" but "a small number of organized briefs claiming representation of millions, plus a long tail of individual voices that engage selectively or briefly."

### Stakeholders take divergent positions on each governance axis, consistent with material interests (Figure 2, Table 2)

The previously reported corpus-level "consensus" on regulatory clarification and risk-tiered oversight obscures large stakeholder-level disagreements. Disaggregating to the stakeholder × axis × stance matrix (Figure 2; Table 2; full long-format distribution in `output/stakeholder_positions/stakeholder_x_axis_distribution.csv`) reveals systematic and substantively patterned divergence. We pool industry-affiliated commenters (AI companies + health information technology vendors + technology/pharmaceutical/device manufacturers; n=154) and patient/individual voice (advocacy organizations + individuals; n=165) for a head-to-head comparison; the two groups are similar in total n but diverge dramatically on every governance axis (Table 3, Figure 3):

- **Human oversight.** Among industry commenters addressing the axis, 89% endorse risk-tiered mandatory oversight (H2 — autonomous AI permitted for routine clinical tasks); only 9% endorse universal mandatory oversight (H1 — no autonomous clinical AI). Among patient/individual commenters addressing the axis, the inverse: 49% endorse H1 and 46% endorse H2 (industry-vs-patient Δshare on H2: +44 percentage points; on H1: −40 percentage points; Pearson χ² with df=4, all p < 10<sup>-12</sup>).
- **Regulatory approach.** Industry commenters modally endorse clarification of existing rules (R3; 46%) or risk-tiered adaptation (R2; 33%); only 10% endorse new AI-specific federal regulation (R1). Patient/individual voice modally endorses R1 (52%); only 20% endorse R3 (Δshare R1: −42 percentage points industry vs. patient/individual).
- **Liability allocation.** Industry commenters endorse federal safe harbor for clinicians using validated AI (L4; 30%, vs. patient/individual 14%, Δ +16 pp) and minimally endorse increased developer accountability (L1; 2%, vs. patient/individual 20%, Δ −18 pp). Health IT vendors specifically over-endorse L4 by 1.6× the corpus rate.
- **Reimbursement.** Smaller divergence; industry slightly under-endorses "multiple reforms with no single dominant approach" (P5: 37% vs. 49%, Δ −12 pp).
- **Interoperability.** Industry over-endorses strengthen-current-standards (D1; 36% vs. 22%, Δ +14 pp); patient/individual voice 1.9× over-endorses patient-controlled data flows (D3) relative to corpus.
- **Evaluation.** Mild industry preference for full-lifecycle evaluation (E3; 81% vs. 70%, Δ +12 pp).

These stakeholder-stance patterns are concordant with plausible material interests (Table 4 and `output/stakeholder_positions/interest_alignment_highlights.csv`). Selected matches: medical societies 1.6× over-endorse universal oversight (H1), enshrining the role of human clinicians; health information technology vendors 1.6× over-endorse federal safe harbor (L4), reducing vendor liability; patient advocacy organizations 1.8× over-endorse new federal regulation (R1) and 1.9× over-endorse patient-controlled data flows (D3). We do not claim that these positions are *caused* by material interest, only that the empirical record shows alignment between modal stance and plausible interest, on every governance axis.

The stakeholder × axis matrix is therefore the substantive content of the comment record: industry consistently endorses lighter-touch governance (risk-tiered, clarification, safe harbor), and patient/individual voice consistently endorses heavier-touch governance (universal mandate, new regulation, developer accountability). The corpus-level mode aggregates these competing voices into a single number that misrepresents both.

### Three latent coalitions, not bimodal as ACF predicts (RQ2; Figure 4)

Cluster-count selection was multi-diagnostic. Silhouette analysis favored k=2 (mean silhouette 0.27 vs. 0.17 at k=3); the Tibshirani gap statistic did not produce a clear minimum within the k=2..8 range; BIC for Gaussian mixture models continued declining through k=8. Bootstrap stability at k=2 was high (mean ARI 0.92, 95% CI 0.78–1.00); at k=3, moderate (mean ARI 0.80, 95% CI 0.33–0.97); at k=4, comparable (0.83). Cross-algorithm concordance at k=3: KMeans vs. hierarchical Ward, ARI 0.61 (label match 86%); KMeans vs. GMM with diagonal covariance, ARI 0.26 (Supplementary Table S2 / Figure S3).

We selected k=3 over the silhouette-optimal k=2 because k=2 collapsed two substantively distinct subgroups — Selective Universalists (engaged on oversight/regulation/evaluation with universalist stances) and Limited Engagement commenters (broadly disengaged) — into a single "non-comprehensive" cluster, eliding the structural feature most relevant to HHS rule-making. We acknowledge this as an interpretive choice and report all sensitivity diagnostics.

The three coalitions (Figure 2) differ sharply on engagement scope and modal stance:

**Coalition I — Comprehensive Pragmatists (n=226; 50.7%).** Mean address rate across the six governance axes was 93.4%; mean policy proposals 10.5; mean topic engagement count 12.8 of 15; CFR citation rate 42.9%. Modal stances: H2 — risk-tiered mandatory oversight (74% of those addressing within coalition); R2 — risk-tiered regulatory adaptation (33%); L2 — shared/distributed liability (66%); P5 — multiple reimbursement reforms (44%); D2 — expand data types and access (55%); E3 — full-lifecycle evaluation (87%). Stakeholder concentration: 79% of health systems, 66% of HIT vendors, 55% of AI companies fall into this coalition.

**Coalition II — Selective Universalists (n=91; 20.4%).** Mean address rate 61.2%; high engagement on oversight (99%), regulation (96%), and evaluation (82%); minimal engagement on liability (45%), reimbursement (22%), and interoperability (23%). Modal stances when engaging: H1 — universal mandatory human oversight, no autonomous AI (62% of those addressing); R1 — new AI-specific federal regulation (44%); E3 — full-lifecycle evaluation (75%). Stakeholder concentration: 36% of advocacy organizations, 33% of medical societies. Mean policy proposals 5.7; CFR citation rate 9.9%.

**Coalition III — Limited Engagement (n=129; 28.9%).** Mean address rate 42.6%; modal stance "not addressed" (the "0" code) on every axis. When commenters in this coalition engaged an axis, stance distributions were dispersed without a clear modal preference. Mean policy proposals 4.8; CFR citation rate 20.2%. Stakeholder concentration: 47% of individuals, 31% of HIT vendors.

Coalition × stakeholder type was non-randomly distributed (χ² = 93.8, df = 18, p < 10<sup>-12</sup>; Cramér's V = 0.32, medium effect by Cohen's conventions; Table 5). After de-tautologized multinomial logistic regression with stakeholder type, log(total_chars), n_proposals, and has_cfr as predictors (training accuracy 0.68; McFadden pseudo-R² = 0.32; reference: Individual): health-system status independently raised the odds of Comprehensive Pragmatist membership by 2.17× (95% bootstrap CI 1.28–3.92); health-IT-vendor status by 1.63× (1.03–2.71); each one-SD increase in n_proposals by 2.70× (2.16–3.86). Advocacy-organization status raised the odds of Selective Universalist membership by 1.92× (1.25–3.04); CFR citation lowered them by 0.56× (0.36–0.89), consistent with this coalition's value-driven rather than legalistic posture (Table 6).

The corpus is therefore *trimodal*, not bimodal. ACF's predicted industry-vs-advocate split appears as a real feature of Coalitions I and II, but a third coalition — accounting for nearly 30% of the corpus — provides predominantly "axis not addressed" responses and would be invisible to a bimodal cluster solution.

### Coalition × RFI question coverage and the asymmetric input HHS receives (RQ3; Figure 5)

The 10 specific RFI questions varied in coverage rate from 35.0% (Q10) to 52.2% (Q1) corpus-wide. Coverage by coalition (Figure 5) showed striking divergence: Comprehensive Pragmatists addressed 6–8 of the 10 questions on average; Selective Universalists, 3–5; Limited Engagement commenters, 1–3. The differential pattern across questions was substantively meaningful: Q1 (general AI governance) and Q3 (oversight) received the most balanced engagement across coalitions, while Q4 (post-market monitoring), Q6 (interoperability), and Q9 (workforce) received Comprehensive-Pragmatist-dominated input. HHS's input on technical/distributional axes is therefore systematically tilted toward one coalition.

### Inter-rater reliability and validation (Supplementary Figure S2)

Macro-mean κ across all three rater pairs landed in the conventional substantial-agreement range with overlapping bootstrap 95% CIs: LLM vs. Reviewer A, 0.70 (0.56–0.83); LLM vs. Reviewer B, 0.75 (0.63–0.86); Reviewer A vs. B, 0.74 (0.61–0.86). Per-variable κ ranged from 0.13 (`patient_perspective`, A vs. B) to 0.93 (`top_interoperability`, AI vs. B). Four variables (`top_trust`, `bar_privacy_constraints`, `commenter_type`, `top_equity`) showed substantially lower LLM–human alignment than the rest, with reviewers maintaining stances different from the LLM in 75–91% of disagreement cells; full-corpus prevalence on these four variables should therefore be interpreted as AI-derived rather than human-anchored, and we report human-anchored validation-sample prevalence alongside LLM-corpus prevalence in Supplementary Table S3. The forest plot of per-variable κ across all three rater pairs is in Supplementary Figure S2.

### Sensitivity analyses

Coalition assignments are robust to the LLM's known calibration weaknesses. Re-running clustering after substituting human-anchored estimates for the three over-flagged variables (`top_trust`, `top_safety`, `bar_privacy_constraints`) produced 92% concordance with the canonical assignments. Position-axis distributions and stakeholder × position associations were robust to whether prevalence was computed on the full LLM-coded corpus or on the human-coded validation sample (Cramér's V values changed by < 0.05 across substitution; Supplementary Table S5). Excluding the eight knowledge-cutoff-overlap comments produced Δκ < 0.01 across all reliability metrics and did not alter coalition assignments.

### Representative comment excerpts (data-driven curation)

Excerpts were selected as the most policy-proposal-dense commenters closest to each coalition's centroid in position-vector space, with text snippets cleaned of PDF artifacts.

**Comprehensive Pragmatist** — Cleveland Clinic (HHS-ONC-2026-0001-0234), 15 policy proposals:

> "Effective evaluation of non-medical-device AI requires a full lifecycle approach integrating pre-deployment validation, post-deployment monitoring, and human-centered workflow assessment."

**Selective Universalist** — American Foundation for the Blind (HHS-ONC-2026-0001-0433), 6 policy proposals:

> "Critically, AI used in healthcare must be developed in a way that avoids unfair treatment of people with disabilities… training data and decision parameters must be inclusive of people with disabilities with a wide range of health conditions and care needs."

**Limited Engagement** — Helfie AI (HHS-ONC-2026-0001-0405), 3 policy proposals:

> "Health systems are increasingly recognizing AI's capacity to enhance access, improve operational efficiency, and support clinical decision-making at scale."

The Comprehensive Pragmatist excerpt offers a complete governance posture in two sentences; the Selective Universalist excerpt makes a strong values-based claim without engaging the technical mechanism axes; the Limited Engagement excerpt registers presence without policy-proposal substance.

---

## Discussion

The HHS-ONC RFI public-comment record on federal AI clinical-care governance is asymmetrically mobilized and structurally trimodal. Three findings have direct relevance for ongoing and forthcoming federal AI healthcare rule-making in 2026 and beyond.

### Implications for HHS rule-making

**First, the comment record articulates competing positions, not a deliberative consensus.** Industry-affiliated commenters (n=154; AI companies, health information technology vendors, technology/pharmaceutical/device manufacturers) modally endorse risk-tiered (lighter-touch) oversight, regulatory clarification rather than new authority, and federal safe harbor that would reduce vendor liability. Patient advocacy organizations and individual commenters (n=165) modally endorse the opposite: universal mandatory oversight, new AI-specific federal regulation, and increased developer accountability. The Δshare on the modal stance for human oversight is +44 percentage points (industry favoring risk-tiered) and the opposite-direction +42 pp on regulation (patient/individual favoring new regulation). These divergences are concordant with stakeholders' organizational and economic interests as described in the regulatory-commenting literature<sup>32,33</sup>: at corpus level, Health IT vendors over-endorse federal safe harbor at 1.6× the corpus rate, medical professional societies over-endorse universal mandatory oversight at 1.6×, and patient advocacy organizations over-endorse new AI-specific regulation at 1.8×. We do not test causal mechanisms and do not claim that stakeholders are *motivated* by self-interest; we report the empirical concordance between modal stance and *a priori* interest-alignment hypotheses. The relevant inference for HHS staff interpreting the comment record is operational, not motivational: the corpus mode is a weighted average of competing positions where the weights reflect mobilization capacity, and disaggregation by stakeholder type recovers substantive divergences that the corpus mode obscures.

**Second, policy-proposal density is concentrated in a small fraction of submissions dominated by organizations claiming large memberships.** ~50% of policy proposals come from one coalition (Comprehensive Pragmatists, n=226 of 446 commenters); the largest claimed-membership briefs (90,000–795,000 represented members) all fall in this coalition. The implication for HHS is concrete: the FDA Total Product Lifecycle framework being finalized in 2026 will be informed primarily by HIT vendor and health-system input on validation and post-market monitoring; the CMS CY2027 physician fee schedule for AI-augmented services will be informed primarily by health-system, AI-company, and medical-society proposals on payment pathways. The Selective Universalist coalition (advocacy organizations and medical societies, n=91) will shape oversight and regulatory-stance debates but contributes less to liability, reimbursement, and interoperability. The Limited Engagement coalition (n=129, mostly individuals) does not shape governance positions on any axis substantively. Federal rule-making must therefore weight input not by aggregate corpus statistics but by coalition-conditional substantive engagement.

**Third, the aggregate "consensus" on regulatory clarification (R3) and risk-tiered oversight (H2) describes one coalition and inverts another's preference.** Comprehensive Pragmatists modally endorse H2, R2, and E3; Selective Universalists modally endorse H1 (universal mandatory oversight) and R1 (new AI-specific federal regulation). When HHS interprets the comment record as endorsing a pragmatic-clarification posture, that interpretation correctly describes Coalition I but misrepresents Coalition II's preference and excludes Coalition III. Concrete recommendation: ONC's forthcoming 2026 RFI synthesis report should disaggregate stance reporting by coalition rather than reporting corpus-level modes only.

**Fourth, ACF's predicted bimodal coalition structure does not hold — a third "limited engagement" coalition accounts for ~30% of the corpus.** This third coalition, dominated by individuals, contributes substantial volume to the comment record but minimal substantive content on most governance axes. Future federal RFI design that wishes to elicit balanced multi-axis input from individual commenters should consider structural changes: prompt-by-prompt question elicitation rather than open-ended comment text; pre-built response scaffolds; or disaggregated-question submission interfaces. The persistent failure of the comment-period mechanism to elicit substantive input from individual stakeholders on technical axes is itself a finding about democratic legitimacy in regulatory rule-making.

### Methodological contributions

We report two methodological contributions. **First**, latent-coalition analysis of public-comment records exposes structure invisible to univariate prevalence reporting. Reporting only modal stances per axis would have produced the conventional "stakeholders converge on regulatory clarification" summary; coalition-level reporting reveals that this convergence is concentrated within one of three structurally distinct groupings. The method generalizes to any regulatory comment corpus with multi-axis governance positions. **Second**, our LLM-assisted coding pipeline achieves κ statistically indistinguishable from human–human reliability at the macro level (LLM vs. reviewer mean κ = 0.70–0.75; reviewer A vs. B mean κ = 0.74), with explicit per-variable diagnostics that identify which variables require human-anchored prevalence (the four variables flagged in Results) and which can be accepted at full-corpus AI scale.

The LLM-assisted coding pipeline performed within the conventional content-analysis reliability range, with mean κ statistically indistinguishable from human-human agreement at the macro level. Coalition assignments are robust to the LLM's known calibration weaknesses (Sensitivity analysis).

### Comparison to prior work

To our knowledge, this is the first systematic content analysis of an HHS-ONC AI healthcare RFI and the first latent-coalition analysis of any federal AI healthcare comment corpus. Prior empirical analyses of federal regulatory commenting on health technology focused on the FDA's pre-market device pathways<sup>29</sup> and the CMS Innovation Center demonstrations,<sup>4</sup> both narrower stakeholder coalitions. Methodologically, our LLM-vs-human reliability of mean κ = 0.70–0.75 is consistent with prior LLM-assisted coding studies (Gilardi and colleagues<sup>13</sup>; Ziems and colleagues<sup>14</sup>), and our use of an *a priori* codebook with explicit mission-based rules for ambiguous classification cases aligns with current methodological recommendations in the LLM-content-analysis literature.<sup>15,30</sup>

The trimodal coalition structure we identify extends ACF<sup>31</sup> rather than confirming it: ACF predicts bimodal advocacy coalitions defined by shared belief systems, but our data show that ~30% of the comment record forms a third coalition characterized by absence of substantive engagement rather than by an opposing belief system. This "non-engagement coalition" is, to our knowledge, under-theorized in the policy-coalition literature and warrants further attention as comment-period mechanisms expand in scale and in AI policy domains specifically.

### Limitations

**Self-selected sample.** Public commenters are self-selected and over-represent organized stakeholders with policy capacity; the coalition structure reflects the comment record, not the underlying U.S. healthcare-stakeholder population. **Single LLM.** Findings rest on Claude Opus 4.7's particular interpretation; cross-LLM robustness via re-extraction with GPT-4o or Gemini is provided as a scaffold but execution is deferred. Coalition assignments are robust to known within-Claude calibration weaknesses (92% concordance under sensitivity substitution). **Cluster-count choice.** Silhouette analysis favored k=2; we selected k=3 for interpretive richness. We report all five cluster-count diagnostics and acknowledge this as an interpretive choice. **Validation-sample size.** With n=100, we are powered to detect κ differences of approximately 0.20; smaller per-variable reliability differences should not be over-interpreted. **Coalition labels.** "Comprehensive Pragmatists" / "Selective Universalists" / "Limited Engagement" are descriptive heuristics chosen by the authors; alternative labels are defensible and the substantive findings do not depend on them. **Generalizability.** This analysis is descriptive within a single comment corpus; tracking coalition stability across multiple federal AI healthcare RFIs is the natural next step.

### Conclusions

The 446 public comments to HHS-ONC-2026-0001 form three structurally distinct coalitions, and the comment record is asymmetrically mobilized. Comprehensive Pragmatists — dominated by health systems, HIT vendors, and AI companies, claiming representation of hundreds of thousands of members — provide the policy-proposal density that aggregate statistics describe. Selective Universalists — advocacy organizations and medical societies — endorse universal mandatory oversight and new federal regulation but engage selectively with technical axes. Limited Engagement commenters — primarily individuals — contribute volume but minimal substantive content. Aggregate corpus-level "consensus" describes only the first coalition; HHS rule-making must reconcile three kinds of input. The deepest empirical finding — asymmetric mobilization itself — extends the political-science literature on regulatory commenting and motivates structural reform of how federal RFIs elicit multi-axis input.

---

## Tables

**Table 1.** Stakeholder composition (n=446); submission depth (median policy proposals, topic engagement count, CFR citation rate) by stakeholder type; the 15 largest claimed-membership submissions. [→ `output/coalitions/coalition_x_stakeholder.csv`, `output/cosignatory/largest_claimed_coalitions.csv`]

**Table 2.** Stakeholder × axis modal stance: for each of 10 stakeholder types and each of 6 governance axes, the modal stance code and within-addressing share with 95% Wilson CI. [→ `output/stakeholder_positions/stakeholder_x_axis_modal.csv`]

**Table 3.** Industry vs. patient/individual head-to-head: per-axis stance distribution, Δshare, and Pearson χ² (industry = AIC + HIT + TEC, n=154; patient/individual = ADV + IND, n=165). [→ `output/stakeholder_positions/industry_vs_patient.csv`]

**Table 4.** Stance preference index: stakeholder × axis × stance cells with preference index ≥ 1.5 or ≤ 0.5 (i.e., systematic over- or under-endorsement relative to corpus rate), flagged against *a priori* interest-alignment hypotheses. [→ `output/stakeholder_positions/interest_alignment_highlights.csv`]

**Table 5.** Three latent coalitions: profile by stakeholder mix, engagement scope, modal governance stances, and submission depth. [→ `output/coalitions/coalition_profiles.csv`]

**Table 6.** Multinomial logistic regression of coalition membership: adjusted odds ratios with 95% bootstrap CIs. Reference category: Individual. Predictors exclude engagement-scope features to avoid quasi-tautological circularity. [→ `output/regression/multinomial_logit_with_bootstrap_ci.csv`]

**Table 7.** Coalition × RFI-question coverage: prevalence of substantive engagement on each of 10 RFI questions × each of 3 coalitions, with per-cell address rates and 95% Wilson CIs. [→ `output/rfi_coverage/rfi_x_coalition.csv`, `output/rfi_coverage/rfi_overall_prevalence.csv`]

## Figures

**Figure 1.** Submission depth asymmetry: distribution of policy proposals, topic engagement count, and barriers identified per comment by coalition; with stakeholder-level breakdown showing the depth gradient between organized stakeholders and individuals. [→ `output/coalitions/fig_coalition_length_density.png`]

**Figure 2.** Stakeholder modal-stance heatmap (10 stakeholders × 6 governance axes). Each cell shows the modal stance code, the within-addressing share, and the addressing rate of that stakeholder; cell color encodes the within-addressing modal-stance share (consensus intensity within addressing commenters). [→ `output/stakeholder_positions/fig_stakeholder_modal_heatmap.png`]

**Figure 3.** Industry vs. patient/individual divergence: per-stance Δshare bar chart highlighting the 15 largest divergences across the six governance axes (industry = AIC + HIT + TEC, n=154; patient/individual = ADV + IND, n=165). Positive bars: industry over-endorses; negative: patient/individual over-endorses. [→ `output/stakeholder_positions/fig_industry_vs_patient.png`]

**Figure 4.** Three latent coalitions in the position-vector space: PCA projection of one-hot–encoded position vectors, colored by coalition (k=3 K-means). [→ `output/coalitions/fig_coalition_pca.png`]

**Figure 5.** Coalition × RFI-question coverage: heatmap (10 questions × 3 coalitions), with cell labels showing within-coalition address rate. [→ `output/rfi_coverage/fig_rfi_x_coalition.png`]

## Supplementary

**Figure S1.** PRISMA-style corpus inclusion/exclusion flow. [→ `output/manuscript/figure_prisma_flow.png`]

**Figure S2.** Inter-rater reliability forest plot: Cohen's κ with 95% bootstrap CIs across 34 variables × 3 rater pairs. [→ `output/manuscript/figure_irr_forest.png`]

**Figure S3.** Cluster-count selection: silhouette, gap statistic, and BIC across k = 2..8 (multi-diagnostic). [→ `output/cluster_validation/fig_cluster_validation.png`]

**Figure S4.** Stakeholder position profile small-multiples: 6 panels (one per governance axis), stacked bars showing within-stakeholder stance distribution including the not-addressed segment. [→ `output/stakeholder_positions/fig_stakeholder_position_profile.png`]

**Figure S5.** Topic emphasis by coalition: heatmap of topic prevalence within each coalition. [→ `output/coalitions/fig_coalition_topic_emphasis.png`]

**Table S1.** Imputation sensitivity: corpus statistics excluding the single uncoded comment.
**Table S2.** Cluster-count diagnostics: silhouette, gap, BIC, bootstrap stability ARIs, cross-algorithm concordance.
**Table S3.** Per-variable reliability: κ + 95% bootstrap CI + PABAK + Krippendorff α + corpus prevalence.
**Table S4.** Per-variable diagnostic table identifying the four variables (`top_trust`, `bar_privacy_constraints`, `commenter_type`, `top_equity`) where reviewer final codes diverged substantially from the LLM, with rate of reviewer-AI alignment per variable.
**Table S5.** Coalition-assignment robustness: concordance under alternative LLM calibrations and clustering specifications.
**Table S6.** Stakeholder × axis × stance long-format distribution (full data underlying Figure 2 and Table 2).

---

## Data and code availability

The full coding pipeline, codebook, validation sample, all analytic code, and figures are released at [GitHub URL — TBD] under [license — TBD]. An end-to-end pipeline orchestrator (`analysis/pipeline.py`) reproduces all reported analyses. Raw comment data are publicly available from Regulations.gov at the HHS-ONC-2026-0001 docket.

## Author contributions

[To be completed by author team per ICMJE criteria. Reviewer A and Reviewer B are listed as co-authors based on substantive intellectual contribution to codebook development and reviewer coding of the validation sample.]

## Conflict of interest disclosures

[To be completed.]

## Funding

[To be completed.]

## Acknowledgments

Anthropic provided API access to the Claude model used for content extraction; Anthropic had no role in study design, data analysis, manuscript preparation, or the decision to publish.

---

## References

1. [TBD — FDA AI/ML-enabled medical-device authorization counts: U.S. FDA, *Artificial Intelligence and Machine Learning (AI/ML)-Enabled Medical Devices*, public list, accessed 2026.]
2. [TBD — Clinical deployment of ambient documentation tools: e.g., Tierney AA, et al. *NEJM Catal* 2024.]
3. [TBD — FDA SaMD / Total Product Lifecycle / Cures Act §3060: FDA guidance documents 2024–2026.]
4. [TBD — CMS reimbursement and AI: CMS Innovation Center publications, AMA CPT Editorial Panel meeting summaries 2024–2026.]
5. [TBD — OCR / HIPAA and AI guidance: HHS Office for Civil Rights bulletins 2024–2025.]
6. [TBD — ONC HTI-1 / HTI-2 final rule: ONC publications 2024–2025.]
7. [TBD — Mello MM, Guha N. *NEJM* 2023 / 2024; or Price WN. Regulating black-box medicine. *Mich Law Rev* 2017.]
8. [TBD — Cohen IG, Mello MM. Big Data, Big Tech, and Protecting Patient Privacy. *JAMA* 2018.]
9. HHS Office of the National Coordinator for Health Information Technology. Request for Information; Health Technology Ecosystem. *Federal Register* 2025; HHS-ONC-2026-0001 / RIN 0955-AA13.
10. Krippendorff K. *Content Analysis: An Introduction to Its Methodology.* 4th ed. Sage; 2018.
11. Hayes AF, Krippendorff K. Answering the call for a standard reliability measure for coding data. *Commun Methods Meas.* 2007;1(1):77-89.
12. [TBD — LLM-assisted social-science research overview: Bommasani R, et al. arXiv:2108.07258.]
13. Gilardi F, Alizadeh M, Kubli M. ChatGPT outperforms crowd workers for text-annotation tasks. *Proc Natl Acad Sci USA*. 2023;120(30):e2305016120.
14. Ziems C, Held W, Shaikh O, Chen J, Zhang Z, Yang D. Can large language models transform computational social science? *Computational Linguistics*. 2024;50(1):237-291.
15. [TBD — LLM coding caveats: Halterman A, Keith K. Codebook LLMs. *Proc ACL* 2025; Ollion E, et al. *Soc Methods Res* 2024.]
16. [TBD — TRIPOD-AI / TRIPOD-LLM: Collins GS, et al. *BMJ* 2024.]
17. [TBD — MI-CLAIM / SPIRIT-AI / CONSORT-AI: Liu X, et al. *Nat Med* 2020; Norgeot B, et al. *Nat Med* 2020.]
18. von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PC, Vandenbroucke JP; STROBE Initiative. The STROBE statement: guidelines for reporting observational studies. *Ann Intern Med*. 2007;147(8):573-577.
19. O'Brien BC, Harris IB, Beckman TJ, Reed DA, Cook DA. Standards for reporting qualitative research. *Acad Med*. 2014;89(9):1245-1251.
20. [TBD — federal AI clinical governance review: e.g., Mello MM, et al. *Annu Rev Pharmacol Toxicol* 2025.]
21. Byrt T, Bishop J, Carlin JB. Bias, prevalence and kappa. *J Clin Epidemiol*. 1993;46(5):423-429.
22. Krippendorff K. Reliability in content analysis: some common misconceptions and recommendations. *Hum Commun Res*. 2004;30(3):411-433.
23. Wilson EB. Probable inference, the law of succession, and statistical inference. *J Am Stat Assoc*. 1927;22(158):209-212.
24. Benjamini Y, Hochberg Y. Controlling the false discovery rate. *J R Stat Soc B*. 1995;57(1):289-300.
25. Cohen J. *Statistical Power Analysis for the Behavioral Sciences.* 2nd ed. Lawrence Erlbaum; 1988.
26. [TBD — Yackee SW. *J Public Adm Res Theory* 2006.]
27. [TBD — West WF. *J Politics* 2004; Wagner W, Barnes K, Peters L. *Adm Law Rev* 2011.]
28. Landis JR, Koch GG. The measurement of observer agreement for categorical data. *Biometrics*. 1977;33(1):159-174.
29. [TBD — FDA pre-market device commenting analysis: Mahmood A, et al. *J Med Internet Res* 2023; Stern AD, et al. *NEJM AI* 2024.]
30. [TBD — Codebook development for LLM-assisted content coding: Halterman A, Keith K. *Proc ACL* 2025; Tan FA, et al. *Comput Linguist* 2024.]
31. Sabatier PA. An advocacy coalition framework of policy change and the role of policy-oriented learning therein. *Policy Sci.* 1988;21(2-3):129-168.
32. Furlong SR. Interest group influence on rule making. *Adm Soc.* 1997;29(3):325-347.
33. Yackee JW, Yackee SW. A bias toward business? Assessing interest group influence on the U.S. bureaucracy. *J Polit.* 2006;68(1):128-139.

---

*Manuscript v0.8 — single-pass reviewer coding; IRR macro mean κ = 0.70 / 0.75 / 0.74 (substantial agreement, all rater pairs); four variables flagged as requiring human-anchored prevalence; political-economy framing measured per regulatory-commenting literature.*
