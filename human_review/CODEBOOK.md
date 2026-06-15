# Codebook — HHS AI RFI comment coding

Use this when performing human review. Codes must match exactly (e.g. `H2`, `R3`, `IND`) for comparison with LLM output.

---

## Commenter type (commenter_type)

| Code | Label |
|------|--------|
| MPS | Medical professional society |
| HSP | Health system/provider |
| HIT | Health IT |
| AIC | AI company |
| TEC | Tech/pharma/device |
| PAY | Payer/insurer |
| ADV | Patient/consumer advocate |
| LAB | Labor organization |
| POL | Policy/academic/consulting |
| IND | Individual |

**Rule:** Primary organizational type of the commenter. Use submission metadata and any stated organization in the text.

---

## Perspective flags (0 or 1)

These flags describe **who is speaking** (the commenter), not what the comment discusses. A commenter's vantage point is determined by who they are organized to represent, not by the topical focus of their comment. Patient-impact language inside a clinician society's submission does not flip `patient_perspective` to 1 — that content shows up in topic flags (`top_safety`, `top_equity`, etc.).

### clinical_perspective

`clinical_perspective = 1` if the commenter is either:
- an individual writing from their own direct clinical practice experience, OR
- an organization whose **primary mission is representing clinicians** (clinician professional societies, nursing associations, specialty colleges).

**Examples — `clinical_perspective = 1`:** AMA, American Society of Nephrology (ASN), American Urological Association (AUA), American Nurses Association (ANA), Pennsylvania Medical Society, AMIA, AOA, individual physicians/nurses describing their own practice.

**Examples — `clinical_perspective = 0`:** Health systems and providers (Memorial Sloan Kettering, Advocate Health), vendors and health IT (IBM, UpDoc, MEDvidi, Epic), patient advocacy organizations (ACS CAN, Susan G. Komen), policy/academic/consulting groups, individuals writing as patients or laypeople.

### patient_perspective

`patient_perspective = 1` if the commenter is either:
- an individual writing from their own or a family member's experience as patient or caregiver, OR
- an organization whose **primary mission is patient or caregiver representation**.

**Examples — `patient_perspective = 1`:** ACS Cancer Action Network, Susan G. Komen, AARP, National Council for Mental Wellbeing, Community Oncology Alliance (hybrid clinician/patient advocacy — the patient-advocacy mission is constitutive), American Kidney Fund, National Kidney Foundation, individuals describing their own or a family member's care experience.

**Examples — `patient_perspective = 0`:** Clinician professional societies (AUA, AMA, ASN, state medical societies — they speak *as clinicians*, even when discussing patient impact), vendors and health IT (IBM, MEDvidi, UpDoc, GenHealth, basys.ai), health systems and providers (MSK, Advocate Health), consulting firms, policy/academic organizations.

**Borderline cases:**
- Hybrid clinician/patient organizations (e.g., Community Oncology Alliance) → `1` if patient/caregiver representation is an explicit, constitutive mission pillar (not just rhetorical).
- Disease-specific clinician societies → `0`. The condition's patient-impact does not transfer to the society's perspective.
- Service-providing organizations for a patient population (e.g., PACE programs, hospices) → judge by mission. If patient/caregiver advocacy is the primary mission, `1`; if the primary purpose is service delivery, `0`.

### Why both can be 0

A submission can have `clinical_perspective = 0` AND `patient_perspective = 0`. Vendors, health IT companies, policy think tanks, payers, and consulting firms all fit here. These flags are not exhaustive — they identify the two perspective types of substantive interest, not all possible commenter framings.

### Why both can be 1

A clinician writing about their own experience as a patient or caregiver of a family member can score 1 on both flags simultaneously. Use the comment text to determine; if the perspective is explicitly mixed, code both 1.

---

## Topics (top_*): 0 or 1

Code **1** only with **2+ sentences of substantive engagement**, not passing mentions.

| Variable | Description |
|----------|-------------|
| top_regulation | Federal regulatory approach: scope, risk classification, stringency |
| top_evaluation | How to evaluate/monitor AI: pre-market, post-deployment, metrics, standards |
| top_reimbursement | CMS payment: billing codes, coverage, FFS barriers, value-based payment |
| top_transparency | Explainability, interpretability, black-box, model disclosure |
| top_workflow | Clinical workflow: EHR integration, usability, disruption, alert fatigue |
| top_trust | Building/maintaining trust (clinicians, patients, institutions, public) |
| top_safety | Patient safety: adverse events, harms, error reporting, near-misses |
| top_fda_scope | What is/isn’t FDA-regulated: SaMD, CDS exemption, 21st CCA boundaries |
| top_interoperability | Data standards (FHIR, USCDI, TEFCA), exchange, fragmentation |
| top_equity | Algorithmic bias, disparities, underrepresented populations, fairness |
| top_liability | Legal responsibility: malpractice, shared liability, safe harbor |
| top_admin_burden | AI for admin: ambient documentation, prior auth, billing automation |
| top_standards | Accreditation, certification, credentialing, industry-led standards |
| top_workforce | Impact on workers: displacement, augmentation, training, burnout |
| top_privacy | HIPAA, de-identification, re-identification risk, data rights |

---

## Barriers (bar_*): 0 or 1

Code **1** only if the commenter **explicitly frames this as an obstacle or problem** blocking AI adoption.

| Variable | Description |
|----------|-------------|
| bar_reg_uncertainty | Regulatory ambiguity as barrier: unclear rules, uncertain FDA scope |
| bar_liability_risk | Liability uncertainty as barrier: unclear responsibility, chilling effect |
| bar_payment_misalign | Payment disincentives: FFS penalizes efficiency, no coverage pathway |
| bar_data_fragmentation | Data fragmentation: silos, incomplete records, interoperability gaps |
| bar_clinician_trust | Clinician skepticism: black-box, workflow disruption, lack of evidence |
| bar_bias_equity | Bias/equity risk: AI may perpetuate disparities, unrepresentative data |
| bar_privacy_constraints | Privacy rules as barrier: HIPAA limits data sharing for AI |
| bar_cost_resources | Cost/resource constraints: implementation costs, infrastructure, rural burden |

---

## Governance positions

Code the **dominant** position per axis (most emphasis or primary recommendation). If the comment does **not** address the axis, use the **0** code.

### pos_oversight (Human oversight)

| Code | Meaning |
|------|--------|
| H0 | Not addressed |
| H1 | Required for ALL clinical AI — human in every AI-influenced decision |
| H2 | Required for HIGH-RISK only; flexible for low-risk |
| H3 | Recommended but NOT MANDATED — organizations decide |
| H4 | Not always necessary — validated AI can operate autonomously |

### pos_regulation (Regulatory approach)

| Code | Meaning |
|------|--------|
| R0 | Not addressed |
| R1 | New AI-SPECIFIC regulation needed |
| R2 | RISK-TIERED adaptation of existing frameworks |
| R3 | CLARIFY existing rules (guidance, ambiguity) |
| R4 | REDUCE regulatory burden |
| R5 | INDUSTRY SELF-GOVERNANCE preferred |

### pos_liability (Liability)

| Code | Meaning |
|------|--------|
| L0 | Not addressed |
| L1 | Increase DEVELOPER/VENDOR accountability |
| L2 | SHARED or distributed liability framework |
| L3 | CURRENT LAW adequate |
| L4 | FEDERAL SAFE HARBOR for compliant use |
| L5 | NEW LEGAL FRAMEWORK needed |

### pos_reimbursement (Reimbursement)

| Code | Meaning |
|------|--------|
| P0 | Not addressed |
| P1 | AI-SPECIFIC payment pathways (codes, add-ons) |
| P2 | Integrate into VALUE-BASED models |
| P3 | Remove FFS BARRIERS |
| P4 | CMS PILOT/DEMONSTRATION programs |
| P5 | MULTIPLE reforms (no single dominant) |

### pos_interoperability (Interoperability)

| Code | Meaning |
|------|--------|
| D0 | Not addressed |
| D1 | STRENGTHEN/ENFORCE current standards (FHIR, USCDI, TEFCA) |
| D2 | EXPAND data types and access |
| D3 | PATIENT-CONTROLLED access |
| D4 | PREVENT GATEKEEPING by vendors/systems |
| D5 | Build FEDERAL data infrastructure |

### pos_evaluation (Evaluation and monitoring)

| Code | Meaning |
|------|--------|
| E0 | Not addressed |
| E1 | PRE-MARKET validation primary |
| E2 | POST-MARKET monitoring primary |
| E3 | FULL LIFECYCLE (pre- + post-market) required |
| E4 | DEVELOPER/INDUSTRY-LED evaluation |
| E5 | INDEPENDENT/FEDERAL evaluation infrastructure |

---

## Supplementary

- **n_proposals:** Count of specific actionable policy proposals (0–15).
- **has_cfr_citation:** 1 if comment cites specific CFR sections, else 0.
- **evidence_type:** One of: `peer_reviewed`, `industry_data`, `government_data`, `clinical_anecdote`, `none`, `mixed`.
- **rfi_questions:** List of RFI question numbers explicitly addressed (1–10). Empty list if none.
