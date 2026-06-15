from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class CommenterType(str, Enum):
    MPS = "MPS"
    HSP = "HSP"
    HIT = "HIT"
    AIC = "AIC"
    TEC = "TEC"
    PAY = "PAY"
    ADV = "ADV"
    LAB = "LAB"
    POL = "POL"
    IND = "IND"


class OversightPosition(str, Enum):
    H0 = "H0"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"

class RegulatoryPosition(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"

class LiabilityPosition(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"

class ReimbursementPosition(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"

class InteroperabilityPosition(str, Enum):
    D0 = "D0"
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"
    D5 = "D5"

class EvaluationPosition(str, Enum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"

class EvidenceType(str, Enum):
    PEER_REVIEWED = "peer_reviewed"
    INDUSTRY_DATA = "industry_data"
    GOVERNMENT_DATA = "government_data"
    CLINICAL_ANECDOTE = "clinical_anecdote"
    NONE = "none"
    MIXED = "mixed"


class TopicCoverage(BaseModel):
    top_regulation: int = Field(description="Federal regulatory approach to clinical AI: scope, risk classification, stringency, mechanism")
    top_evaluation: int = Field(description="How to evaluate/monitor AI: pre-market validation, post-deployment monitoring, metrics, standards")
    top_reimbursement: int = Field(description="CMS payment policy: billing codes, coverage, FFS barriers, value-based payment")
    top_transparency: int = Field(description="Explainability, interpretability, black-box concerns, model disclosure")
    top_workflow: int = Field(description="Clinical workflow integration: EHR integration, usability, workflow disruption, alert fatigue")
    top_trust: int = Field(description="Building/maintaining trust among clinicians, patients, institutions, or the public")
    top_safety: int = Field(description="Patient safety: adverse events, known harms, error reporting, near-misses")
    top_fda_scope: int = Field(description="What is/isn't FDA-regulated: SaMD, CDS exemption, 21st Century Cures Act boundaries")
    top_interoperability: int = Field(description="Data standards (FHIR, USCDI, TEFCA), exchange, fragmentation, access")
    top_equity: int = Field(description="Algorithmic bias, health disparities, underrepresented populations, fairness")
    top_liability: int = Field(description="Legal responsibility for AI errors: malpractice, shared liability, safe harbor")
    top_admin_burden: int = Field(description="AI for admin tasks: ambient documentation, prior auth, billing automation")
    top_standards: int = Field(description="Accreditation, certification, credentialing, industry-led standards")
    top_workforce: int = Field(description="Impact on workers: displacement, augmentation, training, burnout, staffing")
    top_privacy: int = Field(description="HIPAA, de-identification, re-identification risk, data rights")


class BarriersIdentified(BaseModel):
    bar_reg_uncertainty: int = Field(description="Regulatory ambiguity as barrier: unclear rules, unpredictable oversight, uncertain FDA scope")
    bar_liability_risk: int = Field(description="Liability uncertainty as barrier: unclear responsibility, chilling effect")
    bar_payment_misalign: int = Field(description="Payment disincentives as barrier: FFS penalizes efficiency, no coverage pathway")
    bar_data_fragmentation: int = Field(description="Data fragmentation as barrier: silos, incomplete records, interoperability gaps")
    bar_clinician_trust: int = Field(description="Clinician skepticism as barrier: black-box concerns, workflow disruption, lack of evidence")
    bar_bias_equity: int = Field(description="Bias/equity risk as barrier: AI may perpetuate disparities, unrepresentative data")
    bar_privacy_constraints: int = Field(description="Privacy rules as barrier: HIPAA limits data sharing for AI development")
    bar_cost_resources: int = Field(description="Cost/resource constraints as barrier: implementation costs, infrastructure gaps, rural burden")


class GovernancePositions(BaseModel):
    pos_oversight: OversightPosition = Field(description="Human oversight requirement for clinical AI")
    pos_regulation: RegulatoryPosition = Field(description="Federal regulatory approach to clinical AI")
    pos_liability: LiabilityPosition = Field(description="Liability allocation for AI-influenced care")
    pos_reimbursement: ReimbursementPosition = Field(description="Payment and reimbursement approach")
    pos_interoperability: InteroperabilityPosition = Field(description="Data interoperability priority")
    pos_evaluation: EvaluationPosition = Field(description="AI evaluation and monitoring model")


class Supplementary(BaseModel):
    n_proposals: int = Field(description="Count of specific actionable policy proposals (0-15)")
    has_cfr_citation: int = Field(description="1 if comment cites specific CFR sections, 0 otherwise")
    evidence_type: EvidenceType = Field(description="Primary type of evidence cited")
    rfi_questions: list[int] = Field(description="Which RFI questions (1-10) explicitly addressed. Empty list if none.")


class CommentExtraction(BaseModel):
    commenter_type: CommenterType = Field(description="Primary organizational type of the commenter")
    clinical_perspective: bool = Field(
        description=(
            "True if the commenter is either (a) an individual writing from their own "
            "direct clinical practice experience, OR (b) an organization whose primary "
            "mission is representing clinicians (clinician professional societies, "
            "nursing associations, specialty colleges — e.g., AMA, ASN, AUA, ANA, AMIA). "
            "False for health systems, vendors, patient advocacy orgs, policy/academic "
            "organizations, and individuals not writing from clinical practice."
        )
    )
    patient_perspective: bool = Field(
        description=(
            "True if the commenter is either (a) an individual writing from their own or "
            "a family member's experience as patient or caregiver, OR (b) an organization "
            "whose primary mission is patient or caregiver representation (e.g., ACS Cancer "
            "Action Network, Susan G. Komen, AARP, National Council for Mental Wellbeing, "
            "Community Oncology Alliance, American Kidney Fund). False for clinician "
            "professional societies (AUA, AMA, ASN — even when they discuss patient impact "
            "they speak AS clinicians), vendors / health IT, health systems, consulting firms. "
            "The test: who is the org organized to represent, not what the comment discusses."
        )
    )
    organization_from_document: str | None = Field(
        default=None,
        description="Organization name as explicitly stated in the full text of the comment (inline text and/or extracted text from attachments). Empty if not stated or ambiguous."
    )
    topics: TopicCoverage
    barriers: BarriersIdentified
    positions: GovernancePositions
    supplementary: Supplementary
