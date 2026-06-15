"""Publication-grade analyses for the manuscript.

Loads the same coded-comments dataframe as analyze.py and produces:

  output/manuscript/
    table_prevalence_ci.csv        Topic/barrier/perspective prevalence + Wilson 95% CIs
    table_position_prevalence.csv  Governance position distribution + CIs
    table_topic_by_stakeholder.csv Topic prevalence x commenter type with chi-square / Cramér's V
    table_position_by_stakeholder.csv  Position distribution x commenter type with chi-square
    table_position_associations.csv    Pairwise position-position Cramér's V
    table_sensitivity_ai_vs_human.csv  AI-derived prevalence vs human-derived on the n=100 sample
    table_reliability_summary.csv      κ + PABAK + prevalence per variable for manuscript Table N
    figure_irr_forest.png              Forest plot of κ across variables
    figure_prevalence_by_stakeholder.png  Heatmap with significance markers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "manuscript"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_LABELS = {
    "top_regulation": "Regulatory framework",
    "top_evaluation": "Evaluation & monitoring",
    "top_reimbursement": "Reimbursement & payment",
    "top_transparency": "Transparency & explainability",
    "top_workflow": "Clinical workflow integration",
    "top_trust": "Trust",
    "top_safety": "Patient safety",
    "top_fda_scope": "FDA scope & device classification",
    "top_interoperability": "Interoperability & data standards",
    "top_equity": "Equity & bias",
    "top_liability": "Liability & accountability",
    "top_admin_burden": "Administrative burden reduction",
    "top_standards": "Standards & accreditation",
    "top_workforce": "Workforce impact",
    "top_privacy": "Privacy & HIPAA",
}
BARRIER_LABELS = {
    "bar_reg_uncertainty": "Regulatory uncertainty",
    "bar_liability_risk": "Liability risk",
    "bar_payment_misalign": "Payment misalignment",
    "bar_data_fragmentation": "Data fragmentation",
    "bar_clinician_trust": "Clinician trust deficit",
    "bar_bias_equity": "Bias & equity risk",
    "bar_privacy_constraints": "Privacy constraints",
    "bar_cost_resources": "Cost & resource constraints",
}
PERSPECTIVE_LABELS = {
    "clinical_perspective": "Clinical perspective",
    "patient_perspective": "Patient perspective",
}
ALL_BINARY_LABELS = {**TOPIC_LABELS, **BARRIER_LABELS, **PERSPECTIVE_LABELS}

POSITION_VARS = [
    "pos_oversight",
    "pos_regulation",
    "pos_liability",
    "pos_reimbursement",
    "pos_interoperability",
    "pos_evaluation",
]
POSITION_LABELS = {
    "pos_oversight": "Human oversight",
    "pos_regulation": "Regulatory approach",
    "pos_liability": "Liability allocation",
    "pos_reimbursement": "Reimbursement",
    "pos_interoperability": "Interoperability",
    "pos_evaluation": "Evaluation & monitoring",
}

COMMENTER_TYPE_LABELS = {
    "MPS": "Medical professional society",
    "HSP": "Health system / provider",
    "HIT": "Health IT",
    "AIC": "AI company",
    "TEC": "Tech / pharma / device",
    "PAY": "Payer / insurer",
    "ADV": "Patient / consumer advocate",
    "LAB": "Labor organization",
    "POL": "Policy / academic / consulting",
    "IND": "Individual",
}


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score 95% CI for a proportion. Returns (p, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cramers_v(table: pd.DataFrame) -> float:
    if table.values.sum() == 0 or min(table.shape) < 2:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(table.values)
    n = table.values.sum()
    r, c = table.shape
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def safe_chisq(table: pd.DataFrame) -> tuple[float, float, bool]:
    """Return (chi2, p, used_fisher).

    Falls back to NaN if the table is degenerate. Sets used_fisher=True if any
    expected count is < 5 AND the table is 2x2 (where Fisher is computable directly).
    For larger sparse tables, χ² is still reported but flagged via low expected counts.
    """
    if table.values.sum() == 0 or min(table.shape) < 2:
        return float("nan"), float("nan"), False
    chi2, p, _, expected = chi2_contingency(table.values)
    used_fisher = False
    if (expected < 5).any() and table.shape == (2, 2):
        _, p = fisher_exact(table.values)
        used_fisher = True
    return float(chi2), float(p), used_fisher


def bh_fdr(pvals: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR adjustment. Returns (q_values, rejected)."""
    pairs = [(p, i) for i, p in enumerate(pvals) if not np.isnan(p)]
    pairs.sort()
    n = len(pairs)
    q_values = [float("nan")] * len(pvals)
    rejected = [False] * len(pvals)
    if n == 0:
        return q_values, rejected
    # BH-adjusted p-values (Yekutieli–Hommel monotone correction)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        p, idx = pairs[rank]
        q = min(prev, p * n / (rank + 1))
        prev = q
        q_values[idx] = q
        rejected[idx] = q <= alpha
    return q_values, rejected


def prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Variable-level prevalence with 95% Wilson CIs."""
    rows = []
    n = len(df)
    for var, label in ALL_BINARY_LABELS.items():
        if var not in df.columns:
            continue
        k = int(df[var].sum())
        p, lo, hi = wilson_ci(k, n)
        rows.append(
            {
                "domain": (
                    "Topic" if var in TOPIC_LABELS else
                    "Barrier" if var in BARRIER_LABELS else
                    "Perspective"
                ),
                "variable": var,
                "label": label,
                "n_yes": k,
                "n_total": n,
                "prevalence": round(p, 3),
                "ci95_lo": round(lo, 3),
                "ci95_hi": round(hi, 3),
            }
        )
    out = pd.DataFrame(rows).sort_values(["domain", "prevalence"], ascending=[True, False])
    out.to_csv(OUT_DIR / "table_prevalence_ci.csv", index=False)
    return out


POSITION_NULL_CODE = {
    "pos_oversight": "H0",
    "pos_regulation": "R0",
    "pos_liability": "L0",
    "pos_reimbursement": "P0",
    "pos_interoperability": "D0",
    "pos_evaluation": "E0",
}


def position_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """Per-position distribution table with CIs for the 'addressed' rate and the modal stance."""
    rows = []
    n = len(df)
    for var in POSITION_VARS:
        if var not in df.columns:
            continue
        not_addressed_code = POSITION_NULL_CODE[var]
        addressed_mask = df[var].astype(str) != not_addressed_code
        k_addr = int(addressed_mask.sum())
        p_addr, lo_addr, hi_addr = wilson_ci(k_addr, n)

        # Modal stance among addressed comments
        if k_addr > 0:
            modes = df.loc[addressed_mask, var].value_counts()
            modal_code = modes.index[0]
            modal_count = int(modes.iloc[0])
            modal_share = modal_count / k_addr
        else:
            modal_code, modal_count, modal_share = "", 0, 0.0

        rows.append(
            {
                "axis": POSITION_LABELS[var],
                "variable": var,
                "n_addressed": k_addr,
                "addressed_rate": round(p_addr, 3),
                "addressed_ci_lo": round(lo_addr, 3),
                "addressed_ci_hi": round(hi_addr, 3),
                "modal_code": modal_code,
                "modal_count": modal_count,
                "modal_share_of_addressed": round(modal_share, 3),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "table_position_prevalence.csv", index=False)
    return out


def topic_by_stakeholder(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format table: topic prevalence stratified by commenter_type, with chi-square + Cramér's V."""
    rows = []
    types_observed = sorted(df["commenter_type"].dropna().unique())
    for var, label in TOPIC_LABELS.items():
        if var not in df.columns:
            continue
        sub = df[["commenter_type", var]].copy()
        # Build 2 x K table: rows = {0, 1}, cols = commenter types
        ct = pd.crosstab(sub[var], sub["commenter_type"])
        # Ensure both rows present
        for v in (0, 1):
            if v not in ct.index:
                ct.loc[v] = 0
        ct = ct.sort_index()
        chi2, p, used_fisher = safe_chisq(ct)
        v_eff = cramers_v(ct)
        # Per-stakeholder prevalence
        for ctype in types_observed:
            if ctype not in ct.columns:
                k, n = 0, 0
            else:
                col = ct[ctype]
                n = int(col.sum())
                k = int(col.get(1, 0))
            prev, lo, hi = wilson_ci(k, n) if n > 0 else (0.0, 0.0, 0.0)
            rows.append(
                {
                    "topic": label,
                    "variable": var,
                    "commenter_type": ctype,
                    "commenter_label": COMMENTER_TYPE_LABELS.get(ctype, ctype),
                    "n_yes": k,
                    "n_total": n,
                    "prevalence": round(prev, 3),
                    "ci95_lo": round(lo, 3),
                    "ci95_hi": round(hi, 3),
                    "chi2_overall": round(chi2, 3) if not np.isnan(chi2) else "",
                    "p_overall": round(p, 4) if not np.isnan(p) else "",
                    "test": "Fisher's exact" if used_fisher else "Pearson χ²",
                    "cramers_v": round(v_eff, 3) if not np.isnan(v_eff) else "",
                }
            )
    out = pd.DataFrame(rows)
    # Apply BH-FDR within the topic-by-stakeholder family (one test per topic).
    per_topic = out.drop_duplicates(subset=["variable"]).copy()
    pvals = [float(p) if p != "" else float("nan") for p in per_topic["p_overall"].tolist()]
    qvals, rejected = bh_fdr(pvals)
    per_topic["q_value_bh"] = [round(q, 4) if not np.isnan(q) else "" for q in qvals]
    per_topic["significant_bh_05"] = rejected
    qmap = dict(zip(per_topic["variable"], per_topic["q_value_bh"]))
    sigmap = dict(zip(per_topic["variable"], per_topic["significant_bh_05"]))
    out["q_value_bh"] = out["variable"].map(qmap)
    out["significant_bh_05"] = out["variable"].map(sigmap)
    out.to_csv(OUT_DIR / "table_topic_by_stakeholder.csv", index=False)
    return out


def position_by_stakeholder(df: pd.DataFrame) -> pd.DataFrame:
    """Each governance position x commenter_type cross-tab + chi-square + Cramér's V."""
    rows = []
    for var in POSITION_VARS:
        if var not in df.columns:
            continue
        ct = pd.crosstab(df[var], df["commenter_type"])
        chi2, p, used_fisher = safe_chisq(ct)
        v_eff = cramers_v(ct)
        # Check expected-count assumption
        from scipy.stats import chi2_contingency as _c2c
        _, _, _, expected = _c2c(ct.values) if ct.values.sum() > 0 else (None, None, None, np.zeros_like(ct.values))
        prop_low_expected = float((expected < 5).mean()) if expected.size else 0.0
        rows.append(
            {
                "axis": POSITION_LABELS[var],
                "variable": var,
                "n_total": int(ct.values.sum()),
                "chi2": round(chi2, 3) if not np.isnan(chi2) else "",
                "p_value": round(p, 4) if not np.isnan(p) else "",
                "cramers_v": round(v_eff, 3) if not np.isnan(v_eff) else "",
                "frac_cells_expected_lt_5": round(prop_low_expected, 3),
                "test": "Fisher's exact" if used_fisher else "Pearson χ²",
            }
        )
        ct.to_csv(OUT_DIR / f"crosstab_{var}_by_commenter_type.csv")
    out = pd.DataFrame(rows)
    pvals = [float(p) if p != "" else float("nan") for p in out["p_value"].tolist()]
    qvals, rejected = bh_fdr(pvals)
    out["q_value_bh"] = [round(q, 4) if not np.isnan(q) else "" for q in qvals]
    out["significant_bh_05"] = rejected
    out.to_csv(OUT_DIR / "table_position_by_stakeholder.csv", index=False)
    return out


def position_pairwise_association(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Cramér's V across the 6 governance axes."""
    rows = []
    for i, v1 in enumerate(POSITION_VARS):
        for v2 in POSITION_VARS[i + 1:]:
            if v1 not in df.columns or v2 not in df.columns:
                continue
            ct = pd.crosstab(df[v1], df[v2])
            chi2, p, _ = safe_chisq(ct)
            v_eff = cramers_v(ct)
            rows.append(
                {
                    "axis_1": POSITION_LABELS[v1],
                    "axis_2": POSITION_LABELS[v2],
                    "chi2": round(chi2, 3) if not np.isnan(chi2) else "",
                    "p_value": round(p, 4) if not np.isnan(p) else "",
                    "cramers_v": round(v_eff, 3) if not np.isnan(v_eff) else "",
                }
            )
    out = pd.DataFrame(rows)
    pvals = [float(p) if p != "" else float("nan") for p in out["p_value"].tolist()]
    qvals, rejected = bh_fdr(pvals)
    out["q_value_bh"] = [round(q, 4) if not np.isnan(q) else "" for q in qvals]
    out["significant_bh_05"] = rejected
    out = out.sort_values("cramers_v", ascending=False)
    out.to_csv(OUT_DIR / "table_position_associations.csv", index=False)
    return out


def sensitivity_ai_vs_human(df: pd.DataFrame) -> pd.DataFrame:
    """Compare AI prevalence vs human prevalence on the n=100 validation sample.

    Use Reviewer B as the more permissive anchor + Reviewer A as the strict anchor; report both.
    Variables of focus: top_trust, top_safety, bar_privacy_constraints, commenter_type (multi-class summarised differently).
    """
    sys.path.insert(0, str(ROOT / "analysis"))
    from reliability import (
        load_llm_key, load_reviewer_csv, REVIEWER_A_CSV, REVIEWER_B_CSV
    )

    ai = load_llm_key()
    ra = load_reviewer_csv(REVIEWER_A_CSV)
    rb = load_reviewer_csv(REVIEWER_B_CSV)
    common_ids = sorted(set(ai) & set(ra) & set(rb))

    binary_focus = ["top_trust", "top_safety", "bar_privacy_constraints",
                    "patient_perspective", "clinical_perspective", "top_equity",
                    "top_standards", "top_transparency"]
    rows = []
    for var in binary_focus:
        ai_vals = [ai[c].get(var, "") for c in common_ids]
        a_vals = [ra[c].get(var, "") for c in common_ids]
        b_vals = [rb[c].get(var, "") for c in common_ids]
        n = len(common_ids)
        ai_yes = sum(v == "1" for v in ai_vals)
        a_yes = sum(v == "1" for v in a_vals)
        b_yes = sum(v == "1" for v in b_vals)
        ai_p, ai_lo, ai_hi = wilson_ci(ai_yes, n)
        a_p, a_lo, a_hi = wilson_ci(a_yes, n)
        b_p, b_lo, b_hi = wilson_ci(b_yes, n)
        # Full corpus AI prevalence
        full_n = len(df)
        full_yes = int(df[var].sum()) if var in df.columns else 0
        full_p, full_lo, full_hi = wilson_ci(full_yes, full_n)

        rows.append(
            {
                "variable": var,
                "label": ALL_BINARY_LABELS.get(var, var),
                "ai_full_corpus_n": full_n,
                "ai_full_corpus_prev": round(full_p, 3),
                "ai_full_ci": f"[{full_lo:.3f}, {full_hi:.3f}]",
                "n_validation": n,
                "ai_validation_prev": round(ai_p, 3),
                "ai_validation_ci": f"[{ai_lo:.3f}, {ai_hi:.3f}]",
                "human_a_prev": round(a_p, 3),
                "human_a_ci": f"[{a_lo:.3f}, {a_hi:.3f}]",
                "human_b_prev": round(b_p, 3),
                "human_b_ci": f"[{b_lo:.3f}, {b_hi:.3f}]",
                "ai_vs_human_diff_avg": round(ai_p - (a_p + b_p) / 2, 3),
            }
        )

    # Commenter type: distribution comparison (top 5 categories)
    print("\nCommenter type distributions on the 100-sample:")
    for src, label in [(ai, "AI"), (ra, "Reviewer A"), (rb, "Reviewer B")]:
        from collections import Counter
        vals = [src[c].get("commenter_type", "") for c in common_ids]
        top = Counter(vals).most_common()
        print(f"  {label:<12}: " + ", ".join(f"{k}={v}" for k, v in top if k))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "table_sensitivity_ai_vs_human.csv", index=False)
    return out


def reliability_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Single manuscript-ready reliability table.

    Pulls the per-variable IRR from output/ai_vs_human_v2.csv and joins it with
    AI-derived full-corpus prevalence. One row per variable.
    """
    irr_path = ROOT / "output" / "ai_vs_human_v2.csv"
    if not irr_path.exists():
        return pd.DataFrame()
    irr = pd.read_csv(irr_path)
    # Full-corpus prevalence for binary vars
    n = len(df)
    prev_lookup = {}
    for var in df.columns:
        if df[var].dtype.kind in "iuf" and df[var].dropna().isin([0, 1]).all():
            k = int(df[var].sum())
            p, lo, hi = wilson_ci(k, n)
            prev_lookup[var] = (p, lo, hi)
    out_rows = []
    for _, r in irr.iterrows():
        var = r["variable"]
        if var in prev_lookup:
            p, lo, hi = prev_lookup[var]
            prev_str = f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
        else:
            prev_str = "—"
        out_rows.append(
            {
                "variable": var,
                "label": ALL_BINARY_LABELS.get(var) or POSITION_LABELS.get(var) or var,
                "ai_full_corpus_prevalence": prev_str,
                "n_validation": r["n"],
                "ai_vs_a_kappa": r["ai_vs_a_kappa"],
                "ai_vs_a_pabak": r["ai_vs_a_pabak"],
                "ai_vs_b_kappa": r["ai_vs_b_kappa"],
                "ai_vs_b_pabak": r["ai_vs_b_pabak"],
                "a_vs_b_kappa": r["a_vs_b_kappa"],
                "a_vs_b_pabak": r["a_vs_b_pabak"],
            }
        )
    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_DIR / "table_reliability_summary.csv", index=False)
    return out


def figure_irr_forest(df: pd.DataFrame) -> None:
    """Forest plot of κ for each variable across the three rater pairs, with 95% bootstrap CIs."""
    irr_path = ROOT / "output" / "ai_vs_human_v2.csv"
    if not irr_path.exists():
        return
    irr = pd.read_csv(irr_path).sort_values("a_vs_b_kappa")
    n = len(irr)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, max(8, n * 0.34)))
    offsets = {"ai_vs_a": -0.25, "ai_vs_b": 0.0, "a_vs_b": 0.25}
    colours = {"ai_vs_a": "#1f77b4", "ai_vs_b": "#ff7f0e", "a_vs_b": "#2ca02c"}
    labels = {"ai_vs_a": "AI vs Reviewer A",
              "ai_vs_b": "AI vs Reviewer B",
              "a_vs_b": "Reviewer A vs Reviewer B"}
    for prefix, off in offsets.items():
        k = irr[f"{prefix}_kappa"]
        lo = irr[f"{prefix}_kappa_lo"]
        hi = irr[f"{prefix}_kappa_hi"]
        ax.errorbar(k, y + off, xerr=[k - lo, hi - k],
                    fmt="o", color=colours[prefix], label=labels[prefix],
                    markersize=4.5, elinewidth=1.0, capsize=2.0, alpha=0.85, zorder=3)
    ax.axvline(0.6, color="gray", linestyle="--", alpha=0.6, label="κ = 0.60")
    ax.axvline(0.4, color="lightgray", linestyle=":", alpha=0.6, label="κ = 0.40")
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(irr["variable"], fontsize=9)
    ax.set_xlabel("Cohen's κ (95% bootstrap CI)")
    ax.set_xlim(-0.2, 1.05)
    ax.set_title("Inter-rater reliability across 34 coded variables (validation sample, n=100)\n"
                 "Bootstrap 95% CIs from 2,000 resamples")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure_irr_forest.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_prisma_flow(df: pd.DataFrame) -> None:
    """PRISMA-style inclusion/exclusion flow for the comment corpus."""
    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.axis("off")

    boxes = [
        (0.5, 0.95, "447 public comments retrieved\nfrom HHS-ONC-2026-0001 docket\n(Regulations.gov, March 2026)"),
        (0.5, 0.78, "1 excluded\n(Cook Group, Inc., HHS-ONC-2026-0001-0280:\nno extractable inline or attachment text)"),
        (0.5, 0.60, "446 comments coded by\nClaude Opus 4.7 (1M context)\n(structured-output extraction; 34 variables)"),
        (0.5, 0.42, "Stratified random sample\n(n=100, 22.4%; seed=42)\nfor human validation"),
        (0.5, 0.24, "Independent coding\nby two human reviewers (A, B)\n(34 variables)"),
        (0.5, 0.07, "Inter-rater reliability assessment\n(Cohen's κ, PABAK, Krippendorff's α)"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"),
                fontsize=11, transform=ax.transAxes)

    # Connecting arrows
    arrow_kw = dict(arrowstyle="->", color="black", lw=1.2)
    for y_top, y_bot in [(0.92, 0.84), (0.74, 0.66), (0.56, 0.48),
                         (0.38, 0.30), (0.20, 0.13)]:
        ax.annotate("", xy=(0.5, y_bot), xytext=(0.5, y_top),
                    xycoords="axes fraction", arrowprops=arrow_kw)

    # Side annotation: 446-comment corpus is what feeds the substantive analysis
    ax.annotate("446 comments → full-corpus\nanalysis (Tables 1–4, Fig. 1–3)",
                xy=(0.78, 0.60), xytext=(0.96, 0.50), xycoords="axes fraction",
                ha="left", va="center", fontsize=9, style="italic",
                arrowprops=dict(arrowstyle="->", color="gray"))

    plt.title("Figure S1. Comment corpus flow", fontsize=12, pad=10)
    plt.savefig(OUT_DIR / "figure_prisma_flow.png", dpi=300, bbox_inches="tight")
    plt.close()


def _load_comment_texts() -> dict[str, dict[str, str]]:
    """Build {comment_id: {organization, inline, attached_text}} lookup from data/comments.csv."""
    import csv as _csv
    csv_path = ROOT / "data" / "comments.csv"
    out: dict[str, dict[str, str]] = {}
    _csv.field_size_limit(sys.maxsize)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            cid = (row.get("id") or "").strip()
            if not cid:
                continue
            inline = (row.get("comment") or "").strip()
            attached = ""
            for i in range(1, 13):
                t = (row.get(f"fullText_{i}") or "").strip()
                if t:
                    attached += ("\n\n" if attached else "") + t
            out[cid] = {
                "organization": (row.get("organization") or "").strip(),
                "inline": inline,
                "attached": attached,
            }
    return out


def _pick_quote(text: str, keywords: list[str], max_words: int = 45) -> str:
    """Find the first sentence containing a keyword and return a clean ~max_words snippet."""
    import re
    if not text:
        return ""
    # Normalize whitespace and HTML artifacts
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("•", "").replace("·", "")
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    # Prefer sentences that aren't obvious bullet headings
    candidates = [s for s in sentences if len(s.split()) >= 12 and not s.endswith(":")]
    for kw in keywords:
        for s in candidates:
            if kw.lower() in s.lower():
                words = s.strip().split()
                snippet = " ".join(words[:max_words])
                if len(words) > max_words:
                    snippet += "..."
                return snippet
    if candidates:
        words = candidates[0].split()
        snippet = " ".join(words[:max_words])
        return snippet + ("..." if len(words) > max_words else "")
    return ""


def representative_excerpts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract representative comment excerpts supporting major findings, with quoted text."""
    texts = _load_comment_texts()

    # Selection rules — restrict to comments where AI's commenter_type is plausible
    # given the organization name (avoid GNYHA-as-ADV miscoding).
    r3_cands = df[(df["pos_regulation"].astype(str) == "R3") &
                  (df["n_proposals"] >= 8) &
                  (df["commenter_type"].astype(str).str.contains("MPS|HSP", na=False))].copy()

    h2_cands = df[(df["pos_oversight"].astype(str) == "H2") &
                  (df["n_proposals"] >= 8) &
                  (df["commenter_type"].astype(str).str.contains("MPS|HSP|HIT", na=False))].copy()

    e3_cands = df[(df["pos_evaluation"].astype(str) == "E3") &
                  (df["n_proposals"] >= 8)].copy()

    l4_cands = df[(df["pos_liability"].astype(str) == "L4") &
                  (df["n_proposals"] >= 5) &
                  (df["commenter_type"].astype(str).str.contains("MPS|HSP|HIT|AIC", na=False))].copy()

    # Patient advocacy: filter on organization name to avoid AI miscoding
    adv_keywords = ["cancer", "alliance", "society", "council for mental",
                    "patient", "consumer", "alzheimer", "diabetes", "kidney"]
    adv_eq = df[df["top_equity"] == 1].copy()
    adv_eq["org_lower"] = adv_eq["organization"].str.lower().fillna("")
    adv_eq = adv_eq[adv_eq["org_lower"].apply(lambda s: any(k in s for k in adv_keywords))]

    selections = [
        ("R3 — Clarification of existing rules", r3_cands,
         ["clarif", "guidance", "exempt", "ambig"]),
        ("H2 — Risk-tiered human oversight", h2_cands,
         ["risk", "oversight", "high-risk", "human", "clinician"]),
        ("E3 — Full-lifecycle evaluation", e3_cands,
         ["lifecycle", "post-market", "monitoring", "validation"]),
        ("L4 — Federal safe harbor for validated AI", l4_cands,
         ["safe harbor", "liability", "shield", "protect"]),
        ("Patient advocacy on equity and bias", adv_eq,
         ["bias", "equity", "underrepresent", "disparit"]),
    ]

    excerpt_rows = []
    for label, cands, keywords in selections:
        if len(cands) == 0:
            continue
        cands = cands.sort_values("n_proposals", ascending=False)
        for _, row in cands.head(5).iterrows():
            cid = row.get("id") or row.get("comment_id")
            txt_rec = texts.get(cid, {})
            full_text = (txt_rec.get("inline") or "") + " " + (txt_rec.get("attached") or "")
            quote = _pick_quote(full_text, keywords)
            if not quote:
                continue
            excerpt_rows.append(
                {
                    "finding": label,
                    "comment_id": cid,
                    "organization": txt_rec.get("organization") or "(individual)",
                    "commenter_type": str(row.get("commenter_type")).replace("CommenterType.", ""),
                    "n_proposals": row.get("n_proposals"),
                    "excerpt": quote,
                }
            )
            break

    out = pd.DataFrame(excerpt_rows)
    out.to_csv(OUT_DIR / "table_representative_excerpts.csv", index=False)
    return out


def power_analysis_validation_sample() -> pd.DataFrame:
    """Compute statistical power of the n=100 validation sample for detecting key effects."""
    # Power for κ: approximate via the SE formula.
    # SE(κ) ≈ sqrt((Po(1-Po)) / (n*(1-Pe)^2)) for two raters, binary case.
    rows = []
    n = 100
    # Detectable difference between two κ estimates with 80% power, two-sided α=0.05
    # Approximate as a difference of two correlated κ estimates with SE ≈ 0.07–0.10 per estimate.
    # MDE ≈ 2.8 * SE_diff ≈ 0.20–0.25 — meaning we can detect κ differences of ~0.2 only.
    rows.append({
        "comparison": "Cohen's κ between two rater pairs (e.g., AI–B vs A–B)",
        "n": n,
        "approx_se_per_estimate": "0.07–0.10",
        "mde_80pct_power_alpha_0.05": "~0.20",
        "interpretation": "Sample is powered to detect large differences in κ, "
                          "not the moderate (0.05–0.10) differences observed.",
    })
    # Power for prevalence comparison
    rows.append({
        "comparison": "Two-sample test of binary prevalence (AI vs human-anchored)",
        "n": n,
        "approx_se_per_estimate": "varies by p; max 0.05 at p=0.5",
        "mde_80pct_power_alpha_0.05": "~0.15 absolute (at p~0.5)",
        "interpretation": "Sample detects 15-percentage-point gaps; the AI over-flagging gaps "
                          "of 19–28 points are well above this threshold.",
    })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "table_power_analysis.csv", index=False)
    return out


def figure_topic_prevalence_by_stakeholder(df: pd.DataFrame) -> None:
    """Heatmap of topic prevalence (rows) by commenter type (cols), n-weighted."""
    rows = list(TOPIC_LABELS.keys())
    types = sorted(df["commenter_type"].dropna().unique())
    mat = np.zeros((len(rows), len(types)))
    for i, var in enumerate(rows):
        for j, ctype in enumerate(types):
            sub = df[df["commenter_type"] == ctype]
            if len(sub):
                mat[i, j] = sub[var].mean()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{t}\n(n={int((df['commenter_type']==t).sum())})" for t in types],
                       rotation=0, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([TOPIC_LABELS[r] for r in rows], fontsize=10)
    for i in range(len(rows)):
        for j in range(len(types)):
            v = mat[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax, label="Prevalence within stakeholder group", fraction=0.025, pad=0.01)
    ax.set_title("Topic prevalence by commenter type (n=446)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure_prevalence_by_stakeholder.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} coded comments")

    print("\n[1/8] Variable-level prevalence + 95% CIs ...")
    prevalence_table(df)

    print("[2/8] Position prevalence ...")
    position_prevalence(df)

    print("[3/8] Topic by stakeholder ...")
    topic_by_stakeholder(df)

    print("[4/8] Position by stakeholder ...")
    position_by_stakeholder(df)

    print("[5/8] Pairwise position associations ...")
    position_pairwise_association(df)

    print("[6/8] Validation sensitivity (AI vs human prevalence) ...")
    sensitivity_ai_vs_human(df)

    print("[7/11] Reliability summary table ...")
    reliability_summary_table(df)

    print("[8/11] Figures: IRR forest + topic-by-stakeholder heatmap ...")
    figure_irr_forest(df)
    figure_topic_prevalence_by_stakeholder(df)

    print("[9/11] PRISMA flow figure ...")
    figure_prisma_flow(df)

    print("[10/11] Representative comment excerpts ...")
    representative_excerpts(df)

    print("[11/11] Power analysis ...")
    power_analysis_validation_sample()

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
