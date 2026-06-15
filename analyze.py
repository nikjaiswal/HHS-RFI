"""
Generate all tables and figures for the paper.

Usage:
    python analyze.py
    python analyze.py --tables-only
    python analyze.py --figures-only
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact

from config import OUTPUT_JSONL, TABLES_DIR, FIGURES_DIR, PROJECT_ROOT
from models import CommentExtraction


# ---- Data Loading ----

def load_data() -> pd.DataFrame:
    """Load coded comments into a flat DataFrame."""
    records = []
    with open(OUTPUT_JSONL) as f:
        for line in f:
            raw = json.loads(line)
            meta = raw.pop("_meta")
            extraction = CommentExtraction.model_validate(raw)
            d = extraction.model_dump()

            # Flatten nested models
            flat = {}
            flat.update(meta)
            flat["commenter_type"] = d["commenter_type"]
            flat["clinical_perspective"] = int(d["clinical_perspective"])
            flat["patient_perspective"] = int(d["patient_perspective"])
            flat["organization_from_document"] = d.get("organization_from_document") or ""

            for key, val in d["topics"].items():
                flat[key] = val
            for key, val in d["barriers"].items():
                flat[key] = val
            for key, val in d["positions"].items():
                flat[key] = val

            flat["n_proposals"] = d["supplementary"]["n_proposals"]
            flat["has_cfr_citation"] = d["supplementary"]["has_cfr_citation"]
            flat["evidence_type"] = d["supplementary"]["evidence_type"]
            flat["n_rfi_questions"] = len(d["supplementary"]["rfi_questions"])

            records.append(flat)

    return pd.DataFrame(records)


# ---- Label Mappings ----

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

TYPE_LABELS = {
    "MPS": "Medical professional society",
    "AIC": "AI company",
    "IND": "Individual",
    "POL": "Policy/academic/consulting",
    "HSP": "Health system/provider",
    "HIT": "Health IT",
    "TEC": "Tech/pharma/device",
    "ADV": "Patient/consumer advocate",
    "PAY": "Payer/insurer",
    "LAB": "Labor organization",
}

EVIDENCE_TYPE_LABELS = {
    "peer_reviewed": "Peer-reviewed literature",
    "industry_data": "Industry data",
    "government_data": "Government data",
    "clinical_anecdote": "Clinical anecdote / experience",
    "none": "None cited",
    "mixed": "Mixed",
}

# Governance axis short names (for tables/figures)
AXIS_LABELS = {
    "pos_oversight": "Human oversight",
    "pos_regulation": "Regulation",
    "pos_liability": "Liability",
    "pos_reimbursement": "Reimbursement",
    "pos_interoperability": "Interoperability",
    "pos_evaluation": "Evaluation",
}

# Full position label maps for each axis
POSITION_LABELS = {
    "pos_oversight": {
        "H0": "Not addressed", "H1": "Required for all",
        "H2": "High-risk only", "H3": "Recommended, not mandated",
        "H4": "Not always necessary",
    },
    "pos_regulation": {
        "R0": "Not addressed", "R1": "New AI-specific regulation",
        "R2": "Risk-tiered adaptation", "R3": "Clarify existing rules",
        "R4": "Reduce burden", "R5": "Industry self-governance",
    },
    "pos_liability": {
        "L0": "Not addressed", "L1": "Increase developer liability",
        "L2": "Shared framework", "L3": "Current law adequate",
        "L4": "Federal safe harbor", "L5": "New legal framework",
    },
    "pos_reimbursement": {
        "P0": "Not addressed", "P1": "AI-specific payment",
        "P2": "Value-based models", "P3": "Remove FFS barriers",
        "P4": "CMS demonstrations", "P5": "Multiple reforms",
    },
    "pos_interoperability": {
        "D0": "Not addressed", "D1": "Strengthen current standards",
        "D2": "Expand data types", "D3": "Patient-controlled access",
        "D4": "Prevent gatekeeping", "D5": "Federal infrastructure",
    },
    "pos_evaluation": {
        "E0": "Not addressed", "E1": "Pre-market primary",
        "E2": "Post-market primary", "E3": "Full lifecycle",
        "E4": "Industry-led", "E5": "Independent/federal infrastructure",
    },
}


# ---- Table Generators ----

def table1_commenter_characteristics(df: pd.DataFrame, tables_dir: Path | None = None):
    """Table 1: Commenter type distribution with characteristics."""
    tables_dir = tables_dir or TABLES_DIR
    rows = []
    for ctype in df["commenter_type"].value_counts().index:
        subset = df[df["commenter_type"] == ctype]
        rows.append({
            "commenter_type": ctype,
            "label": TYPE_LABELS.get(ctype, ctype),
            "n": len(subset),
            "pct": round(len(subset) / len(df) * 100, 1),
            "mean_text_length": int(subset["total_chars"].mean()),
            "pct_clinical": round(subset["clinical_perspective"].mean() * 100, 1),
            "pct_patient": round(subset["patient_perspective"].mean() * 100, 1),
            "mean_topics_discussed": round(
                subset[[c for c in subset.columns if c.startswith("top_")]].sum(axis=1).mean(), 1
            ),
            "mean_barriers_identified": round(
                subset[[c for c in subset.columns if c.startswith("bar_")]].sum(axis=1).mean(), 1
            ),
        })
    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "table1_commenter_characteristics.csv", index=False)
    print(f"  Table 1: {len(result)} commenter types")
    return result


def table2_topic_coverage(df: pd.DataFrame, tables_dir: Path | None = None):
    """Table 2: Topic coverage overall and by commenter type."""
    tables_dir = tables_dir or TABLES_DIR
    topic_cols = [c for c in df.columns if c.startswith("top_")]
    rows = []
    for col in topic_cols:
        row = {
            "topic": col,
            "label": TOPIC_LABELS.get(col, col),
            "overall_n": int(df[col].sum()),
            "overall_pct": round(df[col].mean() * 100, 1),
        }
        for ctype in sorted(df["commenter_type"].unique()):
            subset = df[df["commenter_type"] == ctype]
            row[f"pct_{ctype}"] = round(subset[col].mean() * 100, 1) if len(subset) > 0 else 0
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("overall_pct", ascending=False)
    result.to_csv(tables_dir / "table2_topic_coverage.csv", index=False)
    print(f"  Table 2: {len(result)} topics × {df['commenter_type'].nunique()} types")
    return result


def table3_barriers(df: pd.DataFrame, tables_dir: Path | None = None):
    """Table 3: Barrier frequency overall and by commenter type."""
    tables_dir = tables_dir or TABLES_DIR
    bar_cols = [c for c in df.columns if c.startswith("bar_")]
    rows = []
    for col in bar_cols:
        row = {
            "barrier": col,
            "label": BARRIER_LABELS.get(col, col),
            "overall_n": int(df[col].sum()),
            "overall_pct": round(df[col].mean() * 100, 1),
        }
        for ctype in sorted(df["commenter_type"].unique()):
            subset = df[df["commenter_type"] == ctype]
            row[f"pct_{ctype}"] = round(subset[col].mean() * 100, 1) if len(subset) > 0 else 0
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("overall_pct", ascending=False)
    result.to_csv(tables_dir / "table3_barriers.csv", index=False)
    print(f"  Table 3: {len(result)} barriers")
    return result


def table4_governance_positions(df: pd.DataFrame, tables_dir: Path | None = None):
    """Table 4: Position distributions for each governance axis."""
    tables_dir = tables_dir or TABLES_DIR
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    all_rows = []

    for col in pos_cols:
        counts = df[col].value_counts()
        # Total addressing = total minus "not addressed" (X0)
        not_addressed_code = [k for k in counts.index if k.endswith("0")]
        n_addressed = len(df) - sum(counts.get(c, 0) for c in not_addressed_code)

        for code, n in counts.items():
            label_map = POSITION_LABELS.get(col, {})
            all_rows.append({
                "axis": col,
                "axis_label": AXIS_LABELS.get(col, col),
                "code": code,
                "position_label": label_map.get(code, code),
                "n": n,
                "pct_of_total": round(n / len(df) * 100, 1),
                "pct_of_addressed": round(n / n_addressed * 100, 1) if n_addressed > 0 and not code.endswith("0") else None,
                "n_addressed": n_addressed,
            })

    result = pd.DataFrame(all_rows)
    result.to_csv(tables_dir / "table4_governance_positions.csv", index=False)
    print(f"  Table 4: {len(pos_cols)} axes, {len(result)} rows")
    return result


def cramers_v(contingency_table):
    """Calculate Cramér's V from a contingency table."""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))


def table5_cross_tabulation(df: pd.DataFrame, tables_dir: Path | None = None):
    """Table 5: Cross-tab of commenter type × position for most contested axes."""
    tables_dir = tables_dir or TABLES_DIR
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    results = []

    for col in pos_cols:
        # Exclude "not addressed" for the cross-tab
        not_addressed = [v for v in df[col].unique() if str(v).endswith("0")]
        df_addressed = df[~df[col].isin(not_addressed)]

        if len(df_addressed) < 20:
            continue

        # Build contingency table
        ct = pd.crosstab(df_addressed["commenter_type"], df_addressed[col])

        # Chi-square test (Fisher's exact only works for 2x2)
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            try:
                chi2, p_value, dof, expected = chi2_contingency(ct)
                cv = cramers_v(ct)
            except ValueError:
                p_value, cv = None, None
        else:
            p_value, cv = None, None

        results.append({
            "axis": col,
            "axis_label": AXIS_LABELS.get(col, col),
            "n_addressed": len(df_addressed),
            "chi2_p_value": p_value,
            "cramers_v": round(cv, 3) if cv else None,
        })

        # Save the full cross-tab with human-readable labels (not codes)
        ct_with_totals = ct.copy()
        ct_with_totals.index = [TYPE_LABELS.get(i, i) for i in ct_with_totals.index]
        pos_map = POSITION_LABELS.get(col, {})
        ct_with_totals.columns = [pos_map.get(c, c) for c in ct_with_totals.columns]
        ct_with_totals["TOTAL"] = ct_with_totals.sum(axis=1)
        ct_with_totals.to_csv((tables_dir or TABLES_DIR) / f"table5_crosstab_{col}.csv")

    summary = pd.DataFrame(results).sort_values("cramers_v", ascending=False, na_position="last")
    summary.to_csv(tables_dir / "table5_cross_tab_summary.csv", index=False)
    print(f"  Table 5: {len(results)} cross-tabulations")
    return summary


def table6_rfi_coverage(df: pd.DataFrame, tables_dir: Path | None = None):
    """Table 6: How many comments address each RFI question."""
    tables_dir = tables_dir or TABLES_DIR
    # n_rfi_questions is already computed; for per-question we need raw data
    # This is a simpler summary
    summary = {
        "mean_rfi_questions": round(df["n_rfi_questions"].mean(), 1),
        "pct_any_rfi_question": round((df["n_rfi_questions"] > 0).mean() * 100, 1),
        "mean_proposals": round(df["n_proposals"].mean(), 1),
        "pct_with_cfr": round(df["has_cfr_citation"].mean() * 100, 1),
    }
    pd.DataFrame([summary]).to_csv(tables_dir / "table6_supplementary_stats.csv", index=False)
    print(f"  Table 6: supplementary statistics")
    return summary


def table7_stakeholder_position_profiles(df: pd.DataFrame, tables_dir: Path | None = None):
    """Table 7: For key axes, % (and n) of each commenter type in each position — backs Figure 5."""
    tables_dir = tables_dir or TABLES_DIR
    key_axes = ["pos_regulation", "pos_oversight", "pos_evaluation"]
    for col in key_axes:
        not_addressed = [v for v in df[col].unique() if str(v).endswith("0")]
        sub = df[~df[col].isin(not_addressed)]
        if len(sub) < 5:
            continue
        ct = pd.crosstab(sub["commenter_type"], sub[col])
        pct = ct.div(ct.sum(axis=1), axis=0).fillna(0) * 100
        type_order = df["commenter_type"].value_counts().index.tolist()
        type_order = [t for t in type_order if t in pct.index]
        pct = pct.reindex(type_order).dropna(how="all")
        pct.index = [TYPE_LABELS.get(i, i) for i in pct.index]
        pos_map = POSITION_LABELS.get(col, {})
        pct.columns = [pos_map.get(c, c) for c in pct.columns]
        pct.to_csv(tables_dir / f"table7_stakeholder_profile_{col}.csv")
    print(f"  Table 7: stakeholder position profiles ({len(key_axes)} axes)")


# ---- Figure Generators ----

def figure1_topic_heatmap(df: pd.DataFrame, figures_dir: Path | None = None):
    """Heatmap: commenter type (rows) × 15 topics (columns)."""
    figures_dir = figures_dir or FIGURES_DIR
    topic_cols = sorted(
        [c for c in df.columns if c.startswith("top_")],
        key=lambda c: df[c].mean(),
        reverse=True,
    )
    # Order commenter types by n
    type_order = df["commenter_type"].value_counts().index.tolist()

    # Build matrix
    matrix = []
    for ctype in type_order:
        subset = df[df["commenter_type"] == ctype]
        row = [round(subset[col].mean() * 100, 1) for col in topic_cols]
        matrix.append(row)

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"{TYPE_LABELS.get(t, t)} (n={len(df[df['commenter_type']==t])})" for t in type_order],
        columns=[TOPIC_LABELS.get(c, c) for c in topic_cols],
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "% discussing topic"},
        ax=ax,
    )
    ax.set_title("Topic Coverage by Commenter Type (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    fig.savefig((figures_dir or FIGURES_DIR) / "figure1_topic_heatmap.png", dpi=200)
    plt.close()
    print("  Figure 1: topic heatmap")


def figure2_governance_positions(df: pd.DataFrame, figures_dir: Path | None = None):
    """Stacked horizontal bar chart: positions across 6 axes."""
    figures_dir = figures_dir or FIGURES_DIR
    pos_cols = ["pos_oversight", "pos_regulation", "pos_liability",
                "pos_reimbursement", "pos_interoperability", "pos_evaluation"]

    fig, axes = plt.subplots(len(pos_cols), 1, figsize=(12, 10))

    for idx, col in enumerate(pos_cols):
        ax = axes[idx]
        counts = df[col].value_counts()

        # Exclude "not addressed"
        not_addressed_code = [k for k in counts.index if str(k).endswith("0")]
        filtered = {k: v for k, v in counts.items() if k not in not_addressed_code}
        total = sum(filtered.values())

        if total == 0:
            continue

        labels_map = POSITION_LABELS.get(col, {})
        left = 0
        colors = sns.color_palette("Set2", len(filtered))

        for i, (code, n) in enumerate(sorted(filtered.items())):
            pct = n / total * 100
            bar = ax.barh(0, pct, left=left, color=colors[i], edgecolor="white", height=0.6)
            if pct > 8:  # Only label if segment is wide enough
                ax.text(left + pct / 2, 0, f"{labels_map.get(code, code)}\n{pct:.0f}%",
                        ha="center", va="center", fontsize=7, fontweight="bold")
            left += pct

        axis_label = AXIS_LABELS.get(col, col.replace("pos_", "").replace("_", " ").title())
        n_addressed = total
        ax.set_ylabel(f"{axis_label}\n(n={n_addressed})", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("% of comments addressing this axis" if idx == len(pos_cols) - 1 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.suptitle("Governance Position Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig((figures_dir or FIGURES_DIR) / "figure2_governance_positions.png", dpi=200)
    plt.close()
    print("  Figure 2: governance positions")


def figure3_temporal(df: pd.DataFrame, figures_dir: Path | None = None):
    """Bar chart of submission dates."""
    figures_dir = figures_dir or FIGURES_DIR
    dates = pd.to_datetime(df["posted_date"], errors="coerce")
    date_counts = dates.dt.date.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(date_counts)), date_counts.values, color="#2E75B6")
    ax.set_xticks(range(len(date_counts)))
    ax.set_xticklabels([str(d) for d in date_counts.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of comments")
    ax.set_title("Comment Submission Timeline", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig((figures_dir or FIGURES_DIR) / "figure3_temporal.png", dpi=200)
    plt.close()
    print("  Figure 3: temporal distribution")


def figure4_commenter_types(df: pd.DataFrame, figures_dir: Path | None = None):
    """Bar chart: commenter type distribution."""
    figures_dir = figures_dir or FIGURES_DIR
    counts = df["commenter_type"].value_counts()
    labels = [TYPE_LABELS.get(t, t) for t in counts.index]
    colors = sns.color_palette("Set3", len(counts))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(counts)), counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of comments")
    ax.set_title("Commenter Type Distribution", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (bar, n) in enumerate(zip(bars, counts.values)):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, f"n={n}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig((figures_dir or FIGURES_DIR) / "figure4_commenter_types.png", dpi=200)
    plt.close()
    print("  Figure 4: commenter type distribution")


def figure5_stakeholder_position_profiles(df: pd.DataFrame, figures_dir: Path | None = None):
    """Heatmaps: for each of 3 key axes, % of each commenter type taking each position (who wants what)."""
    figures_dir = figures_dir or FIGURES_DIR
    key_axes = ["pos_regulation", "pos_oversight", "pos_evaluation"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    for idx, col in enumerate(key_axes):
        ax = axes[idx]
        not_addressed = [v for v in df[col].unique() if str(v).endswith("0")]
        sub = df[~df[col].isin(not_addressed)]
        if len(sub) < 10:
            ax.set_visible(False)
            continue
        ct = pd.crosstab(sub["commenter_type"], sub[col])
        # Row % (within each commenter type)
        pct = ct.div(ct.sum(axis=1), axis=0).fillna(0) * 100
        # Order types by n (same as elsewhere)
        type_order = df["commenter_type"].value_counts().index.tolist()
        type_order = [t for t in type_order if t in pct.index]
        pct = pct.reindex(type_order).dropna(how="all")
        if pct.empty:
            ax.set_visible(False)
            continue
        pct.index = [TYPE_LABELS.get(i, i) for i in pct.index]
        pos_map = POSITION_LABELS.get(col, {})
        pct.columns = [pos_map.get(c, c) for c in pct.columns]
        sns.heatmap(
            pct,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "% within type"},
            ax=ax,
        )
        ax.set_title(f"{AXIS_LABELS.get(col, col)}\n(n={len(sub)} addressed)", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=9)

    plt.suptitle("Stakeholder position profiles — Who wants what? (% within each commenter type)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig((figures_dir or FIGURES_DIR) / "figure5_stakeholder_position_profiles.png", dpi=200)
    plt.close()
    print("  Figure 5: stakeholder position profiles")


# ---- Summary Stats ----

def print_summary(df: pd.DataFrame):
    """Print key statistics to console."""
    print(f"\n{'='*60}")
    print(f"  ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total comments: {len(df)}")
    print(f"  Commenter types: {df['commenter_type'].nunique()}")

    topic_cols = [c for c in df.columns if c.startswith("top_")]
    print(f"\n  Mean topics per comment: {df[topic_cols].sum(axis=1).mean():.1f}")
    print(f"  Median topics per comment: {df[topic_cols].sum(axis=1).median():.0f}")

    bar_cols = [c for c in df.columns if c.startswith("bar_")]
    print(f"  Mean barriers per comment: {df[bar_cols].sum(axis=1).mean():.1f}")

    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    for col in pos_cols:
        not_addressed = [v for v in df[col].unique() if str(v).endswith("0")]
        pct_addressed = (1 - df[col].isin(not_addressed).mean()) * 100
        most_common = df[~df[col].isin(not_addressed)][col].mode()
        mc = most_common.iloc[0] if len(most_common) > 0 else "N/A"
        label = POSITION_LABELS.get(col, {}).get(mc, mc)
        print(f"  {col}: {pct_addressed:.0f}% addressed, most common = {label}")

    print(f"\n  Evidence types:")
    for etype, n in df["evidence_type"].value_counts().items():
        lbl = EVIDENCE_TYPE_LABELS.get(etype, etype)
        print(f"    {lbl}: {n} ({n/len(df)*100:.1f}%)")

    print(f"\n  Mean proposals per comment: {df['n_proposals'].mean():.1f}")
    print(f"  Comments with CFR citations: {df['has_cfr_citation'].sum()} ({df['has_cfr_citation'].mean()*100:.1f}%)")


# ---- HTML Report ----

def write_index_html(out_dir: Path, df: pd.DataFrame) -> None:
    """Write index.html for initial analysis folder (figures + table links)."""
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    n = len(df)
    topic_cols = [c for c in df.columns if c.startswith("top_")]
    mean_topics = df[topic_cols].sum(axis=1).mean()
    bar_cols = [c for c in df.columns if c.startswith("bar_")]
    mean_barriers = df[bar_cols].sum(axis=1).mean()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HHS AI RFI — Initial Analysis</title>
  <style>
    :root {{ font-family: system-ui, -apple-system, sans-serif; line-height: 1.5; color: #1a1a1a; }}
    body {{ max-width: 1000px; margin: 0 auto; padding: 1.5rem; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
    .meta {{ color: #555; font-size: 0.9rem; margin-bottom: 2rem; }}
    section {{ margin: 2.5rem 0; }}
    section h2 {{ font-size: 1.15rem; border-bottom: 1px solid #ccc; padding-bottom: 0.35rem; }}
    figure {{ margin: 1rem 0; }}
    figure img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
    figure figcaption {{ font-size: 0.9rem; color: #555; margin-top: 0.35rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.75rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    a {{ color: #1967d2; }}
    ul {{ margin: 0.5rem 0; padding-left: 1.25rem; }}
  </style>
</head>
<body>
  <h1>HHS AI RFI Comment Analysis — Initial Findings</h1>
  <p class="meta">Coded comments: <strong>{n}</strong> &nbsp;|&nbsp; Mean topics per comment: <strong>{mean_topics:.1f}</strong> &nbsp;|&nbsp; Mean barriers per comment: <strong>{mean_barriers:.1f}</strong></p>

  <section>
    <h2>Figures</h2>
    <figure>
      <img src="figures/figure1_topic_heatmap.png" alt="Topic coverage by commenter type" width="800">
      <figcaption>Figure 1. Topic coverage by commenter type (%)</figcaption>
    </figure>
    <figure>
      <img src="figures/figure2_governance_positions.png" alt="Governance position distributions" width="800">
      <figcaption>Figure 2. Governance position distributions (among comments that addressed each axis)</figcaption>
    </figure>
    <figure>
      <img src="figures/figure3_temporal.png" alt="Comment submission timeline" width="800">
      <figcaption>Figure 3. Comment submission timeline</figcaption>
    </figure>
    <figure>
      <img src="figures/figure4_commenter_types.png" alt="Commenter type distribution" width="800">
      <figcaption>Figure 4. Commenter type distribution</figcaption>
    </figure>
    <figure>
      <img src="figures/figure5_stakeholder_position_profiles.png" alt="Stakeholder position profiles" width="800">
      <figcaption>Figure 5. Stakeholder position profiles — Who wants what? (% within each commenter type, for Regulation, Oversight, Evaluation)</figcaption>
    </figure>
  </section>

  <section>
    <h2>Tables (CSV)</h2>
    <ul>
      <li><a href="tables/table1_commenter_characteristics.csv">Table 1. Commenter characteristics</a></li>
      <li><a href="tables/table2_topic_coverage.csv">Table 2. Topic coverage overall and by type</a></li>
      <li><a href="tables/table3_barriers.csv">Table 3. Barriers identified</a></li>
      <li><a href="tables/table4_governance_positions.csv">Table 4. Governance positions</a></li>
      <li><a href="tables/table5_cross_tab_summary.csv">Table 5. Cross-tab summary (commenter × position)</a></li>
      <li><a href="tables/table6_supplementary_stats.csv">Table 6. Supplementary (RFI coverage, proposals, CFR)</a></li>
      <li>Table 7. Stakeholder position profiles: <a href="tables/table7_stakeholder_profile_pos_regulation.csv">Regulation</a>, <a href="tables/table7_stakeholder_profile_pos_oversight.csv">Oversight</a>, <a href="tables/table7_stakeholder_profile_pos_evaluation.csv">Evaluation</a></li>
    </ul>
  </section>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"  Index: {out_dir / 'index.html'}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Write tables and figures to this folder (e.g. initial_analysis). Creates out_dir/tables, out_dir/figures, and out_dir/index.html.",
    )
    args = parser.parse_args()

    if not OUTPUT_JSONL.exists():
        print(f"Error: {OUTPUT_JSONL} not found. Run extract.py first.")
        return

    df = load_data()
    print(f"Loaded {len(df)} coded comments\n")

    tables_dir = TABLES_DIR
    figures_dir = FIGURES_DIR
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        tables_dir = out_dir / "tables"
        figures_dir = out_dir / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {out_dir.resolve()}\n")

    if not args.figures_only:
        print("Generating tables...")
        table1_commenter_characteristics(df, tables_dir=tables_dir)
        table2_topic_coverage(df, tables_dir=tables_dir)
        table3_barriers(df, tables_dir=tables_dir)
        table4_governance_positions(df, tables_dir=tables_dir)
        table5_cross_tabulation(df, tables_dir=tables_dir)
        table6_rfi_coverage(df, tables_dir=tables_dir)
        table7_stakeholder_position_profiles(df, tables_dir=tables_dir)

    if not args.tables_only:
        print("\nGenerating figures...")
        figure1_topic_heatmap(df, figures_dir=figures_dir)
        figure2_governance_positions(df, figures_dir=figures_dir)
        figure3_temporal(df, figures_dir=figures_dir)
        figure4_commenter_types(df, figures_dir=figures_dir)
        figure5_stakeholder_position_profiles(df, figures_dir=figures_dir)

    if args.out_dir:
        print("\nWriting report index...")
        write_index_html(out_dir, df)

    print_summary(df)


if __name__ == "__main__":
    main()
