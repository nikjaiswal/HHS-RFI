"""Stakeholder × axis × stance analysis — preserves the 'who is saying what' dimension.

The coalition analysis groups commenters into 3 clusters but loses the per-stakeholder
position information needed for political-economy interpretation. This module restores
that view:

  1. For each (stakeholder type, governance axis) cell, compute the within-stakeholder
     position distribution.
  2. Identify the modal stance per (stakeholder, axis), with within-addressing share
     and 95% Wilson CI.
  3. Compute a "stance preference index" = (within-stakeholder rate of each stance) /
     (corpus rate of that stance among addressing comments). Values > 1 mean a
     stakeholder *over-endorses* a stance relative to the corpus; values < 1 mean
     under-endorse. This is the political-economy lens.
  4. Identify the largest stakeholder × stance divergences (where preference index
     is highest), highlighting plausible interest-alignment hypotheses.
  5. Industry-vs-individual head-to-head: pool {AIC, HIT, TEC} as "industry" and
     {ADV, IND} as "individual/patient voice"; report stance distributions and χ²
     for each axis.

Outputs:
  output/stakeholder_positions/
    stakeholder_x_axis_modal.csv          modal stance per (stakeholder, axis)
    stakeholder_x_axis_distribution.csv   long format: (stakeholder, axis, stance, n, share)
    stance_preference_index.csv           preference index per cell
    interest_alignment_highlights.csv     largest preference-index divergences (>1.5 or <0.5)
    industry_vs_patient.csv               pooled comparison per axis
    fig_stakeholder_modal_heatmap.png     6 axes × 10 stakeholders modal-stance heatmap
    fig_stakeholder_position_profile.png  6-panel small-multiples (one panel per axis)
    fig_industry_vs_patient.png           bar chart: where do industry and patient voice diverge?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "stakeholder_positions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_VARS = ["pos_oversight", "pos_regulation", "pos_liability",
                 "pos_reimbursement", "pos_interoperability", "pos_evaluation"]
POSITION_LABELS = {
    "pos_oversight": "Human oversight",
    "pos_regulation": "Regulatory approach",
    "pos_liability": "Liability allocation",
    "pos_reimbursement": "Reimbursement",
    "pos_interoperability": "Interoperability",
    "pos_evaluation": "Evaluation",
}
POSITION_NULL = {"pos_oversight": "H0", "pos_regulation": "R0", "pos_liability": "L0",
                 "pos_reimbursement": "P0", "pos_interoperability": "D0", "pos_evaluation": "E0"}

POSITION_DETAIL = {
    "H0": "Not addressed", "H1": "Universal mandatory oversight",
    "H2": "Risk-tiered oversight", "H3": "Recommended not mandated", "H4": "Autonomous OK",
    "R0": "Not addressed", "R1": "New AI-specific regulation",
    "R2": "Risk-tiered adaptation", "R3": "Clarify existing rules",
    "R4": "Reduce burden", "R5": "Industry self-governance",
    "L0": "Not addressed", "L1": "Increase developer accountability",
    "L2": "Shared/distributed liability", "L3": "Current law adequate",
    "L4": "Federal safe harbor", "L5": "New legal framework",
    "P0": "Not addressed", "P1": "AI-specific payment pathways",
    "P2": "Value-based payment", "P3": "Remove FFS barriers",
    "P4": "CMS pilot/demonstration", "P5": "Multiple reforms",
    "D0": "Not addressed", "D1": "Strengthen current standards",
    "D2": "Expand data types", "D3": "Patient-controlled data",
    "D4": "Prevent gatekeeping", "D5": "Federal data infrastructure",
    "E0": "Not addressed", "E1": "Pre-market validation primary",
    "E2": "Post-market monitoring primary", "E3": "Full lifecycle",
    "E4": "Developer-led", "E5": "Federal evaluation infrastructure",
}

COMMENTER_LABEL = {
    "MPS": "Medical society", "HSP": "Health system", "HIT": "Health IT",
    "AIC": "AI company", "TEC": "Tech/pharma", "PAY": "Payer",
    "ADV": "Patient advocate", "LAB": "Labor", "POL": "Policy/academic", "IND": "Individual",
}

# Hypotheses about which stances would advantage a stakeholder's economic/political interests.
# We don't claim causal endorsement-from-interest; we flag concordance for discussion.
INTEREST_HYPOTHESES: dict[str, list[tuple[str, str]]] = {
    "AIC": [
        ("L4", "Federal safe harbor reduces AI-developer / clinician liability exposure for vendor products"),
        ("R4", "Reducing regulatory burden lowers compliance cost for AI vendors"),
        ("R5", "Industry self-governance averts new federal rules on AI vendors"),
        ("D4", "Preventing data gatekeeping by EHR incumbents helps vendor market access"),
        ("P1", "AI-specific payment pathways create direct vendor revenue"),
        ("E1", "Pre-market-only evaluation reduces post-deployment burden"),
    ],
    "HIT": [
        ("L4", "Safe harbor reduces vendor liability"),
        ("R4", "Reducing regulatory burden lowers compliance cost"),
        ("D1", "Strengthening current standards favors incumbent EHR/HIT vendors"),
        ("P1", "AI-specific payment pathways create new revenue"),
    ],
    "TEC": [
        ("L4", "Safe harbor for validated AI"),
        ("R4", "Reducing regulatory burden"),
    ],
    "HSP": [
        ("L4", "Safe harbor for clinicians using validated AI protects health systems"),
        ("L2", "Shared liability spreads financial exposure"),
        ("P2", "Value-based payment favors integrated systems"),
    ],
    "MPS": [
        ("L4", "Safe harbor protects member clinicians"),
        ("H1", "Universal oversight enshrines clinician role"),
        ("H2", "Risk-tiered oversight preserves clinician judgment for high-risk"),
    ],
    "ADV": [
        ("H1", "Universal oversight protects patients from autonomous AI risks"),
        ("R1", "New regulation creates federal floor for safety"),
        ("E5", "Independent federal evaluation prevents industry capture"),
        ("D3", "Patient-controlled data flows protect patient agency"),
    ],
    "PAY": [
        ("E1", "Pre-market validation favors payer cost-containment"),
        ("R3", "Clarification favors stable claim adjudication"),
    ],
}


def _str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v)
    return s.split(".")[-1] if "." in s else s


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0, centre - half), min(1, centre + half)


def stakeholder_x_axis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (stakeholder, axis), compute distribution of stances + modal stance."""
    df_x = df.copy()
    df_x["commenter_str"] = df_x["commenter_type"].apply(_str)
    types = sorted(df_x["commenter_str"].unique())

    dist_rows = []
    modal_rows = []
    for t in types:
        sub = df_x[df_x["commenter_str"] == t]
        n_total = len(sub)
        for axis in POSITION_VARS:
            null_code = POSITION_NULL[axis]
            vals = sub[axis].apply(_str)
            n_addressed = (vals != null_code).sum()
            non_null = vals[vals != null_code]
            counts = non_null.value_counts()
            for stance, n in counts.items():
                share = n / n_addressed if n_addressed > 0 else 0
                p, lo, hi = wilson_ci(int(n), int(n_addressed))
                dist_rows.append({
                    "stakeholder": t,
                    "stakeholder_label": COMMENTER_LABEL.get(t, t),
                    "axis": POSITION_LABELS[axis],
                    "stance": stance,
                    "stance_label": POSITION_DETAIL.get(stance, stance),
                    "n_with_stance": int(n),
                    "n_addressing_axis": int(n_addressed),
                    "n_stakeholder_total": int(n_total),
                    "share_of_addressing": round(share, 3),
                    "ci_lo": round(lo, 3),
                    "ci_hi": round(hi, 3),
                })
            if len(counts) > 0:
                modal_stance = counts.index[0]
                modal_n = int(counts.iloc[0])
                modal_share = modal_n / n_addressed
                p, lo, hi = wilson_ci(modal_n, n_addressed)
                modal_rows.append({
                    "stakeholder": t,
                    "stakeholder_label": COMMENTER_LABEL.get(t, t),
                    "axis": POSITION_LABELS[axis],
                    "n_addressing": int(n_addressed),
                    "address_rate": round(n_addressed / n_total, 3),
                    "modal_stance": modal_stance,
                    "modal_stance_label": POSITION_DETAIL.get(modal_stance, modal_stance),
                    "modal_n": modal_n,
                    "modal_share_of_addressing": round(modal_share, 3),
                    "modal_share_ci_lo": round(lo, 3),
                    "modal_share_ci_hi": round(hi, 3),
                })

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_DIR / "stakeholder_x_axis_distribution.csv", index=False)
    modal_df = pd.DataFrame(modal_rows)
    modal_df.to_csv(OUT_DIR / "stakeholder_x_axis_modal.csv", index=False)
    return dist_df, modal_df


def stance_preference_index(dist_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Preference index = within-stakeholder rate / corpus rate, per (stakeholder, axis, stance).

    Values > 1: stakeholder over-endorses this stance vs. the corpus (among comments
    addressing the axis). Values < 1: under-endorses.
    """
    df_x = df.copy()
    df_x["commenter_str"] = df_x["commenter_type"].apply(_str)

    # Corpus rates per axis × stance (among comments addressing the axis)
    corpus_rates = {}
    for axis in POSITION_VARS:
        null_code = POSITION_NULL[axis]
        vals = df_x[axis].apply(_str)
        non_null = vals[vals != null_code]
        n = len(non_null)
        rates = (non_null.value_counts() / n).to_dict() if n else {}
        for stance, rate in rates.items():
            corpus_rates[(POSITION_LABELS[axis], stance)] = rate

    rows = []
    for _, r in dist_df.iterrows():
        key = (r["axis"], r["stance"])
        corpus_rate = corpus_rates.get(key, 0.0)
        if corpus_rate > 0 and r["share_of_addressing"] > 0:
            pref_index = r["share_of_addressing"] / corpus_rate
        else:
            pref_index = float("nan")
        rows.append({
            "stakeholder": r["stakeholder"],
            "stakeholder_label": r["stakeholder_label"],
            "axis": r["axis"],
            "stance": r["stance"],
            "stance_label": r["stance_label"],
            "n_with_stance": r["n_with_stance"],
            "n_addressing_axis": r["n_addressing_axis"],
            "stakeholder_share": r["share_of_addressing"],
            "corpus_share": round(corpus_rate, 3),
            "preference_index": round(pref_index, 3) if not np.isnan(pref_index) else "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "stance_preference_index.csv", index=False)
    return out


def interest_alignment_highlights(pref_df: pd.DataFrame) -> pd.DataFrame:
    """Surface stances where stakeholders systematically over- or under-endorse, and
    flag plausible interest-alignment hypotheses."""
    rows = []
    for _, r in pref_df.iterrows():
        if r["preference_index"] == "" or r["n_with_stance"] < 3:
            continue
        pi = float(r["preference_index"])
        if pi >= 1.5 or pi <= 0.5:
            stake = r["stakeholder"]
            hyp_match = ""
            for stance, rationale in INTEREST_HYPOTHESES.get(stake, []):
                if stance == r["stance"] and pi >= 1.5:
                    hyp_match = rationale
                    break
            rows.append({
                "stakeholder": r["stakeholder_label"],
                "axis": r["axis"],
                "stance": r["stance"],
                "stance_label": r["stance_label"],
                "n_with_stance": r["n_with_stance"],
                "stakeholder_share": r["stakeholder_share"],
                "corpus_share": r["corpus_share"],
                "preference_index": r["preference_index"],
                "direction": "OVER-endorse" if pi >= 1.5 else "UNDER-endorse",
                "interest_alignment_hypothesis": hyp_match,
            })
    out = pd.DataFrame(rows)
    out = out.sort_values("preference_index", ascending=False)
    out.to_csv(OUT_DIR / "interest_alignment_highlights.csv", index=False)
    return out


def industry_vs_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled industry (AIC + HIT + TEC) vs patient/individual voice (ADV + IND)."""
    df_x = df.copy()
    df_x["commenter_str"] = df_x["commenter_type"].apply(_str)

    industry = df_x[df_x["commenter_str"].isin(["AIC", "HIT", "TEC"])]
    patient = df_x[df_x["commenter_str"].isin(["ADV", "IND"])]

    rows = []
    for axis in POSITION_VARS:
        null_code = POSITION_NULL[axis]
        ind_vals = industry[axis].apply(_str)
        pat_vals = patient[axis].apply(_str)
        ind_non_null = ind_vals[ind_vals != null_code]
        pat_non_null = pat_vals[pat_vals != null_code]
        if len(ind_non_null) == 0 or len(pat_non_null) == 0:
            continue
        ind_dist = ind_non_null.value_counts(normalize=True)
        pat_dist = pat_non_null.value_counts(normalize=True)
        # χ² test
        all_codes = sorted(set(ind_non_null) | set(pat_non_null))
        ind_counts = [int((ind_non_null == c).sum()) for c in all_codes]
        pat_counts = [int((pat_non_null == c).sum()) for c in all_codes]
        ct = np.array([ind_counts, pat_counts])
        if ct.sum() > 0 and min(ct.shape) >= 2:
            chi2, p, _, _ = chi2_contingency(ct)
        else:
            chi2, p = float("nan"), float("nan")
        for c in all_codes:
            rows.append({
                "axis": POSITION_LABELS[axis],
                "stance": c,
                "stance_label": POSITION_DETAIL.get(c, c),
                "industry_n": int((ind_non_null == c).sum()),
                "industry_share": round(float(ind_dist.get(c, 0)), 3),
                "patient_n": int((pat_non_null == c).sum()),
                "patient_share": round(float(pat_dist.get(c, 0)), 3),
                "diff_industry_minus_patient": round(
                    float(ind_dist.get(c, 0) - pat_dist.get(c, 0)), 3
                ),
                "axis_chi2": round(float(chi2), 3) if not np.isnan(chi2) else "",
                "axis_p": float(p) if not np.isnan(p) else "",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "industry_vs_patient.csv", index=False)
    return out


def figure_modal_heatmap(modal_df: pd.DataFrame) -> None:
    """Cleaner heatmap: rows = stakeholders, cols = governance axes.

    Each cell shows: stance code (large) + within-addressing share (small).
    Color encodes within-addressing modal-stance share (consensus intensity).
    Stakeholders are grouped semantically: Industry / Health-system / Clinical
    society / Other / Patient-individual. Group separators are drawn as horizontal
    rules to make the structure visually clear.
    """
    # Semantic ordering of stakeholders, with group separators
    groups = [
        ("Industry", ["AIC", "HIT", "TEC"]),
        ("Health systems / payers", ["HSP", "PAY"]),
        ("Clinical / professional", ["MPS"]),
        ("Other organized", ["LAB", "POL"]),
        ("Patient/individual voice", ["ADV", "IND"]),
    ]
    types_order = [t for _, ts in groups for t in ts]
    sep_indices = []
    cum = 0
    for _, ts in groups[:-1]:
        cum += len(ts)
        sep_indices.append(cum - 0.5)

    axes_order = ["Human oversight", "Regulatory approach", "Liability allocation",
                  "Reimbursement", "Interoperability", "Evaluation"]
    pivot_share = modal_df.pivot(index="stakeholder", columns="axis",
                                 values="modal_share_of_addressing").reindex(types_order)[axes_order]
    pivot_stance = modal_df.pivot(index="stakeholder", columns="axis",
                                  values="modal_stance").reindex(types_order)[axes_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_share.values, cmap="Blues", vmin=0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(axes_order)))
    ax.set_xticklabels(axes_order, fontsize=10, rotation=15, ha="right")
    ax.set_yticks(range(len(types_order)))
    ax.set_yticklabels(
        [f"{t} ({COMMENTER_LABEL.get(t, t)})" for t in types_order],
        fontsize=10,
    )
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)

    # Group separators
    for sep in sep_indices:
        ax.axhline(sep, color="black", linewidth=1.4)

    # Cell annotations: stance code (large), then share % (small) on a second line
    for i in range(len(types_order)):
        for j in range(len(axes_order)):
            share = pivot_share.values[i, j]
            stance = pivot_stance.values[i, j]
            if pd.isna(share):
                continue
            color = "white" if share > 0.65 else "black"
            ax.text(j, i - 0.13, str(stance), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
            ax.text(j, i + 0.20, f"{share:.0%}", ha="center", va="center",
                    fontsize=9, color=color, alpha=0.85)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.015)
    cbar.set_label("Modal-stance share among addressing comments", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Modal governance stance per stakeholder × axis (n=446)",
                 fontsize=12, pad=12)

    # Group annotations on the left
    g_starts = [0]
    for _, ts in groups[:-1]:
        g_starts.append(g_starts[-1] + len(ts))
    for (gname, ts), gs in zip(groups, g_starts):
        ax.text(-2.3, gs + (len(ts) - 1) / 2, gname,
                ha="right", va="center", fontsize=9, fontweight="bold",
                color="#444444", rotation=0,
                transform=ax.transData)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_stakeholder_modal_heatmap.png", dpi=260, bbox_inches="tight")
    plt.close()


def figure_position_profile(dist_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """6-panel small-multiples: one panel per axis, stacked bars by stakeholder."""
    df_x = df.copy()
    df_x["commenter_str"] = df_x["commenter_type"].apply(_str)
    types_order = ["HSP", "HIT", "AIC", "TEC", "PAY", "MPS", "POL", "ADV", "LAB", "IND"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for ax_idx, axis in enumerate(POSITION_VARS):
        ax = axes_flat[ax_idx]
        null_code = POSITION_NULL[axis]
        # Count distribution per stakeholder, ALL stances (including not addressed)
        all_stances = []
        all_axis_codes = sorted(set(df_x[axis].apply(_str).unique()))
        if null_code in all_axis_codes:
            all_axis_codes.remove(null_code)
            all_axis_codes = [null_code] + all_axis_codes  # null first
        cmap = plt.get_cmap("tab10")
        bottoms = np.zeros(len(types_order))
        for s_idx, stance in enumerate(all_axis_codes):
            shares = []
            for t in types_order:
                sub = df_x[df_x["commenter_str"] == t]
                if len(sub) == 0:
                    shares.append(0)
                    continue
                vals = sub[axis].apply(_str)
                shares.append((vals == stance).sum() / len(sub))
            shares = np.array(shares)
            color = "lightgray" if stance == null_code else cmap(s_idx - 1)
            label = "Not addressed" if stance == null_code else POSITION_DETAIL.get(stance, stance)
            ax.bar(range(len(types_order)), shares, bottom=bottoms,
                   color=color, edgecolor="white", linewidth=0.4, label=label)
            bottoms += shares
        ax.set_title(POSITION_LABELS[axis], fontsize=11)
        ax.set_xticks(range(len(types_order)))
        ax.set_xticklabels(types_order, fontsize=9)
        ax.set_ylabel("Share of stakeholder")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.85, ncol=1)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Stakeholder position profiles across the six governance axes (n=446)\n"
                 "Each bar = one stakeholder type; segments show stance distribution including not-addressed (gray).",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_stakeholder_position_profile.png", dpi=240, bbox_inches="tight")
    plt.close()


def figure_industry_vs_patient(ivp_df: pd.DataFrame) -> None:
    """Cleaner bar chart, grouped by governance axis.

    Within each axis group, sort stances by industry−patient Δshare and plot horizontal
    bars colored by sign. Light-gray banding separates axes; small ticks show specific
    Δshare values.
    """
    ivp_df = ivp_df.copy()
    ivp_df = ivp_df[(ivp_df["industry_n"] >= 5) | (ivp_df["patient_n"] >= 5)]
    ivp_df = ivp_df[ivp_df["diff_industry_minus_patient"].abs() >= 0.05]

    axes_order = ["Human oversight", "Regulatory approach", "Liability allocation",
                  "Reimbursement", "Interoperability", "Evaluation"]
    rows: list[dict] = []
    for axis in axes_order:
        sub = ivp_df[ivp_df["axis"] == axis].sort_values(
            "diff_industry_minus_patient", ascending=True
        )
        for _, r in sub.iterrows():
            rows.append(r.to_dict())
    plot_df = pd.DataFrame(rows).reset_index(drop=True)
    if plot_df.empty:
        return

    n_rows = len(plot_df)
    fig, ax = plt.subplots(figsize=(10, max(5, n_rows * 0.42)))
    y_positions = np.arange(n_rows)[::-1]

    # Bar colors by sign
    colors = ["#2c7fb8" if d > 0 else "#d7301f" for d in plot_df["diff_industry_minus_patient"]]
    ax.barh(y_positions, plot_df["diff_industry_minus_patient"],
            color=colors, edgecolor="white", linewidth=0.5, height=0.78)

    # Axis-group banding (light gray rectangles behind alternating axis groups)
    for i, axis in enumerate(axes_order):
        idx = plot_df.index[plot_df["axis"] == axis].tolist()
        if not idx:
            continue
        if i % 2 == 1:  # alternate banding
            y_top = y_positions[idx[0]] + 0.5
            y_bottom = y_positions[idx[-1]] - 0.5
            ax.axhspan(y_bottom, y_top, color="#f4f4f4", zorder=0)

    # Tick labels: stance code + short label
    ytick_labels = [f"{r['stance']} — {r['stance_label']}" for _, r in plot_df.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.tick_params(axis="y", which="both", length=0)

    # Right-edge axis grouping labels
    for axis in axes_order:
        idx = plot_df.index[plot_df["axis"] == axis].tolist()
        if not idx:
            continue
        ymid = (y_positions[idx[0]] + y_positions[idx[-1]]) / 2
        ax.text(1.02, ymid, axis, transform=ax.get_yaxis_transform(),
                fontsize=9, fontweight="bold", color="#333333",
                ha="left", va="center")

    # Δ value annotations at bar tip
    for y, d in zip(y_positions, plot_df["diff_industry_minus_patient"]):
        ha = "left" if d > 0 else "right"
        offset = 0.005 if d > 0 else -0.005
        ax.text(d + offset, y, f"{d:+.0%}", ha=ha, va="center", fontsize=8,
                color="black")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ share (industry − patient/individual)", fontsize=11)
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks(np.arange(-0.5, 0.6, 0.1))
    ax.set_xticklabels([f"{int(t*100)}%" for t in np.arange(-0.5, 0.6, 0.1)], fontsize=9)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(
        "Where industry (AIC+HIT+TEC, n=154) and patient/individual voice (ADV+IND, n=165) "
        "diverge on stance share, by governance axis",
        fontsize=11, pad=10,
    )
    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color="#2c7fb8", label="Industry over-endorses"),
        Patch(color="#d7301f", label="Patient/individual over-endorses"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_industry_vs_patient.png", dpi=260, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} comments")

    print("\n[1/5] Stakeholder × axis distribution + modal stance ...")
    dist_df, modal_df = stakeholder_x_axis(df)

    print("[2/5] Stance preference index ...")
    pref_df = stance_preference_index(dist_df, df)

    print("[3/5] Interest-alignment highlights ...")
    highlights = interest_alignment_highlights(pref_df)
    print(f"    {len(highlights)} stakeholder × stance cells with preference index >= 1.5 or <= 0.5")
    print("\nTop interest-aligned over-endorsements:")
    matched = highlights[highlights["interest_alignment_hypothesis"] != ""].head(15)
    for _, r in matched.iterrows():
        print(f"  {r['stakeholder']:<22} → {r['axis']}: {r['stance']} "
              f"({r['stakeholder_share']:.0%} vs corpus {r['corpus_share']:.0%}, "
              f"{r['preference_index']:.1f}×) — {r['interest_alignment_hypothesis']}")

    print("\n[4/5] Industry vs patient/individual head-to-head ...")
    ivp = industry_vs_patient(df)
    print("\nLargest industry vs patient/individual divergences (Δshare):")
    ivp_filtered = ivp.copy()
    ivp_filtered["abs_diff"] = ivp_filtered["diff_industry_minus_patient"].abs()
    ivp_filtered = ivp_filtered[(ivp_filtered["industry_n"] >= 5) |
                                  (ivp_filtered["patient_n"] >= 5)]
    top_ivp = ivp_filtered.nlargest(10, "abs_diff")
    for _, r in top_ivp.iterrows():
        d = r["diff_industry_minus_patient"]
        sign = "+" if d > 0 else ""
        print(f"  {r['axis']}: {r['stance']} ({r['stance_label']:<35})  "
              f"industry {r['industry_share']:.0%}  patient {r['patient_share']:.0%}  "
              f"Δ={sign}{d:.0%}")

    print("\n[5/5] Figures ...")
    figure_modal_heatmap(modal_df)
    figure_position_profile(dist_df, df)
    figure_industry_vs_patient(ivp)

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
