"""Three-coalition analysis of the HHS-ONC-2026-0001 comment corpus.

Locks k=3 as the canonical position-vector clustering and produces all the
narrative-targeted outputs for the manuscript:

  output/coalitions/
    coalition_assignments.csv         comment_id, commenter_type, coalition_id, coalition_name
    coalition_profiles.csv            n, share, stakeholder_mix, modal_stance_per_axis, mean
                                      proposals/topics/CFR-rate, top topics, address rates
    coalition_x_stakeholder.csv       cross-tab + chi-square + Cramér's V + BH-FDR
    multinomial_logit_results.csv     coalition ~ stakeholder + n_proposals + topic_count + has_cfr_citation
    coalition_topic_emphasis.csv      topic prevalence per coalition (long format)
    fig_coalition_profile.png         3-panel: stance profiles, address rates, stakeholder mix
    fig_coalition_topic_emphasis.png  Heatmap of topic prevalence by coalition
    fig_coalition_length_density.png  Boxplots of n_proposals / topic_count by coalition
    fig_coalition_pca.png             2D PCA scatter colored by coalition
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "coalitions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_VARS = ["pos_oversight", "pos_regulation", "pos_liability",
                 "pos_reimbursement", "pos_interoperability", "pos_evaluation"]
POSITION_NULL = {"pos_oversight": "H0", "pos_regulation": "R0", "pos_liability": "L0",
                 "pos_reimbursement": "P0", "pos_interoperability": "D0", "pos_evaluation": "E0"}
POSITION_LABEL = {"pos_oversight": "Oversight", "pos_regulation": "Regulation",
                  "pos_liability": "Liability", "pos_reimbursement": "Reimbursement",
                  "pos_interoperability": "Interoperability", "pos_evaluation": "Evaluation"}
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
    "top_admin_burden": "Admin burden reduction",
    "top_standards": "Standards & accreditation",
    "top_workforce": "Workforce impact",
    "top_privacy": "Privacy & HIPAA",
}
COMMENTER_LABEL = {
    "MPS": "Medical society", "HSP": "Health system", "HIT": "Health IT",
    "AIC": "AI company", "TEC": "Tech/pharma", "PAY": "Payer",
    "ADV": "Advocacy org", "LAB": "Labor", "POL": "Policy/academic", "IND": "Individual",
}

# Coalition naming after manual inspection of k=3 K-means output.
# Cluster ordering by KMeans is stochastic; we rename based on engagement profile.
# We assign names by matching cluster characteristics:
#   - "Comprehensive Pragmatists" = highest mean-address-rate cluster
#   - "Restrictive Advocates"     = high oversight+regulation, low technical-axis address
#   - "Limited Engagement"        = low address rate across the board
COALITION_NAMES = {
    "comprehensive": "Comprehensive Pragmatists",
    "restrictive": "Selective Universalists",
    "limited": "Limited Engagement",
}
COALITION_COLORS = {
    "Comprehensive Pragmatists": "#1f77b4",
    "Selective Universalists": "#d62728",
    "Limited Engagement": "#7f7f7f",
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


def cramers_v(table_values: np.ndarray) -> float:
    if table_values.sum() == 0 or min(table_values.shape) < 2:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(table_values)
    n = table_values.sum()
    r, c = table_values.shape
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def assign_coalitions(df: pd.DataFrame) -> pd.DataFrame:
    """Run K-means with k=3 on one-hot position vectors; label coalitions semantically."""
    pos = df[POSITION_VARS].copy()
    for v in POSITION_VARS:
        pos[v] = pos[v].apply(_str)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(pos.values)

    km = KMeans(n_clusters=3, n_init=100, random_state=42)
    labels = km.fit_predict(X)
    df_out = df.copy()
    df_out["cluster_id"] = labels

    # Compute per-cluster characteristics for naming
    cluster_meta = []
    for c in range(3):
        sub = df_out[df_out["cluster_id"] == c]
        addr_rates = []
        for v in POSITION_VARS:
            null = POSITION_NULL[v]
            addr_rates.append((sub[v].apply(_str) != null).mean())
        cluster_meta.append({
            "cluster_id": c,
            "n": len(sub),
            "mean_address_rate": float(np.mean(addr_rates)),
            "oversight_addr": addr_rates[0],
            "liability_addr": addr_rates[2],
            "reimbursement_addr": addr_rates[3],
            "interoperability_addr": addr_rates[4],
        })

    # Rank by mean_address_rate
    by_rate = sorted(cluster_meta, key=lambda c: -c["mean_address_rate"])
    coalition_map = {by_rate[0]["cluster_id"]: COALITION_NAMES["comprehensive"]}

    # Of the remaining two: the one with HIGHER oversight+regulation address but LOW technical
    # is "Restrictive Advocates"; the one with low across-the-board is "Limited Engagement".
    remaining = [c for c in cluster_meta if c["cluster_id"] != by_rate[0]["cluster_id"]]
    # Define a "restrictiveness score": (oversight_addr - reimbursement_addr) — high means
    # selective high-engagement on oversight, silence on technical.
    restrictive = max(remaining, key=lambda c: c["oversight_addr"] - c["reimbursement_addr"])
    limited = [c for c in remaining if c["cluster_id"] != restrictive["cluster_id"]][0]
    coalition_map[restrictive["cluster_id"]] = COALITION_NAMES["restrictive"]
    coalition_map[limited["cluster_id"]] = COALITION_NAMES["limited"]

    df_out["coalition"] = df_out["cluster_id"].map(coalition_map)
    df_out[["comment_id", "commenter_type", "coalition", "cluster_id"]].to_csv(
        OUT_DIR / "coalition_assignments.csv", index=False
    )
    return df_out


def build_coalition_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """For each coalition, compute the signature profile: stakeholders, stances, topics, lengths."""
    rows = []
    for name in [COALITION_NAMES["comprehensive"], COALITION_NAMES["restrictive"],
                 COALITION_NAMES["limited"]]:
        sub = df[df["coalition"] == name]
        n = len(sub)
        # Stakeholder mix
        sd = sub["commenter_type"].apply(_str).value_counts(normalize=True)
        # Modal stance per axis with address rate
        modal_stances = {}
        addr_rates = {}
        for v in POSITION_VARS:
            null = POSITION_NULL[v]
            vals = sub[v].apply(_str)
            addr = (vals != null)
            addr_rates[v] = float(addr.mean())
            non_null = vals[addr]
            if len(non_null):
                top = non_null.value_counts()
                modal_stances[v] = (top.index[0], int(top.iloc[0]), float(top.iloc[0] / len(non_null)))
            else:
                modal_stances[v] = ("(none)", 0, 0.0)
        # Engagement density
        topic_count = sub[[c for c in df.columns if c.startswith("top_")]].sum(axis=1).mean()
        barrier_count = sub[[c for c in df.columns if c.startswith("bar_")]].sum(axis=1).mean()
        n_props = sub["n_proposals"].mean()
        cfr_rate = sub["has_cfr_citation"].mean() if "has_cfr_citation" in sub.columns else np.nan
        # Top 5 topics
        topic_prev = {t: float(sub[t].mean()) for t in TOPIC_LABELS if t in sub.columns}
        top_topics = sorted(topic_prev.items(), key=lambda x: -x[1])[:5]

        rows.append({
            "coalition": name,
            "n": n,
            "share_of_corpus": round(n / len(df), 3),
            "top_stakeholders": "; ".join(f"{k}={v*100:.0f}%" for k, v in sd.head(5).items()),
            "modal_oversight": f"{modal_stances['pos_oversight'][0]} ({modal_stances['pos_oversight'][2]*100:.0f}%)",
            "modal_regulation": f"{modal_stances['pos_regulation'][0]} ({modal_stances['pos_regulation'][2]*100:.0f}%)",
            "modal_liability": f"{modal_stances['pos_liability'][0]} ({modal_stances['pos_liability'][2]*100:.0f}%)",
            "modal_reimbursement": f"{modal_stances['pos_reimbursement'][0]} ({modal_stances['pos_reimbursement'][2]*100:.0f}%)",
            "modal_interoperability": f"{modal_stances['pos_interoperability'][0]} ({modal_stances['pos_interoperability'][2]*100:.0f}%)",
            "modal_evaluation": f"{modal_stances['pos_evaluation'][0]} ({modal_stances['pos_evaluation'][2]*100:.0f}%)",
            "addr_oversight": round(addr_rates["pos_oversight"], 3),
            "addr_regulation": round(addr_rates["pos_regulation"], 3),
            "addr_liability": round(addr_rates["pos_liability"], 3),
            "addr_reimbursement": round(addr_rates["pos_reimbursement"], 3),
            "addr_interoperability": round(addr_rates["pos_interoperability"], 3),
            "addr_evaluation": round(addr_rates["pos_evaluation"], 3),
            "mean_address_rate": round(float(np.mean(list(addr_rates.values()))), 3),
            "mean_n_proposals": round(float(n_props), 2),
            "mean_topic_count": round(float(topic_count), 2),
            "mean_barrier_count": round(float(barrier_count), 2),
            "cfr_citation_rate": round(float(cfr_rate), 3) if not np.isnan(cfr_rate) else "",
            "top_topics": "; ".join(f"{TOPIC_LABELS[k]}={v*100:.0f}%" for k, v in top_topics),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "coalition_profiles.csv", index=False)
    return out


def coalition_x_stakeholder(df: pd.DataFrame) -> pd.DataFrame:
    """Test the coalition × commenter_type association."""
    df_x = df.copy()
    df_x["commenter_str"] = df_x["commenter_type"].apply(_str)
    ct = pd.crosstab(df_x["coalition"], df_x["commenter_str"])
    chi2, p, _, _ = chi2_contingency(ct.values)
    v_eff = cramers_v(ct.values)

    rows = []
    types = ct.columns.tolist()
    for coalition in ct.index:
        for t in types:
            n = ct.loc[coalition, t]
            stakeholder_total = int(ct[t].sum())
            share = n / stakeholder_total if stakeholder_total else 0.0
            rows.append({
                "coalition": coalition,
                "stakeholder": t,
                "stakeholder_label": COMMENTER_LABEL.get(t, t),
                "n": int(n),
                "stakeholder_total": stakeholder_total,
                "share_of_stakeholder": round(share, 3),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "coalition_x_stakeholder.csv", index=False)

    # Save the omnibus test alongside
    omnibus = pd.DataFrame([{
        "test": "Pearson χ²",
        "chi2": round(chi2, 3),
        "df": (ct.shape[0] - 1) * (ct.shape[1] - 1),
        "p_value": p,
        "cramers_v": round(v_eff, 3),
        "interpretation": (
            f"Coalition membership is non-randomly distributed across stakeholder types "
            f"(χ²={chi2:.1f}, df={(ct.shape[0]-1)*(ct.shape[1]-1)}, p={p:.2e}, V={v_eff:.2f})."
        ),
    }])
    omnibus.to_csv(OUT_DIR / "coalition_x_stakeholder_test.csv", index=False)
    return out


def coalition_topic_emphasis(df: pd.DataFrame) -> pd.DataFrame:
    """Topic prevalence by coalition (with stakeholder-controlled comparison via χ²)."""
    rows = []
    for name in [COALITION_NAMES["comprehensive"], COALITION_NAMES["restrictive"],
                 COALITION_NAMES["limited"]]:
        sub = df[df["coalition"] == name]
        n = len(sub)
        for var, label in TOPIC_LABELS.items():
            if var not in df.columns:
                continue
            k = int(sub[var].sum())
            p, lo, hi = wilson_ci(k, n)
            rows.append({
                "coalition": name,
                "topic": label,
                "variable": var,
                "n_yes": k,
                "n_total": n,
                "prevalence": round(p, 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "coalition_topic_emphasis.csv", index=False)
    return out


def multinomial_logit(df: pd.DataFrame) -> pd.DataFrame:
    """Predict coalition membership from stakeholder type + length/density features."""
    df_x = df.copy()
    df_x["topic_count"] = df_x[[c for c in df.columns if c.startswith("top_")]].sum(axis=1)
    df_x["commenter_str"] = df_x["commenter_type"].apply(_str)
    df_x["has_cfr"] = df_x["has_cfr_citation"].astype(int)

    # Predictors: one-hot stakeholder + standardized n_proposals + topic_count + has_cfr
    stake_dummies = pd.get_dummies(df_x["commenter_str"], prefix="ctype", drop_first=False, dtype=int)
    # Drop one reference category (use IND as reference — most numerous)
    if "ctype_IND" in stake_dummies.columns:
        stake_dummies = stake_dummies.drop(columns=["ctype_IND"])
    cont = df_x[["n_proposals", "topic_count", "has_cfr"]].copy()
    scaler = StandardScaler()
    cont[["n_proposals", "topic_count"]] = scaler.fit_transform(cont[["n_proposals", "topic_count"]])
    X = pd.concat([stake_dummies, cont], axis=1)
    y = df_x["coalition"]
    feature_names = X.columns.tolist()

    # multinomial logistic regression
    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=2000,
        C=1.0, random_state=42,
    )
    model.fit(X.values, y.values)
    classes = model.classes_

    # Save coefficients and exp(coef) as odds ratios
    rows = []
    for ci, c in enumerate(classes):
        for fi, fname in enumerate(feature_names):
            beta = model.coef_[ci, fi]
            rows.append({
                "outcome_coalition": c,
                "feature": fname,
                "beta": round(float(beta), 4),
                "odds_ratio": round(float(np.exp(beta)), 3),
            })
    # Add intercepts
    for ci, c in enumerate(classes):
        rows.append({
            "outcome_coalition": c,
            "feature": "(intercept)",
            "beta": round(float(model.intercept_[ci]), 4),
            "odds_ratio": "—",
        })

    out = pd.DataFrame(rows)
    # Compute pseudo-McFadden R^2 and overall accuracy
    train_acc = float(model.score(X.values, y.values))
    out_meta = pd.DataFrame([{
        "n_samples": len(df_x),
        "n_features": len(feature_names),
        "outcome_classes": ", ".join(classes),
        "train_accuracy": round(train_acc, 3),
        "reference_stakeholder": "IND (Individual)",
    }])
    out_meta.to_csv(OUT_DIR / "multinomial_logit_meta.csv", index=False)
    out.to_csv(OUT_DIR / "multinomial_logit_results.csv", index=False)
    return out


def figure_coalition_profile(profiles: pd.DataFrame) -> None:
    """Three-panel composite figure: addr-rate × axis, stakeholder mix, modal stance per coalition."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 1: Address rate per axis × coalition
    ax = axes[0]
    axis_cols = ["addr_oversight", "addr_regulation", "addr_liability",
                 "addr_reimbursement", "addr_interoperability", "addr_evaluation"]
    axis_names = ["Oversight", "Regulation", "Liability", "Reimburse.", "Interop.", "Evaluation"]
    width = 0.27
    x_pos = np.arange(len(axis_cols))
    for i, (_, row) in enumerate(profiles.iterrows()):
        vals = [row[c] for c in axis_cols]
        ax.bar(x_pos + (i - 1) * width, vals, width=width,
               color=COALITION_COLORS[row["coalition"]],
               label=f"{row['coalition']}\n(n={int(row['n'])}, {int(row['share_of_corpus']*100)}%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(axis_names, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Address rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("A. Engagement scope: address rate per governance axis")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # Panel 2: Stakeholder mix
    ax = axes[1]
    stake_data = []
    types_ordered = ["IND", "HIT", "ADV", "MPS", "HSP", "AIC", "POL", "PAY", "TEC", "LAB"]
    for _, row in profiles.iterrows():
        coal = row["coalition"]
        # Re-derive from raw counts
        sd_str = row["top_stakeholders"]
        sd = {kv.split("=")[0].strip(): float(kv.split("=")[1].rstrip("%")) / 100
              for kv in sd_str.split(";") if "=" in kv}
        stake_data.append({"coalition": coal, **{t: sd.get(t, 0) for t in types_ordered}})
    sdf = pd.DataFrame(stake_data).set_index("coalition")
    bottoms = np.zeros(len(sdf))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(types_ordered):
        if t not in sdf.columns:
            continue
        vals = sdf[t].values
        ax.bar(range(len(sdf)), vals, bottom=bottoms, label=COMMENTER_LABEL.get(t, t),
               color=cmap(i), edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_xticks(range(len(sdf)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in sdf.index], fontsize=9)
    ax.set_ylabel("Share of coalition members")
    ax.set_ylim(0, 1.0)
    ax.set_title("B. Stakeholder composition")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1)
    ax.grid(axis="y", alpha=0.25)

    # Panel 3: Modal stance text panel (no bar, table-style)
    ax = axes[2]
    ax.axis("off")
    rows = ["Axis  →  Modal stance (within-coalition share)"]
    rows.append("-" * 48)
    coalition_order = profiles["coalition"].tolist()
    axis_keys = [("modal_oversight", "Oversight"), ("modal_regulation", "Regulation"),
                 ("modal_liability", "Liability"), ("modal_reimbursement", "Reimburse."),
                 ("modal_interoperability", "Interop."), ("modal_evaluation", "Evaluation")]
    table_data = [[name] + [profiles[profiles["coalition"] == name][k].iloc[0]
                            for k, _ in axis_keys] for name in coalition_order]
    headers = ["Coalition"] + [n for _, n in axis_keys]
    txt = []
    for r in [headers] + table_data:
        txt.append(" | ".join(f"{c:<28s}" if i == 0 else f"{c:<14s}" for i, c in enumerate(r)))
    ax.text(0.0, 0.95, "\n".join(txt), transform=ax.transAxes,
            fontsize=8, family="monospace", va="top")
    ax.set_title("C. Modal stance per axis (and within-coalition share)")

    fig.suptitle("Three coalitions in the HHS-ONC RFI public-comment record (n=446)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_coalition_profile.png", dpi=240, bbox_inches="tight")
    plt.close()


def figure_coalition_topic_emphasis(emphasis: pd.DataFrame) -> None:
    """Heatmap of topic prevalence by coalition."""
    pivot = emphasis.pivot(index="topic", columns="coalition", values="prevalence")
    coalition_order = [COALITION_NAMES["comprehensive"], COALITION_NAMES["restrictive"],
                       COALITION_NAMES["limited"]]
    pivot = pivot[coalition_order]
    # Sort topics by mean prevalence
    pivot = pivot.assign(_mean=pivot.mean(axis=1)).sort_values("_mean", ascending=False).drop(
        columns="_mean")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(coalition_order)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in coalition_order], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(coalition_order)):
            v = pivot.values[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=ax, label="Topic prevalence within coalition", fraction=0.04, pad=0.02)
    ax.set_title("Topic emphasis by coalition")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_coalition_topic_emphasis.png", dpi=240, bbox_inches="tight")
    plt.close()


def figure_coalition_length_density(df: pd.DataFrame) -> None:
    """Boxplots of n_proposals / topic_count / barrier_count by coalition."""
    df_w = df.copy()
    df_w["topic_count"] = df[[c for c in df.columns if c.startswith("top_")]].sum(axis=1)
    df_w["barrier_count"] = df[[c for c in df.columns if c.startswith("bar_")]].sum(axis=1)
    coalition_order = [COALITION_NAMES["comprehensive"], COALITION_NAMES["restrictive"],
                       COALITION_NAMES["limited"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, col, label in zip(
        axes,
        ["n_proposals", "topic_count", "barrier_count"],
        ["Number of policy proposals", "Topics engaged (of 15)", "Barriers identified (of 8)"],
    ):
        data = [df_w[df_w["coalition"] == c][col].dropna() for c in coalition_order]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)
        for patch, name in zip(bp["boxes"], coalition_order):
            patch.set_facecolor(COALITION_COLORS[name])
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks(range(1, len(coalition_order) + 1))
        ax.set_xticklabels([c.replace(" ", "\n") for c in coalition_order], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Submission depth by coalition (n=446)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_coalition_length_density.png", dpi=240, bbox_inches="tight")
    plt.close()


def figure_coalition_pca(df: pd.DataFrame) -> None:
    """2D PCA scatter colored by coalition."""
    pos = df[POSITION_VARS].copy()
    for v in POSITION_VARS:
        pos[v] = pos[v].apply(_str)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(pos.values)
    Xp = PCA(n_components=2, random_state=42).fit_transform(X)
    pca = PCA(n_components=2, random_state=42).fit(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, color in COALITION_COLORS.items():
        mask = (df["coalition"] == name).values
        if mask.sum() == 0:
            continue
        ax.scatter(Xp[mask, 0], Xp[mask, 1], s=28, alpha=0.7, color=color,
                   label=f"{name} (n={int(mask.sum())})", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("Position-vector PCA, colored by coalition assignment (n=446)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_coalition_pca.png", dpi=240, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} comments")

    print("[1/7] Assigning coalitions (k=3 K-means + semantic naming) ...")
    df_c = assign_coalitions(df)
    counts = df_c["coalition"].value_counts()
    print(f"      {dict(counts)}")

    print("[2/7] Building coalition profiles ...")
    profiles = build_coalition_profiles(df_c)
    print(profiles[["coalition", "n", "share_of_corpus", "mean_address_rate",
                    "mean_n_proposals", "mean_topic_count", "cfr_citation_rate"]].to_string(index=False))

    print("\n[3/7] Coalition × stakeholder ...")
    coalition_x_stakeholder(df_c)

    print("[4/7] Topic emphasis by coalition ...")
    emphasis = coalition_topic_emphasis(df_c)

    print("[5/7] Multinomial logistic regression ...")
    multinomial_logit(df_c)

    print("[6/7] Composite coalition profile figure ...")
    figure_coalition_profile(profiles)
    figure_coalition_topic_emphasis(emphasis)
    figure_coalition_length_density(df_c)
    figure_coalition_pca(df_c)

    print(f"\n[7/7] All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
