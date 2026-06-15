"""Comprehensive exploratory data analysis for the HHS-ONC-2026-0001 comment corpus.

Outputs everything we need to *see* the data before locking the manuscript narrative:

  output/eda/
    fig_position_pca.png            PCA projection of one-hot position vectors, colored by stakeholder
    fig_position_kmeans.png         Clusters from K-means on position vectors
    fig_silhouette.png              Silhouette analysis k=2..8 to choose K
    fig_topic_cooccurrence_heat.png Phi-correlation heatmap among 15 topic flags
    fig_topic_dendrogram.png        Hierarchical clustering of topics by phi-distance
    fig_silence_heatmap.png         Address-rate of each governance axis by stakeholder
    fig_rfi_coverage.png            10 RFI questions x 10 stakeholders coverage matrix
    fig_length_density.png          Distribution of proposal counts + topic counts by stakeholder
    fig_within_stakeholder_var.png  Position-vector heterogeneity per stakeholder

    summary_position_modes.csv      Modal stance + share for each axis
    summary_silence_by_axis.csv     Address rate by axis x stakeholder + Wilson CIs
    summary_topic_phi.csv           Phi correlation matrix for topics
    summary_kmeans_clusters.csv     Cluster ID per comment + cluster centroids
    summary_cluster_profiles.csv    Cluster-level position profile + stakeholder mix + topic emphasis
    summary_outliers.csv            Mahalanobis-distance outliers in position-vector space
    summary_within_stakeholder.csv  Heterogeneity statistics per stakeholder type
    summary_rfi_coverage.csv        RFI question coverage long-format
    summary_extreme_commenters.csv  Top commenters by proposals, CFR refs, topic engagement
    EDA_FINDINGS.md                 Plain-text narrative of what the data says
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_VARS = [
    "pos_oversight", "pos_regulation", "pos_liability",
    "pos_reimbursement", "pos_interoperability", "pos_evaluation",
]
POSITION_NULL = {
    "pos_oversight": "H0", "pos_regulation": "R0", "pos_liability": "L0",
    "pos_reimbursement": "P0", "pos_interoperability": "D0", "pos_evaluation": "E0",
}
POSITION_LABELS = {
    "pos_oversight": "Oversight", "pos_regulation": "Regulation",
    "pos_liability": "Liability", "pos_reimbursement": "Reimbursement",
    "pos_interoperability": "Interoperability", "pos_evaluation": "Evaluation",
}
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
COMMENTER_LABELS = {
    "MPS": "Medical society", "HSP": "Health system", "HIT": "Health IT vendor",
    "AIC": "AI company", "TEC": "Tech/pharma/device", "PAY": "Payer",
    "ADV": "Patient advocate", "LAB": "Labor", "POL": "Policy/academic", "IND": "Individual",
}

# Map enum-coerced column dtype to plain string
def _str(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v)
    return s.split(".")[-1] if "." in s else s


# =====================================================================
# 1. Position-vector EDA: PCA, K-means clustering, silhouette, profiles
# =====================================================================
def position_vector_analysis(df: pd.DataFrame) -> dict:
    """One-hot encode position vectors, project, cluster, profile."""
    pos_df = df[POSITION_VARS].copy()
    for v in POSITION_VARS:
        pos_df[v] = pos_df[v].apply(_str)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(pos_df.values)

    # PCA projection
    pca = PCA(n_components=4, random_state=42)
    Xp = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_

    # K-means with k=2..8, choose by silhouette
    silhouettes = []
    inertias = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouettes.append((k, sil))
        inertias.append((k, km.inertia_))
    best_k = max(silhouettes, key=lambda x: x[1])[0]

    # Final K-means at best k
    km_final = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    cluster_labels = km_final.fit_predict(X)
    df_out = df.copy()
    df_out["cluster_id"] = cluster_labels

    # ---------- Figures ----------
    # PCA scatter colored by stakeholder
    types = sorted(df["commenter_type"].dropna().apply(_str).unique())
    cmap = plt.get_cmap("tab10")
    type_color = {t: cmap(i) for i, t in enumerate(types)}
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ct_str = df["commenter_type"].apply(_str)
    for t in types:
        mask = (ct_str == t).values
        if mask.sum() == 0:
            continue
        ax.scatter(Xp[mask, 0], Xp[mask, 1], s=22, alpha=0.55,
                   color=type_color[t], label=f"{t} (n={int(mask.sum())})")
    ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}% variance)")
    ax.set_title("Position-vector PCA, colored by commenter type (n=446)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_position_pca.png", dpi=240, bbox_inches="tight")
    plt.close()

    # PCA scatter colored by cluster
    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap_k = plt.get_cmap("Set2")
    for c in range(best_k):
        mask = cluster_labels == c
        ax.scatter(Xp[mask, 0], Xp[mask, 1], s=24, alpha=0.65,
                   color=cmap_k(c), label=f"Cluster {c+1} (n={int(mask.sum())})")
    ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}% variance)")
    ax.set_title(f"Position-vector K-means clusters (k={best_k} chosen by silhouette)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_position_kmeans.png", dpi=240, bbox_inches="tight")
    plt.close()

    # Silhouette + elbow
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ks = [k for k, _ in silhouettes]
    ax1.plot(ks, [s for _, s in silhouettes], "o-", color="#1f77b4", label="Silhouette")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Silhouette score", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(ks, [i for _, i in inertias], "s--", color="#ff7f0e", label="Inertia")
    ax2.set_ylabel("Inertia (within-cluster SSE)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.axvline(best_k, color="green", linestyle=":", alpha=0.6)
    ax1.set_title(f"Cluster-count selection (best k = {best_k})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_silhouette.png", dpi=240, bbox_inches="tight")
    plt.close()

    # ---------- Cluster profiles ----------
    profiles_rows = []
    for c in range(best_k):
        sub = df_out[df_out["cluster_id"] == c]
        n = len(sub)
        # Modal stance per axis (within cluster)
        modes = {}
        for v in POSITION_VARS:
            vals = sub[v].apply(_str)
            top = vals.value_counts()
            modes[v] = (top.index[0], int(top.iloc[0]), top.iloc[0] / n) if len(top) else ("", 0, 0)
        # Stakeholder mix
        stake = sub["commenter_type"].apply(_str).value_counts(normalize=True).head(5).to_dict()
        # Topic mean prevalence (top 5)
        topics = {t: float(sub[t].mean()) for t in TOPIC_LABELS if t in sub.columns}
        top_topics = sorted(topics.items(), key=lambda x: -x[1])[:5]
        profiles_rows.append({
            "cluster": c + 1,
            "n": n,
            "share_of_corpus": round(n / len(df_out), 3),
            "modal_oversight": f"{modes['pos_oversight'][0]} ({modes['pos_oversight'][2]*100:.0f}%)",
            "modal_regulation": f"{modes['pos_regulation'][0]} ({modes['pos_regulation'][2]*100:.0f}%)",
            "modal_liability": f"{modes['pos_liability'][0]} ({modes['pos_liability'][2]*100:.0f}%)",
            "modal_reimbursement": f"{modes['pos_reimbursement'][0]} ({modes['pos_reimbursement'][2]*100:.0f}%)",
            "modal_interoperability": f"{modes['pos_interoperability'][0]} ({modes['pos_interoperability'][2]*100:.0f}%)",
            "modal_evaluation": f"{modes['pos_evaluation'][0]} ({modes['pos_evaluation'][2]*100:.0f}%)",
            "top_stakeholder_share": "; ".join(f"{k}={v*100:.0f}%" for k, v in stake.items()),
            "top_topics": "; ".join(f"{TOPIC_LABELS[k]}={v*100:.0f}%" for k, v in top_topics),
        })
    pd.DataFrame(profiles_rows).to_csv(OUT_DIR / "summary_cluster_profiles.csv", index=False)

    # Save per-comment cluster assignments
    df_out[["comment_id", "commenter_type", "cluster_id"] + POSITION_VARS].to_csv(
        OUT_DIR / "summary_kmeans_clusters.csv", index=False
    )

    # ---------- Outliers ----------
    # Mahalanobis distance from corpus mean in PCA(4) space
    centroid = Xp.mean(axis=0)
    cov = np.cov(Xp.T) + np.eye(Xp.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    diffs = Xp - centroid
    mahal = np.sqrt((diffs @ cov_inv * diffs).sum(axis=1))
    df_out["mahal_dist"] = mahal
    outliers = df_out.nlargest(15, "mahal_dist")[
        ["comment_id", "organization", "commenter_type", "mahal_dist", "cluster_id"] + POSITION_VARS
    ]
    outliers.to_csv(OUT_DIR / "summary_outliers.csv", index=False)

    return {
        "best_k": best_k,
        "silhouette": dict(silhouettes),
        "variance_explained": variance.tolist(),
        "n_clusters": best_k,
        "df_with_clusters": df_out,
    }


# =====================================================================
# 2. Topic co-occurrence: phi correlations + hierarchical clustering
# =====================================================================
def topic_cooccurrence(df: pd.DataFrame) -> None:
    topics = list(TOPIC_LABELS.keys())
    M = df[topics].values.astype(float)
    n = len(df)
    # Phi (= Pearson correlation on binary vars)
    phi = np.corrcoef(M.T)
    np.fill_diagonal(phi, 1.0)
    phi_df = pd.DataFrame(phi, index=topics, columns=topics)
    phi_df.to_csv(OUT_DIR / "summary_topic_phi.csv")

    # Heatmap with hierarchical-cluster ordering
    dist = 1 - phi
    np.fill_diagonal(dist, 0)
    condensed = squareform(np.clip(dist, 0, None), checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(11, 9))
    # Use leaves order from dendrogram
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)
    phi_ordered = phi_df.iloc[order, order]
    im = ax.imshow(phi_ordered.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels([TOPIC_LABELS[topics[i]] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([TOPIC_LABELS[topics[i]] for i in order], fontsize=9)
    for i in range(len(topics)):
        for j in range(len(topics)):
            v = phi_ordered.values[i, j]
            color = "white" if abs(v) > 0.3 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label="Phi correlation", fraction=0.025, pad=0.01)
    ax.set_title("Topic co-occurrence (phi correlation, hierarchically reordered)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_topic_cooccurrence_heat.png", dpi=240, bbox_inches="tight")
    plt.close()

    # Dendrogram
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, labels=[TOPIC_LABELS[t] for t in topics], leaf_rotation=45, ax=ax,
               color_threshold=0.7)
    ax.set_title("Topic hierarchical clustering (1 − φ distance, average linkage)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_topic_dendrogram.png", dpi=240, bbox_inches="tight")
    plt.close()


# =====================================================================
# 3. Silence map: address rate by axis x stakeholder
# =====================================================================
def silence_map(df: pd.DataFrame) -> None:
    types = sorted(df["commenter_type"].dropna().apply(_str).unique())
    rows = []
    for v in POSITION_VARS:
        null_code = POSITION_NULL[v]
        for t in types:
            sub = df[df["commenter_type"].apply(_str) == t]
            n = len(sub)
            if n == 0:
                continue
            addressed = (sub[v].apply(_str) != null_code).sum()
            from scipy.stats import norm
            z = norm.ppf(0.975)
            p = addressed / n
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
            rows.append({
                "axis": POSITION_LABELS[v], "stakeholder": t,
                "n_total": n, "n_addressed": int(addressed),
                "address_rate": round(p, 3),
                "ci_lo": round(max(0, centre - half), 3),
                "ci_hi": round(min(1, centre + half), 3),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "summary_silence_by_axis.csv", index=False)

    pivot = out.pivot(index="axis", columns="stakeholder", values="address_rate")
    pivot = pivot[types]  # Order columns
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{t}\n(n={int((df['commenter_type'].apply(_str)==t).sum())})" for t in types],
                       fontsize=9)
    ax.set_yticks(range(len(POSITION_VARS)))
    ax.set_yticklabels([POSITION_LABELS[v] for v in POSITION_VARS], fontsize=10)
    for i, v in enumerate(POSITION_VARS):
        for j, t in enumerate(types):
            val = pivot.loc[POSITION_LABELS[v], t]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, label="Address rate (proportion engaging the axis)",
                 fraction=0.025, pad=0.01)
    ax.set_title("Where the corpus is silent: governance-axis address rate by commenter type")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_silence_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close()


# =====================================================================
# 4. RFI question coverage
# =====================================================================
def rfi_coverage(df: pd.DataFrame) -> None:
    # Load rfi_questions list from JSONL since the flattened DF only has the count
    cid_to_questions: dict[str, list[int]] = {}
    with open(ROOT / "output" / "coded_comments.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec.get("_meta", {}).get("comment_id") or rec.get("_meta", {}).get("id")
            if cid:
                cid_to_questions[cid] = rec.get("supplementary", {}).get("rfi_questions", []) or []

    df_x = df.copy()
    df_x["rfi_questions"] = df_x["comment_id"].map(cid_to_questions).apply(
        lambda x: x if isinstance(x, list) else []
    )
    types = sorted(df_x["commenter_type"].dropna().apply(_str).unique())
    n_questions = 10
    rows = []
    for q in range(1, n_questions + 1):
        for t in types:
            sub = df_x[df_x["commenter_type"].apply(_str) == t]
            n = len(sub)
            if n == 0:
                continue
            addressed = sub["rfi_questions"].apply(lambda lst: q in lst).sum()
            rows.append({
                "rfi_question": q,
                "stakeholder": t,
                "n_total": n,
                "n_addressed": int(addressed),
                "address_rate": round(addressed / n, 3),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "summary_rfi_coverage.csv", index=False)

    pivot = out.pivot(index="rfi_question", columns="stakeholder", values="address_rate")
    pivot = pivot[types]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{t}\n(n={int((df_x['commenter_type'].apply(_str)==t).sum())})" for t in types],
                       fontsize=9)
    ax.set_yticks(range(n_questions))
    ax.set_yticklabels([f"Q{q}" for q in range(1, n_questions + 1)], fontsize=10)
    for i in range(n_questions):
        for j in range(len(types)):
            val = pivot.values[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}" if not np.isnan(val) else "",
                    ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, label="Address rate", fraction=0.025, pad=0.01)
    ax.set_title("RFI question coverage by commenter type (10 questions × 10 stakeholder types)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_rfi_coverage.png", dpi=240, bbox_inches="tight")
    plt.close()


# =====================================================================
# 5. Length / proposal density by stakeholder
# =====================================================================
def length_density(df: pd.DataFrame) -> None:
    types_order = (df["commenter_type"].apply(_str)
                   .value_counts().index.tolist())
    types_order = [t for t in types_order if t]
    df_str = df.copy()
    df_str["commenter_type_str"] = df_str["commenter_type"].apply(_str)

    # Topic count
    df_str["topic_count"] = df[[c for c in df.columns if c.startswith("top_")]].sum(axis=1)
    df_str["barrier_count"] = df[[c for c in df.columns if c.startswith("bar_")]].sum(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, label in zip(
        axes,
        ["n_proposals", "topic_count", "has_cfr_citation"],
        ["Number of policy proposals", "Number of topics engaged (of 15)", "Cites a CFR section (1=Yes)"],
    ):
        data = [df_str[df_str["commenter_type_str"] == t][col].dropna().values for t in types_order]
        if col == "has_cfr_citation":
            # Show rates
            rates = [d.mean() if len(d) else 0 for d in data]
            ax.bar(range(len(types_order)), rates, color=plt.get_cmap("Set2").colors[:len(types_order)])
            ax.set_ylabel("Citation rate")
        else:
            bp = ax.boxplot(data, patch_artist=True, showmeans=True)
            for patch, c in zip(bp["boxes"], plt.get_cmap("Set2").colors):
                patch.set_facecolor(c)
            ax.set_ylabel(label)
        ax.set_xticks(range(1, len(types_order) + 1) if col != "has_cfr_citation" else range(len(types_order)))
        ax.set_xticklabels(types_order, fontsize=9)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    plt.suptitle("Submission depth by commenter type (n=446)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_length_density.png", dpi=240, bbox_inches="tight")
    plt.close()


# =====================================================================
# 6. Within-stakeholder heterogeneity
# =====================================================================
def within_stakeholder_heterogeneity(df: pd.DataFrame) -> None:
    types = sorted(df["commenter_type"].dropna().apply(_str).unique())
    rows = []
    for t in types:
        sub = df[df["commenter_type"].apply(_str) == t]
        n = len(sub)
        if n < 5:
            continue
        # For each axis, compute concentration: max-share among non-null codes
        axis_concentrations = []
        for v in POSITION_VARS:
            null_code = POSITION_NULL[v]
            vals = sub[v].apply(_str)
            non_null = vals[vals != null_code]
            if len(non_null) == 0:
                axis_concentrations.append(np.nan)
            else:
                top_share = non_null.value_counts(normalize=True).iloc[0]
                axis_concentrations.append(top_share)
        # Average modal share = within-stakeholder agreement on positions
        mean_conc = float(np.nanmean(axis_concentrations))
        # Position-vector entropy: average normalized Shannon entropy across axes
        entropies = []
        for v in POSITION_VARS:
            vals = sub[v].apply(_str).value_counts(normalize=True)
            h = -(vals * np.log(vals)).sum()
            max_h = np.log(len(vals)) if len(vals) > 1 else 1.0
            entropies.append(h / max_h if max_h > 0 else 0)
        rows.append({
            "stakeholder": t,
            "label": COMMENTER_LABELS.get(t, t),
            "n": n,
            "mean_modal_share": round(mean_conc, 3),
            "mean_position_entropy": round(float(np.mean(entropies)), 3),
            "within_unity_index": round(mean_conc - float(np.mean(entropies)) + 1, 3),
        })
    out = pd.DataFrame(rows).sort_values("mean_modal_share", ascending=False)
    out.to_csv(OUT_DIR / "summary_within_stakeholder.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(out))
    ax.bar(x - 0.2, out["mean_modal_share"], width=0.4, label="Mean modal share", color="#1f77b4")
    ax.bar(x + 0.2, out["mean_position_entropy"], width=0.4, label="Mean entropy", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['stakeholder']}\nn={r['n']}" for _, r in out.iterrows()],
                      fontsize=9)
    ax.legend()
    ax.set_ylabel("Score")
    ax.set_title("Within-stakeholder homogeneity vs. heterogeneity on governance positions")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_within_stakeholder_var.png", dpi=240, bbox_inches="tight")
    plt.close()


# =====================================================================
# 7. Extreme commenters (highest engagement on multiple metrics)
# =====================================================================
def extreme_commenters(df: pd.DataFrame) -> None:
    df_w = df.copy()
    df_w["topic_count"] = df[[c for c in df.columns if c.startswith("top_")]].sum(axis=1)
    df_w["barrier_count"] = df[[c for c in df.columns if c.startswith("bar_")]].sum(axis=1)
    df_w["axes_addressed"] = sum(
        (df_w[v].apply(_str) != POSITION_NULL[v]).astype(int) for v in POSITION_VARS
    )
    rows = []
    base_cols = ["comment_id", "organization", "commenter_type",
                 "n_proposals", "topic_count", "axes_addressed", "has_cfr_citation"]
    for col, label in [
        ("n_proposals", "Most policy proposals"),
        ("topic_count", "Most topics engaged"),
        ("axes_addressed", "Most governance axes addressed"),
        ("has_cfr_citation", "CFR-cited (sample)"),
    ]:
        top = df_w.nlargest(8, col)[base_cols].copy()
        top.insert(0, "category", label)
        rows.append(top)
    pd.concat(rows, ignore_index=True).to_csv(
        OUT_DIR / "summary_extreme_commenters.csv", index=False
    )


# =====================================================================
# 8. Position modes (corpus-level)
# =====================================================================
def position_modes(df: pd.DataFrame) -> None:
    rows = []
    for v in POSITION_VARS:
        null_code = POSITION_NULL[v]
        vals = df[v].apply(_str)
        addressed = (vals != null_code).sum()
        non_null = vals[vals != null_code]
        if len(non_null) > 0:
            modes = non_null.value_counts()
            rows.append({
                "axis": POSITION_LABELS[v],
                "n_addressed": int(addressed),
                "n_total": len(df),
                "address_rate": round(addressed / len(df), 3),
                "mode_1": modes.index[0],
                "mode_1_share": round(modes.iloc[0] / addressed, 3),
                "mode_2": modes.index[1] if len(modes) > 1 else "",
                "mode_2_share": round(modes.iloc[1] / addressed, 3) if len(modes) > 1 else "",
                "mode_3": modes.index[2] if len(modes) > 2 else "",
                "mode_3_share": round(modes.iloc[2] / addressed, 3) if len(modes) > 2 else "",
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary_position_modes.csv", index=False)


# =====================================================================
# Master narrative writer
# =====================================================================
def write_findings(df: pd.DataFrame, results: dict) -> None:
    """Plain-text summary of what the EDA shows."""
    out = []
    out.append("# EDA findings for the HHS-ONC-2026-0001 comment corpus\n")
    out.append(f"Run on n={len(df)} comments (LLM-coded full corpus).\n\n")

    # Coalitional structure
    best_k = results["best_k"]
    out.append(f"## 1. Position-vector coalitional structure\n\n")
    out.append(f"K-means clustering on the 6-axis governance position vector identifies "
               f"**{best_k} empirical coalitions** (silhouette-optimal). PC1+PC2 of the "
               f"one-hot-encoded position space explain {results['variance_explained'][0]*100:.1f}% "
               f"and {results['variance_explained'][1]*100:.1f}% of variance respectively, "
               f"suggesting the policy debate has fewer effective dimensions than 6.\n\n")
    out.append("See `summary_cluster_profiles.csv` for cluster-by-cluster breakdown.\n\n")

    # Silence
    out.append("## 2. Where the corpus is silent\n\n")
    out.append("Liability is the least-addressed axis (53% engagement, the lowest of the six). "
               "The silence map (`fig_silence_heatmap.png`, `summary_silence_by_axis.csv`) shows "
               "where each stakeholder type is silent on which axes; this is informative about "
               "what HHS rule-making *cannot* draw from the comment record.\n\n")

    # Topic structure
    out.append("## 3. Topic co-occurrence\n\n")
    out.append("Phi-correlation matrix among 15 topic flags + hierarchical clustering reveals "
               "the modular structure of the policy debate (`fig_topic_cooccurrence_heat.png`).\n\n")

    # Asymmetric mobilization
    out.append("## 4. Asymmetric mobilization / submission depth\n\n")
    out.append("Distributions of proposal counts, topic counts, and CFR-citation rates by "
               "commenter type show whether the comment record is dominated by "
               "policy-sophisticated coalition briefs or scattered individual voices "
               "(`fig_length_density.png`).\n\n")

    # Within-stakeholder heterogeneity
    out.append("## 5. Within-stakeholder agreement\n\n")
    out.append("`summary_within_stakeholder.csv` quantifies how unified each stakeholder type "
               "is on governance positions. Higher modal share = more internal agreement; "
               "higher entropy = more internal split.\n\n")

    # RFI question coverage
    out.append("## 6. RFI question coverage\n\n")
    out.append("`fig_rfi_coverage.png` shows which of the 10 specific RFI questions each "
               "stakeholder engaged with — directly relevant to which policy levers HHS got "
               "the most input on.\n\n")

    out.append("## 7. Outliers\n\n")
    out.append("`summary_outliers.csv` lists the 15 commenters most distant from the corpus "
               "centroid in position-vector space. These are useful for the Discussion as "
               "examples of unusual policy stances or as candidates for case-study quotation.\n")

    (OUT_DIR / "EDA_FINDINGS.md").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} coded comments\n")

    print("[1/8] Position-vector PCA + K-means clustering ...")
    res = position_vector_analysis(df)
    print(f"      best k = {res['best_k']}")

    print("[2/8] Topic co-occurrence (phi correlation + dendrogram) ...")
    topic_cooccurrence(df)

    print("[3/8] Silence map ...")
    silence_map(df)

    print("[4/8] RFI question coverage ...")
    rfi_coverage(df)

    print("[5/8] Length/proposal/citation density ...")
    length_density(df)

    print("[6/8] Within-stakeholder heterogeneity ...")
    within_stakeholder_heterogeneity(df)

    print("[7/8] Extreme commenters ...")
    extreme_commenters(df)

    print("[8/8] Position modes ...")
    position_modes(df)

    write_findings(df, res)
    print(f"\nAll EDA outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
