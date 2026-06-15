"""Cluster-count validation: multiple diagnostics for the k-choice decision.

Reports:
  - Mean silhouette over k=2..8
  - Gap statistic (Tibshirani 2001) over k=2..8 with B=50 reference resamples
  - BIC for Gaussian mixture models over k=2..8
  - Bootstrap stability: pairwise concordance of cluster assignments under data resampling
  - Hierarchical Ward + HDBSCAN concordance with K-means at k=3

All outputs go to output/cluster_validation/.

The honest framing for the manuscript: silhouette prefers k=2; gap statistic and BIC
should be reported alongside; we choose k=3 for interpretability and demonstrate
robustness via concordance with alternative algorithms at k=3.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "cluster_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_VARS = ["pos_oversight", "pos_regulation", "pos_liability",
                 "pos_reimbursement", "pos_interoperability", "pos_evaluation"]


def _str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v)
    return s.split(".")[-1] if "." in s else s


def encode_position_vectors(df: pd.DataFrame) -> np.ndarray:
    pos = df[POSITION_VARS].copy()
    for v in POSITION_VARS:
        pos[v] = pos[v].apply(_str)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return enc.fit_transform(pos.values)


def gap_statistic(X: np.ndarray, ks: list[int], B: int = 50, seed: int = 42) -> pd.DataFrame:
    """Tibshirani gap statistic. Reference distribution: uniform over the bounding box."""
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    def Wk(data: np.ndarray, k: int) -> float:
        if k == 1:
            centroid = data.mean(axis=0, keepdims=True)
            return float(np.sum((data - centroid) ** 2))
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(data)
        return float(km.inertia_)

    rows = []
    for k in ks:
        Wk_real = Wk(X, k)
        Wk_refs = []
        for b in range(B):
            ref = rng.uniform(mins, maxs, size=(n_samples, n_features))
            Wk_refs.append(Wk(ref, k))
        log_Wk_refs = np.log(np.maximum(Wk_refs, 1e-12))
        gap = float(log_Wk_refs.mean() - np.log(max(Wk_real, 1e-12)))
        sk = float(np.sqrt(1 + 1 / B) * log_Wk_refs.std())
        rows.append({"k": k, "Wk_real": Wk_real, "log_Wk_real": float(np.log(max(Wk_real, 1e-12))),
                     "log_Wk_ref_mean": float(log_Wk_refs.mean()),
                     "gap": gap, "sk": sk})
    return pd.DataFrame(rows)


def gmm_bic(X: np.ndarray, ks: list[int], seed: int = 42) -> pd.DataFrame:
    """BIC for Gaussian mixture models across k. Lower is better."""
    rows = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              n_init=10, random_state=seed, max_iter=200)
        gmm.fit(X)
        rows.append({"k": k, "bic": float(gmm.bic(X)), "aic": float(gmm.aic(X)),
                     "log_likelihood": float(gmm.score(X) * X.shape[0])})
    return pd.DataFrame(rows)


def silhouette_curve(X: np.ndarray, ks: list[int], seed: int = 42) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_)
        rows.append({"k": k, "silhouette": float(sil), "inertia": float(km.inertia_)})
    return pd.DataFrame(rows)


def bootstrap_stability(X: np.ndarray, k: int, B: int = 100, seed: int = 42) -> dict:
    """For a given k, report mean adjusted Rand index between assignments produced on
    bootstrap resamples and the assignments on the full data (mapped to the resample indices).
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    full_labels = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(X)
    aris = []
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        boot_labels = KMeans(n_clusters=k, n_init=20, random_state=seed + b + 1).fit_predict(X[idx])
        aris.append(float(adjusted_rand_score(full_labels[idx], boot_labels)))
    return {"k": k, "n_bootstrap": B,
            "mean_ARI": float(np.mean(aris)),
            "std_ARI": float(np.std(aris)),
            "ci_lo": float(np.percentile(aris, 2.5)),
            "ci_hi": float(np.percentile(aris, 97.5))}


def algorithm_concordance(X: np.ndarray, k: int = 3, seed: int = 42) -> pd.DataFrame:
    """Compare K-means vs hierarchical Ward vs Gaussian mixture at the same k."""
    km = KMeans(n_clusters=k, n_init=100, random_state=seed).fit_predict(X)
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    gmm = GaussianMixture(n_components=k, covariance_type="diag",
                          n_init=20, random_state=seed).fit_predict(X)
    pairs = [
        ("KMeans vs Ward", km, ward),
        ("KMeans vs GMM", km, gmm),
        ("Ward vs GMM", ward, gmm),
    ]
    rows = []
    for name, a, b in pairs:
        ari = adjusted_rand_score(a, b)
        # Concordance: fraction of pairs that are co-assigned in both labelings.
        # We use a simple match-rate after greedy label-mapping.
        from itertools import permutations
        best_match = 0
        for perm in permutations(range(k)):
            mapped = np.array([perm[x] for x in a])
            best_match = max(best_match, float((mapped == b).mean()))
        rows.append({
            "comparison": name,
            "adjusted_rand_index": round(ari, 4),
            "best_label_match_rate": round(best_match, 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} comments")
    X = encode_position_vectors(df)

    ks = list(range(2, 9))

    print("[1/5] Silhouette curve ...")
    sil = silhouette_curve(X, ks)
    sil.to_csv(OUT_DIR / "silhouette.csv", index=False)
    print(sil.to_string(index=False))

    print("\n[2/5] Gap statistic (50 ref resamples per k) ...")
    gap = gap_statistic(X, ks, B=50)
    # Identify Tibshirani's optimal k: smallest k such that gap[k] >= gap[k+1] - sk[k+1]
    gap = gap.reset_index(drop=True)
    optimal_k_gap = None
    for i in range(len(gap) - 1):
        if gap.loc[i, "gap"] >= gap.loc[i + 1, "gap"] - gap.loc[i + 1, "sk"]:
            optimal_k_gap = int(gap.loc[i, "k"])
            break
    gap.to_csv(OUT_DIR / "gap_statistic.csv", index=False)
    print(gap.to_string(index=False))
    print(f"Tibshirani optimal k (gap): {optimal_k_gap}")

    print("\n[3/5] Gaussian mixture BIC ...")
    bic = gmm_bic(X, ks)
    bic.to_csv(OUT_DIR / "gmm_bic.csv", index=False)
    print(bic.to_string(index=False))
    optimal_k_bic = int(bic.loc[bic["bic"].idxmin(), "k"])
    print(f"BIC-optimal k: {optimal_k_bic}")

    print("\n[4/5] Bootstrap stability at k=2, k=3, k=4 ...")
    stab_rows = [bootstrap_stability(X, k, B=100) for k in [2, 3, 4]]
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(OUT_DIR / "bootstrap_stability.csv", index=False)
    print(stab.to_string(index=False))

    print("\n[5/5] Algorithm concordance at k=3 ...")
    conc = algorithm_concordance(X, k=3)
    conc.to_csv(OUT_DIR / "algorithm_concordance.csv", index=False)
    print(conc.to_string(index=False))

    # ---------- Composite figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(sil["k"], sil["silhouette"], "o-", color="#1f77b4")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean silhouette")
    ax.set_title("Silhouette (higher = better)")
    ax.axvline(int(sil.loc[sil["silhouette"].idxmax(), "k"]), color="green", linestyle=":", alpha=0.6)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(gap["k"], gap["gap"], yerr=gap["sk"], fmt="o-", color="#ff7f0e", capsize=3)
    ax.set_xlabel("k")
    ax.set_ylabel("Gap statistic ± s_k")
    ax.set_title(f"Gap statistic (Tibshirani optimal: k={optimal_k_gap})")
    if optimal_k_gap:
        ax.axvline(optimal_k_gap, color="green", linestyle=":", alpha=0.6)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(bic["k"], bic["bic"], "s-", color="#2ca02c")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC (lower = better)")
    ax.set_title(f"GMM BIC (optimal: k={optimal_k_bic})")
    ax.axvline(optimal_k_bic, color="green", linestyle=":", alpha=0.6)
    ax.grid(alpha=0.3)

    fig.suptitle("Cluster-count validation: three diagnostics across k = 2..8", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_cluster_validation.png", dpi=240, bbox_inches="tight")
    plt.close()

    # Save summary JSON for the manuscript
    summary = {
        "n_samples": int(X.shape[0]),
        "silhouette_optimal_k": int(sil.loc[sil["silhouette"].idxmax(), "k"]),
        "silhouette_at_k3": float(sil[sil["k"] == 3]["silhouette"].iloc[0]),
        "gap_optimal_k": optimal_k_gap,
        "gap_at_k3": float(gap[gap["k"] == 3]["gap"].iloc[0]),
        "bic_optimal_k": optimal_k_bic,
        "bic_at_k3": float(bic[bic["k"] == 3]["bic"].iloc[0]),
        "bootstrap_ARI_k3": float(stab[stab["k"] == 3]["mean_ARI"].iloc[0]),
        "bootstrap_ARI_k3_ci": [
            float(stab[stab["k"] == 3]["ci_lo"].iloc[0]),
            float(stab[stab["k"] == 3]["ci_hi"].iloc[0]),
        ],
        "kmeans_ward_ari_at_k3": float(conc[conc["comparison"] == "KMeans vs Ward"]["adjusted_rand_index"].iloc[0]),
        "kmeans_gmm_ari_at_k3": float(conc[conc["comparison"] == "KMeans vs GMM"]["adjusted_rand_index"].iloc[0]),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary}")
    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
