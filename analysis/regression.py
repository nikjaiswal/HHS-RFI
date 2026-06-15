"""Multinomial logistic regression of coalition membership — de-tautologized.

Critique fix: the previous regression used topic_engagement_count as a predictor, but
coalitions are partly defined by engagement scope, making that predictor quasi-tautological
with the outcome. This module removes topic_count from the predictor set and reports:

  - Adjusted odds ratios with 95% percentile bootstrap CIs (1000 resamples)
  - Reference category: Individual (IND)
  - Predictors: stakeholder type (one-hot, drop IND), z-standardized n_proposals,
    z-standardized total_chars (where available), CFR-citation indicator
  - McFadden pseudo-R² and training accuracy

All outputs go to output/regression/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v)
    return s.split(".")[-1] if "." in s else s


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build predictors: stakeholder one-hot (drop IND) + z(n_proposals) + z(total_chars) + has_cfr."""
    df_x = df.copy()
    df_x["commenter_str"] = df_x["commenter_type"].apply(_str)
    df_x["has_cfr"] = df_x["has_cfr_citation"].astype(int)
    df_x["log_total_chars"] = np.log1p(df_x["total_chars"].fillna(0))

    stake_dummies = pd.get_dummies(df_x["commenter_str"], prefix="ctype",
                                   drop_first=False, dtype=int)
    if "ctype_IND" in stake_dummies.columns:
        stake_dummies = stake_dummies.drop(columns=["ctype_IND"])

    cont = df_x[["n_proposals", "log_total_chars", "has_cfr"]].copy()
    scaler = StandardScaler()
    cont[["n_proposals", "log_total_chars"]] = scaler.fit_transform(
        cont[["n_proposals", "log_total_chars"]]
    )
    X = pd.concat([stake_dummies, cont], axis=1)
    feature_names = X.columns.tolist()
    return X, feature_names


def fit_logit(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=2000,
        C=1.0, random_state=42,
    )
    model.fit(X, y)
    classes = model.classes_.tolist()
    # Coefficients
    rows = []
    for ci, c in enumerate(classes):
        for fi, fname in enumerate(feature_names):
            rows.append({
                "outcome_coalition": c,
                "feature": fname,
                "beta": float(model.coef_[ci, fi]),
                "odds_ratio": float(np.exp(model.coef_[ci, fi])),
            })
    coef_df = pd.DataFrame(rows)

    # McFadden pseudo R²: 1 - L_full / L_null
    log_loss_full = -np.sum(np.log(np.maximum(model.predict_proba(X)[
        np.arange(len(y)), [classes.index(yi) for yi in y]
    ], 1e-12)))
    null_probs = pd.Series(y).value_counts(normalize=True)
    log_loss_null = -sum(np.log(null_probs[c]) * (y == c).sum() for c in classes)
    mcfadden = 1 - log_loss_full / log_loss_null

    return {
        "model": model,
        "classes": classes,
        "feature_names": feature_names,
        "coef_df": coef_df,
        "train_accuracy": float(model.score(X, y)),
        "mcfadden_r2": float(mcfadden),
    }


def bootstrap_ors(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                  B: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    classes = sorted(set(y))
    or_samples = {(c, f): [] for c in classes for f in feature_names}

    for b in range(B):
        idx = rng.randint(0, n, size=n)
        try:
            m = LogisticRegression(
                multi_class="multinomial", solver="lbfgs", max_iter=1000,
                C=1.0, random_state=seed + b,
            ).fit(X[idx], y[idx])
        except Exception:
            continue
        # Re-align coef rows to canonical class order
        m_classes = m.classes_.tolist()
        for ci, c in enumerate(m_classes):
            for fi, fname in enumerate(feature_names):
                or_samples[(c, fname)].append(float(np.exp(m.coef_[ci, fi])))

    rows = []
    for (c, fname), vals in or_samples.items():
        if len(vals) < 50:
            rows.append({
                "outcome_coalition": c, "feature": fname,
                "or_median": float("nan"), "or_ci_lo": float("nan"), "or_ci_hi": float("nan"),
                "n_bootstrap_valid": len(vals),
            })
            continue
        rows.append({
            "outcome_coalition": c, "feature": fname,
            "or_median": float(np.median(vals)),
            "or_ci_lo": float(np.percentile(vals, 2.5)),
            "or_ci_hi": float(np.percentile(vals, 97.5)),
            "n_bootstrap_valid": len(vals),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_data()
    coalition_path = ROOT / "output" / "coalitions" / "coalition_assignments.csv"
    if not coalition_path.exists():
        print("ERROR: coalition_assignments.csv missing — run coalition_analysis.py first.")
        return
    coalitions = pd.read_csv(coalition_path)
    df = df.merge(coalitions[["comment_id", "coalition"]], on="comment_id", how="inner")
    print(f"Loaded {len(df)} comments with coalition assignments")

    X_df, feature_names = build_design_matrix(df)
    X = X_df.values.astype(float)
    y = df["coalition"].values

    print("\n[1/3] Fitting multinomial logit (de-tautologized: no topic_count)...")
    fit = fit_logit(X, y, feature_names)
    print(f"  Reference category: IND (Individual)")
    print(f"  Predictors: {feature_names}")
    print(f"  Train accuracy: {fit['train_accuracy']:.3f}")
    print(f"  McFadden pseudo-R²: {fit['mcfadden_r2']:.3f}")

    print("\n[2/3] Bootstrap CIs (B=1000) for adjusted odds ratios ...")
    boot = bootstrap_ors(X, y, feature_names, B=1000)

    # Merge point estimates with bootstrap CIs
    final = fit["coef_df"].merge(boot, on=["outcome_coalition", "feature"], how="left")
    final = final[["outcome_coalition", "feature", "beta", "odds_ratio",
                   "or_median", "or_ci_lo", "or_ci_hi", "n_bootstrap_valid"]]
    final = final.round(4)
    final.to_csv(OUT_DIR / "multinomial_logit_with_bootstrap_ci.csv", index=False)

    print("\n[3/3] Highlights — significant predictors (95% CI excludes 1.0):")
    sig = final[(final["or_ci_lo"] > 1.0) | (final["or_ci_hi"] < 1.0)].copy()
    sig = sig.sort_values(["outcome_coalition", "or_median"], ascending=[True, False])
    print(sig[["outcome_coalition", "feature", "odds_ratio",
               "or_ci_lo", "or_ci_hi"]].to_string(index=False))

    # Save metadata for manuscript
    meta = {
        "n_samples": int(X.shape[0]),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "outcome_classes": fit["classes"],
        "reference_stakeholder": "IND (Individual)",
        "train_accuracy": fit["train_accuracy"],
        "mcfadden_pseudo_r2": fit["mcfadden_r2"],
        "predictors_excluded": ["topic_count (excluded as quasi-tautological with engagement-defined outcome)"],
        "n_significant_predictors_95ci": len(sig),
    }
    (OUT_DIR / "multinomial_logit_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
