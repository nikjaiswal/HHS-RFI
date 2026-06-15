"""Coalition × RFI-question coverage analysis.

The RFI poses 10 specific questions. Each comment's coded data includes
`rfi_questions: list[int]` — the questions the comment substantively engaged
with. We map question coverage per coalition, by stakeholder, and overall to
identify which RFI questions HHS got the deepest input on and from whom.

Outputs:
  output/rfi_coverage/
    rfi_x_coalition.csv         coverage rate per RFI question x coalition
    rfi_x_stakeholder.csv       coverage rate per RFI question x commenter type
    rfi_overall_prevalence.csv  per-question prevalence in the full corpus + 95% Wilson CI
    fig_rfi_x_coalition.png     heatmap (10 questions x 3 coalitions)
    fig_rfi_x_stakeholder.png   heatmap (10 questions x 10 stakeholders)
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
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "rfi_coverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_rfi_assignments(jsonl_path: Path) -> dict[str, list[int]]:
    cid_to_q: dict[str, list[int]] = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("_meta", {})
            cid = meta.get("comment_id") or meta.get("id")
            if cid:
                cid_to_q[cid] = rec.get("supplementary", {}).get("rfi_questions", []) or []
    return cid_to_q


def main() -> None:
    df = load_data()
    rfi = load_rfi_assignments(ROOT / "output" / "coded_comments.jsonl")
    df["rfi_questions"] = df["comment_id"].map(rfi).apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Merge coalition assignments
    coalitions = pd.read_csv(ROOT / "output" / "coalitions" / "coalition_assignments.csv")
    df = df.merge(coalitions[["comment_id", "coalition"]], on="comment_id", how="inner")
    print(f"Loaded {len(df)} comments with RFI lists and coalition assignments")

    n_questions = 10

    # Overall prevalence
    rows = []
    for q in range(1, n_questions + 1):
        addressed = sum(1 for lst in df["rfi_questions"] if q in lst)
        p, lo, hi = wilson_ci(addressed, len(df))
        rows.append({
            "rfi_question": q,
            "n_addressed": addressed,
            "n_total": len(df),
            "prevalence": round(p, 3),
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
        })
    overall = pd.DataFrame(rows)
    overall.to_csv(OUT_DIR / "rfi_overall_prevalence.csv", index=False)
    print("\nOverall prevalence per RFI question:")
    print(overall.to_string(index=False))

    # By coalition
    coalition_order = ["Comprehensive Pragmatists", "Selective Universalists", "Limited Engagement"]
    rows = []
    for q in range(1, n_questions + 1):
        for c in coalition_order:
            sub = df[df["coalition"] == c]
            n = len(sub)
            if n == 0:
                continue
            addressed = sum(1 for lst in sub["rfi_questions"] if q in lst)
            p, lo, hi = wilson_ci(addressed, n)
            rows.append({
                "rfi_question": q,
                "coalition": c,
                "n_addressed": addressed,
                "n_total": n,
                "prevalence": round(p, 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
            })
    coal_df = pd.DataFrame(rows)
    coal_df.to_csv(OUT_DIR / "rfi_x_coalition.csv", index=False)

    # Heatmap: question × coalition
    pivot = coal_df.pivot(index="rfi_question", columns="coalition", values="prevalence")
    pivot = pivot[coalition_order]
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    im = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(coalition_order)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in coalition_order],
                       fontsize=10)
    ax.set_yticks(range(n_questions))
    ax.set_yticklabels([f"Question {q}" for q in range(1, n_questions + 1)], fontsize=10)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)
    for i in range(n_questions):
        for j in range(len(coalition_order)):
            v = pivot.values[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Coverage rate within coalition", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Coverage of 10 specific RFI questions by coalition (n=446)",
                 fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_rfi_x_coalition.png", dpi=260, bbox_inches="tight")
    plt.close()

    # By stakeholder type — for completeness
    df["commenter_str"] = df["commenter_type"].apply(_str)
    types = sorted(df["commenter_str"].unique())
    rows = []
    for q in range(1, n_questions + 1):
        for t in types:
            sub = df[df["commenter_str"] == t]
            n = len(sub)
            if n == 0:
                continue
            addressed = sum(1 for lst in sub["rfi_questions"] if q in lst)
            rows.append({
                "rfi_question": q,
                "stakeholder": t,
                "n_addressed": addressed,
                "n_total": n,
                "prevalence": round(addressed / n, 3),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "rfi_x_stakeholder.csv", index=False)

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
