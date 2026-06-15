"""Co-signatory and multi-organization brief detection.

Many comments are submitted as coalition briefs signed by multiple organizations.
We detect these via heuristic patterns in the comment text:
  - "and X other organizations"
  - "the undersigned"
  - "joint comments"
  - lists of organizations in the opening
  - "represents N members"

Outputs:
  output/cosignatory/
    cosignatory_briefs.csv   per-comment flags + detected signatories where extractable
    summary.csv              counts by coalition + stakeholder
    fig_coalition_cosig.png  bar chart of coalition-brief share per coalition
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "cosignatory"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v)
    return s.split(".")[-1] if "." in s else s


def load_comment_texts(path: Path) -> dict[str, str]:
    """Build {comment_id: full_text} from data/comments.csv."""
    csv.field_size_limit(sys.maxsize)
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("id") or "").strip()
            if not cid:
                continue
            inline = (row.get("comment") or "").strip()
            attached = ""
            for i in range(1, 13):
                t = (row.get(f"fullText_{i}") or "").strip()
                if t:
                    attached += " " + t
            out[cid] = (inline + " " + attached).strip()
    return out


# Heuristic indicators of multi-organization briefs
COALITION_PATTERNS = [
    re.compile(r"\bthe undersigned\s+(?:organizations?|associations?|members?|groups?|signatories?)\b", re.I),
    re.compile(r"\b(?:we|the\s+\d+\s+|[A-Z][a-z]+\s+\d+\s+)?(?:organizations?|associations?|signatories)\s+(?:listed|below|jointly)\b", re.I),
    re.compile(r"\bjointly\s+submit\b", re.I),
    re.compile(r"\bjoint(?:ly)?\s+(?:comment|submission|letter|response)s?\b", re.I),
    re.compile(r"\b(?:on behalf of|representing)\s+(?:our|the)\s+(?:\d{1,3}|\w+)\s+(?:member|organization|group|coalition)\b", re.I),
    re.compile(r"\bsigned\s+by\s+(?:more\s+than\s+)?\d{2,}\s+(?:organizations?|associations?)\b", re.I),
    re.compile(r"\b(?:and\s+\d{1,3}\s+other\s+(?:organizations?|signatories?))\b", re.I),
    re.compile(r"\bcoalition\s+of\s+(?:more\s+than\s+)?\d{2,}\b", re.I),
]

MEMBER_PATTERNS = [
    re.compile(r"\b(?:representing|with|of)\s+(?:more\s+than\s+|over\s+|approximately\s+)?([\d,]{2,7}\+?)\s+(?:member|hospital|provider|physician|nurse|patient|consumer|advocate|clinician)\w*\b", re.I),
    re.compile(r"\b([\d,]{2,7}\+?)\s+(?:member|hospital|provider|physician|nurse|patient|consumer|advocate|clinician)s?\b", re.I),
]


def detect_coalition_brief(text: str) -> dict:
    """Return {is_coalition_brief, n_pattern_matches, member_count_hint, evidence_phrase}."""
    if not text:
        return {"is_coalition_brief": False, "n_pattern_matches": 0,
                "member_count_hint": "", "evidence_phrase": ""}
    matches = []
    for pat in COALITION_PATTERNS:
        m = pat.search(text)
        if m:
            matches.append(m.group(0))
    member_count = ""
    for pat in MEMBER_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                member_count = m.group(1).replace(",", "").rstrip("+")
                if member_count.isdigit() and int(member_count) >= 100:
                    break
            except (IndexError, AttributeError):
                continue
            member_count = ""
    return {
        "is_coalition_brief": len(matches) > 0,
        "n_pattern_matches": len(matches),
        "member_count_hint": member_count,
        "evidence_phrase": "; ".join(matches[:2])[:300],
    }


def main() -> None:
    df = load_data()
    coalitions = pd.read_csv(ROOT / "output" / "coalitions" / "coalition_assignments.csv")
    df = df.merge(coalitions[["comment_id", "coalition"]], on="comment_id", how="inner")
    texts = load_comment_texts(ROOT / "data" / "comments.csv")

    print(f"Scanning {len(df)} comments for coalition-brief patterns ...")
    rows = []
    for _, r in df.iterrows():
        t = texts.get(r["comment_id"], "")
        flags = detect_coalition_brief(t)
        rows.append({
            "comment_id": r["comment_id"],
            "organization": r.get("organization", ""),
            "commenter_type": _str(r["commenter_type"]),
            "coalition": r["coalition"],
            "n_proposals": r.get("n_proposals"),
            **flags,
        })
    cb = pd.DataFrame(rows)
    cb.to_csv(OUT_DIR / "cosignatory_briefs.csv", index=False)

    # Summary by coalition
    summary = cb.groupby("coalition").agg(
        n_comments=("comment_id", "count"),
        n_coalition_briefs=("is_coalition_brief", "sum"),
    ).reset_index()
    summary["coalition_brief_rate"] = (summary["n_coalition_briefs"] / summary["n_comments"]).round(3)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nCoalition-brief rate by coalition:")
    print(summary.to_string(index=False))

    # Summary by stakeholder
    stake_summary = cb.groupby("commenter_type").agg(
        n_comments=("comment_id", "count"),
        n_coalition_briefs=("is_coalition_brief", "sum"),
    ).reset_index()
    stake_summary["coalition_brief_rate"] = (
        stake_summary["n_coalition_briefs"] / stake_summary["n_comments"]
    ).round(3)
    stake_summary = stake_summary.sort_values("coalition_brief_rate", ascending=False)
    stake_summary.to_csv(OUT_DIR / "summary_by_stakeholder.csv", index=False)
    print("\nCoalition-brief rate by stakeholder type:")
    print(stake_summary.to_string(index=False))

    # Member-count-hint extracts (the largest claimed coalitions)
    members = cb[cb["member_count_hint"] != ""].copy()
    members["member_count_int"] = pd.to_numeric(members["member_count_hint"], errors="coerce")
    largest = members.dropna(subset=["member_count_int"]).nlargest(15, "member_count_int")[
        ["comment_id", "organization", "commenter_type", "coalition",
         "member_count_hint", "evidence_phrase"]
    ]
    largest.to_csv(OUT_DIR / "largest_claimed_coalitions.csv", index=False)
    print("\nTop 15 largest claimed coalitions / memberships:")
    print(largest.to_string(index=False))

    # Figure
    coalition_order = ["Comprehensive Pragmatists", "Selective Universalists", "Limited Engagement"]
    rates = [
        summary.loc[summary["coalition"] == c, "coalition_brief_rate"].iloc[0]
        if c in summary["coalition"].values else 0.0
        for c in coalition_order
    ]
    counts = [
        summary.loc[summary["coalition"] == c, "n_coalition_briefs"].iloc[0]
        if c in summary["coalition"].values else 0
        for c in coalition_order
    ]
    totals = [
        summary.loc[summary["coalition"] == c, "n_comments"].iloc[0]
        if c in summary["coalition"].values else 0
        for c in coalition_order
    ]
    colors = ["#1f77b4", "#d62728", "#7f7f7f"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(range(len(coalition_order)), rates, color=colors)
    ax.set_xticks(range(len(coalition_order)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in coalition_order], fontsize=10)
    ax.set_ylabel("Share of comments that are coalition briefs")
    ax.set_ylim(0, max(rates) * 1.3 if rates else 1.0)
    ax.set_title("Multi-organization coalition-brief rate by coalition")
    for i, (rate, n_briefs, total) in enumerate(zip(rates, counts, totals)):
        ax.text(i, rate + 0.005, f"{int(n_briefs)}/{int(total)}\n({rate:.0%})",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_coalition_cosig.png", dpi=240, bbox_inches="tight")
    plt.close()

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
