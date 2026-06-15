"""Curate representative comment excerpts for the manuscript.

Improvements over the prior version:
  - Picks comments closest to each coalition's centroid in position-vector space
  - Pulls multiple candidate sentences per comment (not just first match)
  - Cleans PDF whitespace and bullet artifacts
  - Excludes comments where the LLM commenter_type is suspect (e.g., AI flagged as
    something the org name doesn't support)
  - Per-coalition output: 3-4 candidate excerpts each, with metadata, ready for
    hand-selection by the author team.

Outputs:
  output/excerpts/
    candidates.csv        per-coalition candidate excerpts ranked by closeness
    chosen.csv            single chosen excerpt per coalition (default heuristic)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "excerpts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_VARS = ["pos_oversight", "pos_regulation", "pos_liability",
                 "pos_reimbursement", "pos_interoperability", "pos_evaluation"]


def _str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v)
    return s.split(".")[-1] if "." in s else s


def load_texts(path: Path) -> dict[str, str]:
    csv.field_size_limit(sys.maxsize)
    out: dict[str, str] = {}
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


def clean_text(t: str) -> str:
    """Remove HTML, collapse whitespace, strip artifacts."""
    if not t:
        return ""
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&[a-z]+;", " ", t)
    t = re.sub(r"[•·▪◆■□]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def split_sentences(t: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


KEYWORDS = {
    "Comprehensive Pragmatists": [
        "lifecycle", "post-market", "monitoring", "validate", "validation",
        "framework", "risk-tiered", "shared", "standards", "interoperab",
        "harmonize", "align", "comprehensive",
    ],
    "Selective Universalists": [
        "must", "should require", "mandator", "universal", "human in the loop",
        "patient safety", "clinician oversight", "new regulation", "robust",
        "enforce", "require",
    ],
    "Limited Engagement": [
        # Limited-engagement comments are typically narrow; pick distinctive sentences
        "support", "important", "critical", "AI in healthcare", "thank you",
    ],
}


def pick_sentences(text: str, keywords: list[str], max_words: int = 50,
                   n: int = 3) -> list[str]:
    if not text:
        return []
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    candidates = [s for s in sentences if 12 <= len(s.split()) <= 80]
    chosen = []
    seen = set()
    for kw in keywords:
        if len(chosen) >= n:
            break
        for s in candidates:
            if kw.lower() in s.lower() and s not in seen:
                words = s.split()
                snip = " ".join(words[:max_words])
                if len(words) > max_words:
                    snip += "..."
                chosen.append(snip)
                seen.add(s)
                if len(chosen) >= n:
                    break
    # Fall back to first non-trivial sentence if nothing matched keywords
    if not chosen and candidates:
        s = candidates[0]
        words = s.split()
        chosen.append(" ".join(words[:max_words]) + ("..." if len(words) > max_words else ""))
    return chosen


def main() -> None:
    df = load_data()
    coalitions = pd.read_csv(ROOT / "output" / "coalitions" / "coalition_assignments.csv")
    df = df.merge(coalitions[["comment_id", "coalition"]], on="comment_id", how="inner")
    texts = load_texts(ROOT / "data" / "comments.csv")

    # Re-fit K-means to compute distance from centroid
    pos = df[POSITION_VARS].copy()
    for v in POSITION_VARS:
        pos[v] = pos[v].apply(_str)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(pos.values)

    km = KMeans(n_clusters=3, n_init=100, random_state=42).fit(X)
    centroids = km.cluster_centers_
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    df["dist_to_assigned_centroid"] = [
        distances[i, km.labels_[i]] for i in range(len(df))
    ]

    rows = []
    coalition_order = ["Comprehensive Pragmatists", "Selective Universalists", "Limited Engagement"]
    for c in coalition_order:
        sub = df[df["coalition"] == c].copy()
        sub = sub.sort_values("dist_to_assigned_centroid")
        # Only consider comments with substantive text (filter out bare stubs)
        candidates = []
        for _, r in sub.iterrows():
            cid = r["comment_id"]
            text = texts.get(cid, "")
            cleaned = clean_text(text)
            if len(cleaned.split()) < 80:
                continue
            org = (r.get("organization") or "").strip()
            org_lc = org.lower()
            ctype = _str(r["commenter_type"])
            # Drop suspect commenter_type assignments
            # (basic sanity: ADV label but org name suggests a hospital association)
            if ctype == "ADV" and any(x in org_lc for x in
                                       ["hospital", "medical center", "health system", "clinic"]):
                continue
            quotes = pick_sentences(text, KEYWORDS[c], n=3)
            if not quotes:
                continue
            candidates.append({
                "coalition": c,
                "comment_id": cid,
                "organization": org or "(individual)",
                "commenter_type": ctype,
                "n_proposals": int(r.get("n_proposals", 0) or 0),
                "dist_to_centroid": round(float(r["dist_to_assigned_centroid"]), 3),
                "excerpt_1": quotes[0] if len(quotes) > 0 else "",
                "excerpt_2": quotes[1] if len(quotes) > 1 else "",
                "excerpt_3": quotes[2] if len(quotes) > 2 else "",
            })
            if len(candidates) >= 5:
                break
        rows.extend(candidates)

    cands = pd.DataFrame(rows)
    cands.to_csv(OUT_DIR / "candidates.csv", index=False)

    # Default chosen: highest n_proposals among the top 5 candidates per coalition,
    # excluding individuals (since we want representative organizational voices).
    chosen_rows = []
    for c in coalition_order:
        sub = cands[cands["coalition"] == c]
        if len(sub) == 0:
            continue
        # Prefer non-individual organizations
        non_ind = sub[sub["commenter_type"] != "IND"]
        if len(non_ind) > 0:
            pick = non_ind.sort_values(["n_proposals", "dist_to_centroid"],
                                       ascending=[False, True]).iloc[0]
        else:
            pick = sub.iloc[0]
        chosen_rows.append(pick.to_dict())
    pd.DataFrame(chosen_rows).to_csv(OUT_DIR / "chosen.csv", index=False)

    print("\nChosen excerpts (default — author team should review and possibly substitute):\n")
    for r in chosen_rows:
        print(f"=== {r['coalition']} ===")
        print(f"  {r['organization']} ({r['commenter_type']}, {r['comment_id']}, "
              f"n_proposals={r['n_proposals']})")
        print(f"  → \"{r['excerpt_1']}\"")
        if r.get("excerpt_2"):
            print(f"  Alt: \"{r['excerpt_2']}\"")
        print()

    print(f"Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
