"""
Generate a stratified 45-comment sample for human validation coding.
"""

import json
import random
from pathlib import Path

import pandas as pd

from config import OUTPUT_JSONL, VALIDATION_DIR
from models import CommentExtraction

random.seed(42)


def main():
    # Load coded data (flatten nested structure like analyze.load_data)
    records = []
    with open(OUTPUT_JSONL) as f:
        for line in f:
            raw = json.loads(line)
            meta = raw.pop("_meta")
            extraction = CommentExtraction.model_validate(raw)
            d = extraction.model_dump()
            flat = dict(meta)
            flat["commenter_type"] = d["commenter_type"]
            flat["clinical_perspective"] = int(d["clinical_perspective"])
            flat["patient_perspective"] = int(d["patient_perspective"])
            flat["organization_from_document"] = d.get("organization_from_document") or ""
            for k, v in d["topics"].items():
                flat[k] = v
            for k, v in d["barriers"].items():
                flat[k] = v
            for k, v in d["positions"].items():
                flat[k] = v
            flat["n_proposals"] = d["supplementary"]["n_proposals"]
            flat["has_cfr_citation"] = d["supplementary"]["has_cfr_citation"]
            flat["evidence_type"] = d["supplementary"]["evidence_type"]
            flat["rfi_questions"] = d["supplementary"]["rfi_questions"]
            records.append(flat)

    df = pd.DataFrame(records)

    # Stratified sample: ~5 per commenter type
    sample_ids = []
    for ctype in df["commenter_type"].unique():
        subset = df[df["commenter_type"] == ctype]
        n_sample = min(5, len(subset))
        sampled = subset.sample(n=n_sample, random_state=42)
        sample_ids.extend(sampled["comment_id"].tolist())

    sample_df = df[df["comment_id"].isin(sample_ids)].copy()

    # Save sample with LLM codes (for comparison)
    sample_df.to_csv(VALIDATION_DIR / "sample_45_with_llm_codes.csv", index=False)

    # Save blank coding sheet (all variables reviewers need)
    coding_cols = ["comment_id", "organization", "organization_from_document",
                   "commenter_type", "clinical_perspective", "patient_perspective"]
    coding_cols += [c for c in df.columns if c.startswith("top_")]
    coding_cols += [c for c in df.columns if c.startswith("bar_")]
    coding_cols += [c for c in df.columns if c.startswith("pos_")]
    coding_cols += ["n_proposals", "has_cfr_citation", "evidence_type", "rfi_questions"]

    blank = sample_df[["comment_id", "organization"]].copy()
    for col in coding_cols:
        if col not in blank.columns:
            blank[col] = ""

    blank.to_csv(VALIDATION_DIR / "coding_sheet_blank.csv", index=False)

    print(f"Validation sample: {len(sample_df)} comments")
    print(f"  Types represented: {sample_df['commenter_type'].nunique()}")
    for ctype, n in sample_df["commenter_type"].value_counts().items():
        print(f"    {ctype}: {n}")
    print(f"\n  Saved to {VALIDATION_DIR}/")


if __name__ == "__main__":
    main()
