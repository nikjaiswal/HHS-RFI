"""
Link coded_comments.jsonl to source data (CSV and JSON).

Produces:
- output/comments_with_codes.csv — every row from data/comments.csv with coded_* columns
  joined on id = comment_id. Rows with no coding get empty coded_* columns.
- output/all_comments_with_codes.json — data/all_comments.json with a "coded" key added
  to each comment that has a matching coded record (id or documentId = comment_id).

Run from project root: venv/bin/python scripts/link_coded_to_sources.py
"""

import json
import sys
from pathlib import Path

# Run from project root so config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, OUTPUT_JSONL
from models import CommentExtraction


def load_coded_flat():
    """Load coded_comments.jsonl and return a flat DataFrame (one row per comment)."""
    records = []
    with open(OUTPUT_JSONL) as f:
        for line in f:
            raw = json.loads(line)
            meta = raw.pop("_meta")
            extraction = CommentExtraction.model_validate(raw)
            d = extraction.model_dump()
            flat = {}
            flat["comment_id"] = meta["comment_id"]
            flat["coded_organization"] = meta.get("organization", "")
            flat["coded_organization_from_document"] = d.get("organization_from_document") or ""
            flat["coded_commenter_type"] = d["commenter_type"]
            flat["coded_clinical_perspective"] = int(d["clinical_perspective"])
            flat["coded_patient_perspective"] = int(d["patient_perspective"])
            for key, val in d["topics"].items():
                flat[f"coded_{key}"] = val
            for key, val in d["barriers"].items():
                flat[f"coded_{key}"] = val
            for key, val in d["positions"].items():
                flat[f"coded_{key}"] = val
            flat["coded_n_proposals"] = d["supplementary"]["n_proposals"]
            flat["coded_has_cfr_citation"] = d["supplementary"]["has_cfr_citation"]
            flat["coded_evidence_type"] = d["supplementary"]["evidence_type"]
            flat["coded_n_rfi_questions"] = len(d["supplementary"]["rfi_questions"])
            records.append(flat)
    return pd.DataFrame(records)


def main():
    input_csv = DATA_DIR / "comments.csv"
    input_json = DATA_DIR / "all_comments.json"
    out_csv = OUTPUT_DIR / "comments_with_codes.csv"
    out_json = OUTPUT_DIR / "all_comments_with_codes.json"

    if not OUTPUT_JSONL.exists():
        print(f"Error: {OUTPUT_JSONL} not found. Run extract.py first.")
        return
    if not input_csv.exists():
        print(f"Error: {input_csv} not found.")
        return

    coded = load_coded_flat()
    comments_df = pd.read_csv(input_csv, low_memory=False)

    # Join on id = comment_id; keep all CSV rows, add coded_* columns
    merged = comments_df.merge(
        coded,
        left_on="id",
        right_on="comment_id",
        how="left",
        suffixes=("", "_coded"),
    )
    # Drop the duplicate key from the right
    if "comment_id" in merged.columns:
        merged = merged.drop(columns=["comment_id"])
    merged.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(merged)} rows)")

    if input_json.exists():
        with open(input_json) as f:
            data = json.load(f)
        # Assume list of comment objects
        if isinstance(data, list):
            comments_list = data
        elif isinstance(data, dict) and "comments" in data:
            comments_list = data["comments"]
        else:
            comments_list = data if isinstance(data, list) else []
        by_id = {c.get("id") or c.get("documentId"): c for c in comments_list if c.get("id") or c.get("documentId")}
        coded_by_id = coded.set_index("comment_id")
        for cid, row in coded_by_id.iterrows():
            if cid in by_id:
                by_id[cid]["coded"] = row.to_dict()
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {out_json} ({len(comments_list)} comments, {len(coded)} with codes)")
    else:
        print(f"No {input_json} found; skipped JSON output.")


if __name__ == "__main__":
    main()
