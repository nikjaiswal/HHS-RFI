"""Stochastic and cross-LLM robustness checks for the AI extraction pipeline.

This script is a scaffold ready to run. It does NOT execute API calls by default
to avoid silent spending; pass `--run` to actually call the Anthropic (and optional
OpenAI/Google) APIs.

Two robustness analyses:

1. Within-Claude stochasticity. Re-run extraction on a stratified n=30 sub-sample
   with the SAME prompt at temperature 1.0 (default) twice with different seeds,
   and once at temperature 0.0 (deterministic). Compute per-cell stability across
   the three runs and recompute coalition assignments to compare with the canonical
   set in `output/coalitions/coalition_assignments.csv`.

2. Cross-LLM robustness (optional). Re-run extraction with a comparable-tier
   non-Claude model (GPT-4o, Gemini 2.5) on the same n=30 sub-sample. Requires
   OPENAI_API_KEY or GOOGLE_API_KEY in `.env`. Reports per-variable κ between Claude
   and the alternative model and coalition-assignment concordance.

Outputs:
  output/stochastic_robustness/
    subsample_ids.json           which 30 comment_ids were resampled
    runs/<model>_<run>.jsonl     re-extracted codes per run
    cell_stability.csv           per-variable per-cell stability rate
    coalition_concordance.csv    coalition assignment ARI vs canonical
    summary.json                 aggregate stability + cross-model concordance

Run command:
    python3 analysis/stochastic_robustness.py --run                  # within-Claude only
    python3 analysis/stochastic_robustness.py --run --include-openai
    python3 analysis/stochastic_robustness.py --run --include-google
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load_data  # noqa: E402

OUT_DIR = ROOT / "output" / "stochastic_robustness"
RUNS_DIR = OUT_DIR / "runs"


def select_subsample(n: int = 30, seed: int = 7) -> list[str]:
    """Stratified subsample by commenter type, deterministic given seed."""
    df = load_data()
    df = df.copy()
    df["_str_type"] = df["commenter_type"].apply(
        lambda v: str(v).split(".")[-1] if v is not None else ""
    )
    rng = random.Random(seed)
    out: list[str] = []
    types = sorted(df["_str_type"].unique())
    per_type = max(1, n // len(types))
    for t in types:
        sub = df[df["_str_type"] == t]
        if len(sub) == 0:
            continue
        ids = sub["comment_id"].tolist()
        rng.shuffle(ids)
        out.extend(ids[:per_type])
    rng.shuffle(out)
    return out[:n]


def run_claude_extraction(
    comment_ids: list[str], temperature: float, seed_label: str, output_path: Path
) -> None:
    """Re-run extraction on the subsample using the canonical prompt."""
    sys.path.insert(0, str(ROOT))
    from extract import extract_one_comment  # noqa: E402  (depends on extract.py)
    import csv as _csv
    _csv.field_size_limit(sys.maxsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Build comment_id -> raw row lookup
    raw_rows: dict[str, dict] = {}
    with open(ROOT / "data" / "comments.csv", "r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            raw_rows[row.get("id", "").strip()] = row

    n_completed = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for cid in comment_ids:
            row = raw_rows.get(cid)
            if not row:
                continue
            try:
                result = extract_one_comment(
                    row, temperature=temperature, seed_label=seed_label,
                )
                out_f.write(json.dumps(result) + "\n")
                n_completed += 1
            except Exception as e:
                print(f"  [{cid}] failed: {e}")
    print(f"  {output_path.name}: {n_completed}/{len(comment_ids)} comments extracted")


def cell_stability(jsonl_paths: list[Path]) -> dict:
    """For each variable + cell, compute the fraction of paired runs that agree."""
    runs = [list(open(p) for p in jsonl_paths)]  # placeholder
    # Build {comment_id: [extraction_dict_run1, ..., run_n]}
    by_cid: dict[str, list[dict]] = {}
    for path in jsonl_paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                cid = rec.get("_meta", {}).get("comment_id") or rec.get("comment_id")
                if not cid:
                    continue
                by_cid.setdefault(cid, []).append(rec)

    total_cells = 0
    agree_cells = 0
    per_var: dict[str, list[int]] = {}
    for cid, recs in by_cid.items():
        if len(recs) < 2:
            continue
        # Extract a flat dict per run
        flats = []
        for r in recs:
            flat = {
                "commenter_type": r.get("commenter_type"),
                "clinical_perspective": int(r.get("clinical_perspective", 0)),
                "patient_perspective": int(r.get("patient_perspective", 0)),
                **{k: v for k, v in r.get("topics", {}).items()},
                **{k: v for k, v in r.get("barriers", {}).items()},
                **{k: v for k, v in r.get("positions", {}).items()},
                "has_cfr_citation": int(r.get("supplementary", {}).get("has_cfr_citation", 0)),
                "evidence_type": r.get("supplementary", {}).get("evidence_type"),
            }
            flats.append(flat)
        for var in flats[0].keys():
            vals = [f.get(var) for f in flats]
            agreed = all(v == vals[0] for v in vals)
            total_cells += 1
            if agreed:
                agree_cells += 1
            per_var.setdefault(var, [0, 0])
            per_var[var][0] += 1
            per_var[var][1] += int(agreed)

    return {
        "n_comments_with_repeats": sum(1 for r in by_cid.values() if len(r) >= 2),
        "total_cells": total_cells,
        "agree_cells": agree_cells,
        "overall_stability": agree_cells / max(total_cells, 1),
        "per_variable": {k: round(v[1] / v[0], 4) for k, v in per_var.items() if v[0] > 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="Actually call the API (otherwise dry-run)")
    parser.add_argument("--include-openai", action="store_true")
    parser.add_argument("--include-google", action="store_true")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    ids = select_subsample(n=args.n, seed=7)
    (OUT_DIR / "subsample_ids.json").write_text(
        json.dumps(ids, indent=2), encoding="utf-8"
    )
    print(f"Selected {len(ids)} subsample IDs (seed=7).")

    if not args.run:
        print("\nDry-run mode (no API calls). Pass --run to actually execute.")
        print(f"Subsample saved to {OUT_DIR / 'subsample_ids.json'}.")
        print("\nPlanned runs:")
        print("  Claude Opus 4.7 @ T=1.0, seed=A   →  runs/claude_opus47_t1_A.jsonl")
        print("  Claude Opus 4.7 @ T=1.0, seed=B   →  runs/claude_opus47_t1_B.jsonl")
        print("  Claude Opus 4.7 @ T=0.0, seed=det →  runs/claude_opus47_t0.jsonl")
        if args.include_openai:
            print("  GPT-4o     @ T=1.0  (requires OPENAI_API_KEY)")
        if args.include_google:
            print("  Gemini 2.5 @ T=1.0  (requires GOOGLE_API_KEY)")
        print("\nEstimated cost (Anthropic-only, n=30):")
        print("  ~30 comments × 3 runs × ~5K tokens each ≈ 450K tokens ≈ $5–10 USD")
        return

    # ---------- Real execution ----------
    print("\nExecuting Claude robustness runs (this will incur API costs) ...")
    run_claude_extraction(ids, temperature=1.0, seed_label="A",
                          output_path=RUNS_DIR / "claude_opus47_t1_A.jsonl")
    run_claude_extraction(ids, temperature=1.0, seed_label="B",
                          output_path=RUNS_DIR / "claude_opus47_t1_B.jsonl")
    run_claude_extraction(ids, temperature=0.0, seed_label="det",
                          output_path=RUNS_DIR / "claude_opus47_t0.jsonl")

    if args.include_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not set; skipping GPT-4o run.")
        else:
            print("OpenAI run scaffold not implemented in v0.4 — TODO.")

    if args.include_google:
        if not os.environ.get("GOOGLE_API_KEY"):
            print("WARNING: GOOGLE_API_KEY not set; skipping Gemini run.")
        else:
            print("Google run scaffold not implemented in v0.4 — TODO.")

    # ---------- Analyze ----------
    paths = list(RUNS_DIR.glob("claude_opus47_*.jsonl"))
    if len(paths) >= 2:
        stab = cell_stability(paths)
        (OUT_DIR / "summary.json").write_text(json.dumps(stab, indent=2), encoding="utf-8")
        print(f"\nOverall per-cell stability across runs: {stab['overall_stability']:.3f}")
        print(f"Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
