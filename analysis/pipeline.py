"""End-to-end analysis orchestrator for the HHS-ONC-2026-0001 manuscript.

Runs every analytical step in the correct order. Idempotent: each step writes to
its own output/ subdirectory and can be re-run independently.

Usage:
    python3 analysis/pipeline.py                  # run everything
    python3 analysis/pipeline.py --skip-eda       # skip a stage
    python3 analysis/pipeline.py --only coalitions  # run a single stage

Stages (in dependency order):
    1. reliability       — IRR (κ + bootstrap CI + PABAK + Krippendorff α)
    2. descriptives      — Wilson CIs, χ²/FDR, sensitivity analyses
    3. eda               — exploratory pass (silhouette, topic phi, silence map, ...)
    4. coalitions        — k=3 K-means, profiles, χ² test, multinomial logit
    5. cluster_validation — gap, BIC, bootstrap stability, alt-algorithm concordance
    6. regression        — de-tautologized multinomial logit with bootstrap CIs
    7. rfi_coverage      — coalition × RFI-question coverage
    8. cosignatory       — multi-org coalition-brief detection + claimed-membership scale
    9. excerpts          — representative excerpts via cluster-centroid distance
    10. render_docx      — manuscript.md → manuscript.docx
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGES: list[tuple[str, str]] = [
    ("reliability", "AI-vs-human IRR"),
    ("descriptives", "Wilson CIs, χ², sensitivity"),
    ("eda", "Exploratory pass"),
    ("coalitions", "K=3 latent coalitions"),
    ("cluster_validation", "Multi-diagnostic k-choice"),
    ("regression", "De-tautologized multinomial logit"),
    ("rfi_coverage", "Coalition × RFI-question coverage"),
    ("cosignatory", "Multi-org brief detection"),
    ("excerpts", "Representative excerpts"),
    ("render_docx", "manuscript.md → manuscript.docx"),
]


def run_stage(name: str) -> bool:
    script = ROOT / "analysis" / f"{name}.py"
    if not script.exists():
        print(f"  [SKIP] {name}: {script} not found")
        return False
    print(f"\n=== Running {name} ===")
    rc = subprocess.run([sys.executable, str(script)], cwd=str(ROOT)).returncode
    if rc != 0:
        print(f"  [FAIL] {name} exited with code {rc}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", action="append", default=[],
                        help="Skip a stage (may be passed multiple times)")
    parser.add_argument("--only", action="append", default=[],
                        help="Run only specified stages (may be passed multiple times)")
    args = parser.parse_args()

    stages_to_run = [
        name for name, _ in STAGES
        if (not args.only or name in args.only) and name not in args.skip
    ]
    print(f"Will run: {stages_to_run}\n")

    failed = []
    for name in stages_to_run:
        ok = run_stage(name)
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nFailed stages: {failed}")
        sys.exit(1)
    print("\nAll stages completed successfully.")


if __name__ == "__main__":
    main()
