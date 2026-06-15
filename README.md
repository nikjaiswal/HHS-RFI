# HHS-ONC-2026-0001

Code and data for the analysis of public comments to docket [HHS-ONC-2026-0001](https://www.regulations.gov/docket/HHS-ONC-2026-0001) (HHS Health Sector AI RFI).

## Layout

- `analyze.py`, `config.py`, `extract.py`, `models.py`, `prompt.py`, `utils.py`, `generate_validation_sample.py` — coding pipeline
- `run_extraction.sh` — extraction runner
- `analysis/` — manuscript analyses (run `python3 analysis/pipeline.py`)
- `data/` — input corpus (`comments.csv`, attachments)
- `output/` — coded data + per-stage outputs + manuscript tables/figures
- `validation/` — human-review materials (reviewer codes, LLM key for the validation sample, sample manifest)
- `human_review/CODEBOOK.md` — codebook
- `scripts/` — one-off corpus tools
- `docs/HOW_TO_RUN.md` — run instructions

## Setup

1. Copy `.env.example` to `.env` and set:
   - `ANTHROPIC_API_KEY`
   - `REGULATIONS_GOV_API_KEY`

2. Install dependencies:

   ```
   python3 -m venv venv
   venv/bin/pip install -r requirements.txt
   ```

## Running the analysis

```
python3 analysis/pipeline.py             # all stages
python3 analysis/render_docx.py          # rebuild manuscript.docx from manuscript.md
```

Individual stages with `--only <stage>`: `reliability`, `descriptives`, `eda`, `coalitions`, `cluster_validation`, `regression`, `rfi_coverage`, `cosignatory`, `excerpts`.

## Pulling and coding fresh data

```
python3 scripts/pull_comments.py            # full corpus
python3 scripts/pull_comments.py --resume   # checkpointed
bash run_extraction.sh --resume             # code
python3 scripts/link_coded_to_sources.py    # join codes to source rows
```

## Outputs

- `manuscript.md` — manuscript source (rendered to `manuscript.docx`)
- `output/ai_vs_human_v2.csv` — per-variable IRR
- `output/coalitions/` — latent coalition outputs
- `output/stakeholder_positions/` — stakeholder × axis matrix
- `output/regression/` — multinomial logit
- `output/cluster_validation/`, `output/rfi_coverage/`, `output/cosignatory/`, `output/excerpts/`
- `output/eda/`, `output/manuscript/`

## Requirements

Python 3.10+. See `requirements.txt`.
