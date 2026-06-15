import os
from pathlib import Path

# Project root = folder containing this file (stable for Google Drive sync)
PROJECT_ROOT = Path(__file__).resolve().parent

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
VALIDATION_DIR = PROJECT_ROOT / "validation"

INPUT_CSV = DATA_DIR / "comments.csv"
OUTPUT_JSONL = OUTPUT_DIR / "coded_comments.jsonl"
ERRORS_JSONL = OUTPUT_DIR / "errors.jsonl"

MAX_INPUT_CHARS = 100_000
REQUEST_DELAY = 0.5
MAX_RETRIES = 3

for d in [DATA_DIR, OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, VALIDATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)
