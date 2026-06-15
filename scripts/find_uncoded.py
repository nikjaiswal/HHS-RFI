"""
Find comment IDs that are in the source CSV but have no row in coded_comments.jsonl.
Run from project root: python3 scripts/find_uncoded.py
"""
import csv
import json
import sys
csv.field_size_limit(sys.maxsize)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, OUTPUT_JSONL

def main():
    coded_ids = set()
    with open(OUTPUT_JSONL) as f:
        for line in f:
            try:
                coded_ids.add(json.loads(line)["_meta"]["comment_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    with open(DATA_DIR / "comments.csv") as f:
        reader = csv.DictReader(f)
        csv_ids = [row["id"] for row in reader]
    missing = [id_ for id_ in csv_ids if id_ not in coded_ids]
    if not missing:
        print("All CSV comments have a code.")
        return
    print(f"Comment(s) in CSV with no code ({len(missing)}):")
    for id_ in missing:
        print(f"  {id_}")

if __name__ == "__main__":
    main()
