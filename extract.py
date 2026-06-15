"""
Main extraction pipeline. Checkpointing: each comment is appended to the
output file as soon as it is coded. Use --resume to skip already-done IDs
and continue after an interrupt (e.g. Ctrl+C, crash, or rate limit).

Usage:
    python extract.py --limit 5            # test on 5
    python extract.py --resume             # full run; skip already in output (safe to re-run)
    python extract.py --resume --workers 6   # default 6; backs off on 429 and retries
    python extract.py --id HHS-ONC-2026-0001-0209
"""

import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

import anthropic
from tqdm import tqdm

from config import (
    ANTHROPIC_API_KEY, MODEL, INPUT_CSV, OUTPUT_JSONL,
    ERRORS_JSONL, REQUEST_DELAY, MAX_RETRIES,
)
from models import CommentExtraction
from prompt import SYSTEM_PROMPT, build_user_prompt
from utils import load_comments, get_analyzable_text, get_text_source


def extract_one(
    client: anthropic.Anthropic, row: dict
) -> tuple[dict | None, str | None]:
    """Extract structured data from one comment using tool_use."""
    comment_id = row.get("id", "unknown")
    org = (row.get("organization") or "").strip()
    text = get_analyzable_text(row)
    code_empty = getattr(extract_one, "_code_empty", False)

    if not text.strip():
        if code_empty:
            text = "[No content in submission — comment and attachments had no extractable text.]"
        else:
            return None, "too_short"  # skip only when there is no content at all

    tool = {
        "name": "extract_comment_data",
        "description": (
            "Extract structured coding variables from a public comment "
            "on the HHS AI RFI."
        ),
        "input_schema": CommentExtraction.model_json_schema(),
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=3000,
                system=SYSTEM_PROMPT,
                tools=[tool],
                tool_choice={"type": "tool", "name": "extract_comment_data"},
                messages=[
                    {
                        "role": "user",
                        "content": build_user_prompt(comment_id, org, text),
                    }
                ],
            )

            tool_block = next(
                (b for b in response.content if b.type == "tool_use"),
                None,
            )
            if tool_block is None:
                return None, "no_tool_use_block"

            extraction = CommentExtraction.model_validate(tool_block.input)

            result = extraction.model_dump()
            result["_meta"] = {
                "comment_id": comment_id,
                "organization": org,
                "organization_from_document": result.get("organization_from_document") or "",
                "posted_date": row.get("postedDate", ""),
                "text_source": get_text_source(row),
                "total_chars": len(text),
                "model": MODEL,
                "extracted_at": datetime.now().isoformat(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return result, None

        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code in (500, 502, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(10 * (attempt + 1))
            else:
                return None, f"api_error_{e.status_code}: {e.message}"
        except anthropic.APIError as e:
            return None, f"api_error: {e}"
        except Exception as e:
            return None, f"error: {e}"

    return None, "max_retries_exceeded"


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from HHS AI RFI comments"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent API calls. Default 6. On 429 rate limit we back off (wait 30s/60s/90s) and retry; reduce workers if 429s persist.",
    )
    parser.add_argument(
        "--code-empty",
        action="store_true",
        help="When a comment has no extractable text, send a short placeholder to the API so the row still gets a coded record (e.g. for 447/447 completeness).",
    )
    args = parser.parse_args()
    # Cap workers to avoid burst rate limits (docs: short bursts can trigger 429)
    args.workers = max(1, min(args.workers, 20))
    extract_one._code_empty = getattr(args, "code_empty", False)

    assert ANTHROPIC_API_KEY, "Set ANTHROPIC_API_KEY environment variable"
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Resume support
    done_ids = set()
    if args.resume and OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["_meta"]["comment_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resume: {len(done_ids)} already processed")

    # Load and filter
    rows = load_comments(str(INPUT_CSV))
    if args.id:
        rows = [r for r in rows if r.get("id") == args.id]
    if args.limit:
        rows = rows[: args.limit]

    rows_to_do = [r for r in rows if r.get("id") not in done_ids]
    skipped = len(rows) - len(rows_to_do)

    print(f"\nComments to process: {len(rows_to_do)} (skipping {skipped} already done)")
    print(f"Model: {MODEL}  Workers: {args.workers}\n")

    success = errors = skipped_short = 0
    total_in = total_out = 0
    write_lock = Lock()

    def process_one(row):
        cid = row.get("id", "")
        org = (row.get("organization") or "").strip() or "(individual)"
        result, error = extract_one(client, row)
        return cid, org, result, error

    if args.workers <= 1:
        # Sequential: one at a time with delay (original behavior)
        for row in tqdm(rows_to_do, desc="Extracting"):
            cid, org, result, error = process_one(row)
            if result:
                with open(OUTPUT_JSONL, "a") as f:
                    f.write(json.dumps(result) + "\n")
                success += 1
                total_in += result["_meta"]["input_tokens"]
                total_out += result["_meta"]["output_tokens"]
            elif error == "too_short":
                skipped_short += 1
            else:
                with open(ERRORS_JSONL, "a") as f:
                    f.write(json.dumps({"id": cid, "org": org, "error": error, "time": datetime.now().isoformat()}) + "\n")
                errors += 1
            time.sleep(args.delay)
        skipped += skipped_short  # add too_short to skipped count
    else:
        # Parallel: multiple concurrent calls, lock when writing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, row): row for row in rows_to_do}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                cid, org, result, error = future.result()
                with write_lock:
                    if result:
                        with open(OUTPUT_JSONL, "a") as f:
                            f.write(json.dumps(result) + "\n")
                        success += 1
                        total_in += result["_meta"]["input_tokens"]
                        total_out += result["_meta"]["output_tokens"]
                    elif error == "too_short":
                        skipped_short += 1
                    else:
                        with open(ERRORS_JSONL, "a") as f:
                            f.write(json.dumps({"id": cid, "org": org, "error": error, "time": datetime.now().isoformat()}) + "\n")
                        errors += 1
        skipped += skipped_short  # add too_short to skipped count

    cost = (total_in * 3 + total_out * 15) / 1_000_000
    print(f"\n{'='*50}")
    print(f"Done: {success} | Errors: {errors} | Skipped: {skipped} (already done + too short)")
    print(f"Tokens: {total_in:,} in + {total_out:,} out")
    print(f"Est. cost: ${cost:.2f}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
