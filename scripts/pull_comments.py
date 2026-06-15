#!/usr/bin/env python3
"""
Pull all comments from Regulations.gov for the HHS Health Sector AI RFI docket
(HHS-ONC-2026-0001). Persists full API responses and exports CSV.

Usage:
  python pull_comments.py [--limit N] [--download-attachments] [--resume]
  python pull_comments.py --attachment-urls-only   # write data/attachment_urls.txt from existing JSON
  python pull_comments.py --download-attachments-only [--workers N]  # parallel download from existing JSON

Requires .env with REGULATIONS_GOV_API_KEY (get key at https://api.data.gov/).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

import requests

DOCKET_ID = "HHS-ONC-2026-0001"
API_BASE = "https://api.regulations.gov/v4"
PAGE_SIZE = 250
MAX_PAGES_PER_QUERY = 20  # 5000 items max per query
RATE_DELAY_SEC = 1.2  # stay under 50/min
DATA_DIR = Path("data")
CHECKPOINT_FILE = DATA_DIR / "checkpoint.json"
COMMENTS_JSON_FILE = DATA_DIR / "all_comments.json"
COMMENTS_CSV_FILE = DATA_DIR / "comments.csv"
ATTACHMENTS_DIR = DATA_DIR / "attachments"


def _setup_logging() -> logging.Logger:
    """Console + optional data/pull_comments.log with timestamp and level."""
    log = logging.getLogger("pull_comments")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return log
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    out = logging.StreamHandler(sys.stdout)
    out.setLevel(logging.INFO)
    out.setFormatter(fmt)
    log.addHandler(out)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(DATA_DIR / "pull_comments.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError:
        pass
    return log


logger = _setup_logging()


def get_api_key() -> str:
    if load_dotenv:
        load_dotenv()
    key = os.environ.get("REGULATIONS_GOV_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.strip().startswith("REGULATIONS_GOV_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        logger.error("Set REGULATIONS_GOV_API_KEY in .env or environment. Get a key at https://api.data.gov/")
        sys.exit(1)
    return key


def step_a_get_document_object_ids(session: requests.Session) -> list[str]:
    """Return list of document objectIds in the docket (for commentOnId filter)."""
    r = session.get(
        f"{API_BASE}/documents",
        params={"filter[docketId]": DOCKET_ID, "page[size]": 50},
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    object_ids = []
    for doc in data:
        oid = doc.get("attributes", {}).get("objectId")
        if oid:
            object_ids.append(oid)
    if not object_ids:
        logger.error("No documents with objectId found in docket.")
        sys.exit(1)
    return object_ids


def step_b_get_all_comment_ids(session: requests.Session, object_ids: list[str]) -> list[str]:
    """Paginate comment list for each document objectId; return all comment IDs (documentId)."""
    all_ids = []
    for object_id in object_ids:
        page = 1
        last_modified_ge = None
        while True:
            params = {
                "filter[commentOnId]": object_id,
                "page[size]": PAGE_SIZE,
                "page[number]": page,
                "sort": "lastModifiedDate,documentId",
            }
            if last_modified_ge is not None:
                params["filter[lastModifiedDate][ge]"] = last_modified_ge
            r = session.get(f"{API_BASE}/comments", params=params)
            r.raise_for_status()
            j = r.json()
            data = j.get("data", [])
            meta = j.get("meta", {})
            total = meta.get("totalElements", 0)
            for item in data:
                # Comment id is the documentId used for the detail endpoint (e.g. HHS-ONC-2026-0001-0002)
                doc_id = item.get("id") or item.get("attributes", {}).get("documentId")
                if doc_id and doc_id not in all_ids:
                    all_ids.append(doc_id)
            if not data:
                break
            if total <= PAGE_SIZE * page:
                break
            if page >= MAX_PAGES_PER_QUERY:
                # Next batch: filter by lastModifiedDate >= last item
                last_item = data[-1]
                lm = last_item.get("attributes", {}).get("lastModifiedDate")
                if not lm:
                    break
                # API expects Eastern time format "YYYY-MM-DD HH:MM:SS"
                last_modified_ge = lm.replace("T", " ").replace("Z", "")[:19]
                page = 1
            else:
                page += 1
            time.sleep(RATE_DELAY_SEC)
    return all_ids


def step_c_fetch_comment_details(
    session: requests.Session,
    comment_ids: list[str],
    limit: int | None,
    checkpoint: dict,
    download_attachments: bool,
) -> list[dict]:
    """Fetch detail for each comment (include=attachments). Persist full response. Optional download."""
    results = checkpoint.get("comments", [])
    fetched = set(checkpoint.get("fetched_ids", []))
    to_fetch = [cid for cid in comment_ids if cid not in fetched]
    if limit is not None:
        to_fetch = to_fetch[: limit - len(results)]
    total = len(to_fetch)
    for i, comment_id in enumerate(to_fetch):
        r = session.get(
            f"{API_BASE}/comments/{comment_id}",
            params={"include": "attachments"},
        )
        r.raise_for_status()
        payload = r.json()
        results.append(payload)
        fetched.add(comment_id)
        # Checkpoint
        checkpoint["fetched_ids"] = list(fetched)
        checkpoint["comments"] = results
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"fetched_ids": list(fetched), "comment_count": len(results)}, f)
        if download_attachments and payload.get("included"):
            download_comment_attachments(payload, session)
        logger.info("  [%s/%s] %s", i + 1, total, comment_id)
        time.sleep(RATE_DELAY_SEC)
    return results


# Browser User-Agent and Referer for attachment downloads (downloads.regulations.gov often 403 without these)
DOWNLOAD_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
DOWNLOAD_REFERER = "https://www.regulations.gov/"

BINARY_CONTENT_TYPES = frozenset(
    {"application/pdf", "application/octet-stream", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
)


def _download_file(url: str, session: requests.Session, *, referer: bool = False, user_agent: bool = False) -> tuple[bool, bytes | None]:
    """GET url with optional Referer and User-Agent. Returns (success, content) or (False, None)."""
    headers = {}
    if referer:
        headers["Referer"] = DOWNLOAD_REFERER
    if user_agent:
        headers["User-Agent"] = DOWNLOAD_USER_AGENT
    try:
        r = session.get(url, headers=headers if headers else None, timeout=60)
        if r.status_code != 200:
            return False, None
        # Avoid saving error pages as files
        ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ct and not (ct in BINARY_CONTENT_TYPES or ct.startswith("application/") or ct.startswith("image/")):
            if "text/html" in ct or "json" in ct:
                return False, None
        return True, r.content
    except Exception:
        return False, None


def _build_attachment_tasks(comments: list[dict]) -> list[tuple[str, str, Path, str]]:
    """From comment payloads, build (doc_id, url, path, ext) for each file to download. Prefer PDF per attachment."""
    tasks = []
    for item in comments:
        data = item.get("data", {})
        doc_id = data.get("id") or (data.get("attributes") or {}).get("documentId")
        if not doc_id:
            continue
        out_dir = ATTACHMENTS_DIR / doc_id
        att_idx = 0
        for inc in item.get("included", []):
            if inc.get("type") != "attachments":
                continue
            att_idx += 1
            att = inc.get("attributes") or {}
            formats = att.get("fileFormats") or []
            primary = next(
                (f for f in formats if (f.get("format") or "").lower() == "pdf"),
                formats[0] if formats else None,
            )
            if not primary:
                continue
            url = primary.get("fileUrl")
            if not url:
                continue
            ext = (primary.get("format") or "bin").lower()
            path = out_dir / f"attachment_{att_idx}.{ext}"
            tasks.append((doc_id, url, path, ext))
    return tasks


def _download_one_task(task: tuple[str, str, Path, str]) -> tuple[str, Path, bool, str | None]:
    """Download one (doc_id, url, path, ext). Returns (doc_id, path, success, method_used|error)."""
    doc_id, url, path, _ext = task
    if path.exists() and path.stat().st_size > 0:
        return (doc_id, path, True, "skipped_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    http = requests.Session()
    # Try Referer then User-Agent + Referer (same as download_comment_attachments)
    ok, content = _download_file(url, http, referer=True)
    if ok and content:
        path.write_bytes(content)
        return (doc_id, path, True, "fileUrl_referer")
    ok, content = _download_file(url, http, referer=True, user_agent=True)
    if ok and content:
        path.write_bytes(content)
        return (doc_id, path, True, "fileUrl_ua_referer")
    return (doc_id, path, False, "failed")


def run_download_attachments_only(comments_path: Path = COMMENTS_JSON_FILE, workers: int = 5) -> None:
    """Load comments from JSON and download all attachments in parallel. Skips existing files."""
    if not comments_path.exists():
        logger.error("%s not found. Run pull_comments.py first.", comments_path)
        sys.exit(1)
    with open(comments_path, encoding="utf-8") as f:
        comments = json.load(f)
    tasks = _build_attachment_tasks(comments)
    if not tasks:
        logger.info("No attachments to download.")
        return
    logger.info("Downloading %s attachment(s) with %s worker(s)...", len(tasks), workers)
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one_task, t): t for t in tasks}
        for future in as_completed(futures):
            doc_id, path, success, method = future.result()
            done += 1
            if success:
                if method != "skipped_exists":
                    logger.info("  [%s/%s] %s (%s)", done, len(tasks), path.name, method)
            else:
                failed += 1
                logger.warning("  [%s/%s] %s %s", done, len(tasks), path.name, method)
    logger.info("Done. %s downloaded/skipped, %s failed.", len(tasks) - failed, failed)


def download_comment_attachments(payload: dict, session: requests.Session | None = None) -> None:
    """Download attachment files from comment detail payload into data/attachments/{documentId}/.
    Tries in order: (1) API attachment endpoint for binary, (2) fileUrl with Referer, (3) fileUrl with User-Agent + Referer.
    """
    data = payload.get("data", {})
    doc_id = data.get("id") or data.get("attributes", {}).get("documentId")
    if not doc_id:
        return
    out_dir = ATTACHMENTS_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    http = session or requests.Session()
    for inc in payload.get("included", []):
        if inc.get("type") != "attachments":
            continue
        att_id = inc.get("id")
        att = inc.get("attributes", {})
        for idx, fmt in enumerate(att.get("fileFormats") or []):
            url = fmt.get("fileUrl")
            if not url:
                continue
            ext = fmt.get("format") or "bin"
            filename = f"attachment_{idx + 1}.{ext}"
            path = out_dir / filename
            saved = False
            method_used = None
            # Strategy 1: GET api.regulations.gov/v4/attachments/{id} (may return binary or JSON with fileUrl)
            if att_id and session:
                try:
                    r = session.get(f"{API_BASE}/attachments/{att_id}", timeout=30)
                    if r.status_code == 200:
                        ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                        if ct in BINARY_CONTENT_TYPES or (ct.startswith("application/") and "json" not in ct):
                            path.write_bytes(r.content)
                            saved = True
                            method_used = "api_attachment"
                except Exception:
                    pass
                time.sleep(RATE_DELAY_SEC)
            # Strategy 2: fileUrl with Referer
            if not saved:
                ok, content = _download_file(url, http, referer=True)
                if ok and content:
                    path.write_bytes(content)
                    saved = True
                    method_used = "fileUrl_referer"
            # Strategy 3: fileUrl with User-Agent + Referer
            if not saved:
                ok, content = _download_file(url, http, referer=True, user_agent=True)
                if ok and content:
                    path.write_bytes(content)
                    saved = True
                    method_used = "fileUrl_ua_referer"
            if saved and method_used:
                logger.info("    Downloaded %s (%s)", filename, method_used)
            elif not saved:
                logger.warning("Could not download %s", url)
            time.sleep(RATE_DELAY_SEC)


def flatten_comment_for_csv(item: dict) -> dict:
    """Build a flat row from one comment API response (data + included).
    Includes per-document columns: attachmentUrl_1, attachmentTitle_1, attachmentFormat_1, ...
    and legacy pipe-separated attachmentUrls.
    """
    data = item.get("data", {})
    attrs = data.get("attributes", {})
    row = {"id": data.get("id"), "documentId": attrs.get("documentId") or data.get("id")}
    # All attributes (persist everything the API returns)
    for k, v in attrs.items():
        if k in row:
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v) if v else ""
        row[k] = v
    # Per-document attachment info (one slot per attachment)
    attachment_urls_all = []
    for idx, inc in enumerate(item.get("included", [])):
        if inc.get("type") != "attachments":
            continue
        n = idx + 1
        att = inc.get("attributes") or {}
        title = att.get("title") or ""
        row[f"attachmentTitle_{n}"] = title
        formats = att.get("fileFormats") or []
        # Primary: prefer PDF, else first format
        primary = next((f for f in formats if (f.get("format") or "").lower() == "pdf"), formats[0] if formats else None)
        if primary:
            row[f"attachmentUrl_{n}"] = primary.get("fileUrl") or ""
            row[f"attachmentFormat_{n}"] = primary.get("format") or ""
            size = primary.get("size")
            row[f"attachmentSize_{n}"] = size if size is not None else ""
        else:
            row[f"attachmentUrl_{n}"] = ""
            row[f"attachmentFormat_{n}"] = ""
            row[f"attachmentSize_{n}"] = ""
        row[f"fullText_{n}"] = item.get(f"fullText_{n}", "")
        for fmt in formats:
            u = fmt.get("fileUrl")
            if u:
                attachment_urls_all.append(u)
    row["attachmentUrls"] = "|".join(attachment_urls_all)
    return row


# Preferred column order: core fields first, then per-document attachment columns, then legacy, then rest
CSV_LEAD_COLUMNS = ("id", "documentId", "comment", "organization", "title")
CSV_LEGACY_ATTACHMENT = "attachmentUrls"


def _csv_column_order(keys: set[str]) -> list[str]:
    """Return keys in stable order: lead, attachment columns, fullText_1.., legacy, then rest."""
    lead = [k for k in CSV_LEAD_COLUMNS if k in keys]
    rest = keys - set(lead) - {CSV_LEGACY_ATTACHMENT}
    attachment_keys = [k for k in rest if k.startswith("attachmentUrl_") or k.startswith("attachmentTitle_") or k.startswith("attachmentFormat_") or k.startswith("attachmentSize_")]
    fulltext_keys = [k for k in rest if k.startswith("fullText_")]
    other = rest - set(attachment_keys) - set(fulltext_keys)
    def attachment_sort_key(name: str) -> tuple[int, int]:
        m = re.match(r"attachment(Url|Title|Format|Size)_(\d+)", name)
        if not m:
            return (0, 0)
        kind, num = m.group(1), int(m.group(2))
        kind_order = {"Url": 0, "Title": 1, "Format": 2, "Size": 3}
        return (num, kind_order.get(kind, 0))
    attachment_keys.sort(key=attachment_sort_key)
    fulltext_keys.sort(key=lambda k: (int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else 0))
    result = lead + attachment_keys + fulltext_keys
    if CSV_LEGACY_ATTACHMENT in keys:
        result.append(CSV_LEGACY_ATTACHMENT)
    result.extend(sorted(other))
    return result


def write_csv(comments: list[dict], path: Path) -> None:
    """Write one CSV with all keys from all rows (union of keys), in stable column order."""
    if not comments:
        return
    rows = [flatten_comment_for_csv(c) for c in comments]
    all_keys_set = set()
    for r in rows:
        all_keys_set.update(r.keys())
    fieldnames = _csv_column_order(all_keys_set)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_attachment_urls_only(comments_path: Path = COMMENTS_JSON_FILE, out_path: Path | None = None) -> None:
    """Read all_comments.json and write one attachment URL per line to data/attachment_urls.txt (with optional comment id prefix)."""
    out_path = out_path or DATA_DIR / "attachment_urls.txt"
    if not comments_path.exists():
        logger.error("%s not found. Run pull_comments.py first to fetch comments.", comments_path)
        sys.exit(1)
    with open(comments_path, encoding="utf-8") as f:
        comments = json.load(f)
    lines = []
    for item in comments:
        data = item.get("data", {})
        doc_id = data.get("id") or (data.get("attributes") or {}).get("documentId") or ""
        for inc in item.get("included", []):
            if inc.get("type") != "attachments":
                continue
            for fmt in (inc.get("attributes") or {}).get("fileFormats") or []:
                url = fmt.get("fileUrl")
                if url:
                    lines.append(f"{doc_id}\t{url}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info("Wrote %s attachment URL(s) to %s", len(lines), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[1])
    parser.add_argument("--limit", type=int, default=None, help="Max number of comments to fetch (for testing)")
    parser.add_argument("--download-attachments", action="store_true", help="Download attachment PDFs/files to data/attachments/ (during comment fetch)")
    parser.add_argument("--download-attachments-only", action="store_true", help="Only download attachments from existing all_comments.json (parallel workers)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers for --download-attachments-only (default: 5)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (skip already-fetched comments)")
    parser.add_argument("--attachment-urls-only", action="store_true", help="Write data/attachment_urls.txt from existing all_comments.json and exit")
    args = parser.parse_args()

    if args.attachment_urls_only:
        write_attachment_urls_only()
        return

    if args.download_attachments_only:
        run_download_attachments_only(COMMENTS_JSON_FILE, workers=args.workers)
        return

    api_key = get_api_key()
    session = requests.Session()
    session.headers["X-Api-Key"] = api_key

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {"fetched_ids": [], "comments": []}
    if args.resume and CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            c = json.load(f)
        checkpoint["fetched_ids"] = c.get("fetched_ids", [])
        # Reload full comments from JSON if we have it (so we don't re-fetch)
        if COMMENTS_JSON_FILE.exists():
            with open(COMMENTS_JSON_FILE) as f:
                checkpoint["comments"] = json.load(f)
            # Sync fetched_ids to what we actually have in JSON (in case checkpoint was ahead)
            ids_in_json = []
            for item in checkpoint["comments"]:
                d = item.get("data", {})
                doc_id = d.get("id") or (d.get("attributes") or {}).get("documentId")
                if doc_id:
                    ids_in_json.append(doc_id)
            if ids_in_json:
                checkpoint["fetched_ids"] = ids_in_json
        logger.info("Resume: %s already fetched.", len(checkpoint["fetched_ids"]))
    elif args.resume:
        logger.info("No checkpoint found; starting from scratch.")

    # Step A: document objectIds
    logger.info("Step A: Fetching docket documents...")
    object_ids = step_a_get_document_object_ids(session)
    logger.info("  Found %s document(s).", len(object_ids))
    time.sleep(RATE_DELAY_SEC)

    # Step B: all comment IDs
    logger.info("Step B: Fetching comment list (paginated)...")
    comment_ids = step_b_get_all_comment_ids(session, object_ids)
    logger.info("  Total comments: %s.", len(comment_ids))
    if args.limit:
        logger.info("  Limiting to first %s (--limit).", args.limit)

    # Step C: detail for each (or resume from checkpoint)
    logger.info("Step C: Fetching full detail for each comment...")
    comments = step_c_fetch_comment_details(
        session, comment_ids, args.limit, checkpoint, args.download_attachments
    )

    # Persist full JSON and CSV
    logger.info("Writing outputs...")
    with open(COMMENTS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(comments, f, indent=2, ensure_ascii=False)
    write_csv(comments, COMMENTS_CSV_FILE)
    logger.info("  %s (%s comments)", COMMENTS_JSON_FILE, len(comments))
    logger.info("  %s", COMMENTS_CSV_FILE)
    logger.info("Done.")


if __name__ == "__main__":
    main()
