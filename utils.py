import csv
import sys
import html

csv.field_size_limit(sys.maxsize)


def get_analyzable_text(row: dict) -> str:
    parts = []

    comment = html.unescape((row.get("comment") or "").strip())
    if comment and len(comment) > 50:
        lower = comment.lower()[:30]
        if "see attached" not in lower and "please see" not in lower:
            parts.append("[INLINE COMMENT]\n" + comment)

    fulltext_parts = []
    for i in range(1, 13):
        ft = (row.get(f"fullText_{i}") or "").strip()
        if ft:
            fulltext_parts.append(ft)

    if fulltext_parts:
        parts.append("[ATTACHED DOCUMENT]\n" + "\n\n".join(fulltext_parts))

    text = "\n\n".join(parts)

    if len(text) > 100_000:
        half = 50_000
        text = text[:half] + "\n\n[...MIDDLE SECTION TRUNCATED...]\n\n" + text[-half:]

    return text


def get_text_source(row: dict) -> str:
    comment = (row.get("comment") or "").strip()
    has_comment = (len(comment) > 50 and
                   "see attached" not in comment.lower()[:30] and
                   "please see" not in comment.lower()[:30])
    has_fulltext = any(
        (row.get(f"fullText_{i}") or "").strip()
        for i in range(1, 13)
    )

    if has_comment and has_fulltext:
        return "both"
    elif has_fulltext:
        return "fulltext_only"
    elif has_comment:
        return "comment_only"
    return "insufficient"


def load_comments(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))
