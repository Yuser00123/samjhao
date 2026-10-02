"""
extract.py — turn whatever the student has (pasted text, a PDF page) into clean text.
"""
from __future__ import annotations

import re

MAX_CHARS = 12_000  # ~3k tokens; plenty for a chapter section, keeps free-tier calls fast


def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clamp(text: str, limit: int = MAX_CHARS) -> tuple[str, bool]:
    """Return (text, was_truncated)."""
    text = normalize(text)
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    # don't cut mid-sentence if we can help it
    last_stop = max(cut.rfind(". "), cut.rfind("।"), cut.rfind("\n"))
    if last_stop > limit * 0.7:
        cut = cut[: last_stop + 1]
    return cut.strip(), True


def text_from_pdf(path: str, start_page: int = 1, max_pages: int = 3) -> str:
    """Extract text from `max_pages` pages starting at 1-based `start_page`."""
    from pypdf import PdfReader  # lazy import keeps startup fast

    reader = PdfReader(path)
    total = len(reader.pages)
    start = max(1, int(start_page or 1))
    end = min(total, start + max(1, int(max_pages or 1)) - 1)
    chunks = []
    for i in range(start - 1, end):
        try:
            chunks.append(reader.pages[i].extract_text() or "")
        except Exception:  # noqa: BLE001 — a bad page shouldn't kill the whole upload
            chunks.append("")
    text = normalize("\n\n".join(chunks))
    if not text:
        raise ValueError(
            f"PDF ke page {start}–{end} se text nahi nikla (scanned image lagta hai). "
            "Us page ka text type/paste kar do, ya koi text-based PDF use karo."
        )
    return text


def guess_topic(content: str, fallback: str = "Untitled") -> str:
    """A short label for the Doubt Diary when the model didn't give one."""
    content = normalize(content)
    if not content:
        return fallback
    if len(content) <= 80:
        return content
    first = re.split(r"(?<=[.!?।])\s+|\n", content, maxsplit=1)[0]
    return (first[:70] + "…") if len(first) > 70 else first
