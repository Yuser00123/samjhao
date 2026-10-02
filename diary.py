"""
diary.py — the Doubt Diary.

A plain list of dicts kept in the browser session (gr.State), exportable to JSON/Markdown.
No database, no account, nothing leaves the student's device except the text sent to the model.
Optional: set DIARY_PATH=/some/file.json to also persist to disk when running locally.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List

Diary = List[dict]
DIARY_PATH = os.getenv("DIARY_PATH", "").strip()


def new_diary() -> Diary:
    if DIARY_PATH and os.path.exists(DIARY_PATH):
        try:
            with open(DIARY_PATH, encoding="utf-8") as fh:
                return from_json(fh.read())
        except Exception:  # noqa: BLE001 — a corrupt file shouldn't block the app
            return []
    return []


def _persist(diary: Diary) -> None:
    if DIARY_PATH:
        try:
            with open(DIARY_PATH, "w", encoding="utf-8") as fh:
                fh.write(to_json(diary))
        except Exception:  # noqa: BLE001
            pass


def add(diary: Diary, *, kind: str, topic: str, depth: str, style: str, score: str = "", weak: List[str] | None = None, note: str = "") -> Diary:
    entry = {
        "ts": datetime.now().strftime("%d %b %H:%M"),
        "kind": kind,  # "explain" | "quiz" | "revision"
        "topic": topic.strip()[:120] or "Untitled",
        "depth": depth,
        "style": style,
        "score": score,
        "weak": list(weak or []),
        "note": note.strip()[:300],
    }
    diary = list(diary or []) + [entry]
    _persist(diary)
    return diary


def to_rows(diary: Diary) -> List[list]:
    icons = {"explain": "🧠 Samjha", "quiz": "📝 Quiz", "revision": "🔁 Revision"}
    return [
        [e["ts"], icons.get(e["kind"], e["kind"]), e["topic"], e["depth"], e["score"] or "—"]
        for e in reversed(diary or [])
    ]


def topics(diary: Diary) -> List[str]:
    seen, out = set(), []
    for e in diary or []:
        t = e["topic"]
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def weak_points(diary: Diary) -> List[str]:
    out: List[str] = []
    for e in diary or []:
        out.extend(e.get("weak") or [])
    # most recent first, de-duplicated
    seen, uniq = set(), []
    for w in reversed(out):
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def stats(diary: Diary) -> str:
    d = diary or []
    n_exp = sum(1 for e in d if e["kind"] == "explain")
    n_quiz = sum(1 for e in d if e["kind"] in ("quiz", "revision"))
    n_weak = len(weak_points(d))
    if not d:
        return "Diary abhi khaali hai. Kuch samjho ya quiz do — sab yahan save hota jaayega."
    return f"**{len(topics(d))}** topics · **{n_exp}** explanations · **{n_quiz}** quizzes · **{n_weak}** weak points to revise"


def to_markdown(diary: Diary) -> str:
    lines = ["# Doubt Diary — Samjhao", ""]
    for e in diary or []:
        lines.append(f"## {e['ts']} · {e['kind'].title()} · {e['topic']}")
        lines.append(f"- Depth: {e['depth']} · Style: {e['style']}" + (f" · Score: {e['score']}" if e["score"] else ""))
        if e.get("weak"):
            lines.append("- Got wrong:")
            lines.extend(f"  - {w}" for w in e["weak"])
        if e.get("note"):
            lines.append(f"- Note: {e['note']}")
        lines.append("")
    return "\n".join(lines)


def to_json(diary: Diary) -> str:
    return json.dumps(diary or [], ensure_ascii=False, indent=2)


def from_json(text: str) -> Diary:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Diary file should contain a JSON list.")
    required = {"ts", "kind", "topic", "depth", "style"}
    return [e for e in data if isinstance(e, dict) and required.issubset(e)]
