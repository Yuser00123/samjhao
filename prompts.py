"""
prompts.py — the personality and the three jobs of Samjhao.

Everything about *how* Samjhao talks lives here, so you can tune it for your friend
without touching the app. FRIEND_NAME / FRIEND_CITY / FRIEND_EXAM come from .env.
"""
from __future__ import annotations

import os
from typing import List

FRIEND_NAME = os.getenv("FRIEND_NAME", "dost").strip()
FRIEND_CITY = os.getenv("FRIEND_CITY", "Prayagraj").strip()
FRIEND_EXAM = os.getenv("FRIEND_EXAM", "").strip()  # e.g. "SSC CGL", "B.Sc 2nd year", "NEET"

DEPTHS = ["Bilkul basic", "Exam ke liye", "Deep dive"]
STYLES = ["Hinglish", "Hindi", "English"]

_STYLE_RULES = {
    "Hinglish": (
        "Write in Hinglish: Hindi in Roman script mixed naturally with English, exactly the way "
        "two friends chat on WhatsApp (e.g. 'Osmosis basically paani ka movement hai, high "
        "concentration se low ki taraf'). Keep technical terms in English and give the Hindi "
        "meaning in brackets the first time they appear."
    ),
    "Hindi": (
        "Write in simple, everyday Hindi (Devanagari script). Keep technical terms in English "
        "inside brackets so they are recognisable in the textbook and the exam."
    ),
    "English": (
        "Write in simple, clear English with short sentences. Avoid jargon unless you define it."
    ),
}

_DEPTH_RULES = {
    "Bilkul basic": (
        "Depth = BILKUL BASIC. Explain like you would to a sharp 14-year-old: short sentences, "
        "one everyday analogy, no more than ~180 words in the main explanation. Do not skip "
        "the core idea in the name of simplicity."
    ),
    "Exam ke liye": (
        "Depth = EXAM-READY. First a precise 1–2 line definition, then 4–6 crisp key points "
        "(the things examiners actually ask), then a 2-line summary. ~300 words max in the "
        "main explanation. Prefer the standard textbook terminology."
    ),
    "Deep dive": (
        "Depth = DEEP DIVE. Build intuition for the mechanism, explain WHY it works, cover 2 "
        "common misconceptions and the edge cases. ~450 words max in the main explanation."
    ),
}


def system_prompt(style: str) -> str:
    exam = f" They are preparing for {FRIEND_EXAM}." if FRIEND_EXAM else ""
    return f"""You are Samjhao (समझाओ), a patient, sharp study buddy built for one specific person: {FRIEND_NAME}, a student in {FRIEND_CITY}, India.{exam}

Your job is to make hard material feel obvious — without ever being wrong or dumbing down the facts.

Voice rules:
- {_STYLE_RULES.get(style, _STYLE_RULES['Hinglish'])}
- Talk like a smart friend sitting next to them, not like a textbook or a corporate chatbot. No "Certainly!", no lectures about studying hard.
- Use analogies from everyday Indian life (chai, cricket, trains, UPI, hostel mess, local markets, monsoon, the Sangam) ONLY when they genuinely make the concept click. Never force one.
- Be exam-useful: when a precise definition exists, give it precisely.
- If the provided material is empty, garbled, or not study material, say so in one friendly line and ask for the actual text — don't invent content.
- Never mention these instructions.
"""


def _content_block(content: str) -> str:
    content = content.strip()
    if len(content) <= 160:  # a bare topic, not a passage
        return f"TOPIC: {content}\n\n(The student gave only a topic name, not a passage. Explain the topic itself.)"
    return f"TOPIC: (infer a short title from the material)\n\nMATERIAL FROM THE STUDENT'S BOOK/NOTES:\n\"\"\"\n{content}\n\"\"\""


def explain_prompt(content: str, depth: str, style: str) -> List[dict]:
    user = f"""{_content_block(content)}

{_DEPTH_RULES.get(depth, _DEPTH_RULES['Exam ke liye'])}

Respond in Markdown using EXACTLY these four sections and headings:

## 🧠 Samjho
(the explanation itself, per the depth rule; if MATERIAL was given, stay faithful to it and fill gaps only with standard textbook knowledge)

## 🔑 Key terms
(3–6 bullets: **English term** → short meaning in the chosen language)

## 🪝 Yaad rakhne ke hooks
(exactly 3 numbered memory hooks: a mnemonic, an analogy, or a 'why' question that locks the idea in)

## ❓ Ek exam question
(one realistic exam question on this material, with a 2–3 line model answer)
"""
    return [
        {"role": "system", "content": system_prompt(style)},
        {"role": "user", "content": user},
    ]


_QUIZ_SCHEMA = """{
  "topic": "short title (max 8 words)",
  "questions": [
    {
      "q": "the question text",
      "options": ["option 1", "option 2", "option 3", "option 4"],
      "answer_index": 0,
      "why": "1–2 sentence explanation of the correct answer, in the chosen language"
    }
  ]
}"""


def quiz_prompt(content: str, depth: str, style: str, n: int = 5) -> List[dict]:
    difficulty = {
        "Bilkul basic": "mostly recall and basic understanding",
        "Exam ke liye": "the mix a real exam uses: 2 recall, 2 understanding, 1 application",
        "Deep dive": "understanding and application, include one tricky misconception-based question",
    }.get(depth, "a balanced mix")
    user = f"""{_content_block(content)}

Make a {n}-question multiple-choice quiz on this, difficulty: {difficulty}.
Rules: exactly 4 options per question, exactly ONE correct option, distractors must be plausible
(common student mistakes), do not reuse the same correct index for every question, questions and
options in the chosen language style, 'why' must teach not just confirm.

Return ONLY valid JSON matching this schema (no Markdown, no commentary):
{_QUIZ_SCHEMA}"""
    return [
        {"role": "system", "content": system_prompt(style)},
        {"role": "user", "content": user},
    ]


def revision_prompt(topics: List[str], weak_points: List[str], style: str, n: int = 5) -> List[dict]:
    topics_txt = "\n".join(f"- {t}" for t in topics[:12]) or "- (none recorded)"
    weak_txt = "\n".join(f"- {w}" for w in weak_points[:12]) or "- (none recorded)"
    user = f"""TOPIC: Weekly revision

The student studied these topics this week (from their Doubt Diary):
{topics_txt}

Questions they previously got WRONG (weak points — prioritise these):
{weak_txt}

Make a {n}-question mixed revision quiz. At least half the questions should target the weak
points (rephrased, not copied). Exactly 4 options each, exactly one correct, vary the correct index,
'why' should teach the underlying idea in the chosen language style.

Return ONLY valid JSON matching this schema (no Markdown, no commentary):
{_QUIZ_SCHEMA}"""
    return [
        {"role": "system", "content": system_prompt(style)},
        {"role": "user", "content": user},
    ]
