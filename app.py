"""
Samjhao (समझाओ) — a Hinglish study buddy built for one friend, powered by open-weight Gemma.

Run:  python app.py          (reads .env if present)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Load .env without a hard dependency on python-dotenv
_env = Path(__file__).with_name(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import gradio as gr  # noqa: E402

import diary as D  # noqa: E402
import llm  # noqa: E402
import prompts as P  # noqa: E402
from extract import clamp, guess_topic, text_from_pdf  # noqa: E402

APP_TITLE = f"📚 Samjhao — {P.FRIEND_NAME} ka padhai saathi"
REPO_URL = os.getenv("REPO_URL", "https://github.com/Yuser00123/samjhao").strip()
LETTERS = ["A", "B", "C", "D"]
N_Q = 5

EXAMPLES = [
    [
        "The work-energy theorem states that the net work done by all the forces acting on a particle equals "
        "the change in its kinetic energy: W_net = ΔK = ½mv² − ½mu². The theorem holds for both constant and "
        "variable forces and in both inertial and non-inertial frames, provided pseudo forces are included in "
        "the latter. Work done by conservative forces can be expressed as the negative of the change in "
        "potential energy, which leads to the principle of conservation of mechanical energy when only "
        "conservative forces do work.",
        "Exam ke liye",
        "Hinglish",
    ],
    ["Le Chatelier's principle", "Bilkul basic", "Hinglish"],
    ["Limits, continuity and differentiability — kahan fail hota hai?", "Deep dive", "Hinglish"],
]


# ----------------------------------------------------------------------------- helpers
def _prepare_content(text: str, pdf_path: str | None, start_page: float, max_pages: float) -> tuple[str, str]:
    """Pick pasted text if present, else PDF text. Returns (content, info_line)."""
    text = (text or "").strip()
    info = ""
    if not text and pdf_path:
        text = text_from_pdf(pdf_path, int(start_page or 1), int(max_pages or 3))
        info = f"📄 PDF ke page {int(start_page or 1)} se {int(max_pages or 3)} page padhe. "
    if not text:
        raise ValueError("Pehle kuch paste karo ya PDF upload karo — phir Samjhao dabao. 🙂")
    content, truncated = clamp(text)
    if truncated:
        info += "✂️ Text lamba tha, pehle ~12,000 characters liye. Baaki agli baar."
    return content, info


def _normalize_quiz(data: dict) -> dict:
    questions = []
    for q in data.get("questions", []) or []:
        if not isinstance(q, dict):
            continue
        text = str(q.get("q", "")).strip()
        opts = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
        try:
            idx = int(q.get("answer_index"))
        except (TypeError, ValueError):
            continue
        if not text or len(opts) != 4 or not 0 <= idx < 4:
            continue
        questions.append({"q": text, "options": opts, "answer_index": idx, "why": str(q.get("why", "")).strip()})
    return {"topic": str(data.get("topic", "")).strip()[:80], "questions": questions[:N_Q]}


def _radio_updates(quiz: dict | None):
    """Return N_Q gr.update()s that show/hide the question radios."""
    updates = []
    qs = (quiz or {}).get("questions", [])
    for i in range(N_Q):
        if i < len(qs):
            q = qs[i]
            choices = [f"{LETTERS[j]}) {opt}" for j, opt in enumerate(q["options"])]
            updates.append(gr.update(visible=True, label=f"Q{i + 1}. {q['q']}", choices=choices, value=None))
        else:
            updates.append(gr.update(visible=False, choices=[], value=None, label=""))
    return updates


def _tmpfile(suffix: str, text: str) -> str:
    path = Path(tempfile.gettempdir()) / f"doubt-diary{suffix}"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _diary_outputs(diary):
    return diary, D.to_rows(diary), D.stats(diary)


# ----------------------------------------------------------------------------- handlers
def do_explain(text, pdf_path, start_page, max_pages, depth, style, diary):
    """Streams the explanation, then logs it in the diary."""
    try:
        content, info = _prepare_content(text, pdf_path, start_page, max_pages)
    except Exception as err:  # noqa: BLE001
        yield f"⚠️ {err}", "", *_diary_outputs(diary)
        return

    header = f"> {info}\n\n" if info else ""
    out = ""
    try:
        for chunk in llm.stream(P.explain_prompt(content, depth, style)):
            out += chunk
            yield header + out, content, *_diary_outputs(diary)
    except llm.LLMError as err:
        yield header + out + f"\n\n⚠️ {err}", content, *_diary_outputs(diary)
        return
    except Exception as err:  # noqa: BLE001
        yield header + out + f"\n\n⚠️ Kuch gadbad hui: {str(err)[:200]}", content, *_diary_outputs(diary)
        return

    diary = D.add(diary, kind="explain", topic=guess_topic(content), depth=depth, style=style)
    yield header + out, content, *_diary_outputs(diary)


def do_quiz(content, depth, style):
    """Builds a quiz from the last explained content and jumps to the Quiz tab."""
    if not (content or "").strip():
        gr.Warning("Pehle kuch Samjhao, phir uska quiz banega.")
        return None, "", "", *_radio_updates(None), gr.Tabs(selected="explain")
    try:
        raw = llm.chat(P.quiz_prompt(content, depth, style), json_mode=True)
        quiz = _normalize_quiz(llm.parse_json(raw))
    except llm.LLMError as err:
        return None, f"⚠️ {err}", "", *_radio_updates(None), gr.Tabs(selected="quiz")
    except Exception as err:  # noqa: BLE001
        return None, f"⚠️ Quiz banate waqt error: {str(err)[:200]}", "", *_radio_updates(None), gr.Tabs(selected="quiz")
    if len(quiz["questions"]) < 3:
        return None, "⚠️ Model ne poora quiz nahi diya. Ek baar aur 'Quiz banao' dabao.", "", *_radio_updates(None), gr.Tabs(selected="quiz")
    quiz["depth"], quiz["style"], quiz["kind"] = depth, style, "quiz"
    title = f"### 📝 {quiz['topic'] or guess_topic(content)}\n{len(quiz['questions'])} sawaal · {depth} · {style}"
    return quiz, title, "", *_radio_updates(quiz), gr.Tabs(selected="quiz")


def do_revision(diary, style):
    """Mixed revision quiz from the Doubt Diary (weak points first)."""
    tps, weak = D.topics(diary), D.weak_points(diary)
    if not tps:
        gr.Warning("Diary khaali hai — pehle kuch topics samjho ya quiz do.")
        return None, "", "", *_radio_updates(None), gr.Tabs(selected="diary")
    try:
        raw = llm.chat(P.revision_prompt(tps, weak, style), json_mode=True)
        quiz = _normalize_quiz(llm.parse_json(raw))
    except llm.LLMError as err:
        return None, f"⚠️ {err}", "", *_radio_updates(None), gr.Tabs(selected="quiz")
    except Exception as err:  # noqa: BLE001
        return None, f"⚠️ Revision quiz error: {str(err)[:200]}", "", *_radio_updates(None), gr.Tabs(selected="quiz")
    if len(quiz["questions"]) < 3:
        return None, "⚠️ Model ne poora quiz nahi diya. Dobara try karo.", "", *_radio_updates(None), gr.Tabs(selected="quiz")
    quiz["depth"], quiz["style"], quiz["kind"] = "Revision", style, "revision"
    quiz["topic"] = quiz["topic"] or "Weekly revision"
    title = f"### 🔁 {quiz['topic']}\n{len(quiz['questions'])} sawaal · {len(tps)} topics · {len(weak)} weak points targeted"
    return quiz, title, "", *_radio_updates(quiz), gr.Tabs(selected="quiz")


def do_check(quiz, *answers_and_diary):
    """Scores the quiz, explains every question, logs score + weak points."""
    *answers, diary = answers_and_diary
    if not quiz or not quiz.get("questions"):
        return "⚠️ Pehle quiz banao.", *_diary_outputs(diary)
    qs = quiz["questions"]
    if any(a is None for a in answers[: len(qs)]):
        return "⚠️ Saare sawaalon ka jawab choose karo, phir Check karo.", *_diary_outputs(diary)

    correct, weak, lines = 0, [], []
    for i, q in enumerate(qs):
        picked = int(answers[i])
        right = q["answer_index"]
        ok = picked == right
        correct += ok
        if not ok:
            weak.append(q["q"])
        mark = "✅" if ok else "❌"
        lines.append(f"**{mark} Q{i + 1}. {q['q']}**")
        lines.append(f"- Tumhara jawab: {LETTERS[picked]}) {q['options'][picked]}")
        if not ok:
            lines.append(f"- Sahi jawab: **{LETTERS[right]}) {q['options'][right]}**")
        if q.get("why"):
            lines.append(f"- 💡 {q['why']}")
        lines.append("")

    n = len(qs)
    pct = round(100 * correct / n)
    if pct == 100:
        verdict = "🔥 Full marks! Yeh topic pakka ho gaya."
    elif pct >= 60:
        verdict = "👍 Theek hai — galat wale do-teen points dobara dekh lo."
    else:
        verdict = "🙂 Koi baat nahi, isiliye to practice hai. Niche wale 'why' dhyan se padho, phir Diary se revision quiz do."
    score = f"{correct}/{n}"
    summary = f"## Score: {score} ({pct}%)\n{verdict}\n\n" + "\n".join(lines)

    diary = D.add(
        diary,
        kind=quiz.get("kind", "quiz"),
        topic=quiz.get("topic") or "Quiz",
        depth=quiz.get("depth", ""),
        style=quiz.get("style", ""),
        score=score,
        weak=weak,
    )
    return summary, *_diary_outputs(diary)


def export_md(diary):
    return _tmpfile(".md", D.to_markdown(diary))


def export_json(diary):
    return _tmpfile(".json", D.to_json(diary))


def import_json(file_path, diary):
    if not file_path:
        return _diary_outputs(diary)
    try:
        loaded = D.from_json(Path(file_path).read_text(encoding="utf-8"))
    except Exception as err:  # noqa: BLE001
        gr.Warning(f"Diary file padh nahi paaya: {str(err)[:120]}")
        return _diary_outputs(diary)
    # merge: keep existing + new (de-dup on ts+topic)
    seen = {(e["ts"], e["topic"], e["kind"]) for e in diary or []}
    merged = list(diary or []) + [e for e in loaded if (e["ts"], e["topic"], e["kind"]) not in seen]
    return _diary_outputs(merged)


# ----------------------------------------------------------------------------- UI
CSS = """
.gradio-container { max-width: 880px !important; margin: 0 auto !important; }
#explain-out, #quiz-result { font-size: 1.02rem; line-height: 1.6; }
#title h1 { margin-bottom: 0.1rem; }
footer { display: none !important; }
"""

with gr.Blocks(title="Samjhao") as demo:
    content_state = gr.State("")
    quiz_state = gr.State(None)
    diary_state = gr.State(D.new_diary())

    gr.Markdown(
        f"# {APP_TITLE}\n"
        f"Paste karo, samjho, quiz do, bhool jao to Diary se revise karo. "
        f"**Open-weight Gemma** par chalta hai — free, swappable, aur chaaho to bina internet ke bhi.",
        elem_id="title",
    )

    with gr.Tabs() as tabs:
        # ------------------------------------------------------------------ Explain
        with gr.Tab("🧠 Samjhao", id="explain"):
            text_in = gr.Textbox(
                label="Yahan paste karo — paragraph, notes, ya sirf topic ka naam",
                placeholder="e.g. 'Newton's second law' … ya kitaab ka poora paragraph paste karo",
                lines=7,
            )
            with gr.Accordion("…ya PDF upload karo", open=False):
                pdf_in = gr.File(label="PDF (text-based, scanned nahi)", file_types=[".pdf"], type="filepath")
                with gr.Row():
                    start_page = gr.Number(label="Kis page se", value=1, precision=0, minimum=1)
                    max_pages = gr.Number(label="Kitne page", value=3, precision=0, minimum=1, maximum=10)
            with gr.Row():
                depth = gr.Radio(P.DEPTHS, value="Exam ke liye", label="Kitna deep")
                style = gr.Radio(P.STYLES, value="Hinglish", label="Kis bhasha me")
            with gr.Row():
                explain_btn = gr.Button("Samjhao 🙏", variant="primary", scale=2)
                quiz_btn = gr.Button("Quiz banao 📝", variant="secondary", scale=1)
            explain_out = gr.Markdown(elem_id="explain-out")
            gr.Examples(EXAMPLES, inputs=[text_in, depth, style], label="Try karo")

        # ------------------------------------------------------------------ Quiz
        with gr.Tab("📝 Quiz", id="quiz"):
            quiz_title = gr.Markdown("Pehle 'Samjhao' tab me kuch samjho, phir **Quiz banao** dabao — ya Diary se **Revision test** lo.")
            radios = [gr.Radio(choices=[], label="", type="index", visible=False) for _ in range(N_Q)]
            check_btn = gr.Button("Check karo ✅", variant="primary")
            quiz_result = gr.Markdown(elem_id="quiz-result")

        # ------------------------------------------------------------------ Diary
        with gr.Tab("📓 Doubt Diary", id="diary"):
            diary_stats = gr.Markdown(D.stats(D.new_diary()))
            diary_table = gr.Dataframe(
                headers=["Kab", "Kya", "Topic", "Depth", "Score"],
                datatype=["str", "str", "str", "str", "str"],
                value=D.to_rows(D.new_diary()),
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                revision_btn = gr.Button("Revision test banao 🔁", variant="primary")
                rev_style = gr.Radio(P.STYLES, value="Hinglish", label="Bhasha", scale=1)
            gr.Markdown(
                "Diary sirf is browser session me rehti hai (koi account, koi server-side storage nahi). "
                "Band karne se pehle **download** kar lo; agli baar **upload** karke wahin se continue karo."
            )
            with gr.Row():
                dl_md_btn = gr.Button("⬇️ Download (.md)")
                dl_json_btn = gr.Button("⬇️ Backup (.json)")
            dl_file = gr.File(label="Download ready", visible=True)
            up_json = gr.File(label="⬆️ Purani diary upload karo (.json)", file_types=[".json"], type="filepath")

    gr.Markdown(
        f"<small>Model: <b>{llm.describe()}</b> · Sirf tumhara pasted text model tak jaata hai, aur kuch nahi · "
        f"Offline chahiye? <code>LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=gemma3:4b</code> (Ollama) · "
        f"<a href='{REPO_URL}' target='_blank'>Source</a> · Built for Hacktoberfest 2026</small>"
    )

    # ------------------------------------------------------------------ wiring
    diary_outs = [diary_state, diary_table, diary_stats]
    quiz_outs = [quiz_state, quiz_title, quiz_result, *radios, tabs]

    explain_btn.click(
        do_explain,
        inputs=[text_in, pdf_in, start_page, max_pages, depth, style, diary_state],
        outputs=[explain_out, content_state, *diary_outs],
    )
    text_in.submit(
        do_explain,
        inputs=[text_in, pdf_in, start_page, max_pages, depth, style, diary_state],
        outputs=[explain_out, content_state, *diary_outs],
    )
    quiz_btn.click(do_quiz, inputs=[content_state, depth, style], outputs=quiz_outs)
    revision_btn.click(do_revision, inputs=[diary_state, rev_style], outputs=quiz_outs)
    check_btn.click(do_check, inputs=[quiz_state, *radios, diary_state], outputs=[quiz_result, *diary_outs])
    dl_md_btn.click(export_md, inputs=[diary_state], outputs=[dl_file])
    dl_json_btn.click(export_json, inputs=[diary_state], outputs=[dl_file])
    up_json.upload(import_json, inputs=[up_json, diary_state], outputs=diary_outs)


# ----------------------------------------------------------------------------- server
# Gradio runs on FastAPI. We mount it on our own FastAPI app so we can add plain HTTP
# endpoints next to the UI — e.g. /health for Render's health checks and uptime monitors.
import time  # noqa: E402

from fastapi import FastAPI  # noqa: E402

STARTED_AT = time.time()

api = FastAPI(title="Samjhao", docs_url=None, redoc_url=None, openapi_url=None)


@api.get("/health")
def health():
    """Liveness probe: 200 + {"status": "ok"} whenever the process is up."""
    return {
        "status": "ok",
        "service": "samjhao",
        "model": llm.describe(),
        "uptime_seconds": int(time.time() - STARTED_AT),
    }


demo.queue(default_concurrency_limit=4)
app = gr.mount_gradio_app(
    api,
    demo,
    path="/",
    theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone"),
    css=CSS,
    pwa=True,  # lets your friend "Add to Home Screen" on their phone
    show_error=True,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
