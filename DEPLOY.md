# Deploy Samjhao in ~15 minutes

## 0. Unzip and run it once locally (5 min)

```bash
unzip samjhao.zip && cd samjhao
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set:
- `LLM_API_KEY` → your free key from <https://aistudio.google.com/apikey>
- `REPO_URL` → your GitHub repo URL (used by the footer "Source" link)

Then:

```bash
python test_llm.py     # must print "OK in …s ✅"
python app.py          # http://localhost:7860
```

If `test_llm.py` fails:
| Message | Do this |
|---|---|
| mentions `system` / `developer instruction` | add `FOLD_SYSTEM=1` to `.env` |
| model not found (404) | try `LLM_MODEL=gemma-4-26b-a4b-it` |
| still failing on the OpenAI-compatible route | `pip install google-genai`, set `LLM_PROVIDER=google` |
| API key error | re-copy the key; no quotes, no spaces |

## 1. GitHub — already done

The code lives at **https://github.com/Yuser00123/samjhao** (public, MIT). To push future changes from your own machine:

```bash
git clone https://github.com/Yuser00123/samjhao.git && cd samjhao
# ...edit...
git add . && git commit -m "describe the change" && git push
```

`.gitignore` excludes `.env` (your key) and the `notes/` folder (your post drafts).

## 2. Deploy on Render with the Blueprint (5 min)

1. <https://dashboard.render.com> → **New +** → **Blueprint**.
2. Connect GitHub (first time only) → pick the `samjhao` repo → Render reads `render.yaml`.
3. It will ask for the one secret marked `sync: false`: **`LLM_API_KEY`** → paste your key.
4. Click **Apply**. Build takes 2–4 minutes. Status turns **Live** → open the URL (`https://samjhao-xxxx.onrender.com`).
5. Dashboard → your service → **Environment**: confirm `FRIEND_NAME=Suyash`, `FRIEND_EXAM=JEE`, and update `REPO_URL` to your real repo URL.

Test from your phone. In Chrome → ⋮ → **Add to Home Screen** — it installs like an app (PWA).

### Manual alternative (if Blueprint isn't offered)

**New +** → **Web Service** → pick the repo →
- Runtime: **Python 3** · Region: **Singapore** · Plan: **Free**
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Environment variables: copy every `key/value` from `render.yaml` + add `LLM_API_KEY`

## 3. Things to know about the free plan

- The service **sleeps after 15 min idle**; first request after that takes ~40–60 s. Tell Suyash. (Mentioned in the post.)
- Disk is ephemeral — that's why the Doubt Diary lives in the browser session with export/import.
- Free instance hours are plenty for one user.

## 4. Troubleshooting on Render

| Symptom | Fix |
|---|---|
| Build fails on Python version | In **Environment**, set `PYTHON_VERSION` to `3.12.7` (or remove it to use Render's default 3.x) |
| "Port scan timeout" / never goes Live | App must bind `0.0.0.0:$PORT` — it does; check logs for a Python traceback instead |
| App loads but every answer is "⚠️ API key galat…" | `LLM_API_KEY` missing/typo in **Environment** → fix → **Manual Deploy → Clear build cache & deploy** |
| "Gemma thoda busy hai" often | Free-tier rate limit (~15 req/min). Wait 30 s, or switch `LLM_MODEL` to `gemma-4-26b-a4b-it` |
| Title still says "Dost" | `FRIEND_NAME` env var not set on Render → set it and redeploy |

## 5. Backup host (only if Render is down for you)

Hugging Face Spaces (Gradio SDK): create a Space → upload all `.py` files + `requirements.txt` → Settings → **Secrets** → add `LLM_API_KEY`, `FRIEND_NAME`, `FRIEND_EXAM`. Works identically — but then don't claim *Best Use of Render* in the post.

## Health check & reasoning models

* Render → your service → **Settings → Health Check Path** = `/health` (the Blueprint `render.yaml` already sets this).
  `GET /health` returns `{"status":"ok","service":"samjhao","model":…,"uptime_seconds":…}` — point a free uptime pinger at it so the free instance stays warm.
* Gemma 4 is a *reasoning* model: it thinks before it answers (20–60 s on the free tier) and the Gemini
  OpenAI-compatible endpoint returns those thoughts inline as `<thought>…</thought>`. Samjhao strips them
  and shows a "🤔 Gemma soch raha hai…" timer instead. Transient Google `500 Internal error` responses are retried automatically.
  To experiment with switching thinking off: `python test_llm.py --probe-thinking` and set the winning JSON as `LLM_EXTRA_BODY`.
