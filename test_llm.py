"""
10-second smoke test: is Gemma reachable with the current .env?

    python test_llm.py
"""
import os, sys, time
from pathlib import Path

_env = Path(__file__).with_name(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import llm  # noqa: E402

print(f"Provider : {llm.PROVIDER}")
print(f"Model    : {llm.describe()}")
t0 = time.time()
try:
    reply = llm.chat(
        [
            {"role": "system", "content": "You are a friendly study buddy who answers in Hinglish."},
            {"role": "user", "content": "Ek line me batao: photosynthesis kya hai?"},
        ],
        max_tokens=120,
    )
    print(f"Reply    : {reply.strip()[:300]}")
    print(f"OK in {time.time() - t0:.1f}s ✅")
except llm.LLMError as err:
    print(f"FAILED   : {err}")
    sys.exit(1)

# JSON mode check (what the quiz uses)
try:
    raw = llm.chat(
        [{"role": "user", "content": 'Return ONLY JSON: {"ok": true, "lang": "hinglish"}'}],
        json_mode=True,
        max_tokens=60,
    )
    data = llm.parse_json(raw)
    print(f"JSON mode: {data} ✅")
except Exception as err:  # noqa: BLE001
    print(f"JSON mode: ⚠️ {err}  (quiz may still work — the prompt also asks for JSON)")

# ── Optional: python test_llm.py --probe-thinking ─────────────────────────────
# Gemma 4 on the Gemini API "thinks" before answering (slow, and the thoughts arrive
# as <thought>…</thought> text — Samjhao strips them). This probe tries a few
# LLM_EXTRA_BODY values and reports latency + whether a thought block appeared, so
# you can pick the fastest setting for your provider and put it in .env.
if "--probe-thinking" in sys.argv and llm.PROVIDER == "openai":
    import json
    candidates = [
        None,
        {"reasoning_effort": "none"},
        {"reasoning_effort": "low"},
        {"google": {"thinking_config": {"thinking_level": "minimal"}}},
        {"google": {"thinking_config": {"thinking_budget": 0}}},
    ]
    msgs = [{"role": "user", "content": "Ek line me batao: Ohm's law kya hai? Hinglish me."}]
    print("\nProbing thinking controls (each call may take up to ~90s)…")
    for body in candidates:
        llm.EXTRA_BODY = body
        t0 = time.time()
        try:
            raw = llm.chat(msgs, max_tokens=1024, strip=False)
            saw = "<thought>" in raw or "<think>" in raw
            print(f"  {json.dumps(body):70s} → {time.time() - t0:5.1f}s  thought={'yes' if saw else 'no '}  {llm.strip_thoughts(raw).strip()[:60]!r}")
        except Exception as err:  # noqa: BLE001
            print(f"  {json.dumps(body):70s} → ⚠️ {str(err)[:90]}")
    print("Fastest row with thought=no  → set that JSON as LLM_EXTRA_BODY in .env / Render.")
