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
