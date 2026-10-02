"""
llm.py — one tiny wrapper, three ways to run Gemma.

    LLM_PROVIDER=openai  (default)  any OpenAI-compatible endpoint:
                                     Google AI Studio (Gemma 4), Ollama (offline),
                                     Groq, OpenRouter, Backboard, DigitalOcean Gradient...
    LLM_PROVIDER=google             native google-genai SDK (Gemini API)
    LLM_PROVIDER=mock               canned responses, zero network (UI testing / demo fallback)

Because Gemma's weights are open, the *same* model can be served by Google for free,
by Ollama on a laptop with no internet, or by any other provider — swapping is just
two environment variables. That is the whole point of this file.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, Iterator, List

Message = Dict[str, str]

PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
MODEL = os.getenv("LLM_MODEL", "gemma-4-31b-it").strip()
BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
).strip()
API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or "not-needed"  # Ollama ignores the key but the SDK wants a non-empty string
).strip()
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
# Some Gemma builds (e.g. Gemma 3 on the Gemini API) reject a `system` role.
# Set FOLD_SYSTEM=1 to always merge the system prompt into the first user turn.
FOLD_SYSTEM = os.getenv("FOLD_SYSTEM", "0") == "1"


class LLMError(RuntimeError):
    """Raised with a short, friendly message that the UI can show directly."""


def describe() -> str:
    """Human-readable 'what am I talking to' string for the UI footer."""
    if PROVIDER == "mock":
        return "mock responses (no model connected — set LLM_API_KEY)"
    if PROVIDER == "google":
        return f"{MODEL} via Gemini API (google-genai)"
    host = re.sub(r"^https?://", "", BASE_URL).split("/")[0]
    return f"{MODEL} via {host}"


# ----------------------------------------------------------------------------- helpers
def _fold_system(messages: List[Message]) -> List[Message]:
    """Merge a leading system message into the first user message."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{system}\n\n---\n\n{rest[0]['content']}"}
    else:
        rest.insert(0, {"role": "user", "content": system})
    return rest


def _looks_like_system_role_error(err: Exception) -> bool:
    text = str(err).lower()
    return any(k in text for k in ("system", "developer instruction", "system_instruction"))


def _friendly(err: Exception) -> LLMError:
    text = str(err)
    low = text.lower()
    if "429" in low or "rate" in low or "quota" in low or "resource_exhausted" in low:
        return LLMError("Gemma thoda busy hai (rate limit). 20–30 second ruk ke dobara try karo. 🙏")
    if "401" in low or "403" in low or "api key" in low or "permission" in low:
        return LLMError("API key galat ya missing hai. .env me LLM_API_KEY check karo.")
    if "404" in low or "not found" in low or "model" in low and "found" in low:
        return LLMError(f"Model '{MODEL}' is provider pe nahi mila. LLM_MODEL env var check karo.")
    if "connection" in low or "connect" in low or "timed out" in low or "timeout" in low:
        return LLMError("Model server se connect nahi ho paaya. Internet / Ollama chal raha hai kya?")
    return LLMError(f"Model error: {text[:200]}")


def parse_json(text: str) -> dict:
    """Lenient JSON extraction: strips code fences and trims to the outermost {...}."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMError("Model ne valid JSON nahi diya. Ek baar aur try karo.")
        return json.loads(cleaned[start : end + 1])


# ----------------------------------------------------------------------------- openai-compatible
def _openai_client():
    from openai import OpenAI  # imported lazily so `mock` mode needs no SDK

    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120, max_retries=0)


def _openai_chat(messages: List[Message], json_mode: bool, temperature: float, max_tokens: int, stream: bool):
    client = _openai_client()
    kwargs = dict(model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as err:  # noqa: BLE001 — we translate everything into LLMError
        # 1) provider doesn't support response_format → retry without it (prompt already asks for JSON)
        if json_mode and ("response_format" in str(err).lower() or "json" in str(err).lower()):
            kwargs.pop("response_format", None)
            return client.chat.completions.create(**kwargs)
        # 2) provider rejects the system role → fold it into the user turn and retry once
        if _looks_like_system_role_error(err) and messages and messages[0]["role"] == "system":
            kwargs["messages"] = _fold_system(messages)
            return client.chat.completions.create(**kwargs)
        raise


# ----------------------------------------------------------------------------- google-genai (native)
def _google_parts(messages: List[Message]):
    from google.genai import types

    system = None
    contents = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return system, contents


def _google_chat(messages: List[Message], json_mode: bool, temperature: float, max_tokens: int, stream: bool):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=API_KEY)
    system, contents = _google_parts(_fold_system(messages) if FOLD_SYSTEM else messages)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
        response_mime_type="application/json" if json_mode else None,
    )
    fn = client.models.generate_content_stream if stream else client.models.generate_content
    try:
        return fn(model=MODEL, contents=contents, config=config)
    except Exception as err:  # noqa: BLE001
        if system and _looks_like_system_role_error(err):
            _, contents = _google_parts(_fold_system(messages))
            config.system_instruction = None
            return fn(model=MODEL, contents=contents, config=config)
        raise


# ----------------------------------------------------------------------------- mock
def _mock_text(messages: List[Message], json_mode: bool) -> str:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    topic_match = re.search(r"TOPIC:\s*(.+)", user)
    topic = (topic_match.group(1) if topic_match else "yeh topic").strip()[:60]
    if json_mode:
        qs = []
        for i in range(5):
            qs.append(
                {
                    "q": f"(mock) {topic} ke baare me sawaal #{i + 1}: sahi option kaun sa hai?",
                    "options": ["Pehla option", "Doosra option", "Teesra option", "Chautha option"],
                    "answer_index": i % 4,
                    "why": "Mock mode me koi real jawab nahi hai — LLM_API_KEY set karo to Gemma asli quiz banayega.",
                }
            )
        return json.dumps({"topic": topic, "questions": qs}, ensure_ascii=False)
    return (
        f"## 🧠 Samjho\n\n*(Mock mode — Gemma connected nahi hai. `.env` me `LLM_API_KEY` daalo.)*\n\n"
        f"**{topic}** ko aise socho: jaise chai banane me pehle paani garam hota hai, phir patti, phir doodh — "
        f"har step ka ek order hai. Is topic me bhi pehle *definition*, phir *mechanism*, phir *example*.\n\n"
        "## 🔑 Key terms\n- **Term (English)** → matlab (Hindi)\n\n"
        "## 🪝 Yaad rakhne ke hooks\n1. Ek chhota sa mnemonic\n2. Ek analogy\n3. Ek 'kyun' wala sawaal\n\n"
        "## ❓ Ek exam question\nIs topic ko 3 lines me define karo aur ek example do."
    )


# ----------------------------------------------------------------------------- public API
def chat(messages: List[Message], json_mode: bool = False, temperature: float | None = None, max_tokens: int | None = None) -> str:
    """Return the full assistant reply as a string. Retries politely on rate limits."""
    temperature = TEMPERATURE if temperature is None else temperature
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    if PROVIDER == "mock":
        time.sleep(0.4)
        return _mock_text(messages, json_mode)
    if FOLD_SYSTEM and PROVIDER == "openai":
        messages = _fold_system(messages)

    delays = (0, 3, 8)  # seconds between attempts
    last_err: Exception | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            if PROVIDER == "google":
                resp = _google_chat(messages, json_mode, temperature, max_tokens, stream=False)
                return resp.text or ""
            resp = _openai_chat(messages, json_mode, temperature, max_tokens, stream=False)
            return resp.choices[0].message.content or ""
        except Exception as err:  # noqa: BLE001
            last_err = err
            if "429" not in str(err) and "rate" not in str(err).lower():
                break
    raise _friendly(last_err or RuntimeError("unknown error"))


def stream(messages: List[Message], temperature: float | None = None, max_tokens: int | None = None) -> Iterator[str]:
    """Yield the reply token-by-token (falls back to one chunk when streaming isn't supported)."""
    temperature = TEMPERATURE if temperature is None else temperature
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    if PROVIDER == "mock":
        text = _mock_text(messages, json_mode=False)
        for i in range(0, len(text), 12):
            time.sleep(0.01)
            yield text[i : i + 12]
        return
    if FOLD_SYSTEM and PROVIDER == "openai":
        messages = _fold_system(messages)
    try:
        if PROVIDER == "google":
            for chunk in _google_chat(messages, False, temperature, max_tokens, stream=True):
                if getattr(chunk, "text", None):
                    yield chunk.text
            return
        for chunk in _openai_chat(messages, False, temperature, max_tokens, stream=True):
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except LLMError:
        raise
    except Exception as err:  # noqa: BLE001
        # Some endpoints don't stream; try once more without streaming.
        if "stream" in str(err).lower():
            yield chat(messages, False, temperature, max_tokens)
            return
        raise _friendly(err)
