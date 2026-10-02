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
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))  # reasoning models spend tokens on thinking first
# Some Gemma builds (e.g. Gemma 3 on the Gemini API) reject a `system` role.
# Set FOLD_SYSTEM=1 to always merge the system prompt into the first user turn.
FOLD_SYSTEM = os.getenv("FOLD_SYSTEM", "0") == "1"


# Optional provider-specific request fields as JSON, e.g. to try switching off thinking:
#   LLM_EXTRA_BODY={"reasoning_effort":"none"}
#   LLM_EXTRA_BODY={"google":{"thinking_config":{"thinking_level":"minimal"}}}
# If the provider rejects it (400), the request is retried once without it.
try:
    EXTRA_BODY = json.loads(os.getenv("LLM_EXTRA_BODY", "") or "null")
except json.JSONDecodeError:
    EXTRA_BODY = None


class LLMError(RuntimeError):
    """Raised with a short, friendly message that the UI can show directly."""


# ----------------------------------------------------------------------------- thinking filter
# Reasoning models (Gemma 4, Qwen 3, DeepSeek-R1 …) may emit their chain of thought inline as
# <thought>…</thought> or <think>…</think> before the answer. We never show that to the student.
_OPEN_TAGS = ("<thought>", "<think>")
_CLOSE_TAGS = ("</thought>", "</think>")
_MAX_TAG = max(len(t) for t in _CLOSE_TAGS)


def _find_first(text: str, tags) -> tuple[int, str]:
    best, best_tag = -1, ""
    for tag in tags:
        i = text.find(tag)
        if i != -1 and (best == -1 or i < best):
            best, best_tag = i, tag
    return best, best_tag


def _partial_suffix(text: str, tags) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
    keep = 0
    for tag in tags:
        for k in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:k]):
                keep = max(keep, k)
                break
    return keep


class ThoughtFilter:
    """Streaming-safe remover of <thought>/<think> blocks (tags may be split across chunks)."""

    def __init__(self) -> None:
        self.in_thought = False
        self.saw_thought = False
        self.buf = ""

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = ""
        while True:
            if self.in_thought:
                i, tag = _find_first(self.buf, _CLOSE_TAGS)
                if i == -1:
                    self.buf = self.buf[-(_MAX_TAG - 1):]  # keep a tail in case the close tag is split
                    return out
                self.buf = self.buf[i + len(tag):].lstrip("\n")
                self.in_thought = False
            else:
                i, tag = _find_first(self.buf, _OPEN_TAGS)
                if i == -1:
                    keep = _partial_suffix(self.buf, _OPEN_TAGS)
                    cut = len(self.buf) - keep
                    out += self.buf[:cut]
                    self.buf = self.buf[cut:]
                    return out
                out += self.buf[:i]
                self.buf = self.buf[i + len(tag):]
                self.in_thought = self.saw_thought = True

    def flush(self) -> str:
        out = "" if self.in_thought else self.buf
        self.buf = ""
        return out


def strip_thoughts(text: str) -> str:
    f = ThoughtFilter()
    out = f.feed(text) + f.flush()
    if not out.strip() and f.saw_thought:
        raise LLMError("Gemma sochte-sochte token limit cross kar gaya. Dobara try karo (ya LLM_MAX_TOKENS badhao).")
    return out.lstrip("\n")


def _retryable(err: Exception) -> bool:
    low = str(err).lower()
    return any(k in low for k in ("429", "rate", "quota", "500", "502", "503", "504", "internal", "overloaded", "unavailable", "timed out", "timeout"))


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
    if EXTRA_BODY:
        kwargs["extra_body"] = EXTRA_BODY
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as err:  # noqa: BLE001 — we translate everything into LLMError
        low = str(err).lower()
        # 0) provider rejects our optional extra fields → retry without them
        if "extra_body" in kwargs and ("400" in low or "invalid" in low or "unknown" in low or "reasoning" in low or "thinking" in low):
            kwargs.pop("extra_body", None)
            return client.chat.completions.create(**kwargs)
        # 1) provider doesn't support response_format → retry without it (prompt already asks for JSON)
        if json_mode and ("response_format" in low or "json" in low):
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
def chat(messages: List[Message], json_mode: bool = False, temperature: float | None = None, max_tokens: int | None = None, strip: bool = True) -> str:
    """Return the full assistant reply (thinking removed). Retries on rate limits and transient 5xx."""
    temperature = TEMPERATURE if temperature is None else temperature
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    if PROVIDER == "mock":
        time.sleep(0.4)
        return _mock_text(messages, json_mode)
    if FOLD_SYSTEM and PROVIDER == "openai":
        messages = _fold_system(messages)

    delays = (0, 4, 10)  # seconds between attempts
    last_err: Exception | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            if PROVIDER == "google":
                resp = _google_chat(messages, json_mode, temperature, max_tokens, stream=False)
                text = resp.text or ""
            else:
                resp = _openai_chat(messages, json_mode, temperature, max_tokens, stream=False)
                text = resp.choices[0].message.content or ""
            return strip_thoughts(text) if strip else text
        except LLMError:
            raise
        except Exception as err:  # noqa: BLE001
            last_err = err
            if not _retryable(err):
                break
    raise _friendly(last_err or RuntimeError("unknown error"))


def _raw_stream(messages: List[Message], temperature: float, max_tokens: int) -> Iterator[str]:
    if PROVIDER == "google":
        for chunk in _google_chat(messages, False, temperature, max_tokens, stream=True):
            if getattr(chunk, "text", None):
                yield chunk.text
        return
    for chunk in _openai_chat(messages, False, temperature, max_tokens, stream=True):
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def stream(messages: List[Message], temperature: float | None = None, max_tokens: int | None = None) -> Iterator[str]:
    """Yield the visible reply as it arrives. While the model is still *thinking*, yields empty
    strings so the UI can show a 'soch raha hai…' placeholder instead of the chain of thought.
    Retries transient failures that happen before the first token."""
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

    delays = (0, 4, 10)
    last_err: Exception | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        filt, got_any = ThoughtFilter(), False
        try:
            for raw in _raw_stream(messages, temperature, max_tokens):
                got_any = True
                yield filt.feed(raw)  # "" while inside a thought block
            tail = filt.flush()
            if tail:
                yield tail
            if filt.saw_thought and not got_any:
                raise LLMError("Model se khaali jawab aaya. Dobara try karo.")
            return
        except LLMError:
            raise
        except Exception as err:  # noqa: BLE001
            last_err = err
            if got_any or not _retryable(err):
                break
    # Some endpoints don't stream at all; try once more without streaming.
    if last_err and "stream" in str(last_err).lower():
        yield chat(messages, False, temperature, max_tokens)
        return
    raise _friendly(last_err or RuntimeError("unknown error"))
