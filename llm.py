"""Thin wrapper over the Groq SDK. No framework, no magic."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from groq import Groq

import config

_client: Groq | None = None

# Some reasoning models emit <think>...</think> into content. Strip defensively
# even though our configured models are set to suppress it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.require_api_key())
    return _client


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 2000,
    json_mode: bool = False,
    retries: int = 2,
) -> str:
    """One completion. Returns assistant text with reasoning stripped."""
    model = model or config.FAST_MODEL
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        **config.extra_kwargs(model),
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client().chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            return _THINK_RE.sub("", content).strip()
        except Exception as exc:                      # noqa: BLE001
            last_err = exc
            msg = str(exc).lower()
            if any(s in msg for s in ("rate", "429", "503", "timeout")):
                time.sleep(2 * (attempt + 1))
                continue
            # Model rejected a param we sent — drop it and retry.
            for param in ("reasoning_effort", "include_reasoning", "response_format"):
                if param in msg and param in kwargs:
                    kwargs.pop(param)
                    break
            else:
                raise                                  # unrecognised error: surface it
    raise RuntimeError(f"Groq call failed after {retries + 1} attempts: {last_err}")


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a model response, fences and preamble be damned."""
    if not text:
        raise ValueError("empty model response")
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(cleaned)
    if not match:
        raise ValueError(f"no JSON found in response: {text[:300]}")
    return json.loads(match.group(0))


def json_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    retries: int = 2,
) -> Any:
    """Completion that must return JSON. Self-repairs on a parse failure."""
    convo = list(messages)
    for attempt in range(retries + 1):
        raw = chat(
            convo,
            model=model,
            temperature=temperature if attempt == 0 else 0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise ValueError(f"JSON parse failed after {retries + 1} tries: {exc}") from exc
            convo = convo + [
                {"role": "assistant", "content": raw[:800]},
                {"role": "user", "content":
                 "That was not valid JSON. Reply with the JSON object only — "
                 "no prose, no code fences, no trailing commas."},
            ]
    raise RuntimeError("unreachable")

def est_tokens(text: str) -> int:
    """Rough token count for TPM budgeting. ~4 chars per token is close enough."""
    return len(text) // 4