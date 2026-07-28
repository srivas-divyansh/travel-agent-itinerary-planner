"""Thin wrapper over the Groq SDK. No framework, no magic."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from groq import Groq

import config

_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.require_api_key())
    return _client


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.4,
    max_tokens: int = 2000,
    json_mode: bool = False,
    retries: int = 2,
) -> str:
    """One completion. Returns the assistant text."""
    kwargs: dict[str, Any] = {
        "model": config.MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - surface anything Groq throws
            last_err = exc
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg or "503" in msg:
                time.sleep(2 * (attempt + 1))
                continue
            if json_mode and "response_format" in msg:
                # Some models reject json_object; fall back to prompt-only JSON.
                kwargs.pop("response_format", None)
                continue
            raise
    raise RuntimeError(f"Groq call failed after retries: {last_err}")


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a model response, fences and preamble be damned."""
    if not text:
        raise ValueError("empty model response")
    cleaned = text.strip()
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
    temperature: float = 0.0,
    max_tokens: int = 2000,
    retries: int = 2,
) -> Any:
    """Completion that must return JSON. Retries once on a parse failure."""
    for attempt in range(retries + 1):
        raw = chat(
            messages,
            temperature=temperature if attempt == 0 else 0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            if attempt == retries:
                raise
            messages = messages + [
                {"role": "assistant", "content": raw[:500]},
                {"role": "user", "content": "That was not valid JSON. Reply with the JSON object only, no prose, no code fences."},
            ]
    raise RuntimeError("unreachable")
