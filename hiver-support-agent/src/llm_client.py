"""
Thin, provider-agnostic LLM wrapper. Every provider here is free:

- ollama : fully local, zero signup, zero API key. Default.
- groq   : free tier (as of writing, generous req/day limits on Llama /
           Gemma models), needs a free API key from console.groq.com.
- gemini : free tier via Google AI Studio, needs a free API key.

We keep this to ONE method (`chat`) so the rest of the codebase never
touches provider-specific SDKs directly.
"""
from __future__ import annotations

import json
import logging
import time

import requests

from src import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _call_ollama(system: str, user: str, model: str, temperature: float) -> str:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise LLMError(
            f"Ollama call failed ({resp.status_code}): {resp.text}\n"
            "Is `ollama serve` running, and have you pulled the model with "
            f"`ollama pull {model}`?"
        )
    return resp.json()["message"]["content"]


def _call_groq(system: str, user: str, model: str, temperature: float) -> str:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY not set. Get a free key at console.groq.com")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise LLMError(f"Groq call failed ({resp.status_code}): {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(system: str, user: str, model: str, temperature: float) -> str:
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY not set. Get a free key at aistudio.google.com")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={config.GEMINI_API_KEY}"
    )
    resp = requests.post(
        url,
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature},
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise LLMError(f"Gemini call failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


_PROVIDERS = {
    "ollama": (_call_ollama, config.OLLAMA_MODEL),
    "groq": (_call_groq, config.GROQ_MODEL),
    "gemini": (_call_gemini, config.GEMINI_MODEL),
}


def chat(
    system: str,
    user: str,
    provider: str | None = None,
    temperature: float = 0.2,
    retries: int = 2,
    json_mode: bool = False,
) -> str:
    """Single entry point used by the whole codebase.

    If json_mode=True, we append an instruction to return raw JSON and
    strip markdown fences defensively before returning.
    """
    provider = provider or config.LLM_PROVIDER
    if provider not in _PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'. Choose from {list(_PROVIDERS)}")
    fn, model = _PROVIDERS[provider]

    if json_mode:
        system = system + "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble."

    last_err = None
    for attempt in range(retries + 1):
        try:
            out = fn(system, user, model, temperature)
            if json_mode:
                out = out.strip()
                if out.startswith("```"):
                    out = out.strip("`")
                    out = out.split("\n", 1)[-1] if "\n" in out else out
                    if out.endswith("json"):
                        out = out[:-4]
            return out
        except LLMError as e:
            last_err = e
            log.warning("LLM call attempt %d/%d failed: %s", attempt + 1, retries + 1, e)
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def chat_json(system: str, user: str, provider: str | None = None, temperature: float = 0.0) -> dict:
    raw = chat(system, user, provider=provider, temperature=temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model did not return valid JSON: {raw[:300]}") from e
