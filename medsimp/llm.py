"""A small wrapper around the OpenRouter API.

OpenRouter uses the same API format as OpenAI, so we use the `openai` library and just point it
at OpenRouter's URL. Replies are cached on disk, so re-running the same experiment is free.
"""

import hashlib
import json
import time
from pathlib import Path

from openai import APIStatusError, OpenAI, RateLimitError

from . import config

_client = None
models_used: set[str] = set()   # which models actually answered (free models fall back to others)

DAILY_LIMIT_MESSAGE = (
    "You've used today's free OpenRouter requests. Everything done so far is cached, "
    "so re-run tomorrow and it continues where it stopped (or add credits at openrouter.ai)."
)


class DailyLimitReached(RuntimeError):
    pass


class EmptyReply(Exception):
    """OpenRouter answered "200 OK" but with no reply, e.g. the free model's provider failed mid-request."""


def ask(prompt: str, system: str | None = None, model: str | None = None, use_cache: bool = True) -> str:
    """Send a user prompt (plus an optional system prompt) to the model and return its text reply."""
    model = model or config.OPENROUTER_MODEL
    cache_file = _cache_file(model, system, prompt)
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        models_used.add(cached.get("answered_by", model))
        return cached["reply"]

    reply, answered_by = _ask_with_retries(prompt, system, model)
    models_used.add(answered_by)
    _write_cache(cache_file, model, answered_by, system, prompt, reply)
    return reply


# ---------------------------------------------------------------- talking to OpenRouter

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY.startswith("sk-or-..."):
            raise RuntimeError(
                "No OpenRouter API key found. Copy .env.example to .env and paste your key "
                "(get one at https://openrouter.ai/keys)."
            )
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY, timeout=config.TIMEOUT_SECONDS
        )
    return _client


def _request(prompt: str, system: str | None, model: str) -> tuple[str, str]:
    """One API call. Returns (reply, model that answered); raises EmptyReply if there is no answer."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=config.TEMPERATURE,
        **({"max_tokens": config.MAX_TOKENS} if config.MAX_TOKENS else {}),
        extra_body={
            # OpenRouter-specific: fall back to other models if this one is busy...
            "models": [model] + [m for m in config.OPENROUTER_FALLBACK_MODELS if m != model],
            # ...and control "thinking": off is ~100x faster for this task (see config.yaml).
            # If it's on, keep the thinking text out of the reply.
            "reasoning": {"exclude": True} if config.REASONING else {"enabled": False},
        },
    )
    reply = _reply_text(response)
    if not reply:
        raise EmptyReply(_error_message(response))
    return reply, response.model


def _ask_with_retries(prompt: str, system: str | None, model: str) -> tuple[str, str]:
    """Call _request, retrying temporary failures with growing waits. Stops at once on the daily limit."""
    last_problem = "all models busy"
    for attempt in range(config.MAX_ATTEMPTS):
        try:
            return _request(prompt, system, model)
        except EmptyReply as error:
            last_problem = str(error)
            if _is_daily_limit(last_problem):
                raise DailyLimitReached(DAILY_LIMIT_MESSAGE) from error
            print(f"Empty reply from OpenRouter ({last_problem}), retrying...")
            time.sleep(5 * (attempt + 1))
        except RateLimitError as error:
            if _is_daily_limit(str(error)):
                raise DailyLimitReached(DAILY_LIMIT_MESSAGE) from error
            wait = 15 * (attempt + 1)
            print(f"All models busy, waiting {wait}s and retrying...")
            time.sleep(wait)
        except APIStatusError as error:
            if error.status_code < 500 or attempt == config.MAX_ATTEMPTS - 1:
                raise
            last_problem = f"server error {error.status_code}"
            time.sleep(5)
    raise RuntimeError(f"OpenRouter failed {config.MAX_ATTEMPTS} times ({last_problem}). Try again in a few minutes.")


def _reply_text(response) -> str:
    """The model's answer, or "" if the response has no usable answer."""
    if not response.choices:
        return ""
    return (response.choices[0].message.content or "").strip()


def _error_message(response) -> str:
    """OpenRouter puts the reason for a failed request in an extra "error" field."""
    error = (getattr(response, "model_extra", None) or {}).get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error) if error else "no answer in the response"


def _is_daily_limit(message: str) -> bool:
    return "per-day" in message or "per day" in message


# ---------------------------------------------------------------- cache

def _cache_file(model: str, system: str | None, prompt: str) -> Path:
    """One file per (model, system prompt, user prompt), so any prompt change means a fresh call."""
    key = hashlib.sha256(f"{model}\n{system or ''}\n{prompt}".encode()).hexdigest()[:24]
    return config.CACHE_DIR / "llm" / f"{key}.json"


def _write_cache(path: Path, model: str, answered_by: str, system: str | None, prompt: str, reply: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"model": model, "answered_by": answered_by, "system": system, "prompt": prompt, "reply": reply}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
