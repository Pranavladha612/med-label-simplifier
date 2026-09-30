"""A tiny wrapper around OpenRouter.

OpenRouter uses the same API format as OpenAI, so we use the `openai` library and just point it
at OpenRouter's URL. Responses are cached on disk so re-running the same experiment is free.
"""

import hashlib
import json
import time

from openai import APIStatusError, OpenAI, RateLimitError

from . import config

_client = None
models_used: set[str] = set()   # which models actually answered (free models fall back to others)


class DailyLimitReached(RuntimeError):
    pass


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


def ask(prompt: str, system: str | None = None, model: str | None = None, use_cache: bool = True) -> str:
    """Send a user prompt (plus an optional system prompt) to the model and return its text reply."""
    model = model or config.OPENROUTER_MODEL
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    cache_key = hashlib.sha256(f"{model}\n{system or ''}\n{prompt}".encode()).hexdigest()[:24]
    cache_file = config.CACHE_DIR / "llm" / f"{cache_key}.json"
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        models_used.add(cached.get("answered_by", model))
        return cached["reply"]

    last_problem = "all models busy"
    for attempt in range(config.MAX_ATTEMPTS):
        try:
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
            # OpenRouter sometimes answers "200 OK" but with an error inside and no reply
            # (e.g. the free model's provider failed mid-request). Treat that like a temporary error.
            reply = _reply_text(response)
            if reply:
                break
            last_problem = _error_message(response)
            if "per-day" in last_problem or "per day" in last_problem:
                raise DailyLimitReached(
                    "You've used today's free OpenRouter requests. Everything done so far is cached, "
                    "so re-run tomorrow and it continues where it stopped (or add credits at openrouter.ai)."
                )
            print(f"Empty reply from OpenRouter ({last_problem}), retrying...")
            time.sleep(5 * (attempt + 1))
        except RateLimitError as error:
            if "per-day" in str(error) or "per day" in str(error):
                raise DailyLimitReached(
                    "You've used today's free OpenRouter requests. Everything done so far is cached, "
                    "so re-run tomorrow and it continues where it stopped (or add credits at openrouter.ai)."
                ) from error
            wait = 15 * (attempt + 1)
            print(f"All models busy, waiting {wait}s and retrying...")
            time.sleep(wait)
        except APIStatusError as error:
            if error.status_code >= 500 and attempt < config.MAX_ATTEMPTS - 1:
                time.sleep(5)
                continue
            raise
    else:
        raise RuntimeError(f"OpenRouter failed {config.MAX_ATTEMPTS} times ({last_problem}). Try again in a few minutes.")

    models_used.add(response.model)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"model": model, "answered_by": response.model, "system": system, "prompt": prompt, "reply": reply}, indent=2),
        encoding="utf-8",
    )
    return reply
