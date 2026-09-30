"""Tests for the OpenRouter wrapper, using fake responses (no network, no API key needed)."""

from types import SimpleNamespace

import pytest

from medsimp import config, llm


def good_response(text):
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], model="fake/model", model_extra={})


def error_response(message):
    # What OpenRouter sends when a provider fails: "200 OK", but no choices and an "error" field.
    return SimpleNamespace(choices=None, model="fake/model", model_extra={"error": {"message": message}})


@pytest.fixture
def fake_openrouter(monkeypatch, tmp_path):
    """Replace the real client with one that returns the responses we queue up."""
    queue = []
    completions = SimpleNamespace(create=lambda **kwargs: queue.pop(0))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    return queue


def test_retries_when_reply_has_no_choices(fake_openrouter):
    fake_openrouter += [error_response("Provider returned error"), good_response("Take 1 tablet.")]
    assert llm.ask("simplify this", use_cache=False) == "Take 1 tablet."


def test_gives_up_with_the_real_error_message(fake_openrouter):
    fake_openrouter += [error_response("Provider returned error")] * 4
    with pytest.raises(RuntimeError, match="Provider returned error"):
        llm.ask("simplify this", use_cache=False)


def test_daily_limit_stops_immediately(fake_openrouter):
    fake_openrouter += [error_response("Rate limit exceeded: free-models-per-day")]
    with pytest.raises(llm.DailyLimitReached):
        llm.ask("simplify this", use_cache=False)
