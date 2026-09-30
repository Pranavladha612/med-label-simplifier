"""Tests that config.yaml and prompts/prompts.yaml load and that prompts are built correctly (no API calls)."""

import pytest

from medsimp import config, llm, simplify


def test_config_values_loaded():
    assert config.OPENROUTER_MODEL
    assert config.OPENROUTER_BASE_URL.startswith("https://")
    assert config.TARGET_GRADE == 5
    assert "warnings" in config.SECTIONS
    assert 0 < config.ENTAIL_THRESHOLD < 1
    assert config.PROMPTS_FILE.exists()


def test_prompt_file_has_all_prompts():
    prompts = simplify.load_prompts()
    assert {"system", "simplify", "fix"} <= set(prompts)


@pytest.fixture
def captured(monkeypatch):
    """Capture what would be sent to the LLM instead of calling it."""
    calls = []

    def fake_ask(prompt, system=None, **kwargs):
        calls.append({"prompt": prompt, "system": system})
        return "- Take 1 tablet."

    monkeypatch.setattr(llm, "ask", fake_ask)
    return calls


def test_simplify_prompt_is_filled_in(captured):
    simplify.simplify("take 1 tablet every 4 to 6 hours", "Directions")
    call = captured[0]
    assert "grade 5" in call["system"]                      # {grade} filled in
    assert "<label>take 1 tablet every 4 to 6 hours</label>" in call["prompt"]
    assert "Directions" in call["prompt"]
    assert "{" not in call["prompt"] and "{" not in call["system"]   # no unfilled placeholders


def test_fix_prompt_lists_problems(captured):
    simplify.fix("original text", "previous rewrite", ['Missing quantity "24 hours".', '"8 tablets" is not in the original.'])
    prompt = captured[0]["prompt"]
    assert '- Missing quantity "24 hours".' in prompt
    assert "<rewrite>previous rewrite</rewrite>" in prompt
    assert "{" not in prompt


def test_clean_reply_strips_wrappers():
    assert simplify.clean_reply("Here is the rewrite:\n- Take 1 tablet.") == "- Take 1 tablet."
    assert simplify.clean_reply("```\n- Take 1 tablet.\n```") == "- Take 1 tablet."
    assert simplify.clean_reply("<rewrite>- Take 1 tablet.</rewrite>") == "- Take 1 tablet."
