"""Tests for parallel section processing (no network: the label and per-section work are faked)."""

import time

import pytest

from medsimp import fetch, pipeline
from medsimp.llm import DailyLimitReached

LABEL = {
    "openfda": {"brand_name": ["Test"], "generic_name": ["test"]},
    "dosage_and_administration": ["take 1 tablet"],
    "warnings": ["do not use with alcohol"],
    "stop_use": ["stop use if rash"],
}


@pytest.fixture(autouse=True)
def fake_label(monkeypatch):
    monkeypatch.setattr(fetch, "fetch_label", lambda drug: LABEL)


def test_sections_keep_label_order_even_when_they_finish_out_of_order(monkeypatch):
    delays = {"dosage_and_administration": 0.3, "warnings": 0.0, "stop_use": 0.1}

    def fake_section(name, text, use_nli=True):
        time.sleep(delays[name])
        return name

    monkeypatch.setattr(pipeline, "simplify_section", fake_section)
    progress = []
    result = pipeline.simplify_drug("test", on_progress=progress.append)
    assert result.sections == ["dosage_and_administration", "warnings", "stop_use"]
    assert progress[-1].endswith("(3/3)")


def test_sections_run_at_the_same_time(monkeypatch):
    monkeypatch.setattr(pipeline, "simplify_section", lambda name, text, use_nli=True: time.sleep(0.5) or name)
    start = time.time()
    pipeline.simplify_drug("test")
    assert time.time() - start < 1.2   # one after another would take 1.5 s


def test_errors_still_stop_the_run(monkeypatch):
    def fake_section(name, text, use_nli=True):
        if name == "warnings":
            raise DailyLimitReached("limit")
        return name

    monkeypatch.setattr(pipeline, "simplify_section", fake_section)
    with pytest.raises(DailyLimitReached):
        pipeline.simplify_drug("test")
