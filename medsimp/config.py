"""Loads the settings from config.yaml (and the API key from .env) and exposes them as constants.

Edit config.yaml to change behaviour; this file only reads it.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# --- LLM (OpenRouter) ---
_llm = _cfg["llm"]
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")   # secret: only ever from the environment / .env
OPENROUTER_BASE_URL = _llm["base_url"]
# .env may override the model choice without editing config.yaml
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL") or _llm["model"]
_fallbacks_env = os.getenv("OPENROUTER_FALLBACK_MODELS")
OPENROUTER_FALLBACK_MODELS = (
    [m.strip() for m in _fallbacks_env.split(",") if m.strip()] if _fallbacks_env else list(_llm["fallback_models"])
)
REASONING = bool(_llm.get("reasoning", False))
TEMPERATURE = _llm["temperature"]
MAX_TOKENS = _llm["max_tokens"]
TIMEOUT_SECONDS = _llm["timeout_seconds"]
MAX_ATTEMPTS = _llm["max_attempts"]

# --- Prompts ---
PROMPTS_FILE = PROJECT_ROOT / _cfg["prompts"]["file"]

# --- Simplification ---
_simp = _cfg["simplification"]
TARGET_GRADE = _simp["target_grade"]
MAX_RETRIES = _simp["max_fix_retries"]
MAX_CHUNK_WORDS = _simp["max_chunk_words"]
PARALLEL_SECTIONS = _simp.get("parallel_sections", 1)
SECTIONS = list(_simp["sections"])

# --- Verification (NLI model, runs locally on CPU) ---
_ver = _cfg["verification"]
NLI_MODEL = _ver["nli_model"]
NLI_CANDIDATES = _ver["nli_candidates"]
ENTAIL_THRESHOLD = _ver["entail_threshold"]
CONTRADICT_THRESHOLD = _ver["contradict_threshold"]

# --- Paths ---
CACHE_DIR = PROJECT_ROOT / _cfg["paths"]["cache_dir"]
RESULTS_DIR = PROJECT_ROOT / _cfg["paths"]["results_dir"]
