"""All the settings in one place, so you can tweak them without hunting through the code."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- LLM (OpenRouter) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
# Free models are often busy. If the main model is unavailable, OpenRouter automatically tries these in order.
OPENROUTER_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "qwen/qwen3.8-27b:free,nvidia/nemotron-3-super-120b-a12b:free",
    ).split(",") if m.strip()
]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Simplification ---
TARGET_GRADE = 5          # aim for a 5th-grade reading level
MAX_RETRIES = 2           # how many times to regenerate if verification fails
MAX_CHUNK_WORDS = 250     # long sections are split into pieces this size

# --- Verification (NLI model, runs locally on CPU) ---
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_CANDIDATES = 3        # compare each sentence with the 3 most similar sentences on the other side
ENTAIL_THRESHOLD = 0.5   # "this idea is supported" if entailment prob >= this
CONTRADICT_THRESHOLD = 0.7  # "this contradicts the original" if contradiction prob >= this

# --- Which label sections we simplify (openFDA field names) ---
# OTC (over-the-counter) labels use the first group; prescription labels use the second.
SECTIONS = [
    "dosage_and_administration",
    "warnings",
    "do_not_use",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
    "when_using",
    "stop_use",
    "pregnancy_or_breast_feeding",
    "boxed_warning",
    "contraindications",
    "drug_interactions",
]

# --- Caching (saves API calls while you experiment) ---
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
