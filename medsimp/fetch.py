"""Step 1: download drug labels from the free openFDA API and pull out the sections we care about."""

import json

import requests

from . import config

OPENFDA_URL = "https://api.fda.gov/drug/label.json"


def fetch_label(drug_name: str) -> dict:
    """Return the raw openFDA label for a drug (searched by generic or brand name).

    Results are cached in cache/labels/ so we only download each drug once.
    """
    cache_file = config.CACHE_DIR / "labels" / f"{drug_name.lower().replace(' ', '_')}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    name = drug_name.strip()
    query = f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"'
    response = requests.get(OPENFDA_URL, params={"search": query, "limit": 20}, timeout=30)
    if response.status_code == 404:
        raise LookupError(f"No openFDA label found for '{drug_name}'")
    response.raise_for_status()
    label = _best_match(response.json()["results"], name)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(label, indent=2), encoding="utf-8")
    return label


def _best_match(labels: list[dict], name: str) -> dict:
    """Prefer a label whose brand or generic name is exactly what was asked for.

    Searching "Advil" also finds "Advil Dual Action with Acetaminophen"; we want plain Advil.
    Among exact matches, prefer the one with the most sections we can simplify.
    """
    name = name.lower()

    def names(label):
        openfda = label.get("openfda", {})
        return [n.lower() for n in openfda.get("brand_name", []) + openfda.get("generic_name", [])]

    exact = [label for label in labels if name in names(label)]
    candidates = exact or labels
    return max(candidates, key=lambda label: len(get_sections(label)))


def get_sections(label: dict) -> dict[str, str]:
    """Pick out the sections listed in config.SECTIONS as plain strings.

    openFDA stores each section as a list of strings; we join them into one.
    """
    sections = {}
    for name in config.SECTIONS:
        if name in label:
            text = " ".join(label[name]).strip()
            if text:
                sections[name] = text
    return sections


def drug_display_name(label: dict) -> str:
    openfda = label.get("openfda", {})
    brand = (openfda.get("brand_name") or [""])[0]
    generic = (openfda.get("generic_name") or [""])[0]
    if brand and generic and brand.lower() != generic.lower():
        return f"{brand} ({generic.lower()})"
    return brand or generic or "Unknown drug"


def pretty_section_name(field: str) -> str:
    """'ask_doctor_or_pharmacist' -> 'Ask doctor or pharmacist'."""
    return field.replace("_", " ").capitalize()
