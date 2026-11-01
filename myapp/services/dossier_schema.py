"""Schema normalization for locally generated bird dossiers."""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

IUCN_CODES = {
    "Least Concern": "LEAST_CONCERN",
    "Near Threatened": "NEAR_THREATENED",
    "Vulnerable": "VULNERABLE",
    "Endangered": "ENDANGERED",
    "Critically Endangered": "CRITICALLY_ENDANGERED",
    "Extinct in the Wild": "EXTINCT_IN_THE_WILD",
    "Extinct": "EXTINCT",
    "Data Deficient": "DATA_DEFICIENT",
    "Not Evaluated": "NOT_EVALUATED",
}

SCHEMA = (
    '{"common_name":str,"scientific_name":str|null,"summary":"2-3 sentences",'
    '"habitat":str,"native_range":[str],"iucn_status":"Least Concern|Near Threatened|'
    'Vulnerable|Endangered|Critically Endangered|Data Deficient|null",'
    '"lifespan":{"wild_years":[int,int]|null,"captive_years":[int,int]|null},'
    '"chromosome_count_2n":int|null,'
    '"taxonomy":{"order":str|null,"family":str|null,"genus":str|null,"species":str|null}}'
)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extracts the first JSON object from model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _pair(value: Any) -> Optional[list]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low, high = sorted(int(float(item)) for item in value)
    except (TypeError, ValueError):
        return None
    return [low, high] if 0 < low <= high < 100 else None


def _range(value: Any) -> list:
    if isinstance(value, str):
        value = value.strip("[]").split(",")
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:8]


def normalize_profile(
    payload: Dict[str, Any],
    class_label: str,
    reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalizes model output and anchors authoritative facts to the cache."""
    reference = reference or {}
    taxonomy = dict(reference.get("taxonomy") or {})
    taxonomy.update(payload.get("taxonomy") or {})
    taxonomy.setdefault("class", "Aves")
    iucn = reference.get("iucn_status") or payload.get("iucn_status")
    iucn = iucn if iucn in IUCN_CODES else None
    lifespan = payload.get("lifespan") if isinstance(payload.get("lifespan"), dict) else {}
    chromosomes = payload.get("chromosome_count_2n")
    chromosomes = (
        int(chromosomes)
        if isinstance(chromosomes, (int, float)) and 2 <= chromosomes <= 200
        else None
    )
    common_name = class_label.split(".", 1)[-1].replace("_", " ")

    return {
        "schema_version": 5,
        "label": class_label,
        "slug": re.sub(r"[^a-z0-9]+", "_", common_name.lower()).strip("_"),
        "common_name": payload.get("common_name")
        or reference.get("common_name")
        or common_name,
        "scientific_name": reference.get("scientific_name")
        or payload.get("scientific_name"),
        "summary": payload.get("summary") or "",
        "habitat": payload.get("habitat") or "",
        "taxonomy": taxonomy,
        "iucn_status": iucn,
        "iucn_code": IUCN_CODES.get(iucn or ""),
        "native_range": _range(
            reference.get("native_range") or payload.get("native_range")
        ),
        "lifespan": {
            "wild_years": _pair(lifespan.get("wild_years")),
            "captive_years": _pair(lifespan.get("captive_years")),
        },
        "chromosome_count_2n": chromosomes,
        "image_url": reference.get("image_url"),
        "image_attribution": reference.get("image_attribution"),
        "image_license": reference.get("image_license"),
        "wikipedia_url": reference.get("wikipedia_url"),
        "inaturalist_url": reference.get("inaturalist_url"),
        "gbif_url": reference.get("gbif_url"),
        "ncbi": reference.get("ncbi"),
        "source": "local-model",
        "verified": False,
        "model_id": "local-model",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": ["local-model"],
        "cache_hit": False,
    }


def prose_fallback(
    text: str, class_label: str, reference: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Best-effort profile when the small model ignores the JSON instruction."""
    reference = reference or {}
    summary = text.split("Habitat:", 1)[0].strip()
    habitat_match = re.search(r"Habitat:\s*(.+)", text, re.IGNORECASE)
    range_match = re.search(r"Native Range:\s*(.+)", text, re.IGNORECASE)
    iucn_match = re.search(r"IUCN Status:\s*([A-Za-z ]+)", text, re.IGNORECASE)
    wild_match = re.search(
        r"Wild Lifespan:.*?(\d+)\s*(?:-|to)?\s*(\d+)?", text, re.IGNORECASE
    )
    captive_match = re.search(
        r"Captive Lifespan:.*?(\d+)\s*(?:-|to)?\s*(\d+)?", text, re.IGNORECASE
    )

    def pair(match):
        if not match:
            return None
        low = int(match.group(1))
        high = int(match.group(2)) if match.group(2) else low
        return [min(low, high), max(low, high)]

    native_range = []
    if range_match:
        native_range = [
            item.strip(" .")
            for item in re.split(r",| and ", range_match.group(1))
            if item.strip(" .")
        ]
    payload = {
        "common_name": reference.get("common_name")
        or class_label.split(".", 1)[-1].replace("_", " "),
        "scientific_name": reference.get("scientific_name"),
        "summary": summary[:800],
        "habitat": habitat_match.group(1).strip() if habitat_match else "",
        "native_range": reference.get("native_range") or native_range,
        "iucn_status": reference.get("iucn_status")
        or (iucn_match.group(1).strip() if iucn_match else None),
        "lifespan": {"wild_years": pair(wild_match), "captive_years": pair(captive_match)},
        "chromosome_count_2n": None,
        "taxonomy": reference.get("taxonomy") or {},
    }
    return normalize_profile(payload, class_label, reference)
