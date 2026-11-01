"""Read-only clients for public biodiversity, taxonomy, and encyclopedia APIs."""

import re
import urllib.parse
from typing import Any, Dict, List, Optional

from myapp.services.http_client import get_json

IUCN_LABELS = {
    "LEAST_CONCERN": "Least Concern",
    "NEAR_THREATENED": "Near Threatened",
    "VULNERABLE": "Vulnerable",
    "ENDANGERED": "Endangered",
    "CRITICALLY_ENDANGERED": "Critically Endangered",
    "EXTINCT_IN_THE_WILD": "Extinct in the Wild",
    "EXTINCT": "Extinct",
    "DATA_DEFICIENT": "Data Deficient",
    "NOT_EVALUATED": "Not Evaluated",
}
def fetch_inaturalist(common_name: str) -> Optional[Dict[str, Any]]:
    """Resolves a common name to an iNaturalist species taxon."""
    try:
        payload = get_json(
            "https://api.inaturalist.org/v1/taxa/autocomplete",
            {"q": common_name, "rank": "species", "per_page": 5},
        )
    except Exception as exc:
        print(f"[WARNING] iNaturalist lookup failed for {common_name}: {exc}")
        return None

    bird_results = [
        item for item in payload.get("results") or []
        if item.get("iconic_taxon_name") == "Aves"
    ]
    if not bird_results:
        return None

    wanted = re.sub(r"[^a-z]", "", common_name.lower())
    taxon = next(
        (
            item
            for item in bird_results
            if re.sub(r"[^a-z]", "", (item.get("preferred_common_name") or "").lower()) == wanted
        ),
        bird_results[0],
    )
    photo = taxon.get("default_photo") or {}
    return {
        "taxon_id": taxon.get("id"),
        "scientific_name": taxon.get("name"),
        "common_name": taxon.get("preferred_common_name") or common_name,
        "observations_count": taxon.get("observations_count"),
        "photo_url": photo.get("medium_url") or photo.get("square_url"),
        "photo_attribution": photo.get("attribution"),
        "photo_license": photo.get("license_code"),
        "url": f"https://www.inaturalist.org/taxa/{taxon.get('id')}" if taxon.get("id") else None,
    }


def fetch_wikipedia(
    common_name: str, scientific_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Fetches the Wikipedia REST summary, preferring the common-name article."""
    candidates = []
    for title in (common_name, scientific_name):
        if title:
            candidates.extend([title, title[:1].upper() + title[1:].lower()])

    for title in candidates:
        try:
            slug = urllib.parse.quote(title.replace(" ", "_"))
            payload = get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
        except Exception:
            continue

        extract = payload.get("extract")
        if not extract:
            continue
        thumbnail = payload.get("thumbnail") or {}
        return {
            "title": payload.get("title") or title,
            "extract": extract,
            "description": payload.get("description"),
            "url": (payload.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
            "image_url": thumbnail.get("source"),
        }

    # Fall back to MediaWiki search for hyphenation and common-name variants.
    for query in (common_name, scientific_name):
        if not query:
            continue
        try:
            search = get_json(
                "https://en.wikipedia.org/w/api.php",
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 1,
                    "format": "json",
                },
            )
            hits = search.get("query", {}).get("search") or []
            if not hits:
                continue
            slug = urllib.parse.quote((hits[0].get("title") or "").replace(" ", "_"))
            payload = get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
            if payload.get("extract"):
                thumbnail = payload.get("thumbnail") or {}
                return {
                    "title": payload.get("title") or hits[0].get("title"),
                    "extract": payload["extract"],
                    "description": payload.get("description"),
                    "url": (payload.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
                    "image_url": thumbnail.get("source"),
                }
        except Exception:
            continue
    return None


def fetch_gbif(scientific_name: str) -> Optional[Dict[str, Any]]:
    """Fetches GBIF backbone taxonomy, native regions, and IUCN threat status."""
    try:
        match = get_json(
            "https://api.gbif.org/v1/species/match", {"name": scientific_name}
        )
        if match.get("matchType") in (None, "NONE") or not match.get("usageKey"):
            return None
        usage_key = match["usageKey"]
        payload = get_json(
            f"https://api.gbif.org/v1/species/{usage_key}/distributions",
            {"limit": 100},
        )
    except Exception as exc:
        print(f"[WARNING] GBIF lookup failed for {scientific_name}: {exc}")
        return None

    regions: List[str] = []
    threat_code = None
    for record in payload.get("results") or []:
        if record.get("establishmentMeans") == "INTRODUCED":
            continue
        locality = record.get("locality") or record.get("country")
        if (
            record.get("status") == "PRESENT"
            and locality
            and locality != "Global"
            and locality not in regions
        ):
            regions.append(locality)
        if record.get("threatStatus"):
            threat_code = record["threatStatus"]

    taxonomy = {
        key: match.get(key)
        for key in ("kingdom", "phylum", "class", "order", "family", "genus", "species")
        if match.get(key)
    }
    return {
        "usage_key": usage_key,
        "scientific_name": match.get("scientificName") or scientific_name,
        "accepted_name": match.get("canonicalName") or scientific_name,
        "taxonomic_status": match.get("status"),
        "taxonomy": taxonomy,
        "native_range": regions[:8],
        "iucn_code": threat_code,
        "iucn_status": IUCN_LABELS.get(
            threat_code or "", (threat_code or "").replace("_", " ").title() or None
        ),
        "url": f"https://www.gbif.org/species/{usage_key}",
    }
