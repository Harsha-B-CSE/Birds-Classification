"""NCBI Taxonomy and reference assembly lookup."""

import re
import urllib.parse
from typing import Any, Dict, Optional

from myapp.services.http_client import get_json


def fetch_ncbi(scientific_name: str) -> Optional[Dict[str, Any]]:
    """Resolves the NCBI Taxonomy ID and best reference genome assembly."""
    try:
        search = get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            {"db": "taxonomy", "term": scientific_name, "retmode": "json"},
        )
        tax_ids = search.get("esearchresult", {}).get("idlist") or []
        if not tax_ids:
            return None
        tax_id = tax_ids[0]
        summary = get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            {"db": "taxonomy", "id": tax_id, "retmode": "json"},
        )
        record = summary.get("result", {}).get(tax_id, {})
    except Exception as exc:
        print(f"[WARNING] NCBI taxonomy lookup failed for {scientific_name}: {exc}")
        return None

    result: Dict[str, Any] = {
        "tax_id": tax_id,
        "common_name": record.get("commonname"),
        "division": record.get("division"),
        "url": f"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={tax_id}",
        "genbank_url": (
            "https://www.ncbi.nlm.nih.gov/nuccore/?term="
            + urllib.parse.quote(f"txid{tax_id}[Organism]")
        ),
        "reference_assembly": None,
    }

    try:
        assembly_search = get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            {
                "db": "assembly",
                "term": f"{scientific_name}[Organism]",
                "retmode": "json",
            },
        )
        assembly_ids = assembly_search.get("esearchresult", {}).get("idlist") or []
        if assembly_ids:
            assembly = get_json(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                {"db": "assembly", "id": assembly_ids[0], "retmode": "json"},
            ).get("result", {}).get(assembly_ids[0], {})
            accession = assembly.get("assemblyaccession")
            chromosome_match = re.search(
                r'chromosome_count[^>]*>(\d+)<', assembly.get("meta") or ""
            )
            chromosomes = int(chromosome_match.group(1)) if chromosome_match else 0
            if accession:
                result["reference_assembly"] = {
                    "accession": accession,
                    "name": assembly.get("assemblyname"),
                    "status": assembly.get("assemblystatus"),
                    "chromosomes": chromosomes or None,
                    "url": f"https://www.ncbi.nlm.nih.gov/assembly/{accession}",
                }
    except Exception as exc:
        print(f"[WARNING] NCBI assembly lookup failed for {scientific_name}: {exc}")

    return result
