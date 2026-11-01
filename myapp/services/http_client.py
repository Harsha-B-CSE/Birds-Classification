"""Shared HTTPS JSON client with a certifi-backed trust store."""

import json
import ssl
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

TIMEOUT_SECONDS = 8
USER_AGENT = "FeatheredInsights/1.0 (avian research application)"


def get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Performs a short, user-agent-tagged JSON GET request."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=TIMEOUT_SECONDS, context=SSL_CONTEXT
    ) as response:
        return json.loads(response.read().decode("utf-8"))
