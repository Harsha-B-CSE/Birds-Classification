from typing import Dict, Any
from django.conf import settings
from django.http import HttpRequest

def app_mode(request: HttpRequest) -> Dict[str, Any]:
    """Exposes current deployment and storage mode to all Django templates."""
    use_firebase = getattr(settings, "USE_FIREBASE", False)
    return {
        "USE_FIREBASE": use_firebase,
        "APP_MODE": "Firebase Cloud" if use_firebase else "Local Workstation",
        "APP_MODE_BADGE": "cloud" if use_firebase else "offline",
    }
