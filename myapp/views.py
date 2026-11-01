import json
import re
from functools import lru_cache
from PIL import Image, ImageOps

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render, redirect

from myapp.services.firebase_client import (
    clear_prediction_history,
    get_prediction_history,
    save_prediction_history,
)
from myapp.services.species_info import request_species_profile
from myapp.ml.pipeline import get_pipeline

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,80}$")


@lru_cache(maxsize=1)
def _known_class_names() -> frozenset[str]:
    """Loads the validated 224-species catalog once per process."""
    candidates = [
        settings.BASE_DIR / "myapp" / "classes_224.json",
        settings.BASE_DIR / "training_output" / "classes.json",
        settings.BASE_DIR / "myapp" / "cleaned_classes.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as handle:
                    return frozenset(json.load(handle))
            except Exception:
                pass
    return frozenset()


def predict_view(request: HttpRequest) -> HttpResponse:
    """
    Primary Prediction Endpoint.
    Executes the 3-check validation cascade (Bird Detector -> Swin Transformer -> Energy OOD).
    Persists history to Firestore (or local NVMe cache when offline).
    """
    if request.method == "POST" and request.FILES.get("image"):
        try:
            image_file = request.FILES["image"]
            if image_file.size > MAX_UPLOAD_BYTES:
                return JsonResponse(
                    {"error": "Image exceeds the 15 MB upload limit."}, status=413
                )
            if image_file.content_type not in ALLOWED_IMAGE_TYPES:
                return JsonResponse(
                    {"error": "Unsupported format. Upload JPEG, PNG, or WEBP."},
                    status=415,
                )

            image = ImageOps.exif_transpose(Image.open(image_file)).convert("RGB")

            # Execute 3-Check Intelligence Pipeline
            pipeline = get_pipeline()
            result = pipeline.process(image)

            # Rejection Handling (Non-bird or Out-of-Distribution novel species)
            if not result.get("checks_passed", False):
                return JsonResponse({
                    "status": result.get("status", "REJECTED"),
                    "error": result.get("error", "Specimen could not be validated."),
                    "diagnostics": result.get("diagnostics", {}),
                    "latency_ms": result.get("latency_ms", 0),
                }, status=422)

            # Extract successful prediction metadata
            predicted_class = result["class"]
            predicted_display = result.get("display_name", predicted_class)
            predicted_prob = result["probability"]
            top5 = result["top5"]
            latency_ms = result.get("latency_ms", 0)

            # Persist to history (Firestore in Cloud mode, Local JSON in Offline mode)
            device_id = request.POST.get("device_id") or request.headers.get("X-Device-ID")
            if device_id:
                save_prediction_history(device_id, {
                    "species": predicted_class,
                    "display_name": predicted_display,
                    "confidence": float(predicted_prob),
                    "top5": top5,
                    "latency_ms": latency_ms,
                })

            return JsonResponse({
                "status": "SUCCESS",
                "class": predicted_class,
                "display_name": predicted_display,
                "probability": predicted_prob,
                "top5": top5,
                "latency_ms": latency_ms,
                "diagnostics": result.get("diagnostics", {}),
            })

        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            return JsonResponse({"error": "Prediction failed. Please try again."}, status=500)

    return render(request, "myapp/predict.html")


def species_profile_view(request: HttpRequest) -> JsonResponse:
    """Returns a cached scientific dossier for a validated model class label."""
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    label = (request.GET.get("name") or "").strip()
    if label not in _known_class_names():
        return JsonResponse(
            {"error": "Unknown species label. Use a label returned by /predict/."},
            status=400,
        )

    try:
        profile, generating = request_species_profile(label)
    except Exception as exc:
        print(f"[ERROR] Species enrichment failed for {label}: {exc}")
        return JsonResponse(
            {"error": "Scientific profile is temporarily unavailable."}, status=502
        )

    if generating:
        return JsonResponse(
            {"status": "GENERATING", "label": label},
            status=202,
        )
    if profile is None:
        return JsonResponse(
            {"error": "Scientific profile is unavailable for this species."}, status=502
        )
    return JsonResponse(profile)


def history_view(request: HttpRequest) -> HttpResponse:
    """Renders the device-keyed field log."""
    return render(request, "myapp/history.html")


def history_api_view(request: HttpRequest) -> JsonResponse:
    """Reads or clears the prediction history for one anonymous device ID."""
    device_id = (
        request.GET.get("device_id") or request.headers.get("X-Device-ID") or ""
    ).strip()
    if not DEVICE_ID_PATTERN.fullmatch(device_id):
        return JsonResponse({"error": "A valid device_id is required."}, status=400)

    if request.method == "DELETE":
        cleared = clear_prediction_history(device_id)
        return JsonResponse({"status": "CLEARED", "cleared": cleared})

    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    records = get_prediction_history(device_id)
    return JsonResponse(
        {"device_id": device_id, "count": len(records), "history": records}
    )

# ==============================================================================
# Informational & Legacy Authentication Views
# ==============================================================================

def index(request: HttpRequest) -> HttpResponse:
    return render(request, "myapp/index.html")

def about(request: HttpRequest) -> HttpResponse:
    return render(request, "myapp/about.html")

def aboutus(request: HttpRequest) -> HttpResponse:
    return render(request, "myapp/aboutus.html")

def login_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("predict")
            else:
                messages.error(request, "Invalid credentials")
        else:
            messages.error(request, "Both fields are required")

    return render(request, "myapp/login.html")

def signup_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        username = request.POST.get("username", "")
        email = request.POST.get("email", "")
        password1 = request.POST.get("password1", "")
        password2 = request.POST.get("password2", "")

        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            return redirect("signup")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken!")
            return redirect("signup")

        user = User.objects.create_user(username=username, email=email, password=password1)
        user.save()
        messages.success(request, "Account created successfully! Please login.")
        return redirect("login")

    return render(request, "myapp/signup.html")

def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect("login")
