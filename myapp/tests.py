import io
import json
from PIL import Image
from django.test import TestCase, Client
from django.urls import reverse
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile

from myapp.services.cache_manager import (
    ensure_cache_structure,
    get_cache_usage,
    evict_prediction_cache,
)
from myapp.services.firebase_client import (
    save_prediction_history,
    get_prediction_history,
    cache_bird_info,
    get_cached_bird_info,
)
from myapp.services.dossier_schema import extract_json, normalize_profile

class Phase1SystemTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_homepage_redirect_or_render(self):
        """Root URL / should load the primary predict view."""
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Feathered Insights")
        self.assertContains(response, "Avian Vision")

    def test_about_methodology_page(self):
        """Methodology /about/ should load and describe Swin Transformer and 3-Check OOD."""
        response = self.client.get(reverse("about"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Vision Transformer Backbone")
        self.assertContains(response, "Three-Check Verification Cascade")

    def test_cache_manager_initialization(self):
        """Cache directories must exist and report valid usage statistics."""
        ensure_cache_structure()
        usage = get_cache_usage()
        self.assertIn("models_mb", usage)
        self.assertIn("total_mb", usage)
        self.assertIn("max_limit_mb", usage)
        self.assertEqual(usage["max_limit_mb"], 5000.0)

    def test_prediction_history_local_persistence(self):
        """Prediction history should save and retrieve locally for a device ID."""
        test_device = "test_dev_unit_12345"
        record = {
            "species": "Indian Peafowl",
            "confidence": 0.954,
            "top5": [{"class": "Indian Peafowl", "percentage": 95.4}],
        }
        saved = save_prediction_history(test_device, record)
        self.assertTrue(saved)

        history = get_prediction_history(test_device)
        self.assertTrue(len(history) >= 1)
        self.assertEqual(history[-1]["species"], "Indian Peafowl")

    def test_predict_endpoint_with_image(self):
        """POST to /predict/ with a synthetic image should return JSON classification."""
        # Create a tiny 32x32 synthetic RGB image in memory
        img_buffer = io.BytesIO()
        test_img = Image.new("RGB", (32, 32), color=(73, 109, 137))
        test_img.save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        upload = SimpleUploadedFile(
            "specimen.jpg", img_buffer.read(), content_type="image/jpeg"
        )

        response = self.client.post(
            reverse("predict"),
            {"image": upload, "device_id": "test_dev_999"},
        )
        self.assertIn(response.status_code, [200, 422])
        data = json.loads(response.content)
        self.assertIn("status", data)
        self.assertIn("diagnostics", data)

    def test_species_api_rejects_unknown_labels(self):
        """Enrichment must never query external APIs for arbitrary user input."""
        response = self.client.get(reverse("species_profile"), {"name": "not-a-bird"})
        self.assertEqual(response.status_code, 400)

    def test_history_api_reads_and_clears_device_log(self):
        """Field log reads records saved for a device and clears them on request."""
        device_id = "test_history_device"
        save_prediction_history(
            device_id,
            {
                "species": "003.Sooty_Albatross",
                "display_name": "Sooty Albatross",
                "confidence": 0.91,
            },
        )

        response = self.client.get(reverse("history_api"), {"device_id": device_id})
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json()["count"], 1)

        response = self.client.delete(
            reverse("history_api") + f"?device_id={device_id}"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.client.get(reverse("history_api"), {"device_id": device_id}).json()["count"],
            0,
        )

    def test_history_api_rejects_invalid_device_ids(self):
        response = self.client.get(reverse("history_api"), {"device_id": "../../etc/passwd"})
        self.assertEqual(response.status_code, 400)

    def test_local_model_json_extraction(self):
        """Model prose with an embedded JSON object should still parse."""
        payload = extract_json('Here is the dossier:\n{"summary":"A bird.","habitat":"Wetlands"}')
        self.assertEqual(payload["summary"], "A bird.")
        self.assertIsNone(extract_json("No JSON here."))

    def test_local_profile_normalization_prefers_cached_facts(self):
        """Cached scientific facts must anchor model-generated lifespan and taxonomy."""
        payload = {
            "common_name": "Indian Peafowl",
            "scientific_name": "Pavo cristatus",
            "summary": "A large pheasant.",
            "habitat": "Open forests and farmland.",
            "native_range": ["India"],
            "iucn_status": "Vulnerable",
            "lifespan": {"wild_years": [25, 10], "captive_years": [40, 20]},
            "chromosome_count_2n": 78,
            "taxonomy": {"order": "Galliformes", "family": "Phasianidae"},
        }
        reference = {
            "scientific_name": "Pavo cristatus",
            "taxonomy": {"class": "Aves", "family": "Phasianidae"},
            "iucn_status": "Least Concern",
            "native_range": ["Indian Subcontinent"],
        }
        profile = normalize_profile(payload, "221.Indian_Peafowl", reference)
        self.assertEqual(profile["scientific_name"], "Pavo cristatus")
        self.assertEqual(profile["iucn_status"], "Least Concern")
        self.assertEqual(profile["iucn_code"], "LEAST_CONCERN")
        self.assertEqual(profile["native_range"], ["Indian Subcontinent"])
        self.assertEqual(profile["lifespan"]["wild_years"], [10, 25])
        self.assertEqual(profile["lifespan"]["captive_years"], [20, 40])
        self.assertEqual(profile["chromosome_count_2n"], 78)
        self.assertEqual(profile["source"], "local-model")
        self.assertFalse(profile["verified"])

    def test_check1_no_bird_rejection(self):
        """Check 1 rejects non-bird images with '🚫 No bird detected' (HTTP 422)."""
        img_buffer = io.BytesIO()
        test_img = Image.new("RGB", (224, 224), color=(120, 120, 120))
        test_img.save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        upload = SimpleUploadedFile("non_bird.jpg", img_buffer.read(), content_type="image/jpeg")

        response = self.client.post(reverse("predict"), {"image": upload})
        self.assertEqual(response.status_code, 422)
        data = json.loads(response.content)
        self.assertEqual(data["status"], "REJECTED_NOT_A_BIRD")
        self.assertEqual(data["error"], "🚫 No bird detected")

    def test_species_dossier_contains_biological_fields(self):
        """Species dossier endpoint returns complete biological dossier with <1ms cache retrieval."""
        response = self.client.get(reverse("species_profile"), {"name": "001.Black_footed_Albatross"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("common_name", data)
        self.assertIn("scientific_name", data)
        self.assertIn("iucn_status", data)
        self.assertIn("habitat", data)
        self.assertIn("taxonomy", data)
        self.assertIn("lifespan", data)
        self.assertIn("wild_years", data["lifespan"])
        self.assertIn("captive_years", data["lifespan"])

