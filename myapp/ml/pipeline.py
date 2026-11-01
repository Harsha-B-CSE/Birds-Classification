import time
from typing import Dict, Any, Optional
from PIL import Image
from django.conf import settings

from myapp.ml.detector import BirdDetector
from myapp.ml.classifier import SwinClassifier
from myapp.ml.ood_gate import EnergyOODGate

class BirdIntelligencePipeline:
    """
    Unified 3-Check Avian Intelligence Pipeline:
    1. Check 1 (Detector): Rejects non-bird images ($P < 0.60$).
    2. Check 2 (Classifier): Extracts fine-grained species logits via Swin Transformer.
    3. Check 3 (OOD Gate): Evaluates Helmholtz free energy and confidence floor to abstain on unseen species.
    """
    def __init__(self):
        print("[INFO] Initializing 3-Check Avian Pipeline...")
        self.detector = BirdDetector()
        self.classifier = SwinClassifier()
        self.ood_gate = EnergyOODGate(
            energy_threshold=settings.OOD_ENERGY_THRESHOLD,
            confidence_floor=settings.OOD_CONFIDENCE_FLOOR,
        )
        print("[OK] 3-Check Pipeline initialized successfully.")

    def process(self, image: Image.Image) -> Dict[str, Any]:
        """
        Executes the full 3-check cascade on an input PIL image.
        """
        start_time = time.perf_counter()
        diagnostics = {}

        # -------------------------------------------------------------
        # CHECK 1: Bird Presence Verification
        # -------------------------------------------------------------
        is_bird, bird_prob, detector_meta = self.detector.verify_bird_presence(
            image, threshold=settings.BIRD_PRESENCE_THRESHOLD
        )
        diagnostics["check_1"] = {
            "passed": is_bird,
            "bird_probability": bird_prob,
            "meta": detector_meta,
        }

        if not is_bird:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return {
                "status": "REJECTED_NOT_A_BIRD",
                "checks_passed": False,
                "error": "🚫 No bird detected",
                "diagnostics": diagnostics,
                "latency_ms": elapsed_ms,
            }

        # -------------------------------------------------------------
        # CHECK 2: Species Classification (Swin Transformer)
        # -------------------------------------------------------------
        logits, top5 = self.classifier.predict(image, k=5)
        top1_prob = top5[0]["probability"] if top5 else 0.0
        top1_name = top5[0]["class"] if top5 else "Unknown"
        top1_display = top5[0]["display_name"] if top5 else "Unknown"

        diagnostics["check_2"] = {
            "passed": True,
            "top1_class": top1_name,
            "top1_prob": top1_prob,
        }

        # -------------------------------------------------------------
        # CHECK 3: Energy-Based OOD Gating
        # -------------------------------------------------------------
        ood_passed, ood_reason, ood_meta = self.ood_gate.evaluate(logits, top1_prob)
        diagnostics["check_3"] = {
            "passed": ood_passed,
            "reason": ood_reason,
            "meta": ood_meta,
        }

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        if not ood_passed:
            return {
                "status": f"REJECTED_{ood_reason}",
                "checks_passed": False,
                "error": ood_meta.get("message", "⚠️ Unclear or uncataloged bird image"),
                "top1_candidate": top1_name,
                "top5": top5,
                "diagnostics": diagnostics,
                "latency_ms": elapsed_ms,
            }

        # -------------------------------------------------------------
        # ALL CHECKS PASSED: Return Verified Identification
        # -------------------------------------------------------------
        return {
            "status": "SUCCESS",
            "checks_passed": True,
            "class": top1_name,
            "display_name": top1_display,
            "probability": f"{top1_prob:.4f}",
            "percentage": round(top1_prob * 100, 2),
            "top5": top5,
            "diagnostics": diagnostics,
            "latency_ms": elapsed_ms,
        }

# Singleton instance
_pipeline = None

def get_pipeline() -> BirdIntelligencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BirdIntelligencePipeline()
    return _pipeline
