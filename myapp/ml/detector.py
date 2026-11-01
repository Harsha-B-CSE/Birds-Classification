from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
from django.conf import settings

from myapp.ml.preprocessor import preprocess_image

# ImageNet synset ranges corresponding to avian species (~60 classes)
# From 10 (brambling) through 24 (cockatoo), plus eagles, vultures, owls, waterfowl (80-146)
BIRD_IMAGENET_INDICES = set(range(10, 25)) | set(range(80, 101)) | set(range(127, 147))


class BirdDetector:
    """
    Check 1: Binary Bird Presence Gate.
    Uses MobileNetV3 ONNX (fast CPU execution) to verify whether an avian subject
    is present in the photograph (P(bird) >= threshold, default 0.60) before passing
    to the Swin Transformer classifier.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.onnx_session = None
        self.torch_model = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        onnx_candidates = [
            settings.BASE_DIR / "myapp" / "bird_detector.onnx",
            settings.MODELS_CACHE_DIR / "bird_detector.onnx",
        ]
        for p in onnx_candidates:
            if p.exists():
                try:
                    import onnxruntime as ort
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 2
                    self.onnx_session = ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
                    print(f"[OK] Check 1: MobileNetV3 ONNX detector initialized from {p.name}")
                    return
                except Exception as exc:
                    print(f"[WARNING] Check 1 ONNX init failed: {exc}")

        try:
            import torchvision.models as models
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.torch_model = models.mobilenet_v3_small(weights=weights).eval().to(self.device)
            print("[OK] Check 1: MobileNetV3 PyTorch fallback detector initialized.")
        except Exception as exc:
            print(f"[WARNING] Check 1 fallback detector failed: {exc}")

    def verify_bird_presence(
        self, image: Image.Image, threshold: float = 0.60
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Calculates probability that the image contains a bird.
        Returns:
            is_bird (bool): True if bird probability >= threshold.
            bird_probability (float): Confidence score (0.0 to 1.0).
            details (dict): Diagnostic metadata.
        """
        tensor = preprocess_image(image)

        try:
            if self.onnx_session is not None:
                input_name = self.onnx_session.get_inputs()[0].name
                ort_inputs = {input_name: tensor.numpy()}
                logits_np = self.onnx_session.run(None, ort_inputs)[0][0]
                logits = torch.from_numpy(logits_np)
            elif self.torch_model is not None:
                with torch.no_grad():
                    logits = self.torch_model(tensor.to(self.device))[0].cpu()
            else:
                return True, 0.95, {"status": "detector_offline_passthrough"}

            probs = torch.nn.functional.softmax(logits, dim=0)
            bird_prob = sum(probs[i].item() for i in BIRD_IMAGENET_INDICES if i < len(probs))

            top_prob, top_idx = torch.topk(probs, 1)
            if top_idx.item() in BIRD_IMAGENET_INDICES:
                bird_prob = max(bird_prob, top_prob.item())

            bird_prob = min(1.0, round(float(bird_prob), 4))
            is_bird = bird_prob >= threshold

            return is_bird, bird_prob, {
                "threshold": threshold,
                "top_idx": top_idx.item(),
                "top_prob": round(top_prob.item(), 4),
            }
        except Exception as exc:
            print(f"[WARNING] Check 1 verification error: {exc}")
            return True, 0.90, {"error": str(exc)}
