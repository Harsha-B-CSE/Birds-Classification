import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm
from django.conf import settings

from myapp.ml.preprocessor import preprocess_image

class CUBModel(nn.Module):
    """Custom PyTorch model wrapper with Swin Transformer backbone."""
    def __init__(self, num_classes: int = 200, pretrained: bool = False):
        super(CUBModel, self).__init__()
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.model(x))

class SwinClassifier:
    """
    Check 2: Avian Species Classifier.
    Executes vision transformer inference using either ONNX Runtime (fast CPU)
    or native PyTorch checkpoint. Returns raw logits vector alongside top-k predictions.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_class_names()
        self.onnx_session = None
        self.pytorch_model = None

        self._initialize_runtime()

    def _load_class_names(self) -> List[str]:
        candidates = [
            settings.BASE_DIR / "myapp" / "classes_224.json",
            settings.BASE_DIR / "training_output" / "classes.json",
            settings.BASE_DIR / "myapp" / "cleaned_classes.json",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        names = json.load(f)
                        if len(names) >= 200:
                            return names
                except Exception as e:
                    print(f"[WARNING] Could not load class names from {p}: {e}")
        return [f"Species {i}" for i in range(224)]

    def _initialize_runtime(self):
        # 1. Check for ONNX model in myapp or models cache
        onnx_candidates = [
            settings.BASE_DIR / "myapp" / "swin_b_birds.onnx",
            settings.MODELS_CACHE_DIR / "swin_b_birds.onnx",
        ]
        for p in onnx_candidates:
            if p.exists():
                try:
                    import onnxruntime as ort
                    sess_options = ort.SessionOptions()
                    sess_options.intra_op_num_threads = 4
                    self.onnx_session = ort.InferenceSession(str(p), sess_options, providers=["CPUExecutionProvider"])
                    out_shape = self.onnx_session.get_outputs()[0].shape
                    print(f"[OK] Check 2: Swin ONNX model initialized from {p.name} (Output classes: {out_shape[1]})")
                    return
                except Exception as e:
                    print(f"[WARNING] ONNX initialization failed: {e}")

        # 2. Fall back to PyTorch .pth model
        pth_candidates = [
            settings.BASE_DIR / "myapp" / "swin_b_birds_224.pth",
            settings.BASE_DIR / "training_output" / "best_model.pth",
            settings.BASE_DIR / "myapp" / "swin_model_best.pth",
        ]
        for pth_path in pth_candidates:
            if pth_path.exists():
                try:
                    state_dict = torch.load(pth_path, map_location=self.device)
                    head_w = state_dict.get("model.head.fc.weight", state_dict.get("head.fc.weight"))
                    num_cls = head_w.shape[0] if head_w is not None else len(self.class_names)
                    model = CUBModel(num_classes=num_cls, pretrained=False).to(self.device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.pytorch_model = model
                    print(f"[OK] Check 2: PyTorch Swin model loaded from {pth_path.name} ({num_cls} classes)")
                    return
                except Exception as e:
                    print(f"[WARNING] PyTorch model load failed from {pth_path.name}: {e}")

    def predict(self, image: Image.Image, k: int = 5) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Runs forward pass on specimen image.
        Returns:
            logits (torch.Tensor): Raw 1D unnormalized logits vector.
            top5 (list): Top-k class predictions with probabilities and percentages.
        """
        tensor = preprocess_image(image)

        # Path A: ONNX Runtime
        if self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            ort_inputs = {input_name: tensor.numpy()}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            logits_np = ort_outs[0][0]
            logits = torch.from_numpy(logits_np)
        # Path B: PyTorch Native
        elif self.pytorch_model is not None:
            tensor = tensor.to(self.device)
            with torch.no_grad():
                logits = self.pytorch_model(tensor)[0].cpu()
        else:
            # Fallback uniform logits
            logits = torch.zeros(len(self.class_names))

        probs = torch.nn.functional.softmax(logits, dim=0)
        num_k = min(k, len(probs))
        top_probs, top_indices = torch.topk(probs, num_k)

        top_predictions = []
        for p, idx in zip(top_probs, top_indices):
            cls_idx = idx.item()
            name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"Species {cls_idx}"
            display_name = name.split(".", 1)[-1].replace("_", " ")
            top_predictions.append({
                "class": name,
                "display_name": display_name,
                "index": cls_idx,
                "probability": round(float(p.item()), 4),
                "percentage": round(float(p.item() * 100), 2),
            })

        return logits, top_predictions
