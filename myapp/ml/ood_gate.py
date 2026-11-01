import math
from typing import Tuple, Dict, Any, List
import numpy as np
import torch

class EnergyOODGate:
    """
    Check 3: Helmholtz Free Energy-Based Out-of-Distribution (OOD) Gate.
    Calculates the free energy score over the raw logits produced by the
    Swin Transformer classifier.
    
    Energy Formula:
        E(x; T) = -T * log(sum(exp(z_i / T)))
    
    If E(x; T) exceeds tau_energy, the sample is identified as OOD (an unseen
    or uncataloged species).
    """
    def __init__(
        self,
        temperature: float = 1.0,
        energy_threshold: float = -3.90,
        confidence_floor: float = 0.40,
    ):
        self.temperature = temperature
        self.energy_threshold = energy_threshold  # Tuned on held-out OOD images
        self.confidence_floor = confidence_floor

    def compute_energy_score(self, logits: torch.Tensor) -> float:
        """
        Computes the scalar Helmholtz free energy:
        E(x) = -T * LogSumExp(z / T)
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        # LogSumExp is numerically stable
        scaled_logits = logits / self.temperature
        energy = -self.temperature * torch.logsumexp(scaled_logits, dim=-1).item()
        return round(float(energy), 4)

    def evaluate(
        self,
        logits: torch.Tensor,
        top1_probability: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluates logits against energy and confidence thresholds.
        Returns:
            is_valid (bool): True if prediction passes in-distribution validation.
            reason (str): Description if rejected, or 'PASS'.
            metrics (dict): Energy score and threshold metadata.
        """
        energy_score = self.compute_energy_score(logits)

        # Rejection Criteria 1: Confidence below absolute floor
        if top1_probability < self.confidence_floor:
            return False, "INSUFFICIENT_EVIDENCE", {
                "energy_score": energy_score,
                "confidence": round(top1_probability, 4),
                "threshold": self.confidence_floor,
                "message": "⚠️ Unclear or uncataloged bird image",
            }

        # Rejection Criteria 2: High Helmholtz Free Energy (novel/unseen species)
        if energy_score > self.energy_threshold:
            return False, "OUT_OF_DISTRIBUTION", {
                "energy_score": energy_score,
                "energy_threshold": self.energy_threshold,
                "confidence": round(top1_probability, 4),
                "message": "⚠️ Unclear or uncataloged bird image",
            }

        return True, "PASS", {
            "energy_score": energy_score,
            "energy_threshold": self.energy_threshold,
            "confidence": round(top1_probability, 4),
            "status": "IN_DISTRIBUTION",
        }
