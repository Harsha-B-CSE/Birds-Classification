"""PyTorch dataset and class-balanced sampler for combined bird data."""

from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler


class CombinedBirdDataset(Dataset):
    """Loads manifest samples relative to the project root."""

    def __init__(self, project_root: Path, samples: Sequence[Dict[str, object]], transform):
        self.project_root = project_root
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self.project_root / sample["path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), int(sample["label"])


def class_balanced_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    """Gives every class equal total sampling weight despite dataset imbalance."""
    counts: Dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
