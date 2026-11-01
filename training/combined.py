"""Builds a single 224-class manifest from CUB-200 and Indian Birds 25."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from training.indian_birds import (
    INDIAN_CLASS_ALIASES,
    display_name,
    discover_indian_birds,
    indian_samples,
)

CUB_ROOT_NAME = "CUB_200_2011"
INDIAN_ROOT_NAME = "Indian_Birds_25"


def _read_mapping(path: Path, key_type=str) -> Dict:
    mapping = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.strip().split(" ", 1)
            mapping[key_type(key)] = value
    return mapping


def _read_split(path: Path) -> Dict[int, bool]:
    split = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            image_id, is_train = line.strip().split()
            split[int(image_id)] = is_train == "1"
    return split


def build_combined_manifest(project_root: Path, save: bool = True) -> Dict[str, object]:
    """Returns class names and train/validation samples for the 224-class model."""
    cub_root = project_root / CUB_ROOT_NAME
    indian_root = project_root / INDIAN_ROOT_NAME

    with open(project_root / "myapp" / "cleaned_classes.json", "r", encoding="utf-8") as handle:
        cub_classes: List[str] = json.load(handle)
    if len(cub_classes) != 200:
        raise ValueError("cleaned_classes.json must contain 200 CUB classes.")

    image_paths = _read_mapping(cub_root / "images.txt", int)
    image_labels = _read_mapping(cub_root / "image_class_labels.txt", int)
    train_split = _read_split(cub_root / "train_test_split.txt")

    cub_train: List[Dict[str, object]] = []
    cub_valid: List[Dict[str, object]] = []
    for image_id, relative_path in image_paths.items():
        sample = {
            "path": f"{CUB_ROOT_NAME}/images/{relative_path}",
            "label": int(image_labels[image_id]) - 1,
        }
        (cub_train if train_split[image_id] else cub_valid).append(sample)

    indian_info = discover_indian_birds(indian_root)
    aliases = dict(INDIAN_CLASS_ALIASES)
    class_names = list(cub_classes)
    class_to_index: Dict[str, int] = {}
    for folder in indian_info["classes"]:
        if folder in aliases:
            class_to_index[folder] = class_names.index(aliases[folder])
        else:
            class_to_index[folder] = len(class_names)
            class_names.append(f"{len(class_names) + 1:03d}.{display_name(folder).replace(' ', '_')}")

    indian_train, indian_valid = indian_samples(indian_root, class_to_index)
    train_samples = cub_train + indian_train
    valid_samples = cub_valid + indian_valid

    train_counts = Counter(sample["label"] for sample in train_samples)
    missing = [class_names[i] for i in range(len(class_names)) if train_counts[i] == 0]
    if missing:
        raise ValueError(f"Classes without training images: {missing[:5]}")

    manifest = {
        "version": 1,
        "num_classes": len(class_names),
        "classes": class_names,
        "indian_classes": indian_info["classes"],
        "aliases": aliases,
        "train": train_samples,
        "valid": valid_samples,
        "summary": {
            "cub_train": len(cub_train),
            "cub_valid": len(cub_valid),
            "indian_train": len(indian_train),
            "indian_valid": len(indian_valid),
            "total_train": len(train_samples),
            "total_valid": len(valid_samples),
            "indian_overlap": len(aliases),
        },
    }

    if save:
        output_dir = project_root / "training_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "combined_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        with open(output_dir / "classes.json", "w", encoding="utf-8") as handle:
            json.dump(class_names, handle, indent=2)

    return manifest
