"""Discovery and indexing for the Indian Birds 25 image dataset."""

from pathlib import Path
from typing import Dict, List, Tuple

# Indian folder names that are the same species as an existing CUB-200 class.
INDIAN_CLASS_ALIASES = {
    "White-Breasted-Kingfisher": "083.White_breasted_Kingfisher",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def display_name(folder_name: str) -> str:
    """Converts ``Asian-Green-Bee-Eater`` into ``Asian Green Bee Eater``."""
    return folder_name.replace("-", " ").replace("_", " ").strip()


def discover_indian_birds(root: Path) -> Dict[str, object]:
    """Validates the train/valid folder layout and returns class metadata."""
    train_dir = root / "train"
    valid_dir = root / "valid"
    if not train_dir.is_dir() or not valid_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {root}/train and {root}/valid with one folder per species."
        )

    classes = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    valid_classes = sorted(path.name for path in valid_dir.iterdir() if path.is_dir())
    if classes != valid_classes:
        raise ValueError("Indian Birds train/valid class folders do not match.")
    if len(classes) != 25:
        raise ValueError(f"Expected 25 Indian bird classes, found {len(classes)}.")

    counts = {}
    for split_dir in (train_dir, valid_dir):
        for folder in sorted(split_dir.iterdir()):
            if folder.is_dir():
                counts[f"{split_dir.name}/{folder.name}"] = sum(
                    1 for file in folder.iterdir() if file.suffix.lower() in IMAGE_SUFFIXES
                )

    return {
        "root": root,
        "classes": classes,
        "counts": counts,
    }


def indian_samples(
    root: Path, class_to_index: Dict[str, int]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Builds training and validation sample lists with resolved class indices."""
    train_samples: List[Dict[str, object]] = []
    valid_samples: List[Dict[str, object]] = []

    for split, destination in (("train", train_samples), ("valid", valid_samples)):
        split_dir = root / split
        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = class_to_index[folder.name]
            for file in sorted(folder.iterdir()):
                if file.suffix.lower() in IMAGE_SUFFIXES:
                    destination.append(
                        {
                            "path": file.relative_to(root.parent).as_posix(),
                            "label": label,
                        }
                    )
    return train_samples, valid_samples
