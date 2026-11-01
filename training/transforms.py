"""Progressive augmentation schedule from rules.md and ADR-006."""

from typing import Callable

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def augmentation_phase(epoch: int) -> str:
    """Maps an epoch to the documented augmentation phase."""
    if epoch < 10:
        return "light"
    if epoch < 25:
        return "randaugment"
    if epoch < 40:
        return "mixup"
    return "finetune"


def build_train_transform(epoch: int, image_size: int = 224) -> Callable:
    """Builds the training transform for the current progressive phase."""
    phase = augmentation_phase(epoch)
    operations = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.65, 1.0) if phase != "finetune" else (0.85, 1.0),
            ratio=(3 / 4, 4 / 3),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1 if phase in ("light", "finetune") else 0.25,
            contrast=0.1 if phase in ("light", "finetune") else 0.25,
            saturation=0.1 if phase in ("light", "finetune") else 0.25,
        ),
    ]
    if phase in ("randaugment", "mixup"):
        operations.append(transforms.RandAugment(num_ops=2, magnitude=9))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.25 if phase in ("randaugment", "mixup") else 0.10
            ),
        ]
    )
    return transforms.Compose(operations)


def build_eval_transform(image_size: int = 224) -> Callable:
    """Deterministic resize/crop transform for validation."""
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
