"""Training/validation loops and ONNX export for the combined classifier."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def _mixup(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[permutation], targets, targets[permutation], lam


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    mixup_alpha: float = 0.0,
    max_steps: int = 0,
) -> Tuple[float, float]:
    """Runs one epoch and returns mean loss and training accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    total_steps = len(loader) if not max_steps else min(len(loader), max_steps)
    progress = tqdm(loader, total=total_steps, desc=f"train epoch {epoch}", leave=False)

    for step, (images, targets) in enumerate(progress):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, enabled=scaler.is_enabled(), dtype=torch.float16
        ):
            if mixup_alpha > 0:
                images, targets_a, targets_b, lam = _mixup(images, targets, mixup_alpha)
                logits = model(images)
                loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(
                    logits, targets_b
                )
            else:
                logits = model(images)
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size
        progress.set_postfix(loss=f"{total_loss / seen:.4f}")

    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def validate(
    model: nn.Module, loader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Returns mean validation loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for images, targets in tqdm(loader, desc="validation", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size
    return total_loss / max(seen, 1), correct / max(seen, 1)


def export_onnx(model: nn.Module, output_path: Path, image_size: int = 224) -> None:
    """Exports a trained classifier to ONNX FP32."""
    model = model.to("cpu").eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
