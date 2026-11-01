"""Model initialization and head transfer logic for 224-class bird classifier."""

from pathlib import Path
import torch
import torch.nn as nn
from myapp.ml.classifier import CUBModel


def load_and_adapt_swin(
    pretrained_path: Path,
    num_classes: int = 224,
    device: torch.device = torch.device("cpu"),
    resume_mode: bool = False,
) -> nn.Module:
    """
    Loads CUB-200 Swin model and modifies the classification head to num_classes.

    Args:
        pretrained_path: Path to weights file.
        num_classes: Target output classes.
        device: Target device.
        resume_mode: When True, the checkpoint already has num_classes outputs
                     (e.g. resuming mid-training). Skips the 200→N head transplant.
    """
    if resume_mode:
        # Checkpoint already has the correct N-class head.
        # CUBModel(num_classes=num_classes) builds the right architecture directly.
        model = CUBModel(num_classes=num_classes, pretrained=False)
        if pretrained_path.exists():
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"[OK] Resumed {num_classes}-class weights from {pretrained_path.name}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {pretrained_path}")
        return model.to(device)

    # ── Fresh start: load CUB-200 base then transplant the head ─────────────
    model = CUBModel(num_classes=200, pretrained=False)
    if pretrained_path.exists():
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[OK] Loaded pre-trained CUB-200 weights from {pretrained_path.name}")
    else:
        print(f"[WARN] Pretrained weights not found at {pretrained_path}")

    old_fc = model.model.head.fc
    in_features = old_fc.in_features
    new_fc = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(new_fc.weight, std=0.02)
    nn.init.zeros_(new_fc.bias)

    # Transfer 200 class weights
    new_fc.weight.data[:200] = old_fc.weight.data[:200]
    new_fc.bias.data[:200] = old_fc.bias.data[:200]

    model.model.head.fc = new_fc
    print(f"[OK] Modified head from 200 to {num_classes} classes.")
    return model.to(device)



def configure_stage(
    model: nn.Module,
    stage: int,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Configures parameter freezing and AdamW optimizer for Stage 1 vs Stage 2."""
    if stage == 1:
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.head.fc.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            model.model.head.fc.parameters(), lr=stage1_lr, weight_decay=weight_decay
        )
        print(f"[STAGE 1] Backbone frozen. Training only 224-class head at LR={stage1_lr:.1e}")
    else:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=stage2_lr, weight_decay=weight_decay
        )
        print(f"[STAGE 2] All layers unfrozen. Fine-tuning at LR={stage2_lr:.1e}")
    return optimizer
