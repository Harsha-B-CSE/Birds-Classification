#!/usr/bin/env python
"""
GPU Training Script for Swin Transformer on RTX 3060 (224 Target Classes).
Executes two-stage transfer learning:
  Stage 1 (Epochs 1-5): Frozen backbone, train new 224-class head at LR = 1e-3.
  Stage 2 (Epochs 6-30): Unfreeze all layers with Cosine Annealing at LR = 1e-4.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.combined import build_combined_manifest
from training.dataset import CombinedBirdDataset, class_balanced_sampler
from training.engine import train_one_epoch, validate, export_onnx
from training.model_loader import load_and_adapt_swin, configure_stage
from training.transforms import augmentation_phase, build_eval_transform, build_train_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 224-Class Swin Transformer on RTX 3060")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", default="training_output")
    parser.add_argument("--weights", default="myapp/swin_model_best.pth")
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke test step limit per epoch")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--export-onnx", action="store_true")
    # ── Resume arguments ──────────────────────────────────────────────────────
    parser.add_argument(
        "--resume-epoch", type=int, default=0,
        help="Epoch index to resume FROM (0-based). E.g. 22 resumes at epoch 23."
    )
    parser.add_argument(
        "--resume-weights", default="",
        help="Path to checkpoint .pth to load when resuming. "
             "Defaults to <output-dir>/last_model.pth when --resume-epoch > 0."
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print("=" * 60)
    print(f"[TRAIN] Swin-B 224-Class Avian Classifier | Device: {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)")
    print("=" * 60)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build combined 224-class manifest
    manifest = build_combined_manifest(PROJECT_ROOT, save=True)
    num_classes = manifest["num_classes"]
    train_samples = manifest["train"][: args.limit_train or None]
    valid_samples = manifest["valid"][: args.limit_val or None]

    print(f"[DATA] Total Classes: {num_classes} | Train: {len(train_samples)} | Valid: {len(valid_samples)}")

    train_dataset = CombinedBirdDataset(
        PROJECT_ROOT, train_samples, build_train_transform(0, args.image_size)
    )
    valid_dataset = CombinedBirdDataset(
        PROJECT_ROOT, valid_samples, build_eval_transform(args.image_size)
    )
    labels = [int(s["label"]) for s in train_samples]
    sampler = class_balanced_sampler(labels)

    # 2. Load pre-trained weights and adapt head
    if args.resume_epoch > 0:
        # Resume: pick up the last checkpoint from a previous run
        resume_ckpt = args.resume_weights or str(PROJECT_ROOT / args.output_dir / "last_model.pth")
        print(f"[RESUME] Loading checkpoint: {resume_ckpt}  (continuing from epoch {args.resume_epoch + 1})")
        weights_path = Path(resume_ckpt)
    else:
        weights_path = PROJECT_ROOT / args.weights
    model = load_and_adapt_swin(
        weights_path,
        num_classes=num_classes,
        device=device,
        resume_mode=(args.resume_epoch > 0),
    )

    # 3. Setup Stage 1 or Stage 2 optimizer (depending on resume point)
    if args.resume_epoch >= args.warmup_epochs:
        # Already past warmup — go straight to Stage 2 with correct LR
        current_stage = 2
        optimizer = configure_stage(
            model, stage=2, stage2_lr=args.stage2_lr, weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6
        )
        # Fast-forward the scheduler so the LR matches where we left off
        steps_done = args.resume_epoch - args.warmup_epochs
        for _ in range(steps_done):
            scheduler.step()
        print(f"[RESUME] Stage 2 | scheduler fast-forwarded {steps_done} steps | "
              f"LR = {optimizer.param_groups[0]['lr']:.2e}")
    else:
        current_stage = 1
        optimizer = configure_stage(
            model, stage=1, stage1_lr=args.stage1_lr, weight_decay=args.weight_decay
        )
        scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    log_path = output_dir / "epoch_log.csv"
    if not log_path.exists() or args.resume_epoch == 0:
        with open(log_path, "w", newline="", encoding="utf-8") as h:
            csv.writer(h).writerow(["epoch", "stage", "phase", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])

    # Recover best_val_acc from existing log so we don't regress when resuming
    best_val_acc = 0.0
    if args.resume_epoch > 0 and log_path.exists():
        import csv as _csv
        with open(log_path, newline="", encoding="utf-8") as h:
            for row in _csv.DictReader(h):
                try:
                    best_val_acc = max(best_val_acc, float(row["val_acc"]))
                except (KeyError, ValueError):
                    pass
        print(f"[RESUME] Best val_acc from previous log: {best_val_acc:.4f}")

    # ── DataLoaders are created ONCE outside the loop ───────────────────────
    # Recreating DataLoaders with num_workers > 0 inside the loop causes
    # Windows multiprocessing / CUDA context crashes (seen at epoch 22).
    # train_dataset.transform is mutated each epoch so workers always get
    # the correct augmentation via the shared dataset object reference.
    use_persistent = args.workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )

    start_epoch = args.resume_epoch  # 0 = fresh start
    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()

        # Transition to Stage 2 after warmup epochs
        if epoch == args.warmup_epochs and current_stage == 1:
            current_stage = 2
            optimizer = configure_stage(
                model, stage=2, stage2_lr=args.stage2_lr, weight_decay=args.weight_decay
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6
            )
            # Fast-forward scheduler to match resumed epoch position
            for _ in range(epoch - args.warmup_epochs):
                scheduler.step()

        phase = augmentation_phase(epoch)
        mixup_alpha = 0.2 if phase == "mixup" else 0.0
        train_dataset.transform = build_train_transform(epoch, args.image_size)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, epoch, mixup_alpha, args.max_steps
        )
        val_loss, val_acc = validate(model, valid_loader, criterion, device)

        if scheduler:
            scheduler.step()

        elapsed = time.perf_counter() - started
        current_lr = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="", encoding="utf-8") as h:
            csv.writer(h).writerow([
                epoch + 1, current_stage, phase,
                f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}",
                f"{current_lr:.6f}", f"{elapsed:.1f}"
            ])

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs:02d} [Stage {current_stage}|{phase}] "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} loss={val_loss:.4f} "
            f"lr={current_lr:.2e} ({elapsed:.1f}s)"
        )

        # Save latest and best checkpoints
        torch.save(model.state_dict(), output_dir / "last_model.pth")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            torch.save(model.state_dict(), PROJECT_ROOT / "myapp" / "swin_b_birds_224.pth")
            with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
                json.dump(manifest["classes"], f, indent=2)
            with open(PROJECT_ROOT / "myapp" / "classes_224.json", "w", encoding="utf-8") as f:
                json.dump(manifest["classes"], f, indent=2)
            print(f"  [BEST] Saved new best checkpoint (Val Acc: {best_val_acc:.4f})")

    print("\n[DONE] Training complete. Best Val Acc: {:.4f}".format(best_val_acc))

    if args.export_onnx:
        onnx_dest = PROJECT_ROOT / "myapp" / "swin_b_birds.onnx"
        print(f"[EXPORT] Exporting to ONNX at {onnx_dest}...")
        export_onnx(model, onnx_dest, args.image_size)
        print("[OK] ONNX export complete.")


if __name__ == "__main__":
    main()