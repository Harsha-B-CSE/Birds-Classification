#!/usr/bin/env python
"""
Dataset Audit & Merge Script for Feathered Insights

Audits the Indian Birds 25 dataset for corrupted files and unsupported MIME types,
then merges it with the CUB-200-2011 dataset into a unified ImageFolder-compatible
directory structure at D:\\BirdsCache\\data\\merged_dataset\\.

Original source data is preserved (not deleted).
"""

import argparse
import json
import mimetypes
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image
import hashlib

# Supported image MIME types
SUPPORTED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Supported file extensions
SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDIAN_BIRDS_ROOT = PROJECT_ROOT / "Indian_Birds_25"
CUB200_ROOT = PROJECT_ROOT / "CUB_200_2011"
MERGED_ROOT = Path(r"D:\\BirdsCache\\data\\merged_dataset")

# Class name normalization: folder name -> canonical class name
# Indian Birds folders have hyphens, CUB-200 uses numbered prefixes
def normalize_class_name(folder_name: str) -> str:
    """Convert folder name to canonical class name (spaces instead of hyphens/underscores)."""
    return folder_name.replace("-", " ").replace("_", " ")

def load_class_mapping(indian_root: Path, cub_root: Path) -> Dict[str, int]:
    """
    Build a unified class name -> index mapping.
    CUB-200 classes take indices 0-199, Indian Birds get 200-224.
    """
    # Load CUB-200 classes from classes.txt (200 classes)
    cub_classes = []
    with open(cub_root / "classes.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "001.Black_footed_Albatross" -> "Black footed Albatross"
            parts = line.split(".", 1)
            if len(parts) == 2:
                canonical = normalize_class_name(parts[1])
                cub_classes.append(canonical)
    
    # Load Indian Birds classes from classes.txt (25 classes)
    indian_classes = []
    with open(INDIAN_BIRDS_ROOT / "classes.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indian_classes.append(line.strip())
    
    # Check for duplicates (case-insensitive)
    cub_set = {c.lower() for c in cub_classes}
    indian_unique = []
    for cls in indian_classes:
        if cls.lower() not in cub_set:
            indian_unique.append(cls)
        else:
            print(f"[INFO] Duplicate class (will merge): {cls}")
    
    # Build unified class list: CUB-200 first, then unique Indian Birds
    unified_classes = cub_classes + indian_unique
    class_to_idx = {cls: idx for idx, cls in enumerate(unified_classes)}
    
    print(f"[INFO] CUB-200 classes: {len(cub_classes)}")
    print(f"[INFO] Indian Birds unique classes: {len(indian_unique)}")
    print(f"[INFO] Total unified classes: {len(unified_classes)}")
    
    return class_to_idx


def audit_dataset(root: Path, dataset_name: str) -> Tuple[int, int, List[str]]:
    """
    Audit a dataset for corrupted files and unsupported MIME types.
    Returns: (total_files, corrupted_count, unsupported_files_list)
    """
    corrupted = 0
    unsupported = []
    total = 0
    
    print(f"\n[AUDIT] Scanning {dataset_name} at {root}")
    
    for image_file in root.rglob("*"):
        if not image_file.is_file():
            continue
        total += 1
        
        # Check file extension
        ext = image_file.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            unsupported.append(str(image_file.relative_to(root)))
            continue
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(image_file))
        if mime_type and mime_type not in SUPPORTED_MIME_TYPES:
            unsupported.append(str(image_file.relative_to(root)))
            continue
        
        # Verify image integrity
        try:
            with Image.open(image_file) as img:
                img.verify()
                # Re-open for potential loading (verify() closes the file)
                with Image.open(image_file) as img:
                    img.load()
        except Exception as e:
            corrupted += 1
            print(f"  [CORRUPTED] {image_file.relative_to(root)}: {e}")
    
    print(f"  Total files: {total}")
    print(f"  Corrupted: {corrupted}")
    print(f"  Unsupported MIME/ext: {len(unsupported)}")
    
    return total, corrupted, unsupported


def copy_dataset_split(
    src_root: Path,
    dst_root: Path,
    class_to_idx: Dict[str, int],
    dataset_name: str,
    split: str
) -> int:
    """
    Copy images from a dataset split (train/valid) to the merged directory
    organized by class index (ImageFolder format).
    Returns number of files copied.
    """
    src_split = Path(src_root) / split
    if not src_split.exists():
        print(f"  [WARN] Split not found: {src_split}")
        return 0
    
    copied = 0
    
    for class_folder in sorted(src_root / split).iterdir():
        if not class_folder.is_dir():
            continue
        
        folder_name = class_folder.name
        canonical_name = normalize_class_name(folder_name)
        
        if canonical_name not in class_to_idx:
            print(f"  [SKIP] Class not in unified mapping: {canonical_name}")
            continue
        
        class_idx = class_to_idx[canonical_name]
        class_dir = dst_root / str(class_idx)
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in class_folder.iterdir():
            if not img_file.is_file():
                continue
            if img_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            
            # Generate unique filename to avoid collisions
            # Use hash of original path to ensure uniqueness
            rel_path = img_file.relative_to(class_folder.parent.parent)
            file_hash = hashlib.md5(str(rel_path).encode()).hexdigest()[:8]
            new_name = f"{img_file.stem}_{file_hash}{img_file.suffix}"
            dst_file = class_dir / new_name
            
            try:
                shutil.copy2(img_file, dst_file)
                copied += 1
            except Exception as e:
                print(f"  [ERROR] Failed to copy {img_file}: {e}")
    
    print(f"  [{dataset_name}/{split}] Copied {copied} images")
    return copied


def build_manifest(merged_root: Path, class_to_idx: Dict[str, int]) -> Dict:
    """Build a manifest.json describing the merged dataset."""
    manifest = {
        "version": "1.0",
        "total_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "splits": {
            "train": {},
            "valid": {},
        }
    }
    
    for split in ["train", "valid"]:
        split_root = merged_root / split
        if split_root.exists():
            for class_dir in sorted(split_root.iterdir()):
                if class_dir.is_dir():
                    class_idx = class_dir.name
                    count = len(list(class_dir.glob("*.jpg"))) + \
                            len(list(class_dir.glob("*.jpeg"))) + \
                            len(list(class_dir.glob("*.png"))) + \
                            len(list(class_dir.glob("*.webp")))
                    manifest["splits"][split][class_idx] = {
                        "count": count,
                        "class_name": manifest["idx_to_class"][class_idx]
                    }
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Audit and merge Indian Birds + CUB-200 datasets")
    parser.add_argument("--dry-run", action="store_true", help="Only audit, don't copy files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing merged dataset")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Feathered Insights - Dataset Audit & Merge")
    print("=" * 60)
    
    # Load class mapping
    class_to_idx = load_class_mapping(INDIAN_BIRDS_ROOT, CUB200_ROOT)
    
    # Audit both datasets
    print("\n" + "=" * 60)
    print("PHASE 1: Dataset Audit")
    print("=" * 60)
    
    indian_total, indian_corrupt, indian_unsupported = audit_dataset(
        INDIAN_BIRDS_ROOT, "Indian Birds 25"
    )
    cub_total, cub_corrupt, cub_unsupported = audit_dataset(
        CUB200_ROOT, "CUB-200-2011"
    )
    
    total_files = indian_total + cub_total
    total_corrupt = indian_corrupt + cub_corrupt
    total_unsupported = len(indian_unsupported) + len(cub_unsupported)
    
    print(f"\n[AUDIT SUMMARY]")
    print(f"  Total files scanned: {total_files}")
    print(f"  Total corrupted: {total_corrupt}")
    print(f"  Total unsupported: {total_unsupported}")
    
    if total_corrupt > 0:
        print(f"[WARNING] Found {total_corrupt} corrupted files!")
    
    if indian_unsupported:
        print(f"  Indian Birds unsupported: {indian_unsupported[:5]}...")
    if cub_unsupported:
        print(f"  CUB-200 unsupported: {cub_unsupported[:5]}...")
    
    if args.dry_run:
        print("\n[DRY RUN] Audit complete. No files copied.")
        return
    
    # Prepare merged directory
    if MERGED_ROOT.exists():
        if args.force:
            print(f"\n[CLEANUP] Removing existing merged dataset at {MERGED_ROOT}")
            shutil.rmtree(MERGED_ROOT)
        else:
            print(f"\n[ERROR] Merged dataset already exists at {MERGED_ROOT}. Use --force to overwrite.")
            return
    
    # Create merged directory structure
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    (MERGED_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (MERGED_ROOT / "valid").mkdir(parents=True, exist_ok=True)
    
    # Phase 2: Merge datasets
    print("\n" + "=" * 60)
    print("PHASE 2: Dataset Merge")
    print("=" * 60)
    
    total_copied = 0
    
    # Copy Indian Birds
    print("\n[MERGE] Indian Birds 25 (train)")
    total_copied += copy_dataset_split(
        INDIAN_BIRDS_ROOT, MERGED_ROOT, class_to_idx, "IndianBirds", "train"
    )
    
    print("\n[MERGE] Indian Birds 25 (valid)")
    total_copied += copy_dataset_split(
        INDIAN_BIRDS_ROOT, MERGED_ROOT, class_to_idx, "IndianBirds", "valid"
    )
    
    # Copy CUB-200 (train/valid split is in CUB's train_test_split.txt)
    # We'll use the same split structure as CUB provides
    print("\n[MERGE] CUB-200-2011 (train)")
    total_copied += copy_dataset_split(
        CUB200_ROOT, MERGED_ROOT, class_to_idx, "CUB200", "train"
    )
    
    print("\n[MERGE] CUB-200-2011 (valid)")
    total_copied += copy_dataset_split(
        CUB200_ROOT, MERGED_ROOT, class_to_idx, "CUB200", "valid"
    )
    
    print(f"\n[MERGE SUMMARY] Total images copied: {total_copied}")
    
    # Build and save manifest
    print("\n[MANIFEST] Building dataset manifest...")
    manifest = build_manifest(MERGED_ROOT, class_to_idx)
    manifest_path = MERGED_ROOT / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Saved to {manifest_path}")
    
    # Save class mapping
    mapping_path = MERGED_ROOT / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in class_to_idx.items()}, f, indent=2)
    print(f"[MAPPING] Class mapping saved to {mapping_path}")
    
    print("\n" + "=" * 60)
    print("DATASET MERGE COMPLETE")
    print("=" * 60)
    print(f"Merged dataset location: {MERGED_ROOT}")
    print(f"Total classes: {len(class_to_idx)}")
    print("Original source data preserved at:")
    print(f"  - {INDIAN_BIRDS_ROOT}")
    print(f"  - {CUB200_ROOT}")


if __name__ == "__main__":
    main()