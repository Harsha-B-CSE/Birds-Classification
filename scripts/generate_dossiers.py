#!/usr/bin/env python
"""Precompute local-model bird dossiers into the local/Firestore cache."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["indian", "all"], default="indian")
    parser.add_argument("--labels", default="", help="Comma-separated model labels.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    return parser.parse_args()


def load_labels(args: argparse.Namespace) -> list[str]:
    classes_path = ROOT / "training_output" / "classes.json"
    if not classes_path.exists():
        from training.combined import build_combined_manifest

        classes = build_combined_manifest(ROOT)["classes"]
    else:
        classes = json.loads(classes_path.read_text(encoding="utf-8"))

    if args.labels:
        return [label.strip() for label in args.labels.split(",") if label.strip()]
    if args.set == "all":
        return classes
    # Indian classes are appended after the 200 CUB labels; include the merged kingfisher.
    labels = classes[200:]
    alias = "083.White_breasted_Kingfisher"
    if alias in classes and alias not in labels:
        labels.append(alias)
    return labels


def main() -> None:
    args = parse_args()
    if args.local_only:
        os.environ["USE_FIREBASE"] = "False"
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Bird_class.settings")
    import django

    django.setup()
    from django.conf import settings

    from myapp.services.firebase_client import cache_bird_info, get_firestore_db
    from myapp.services.species_info import fetch_species_profile, species_slug

    labels = load_labels(args)
    if args.limit:
        labels = labels[: args.limit]

    for index, label in enumerate(labels, start=1):
        slug = species_slug(label)
        if args.force:
            cached_file = Path(settings.BIRD_INFO_CACHE_DIR) / f"{slug}.json"
            cached_file.unlink(missing_ok=True)
            db = get_firestore_db()
            if db:
                db.collection("bird_info_cache").document(slug).delete()
        started = time.perf_counter()
        profile = fetch_species_profile(label)
        elapsed = time.perf_counter() - started
        status = profile.get("source", "unknown")
        print(
            f"[{index}/{len(labels)}] {label} -> {profile.get('common_name')} "
            f"({status}, {elapsed:.1f}s)"
        )
        if profile.get("sources") and profile.get("source") == "local-model":
            cache_bird_info(slug, dict(profile))


if __name__ == "__main__":
    main()
