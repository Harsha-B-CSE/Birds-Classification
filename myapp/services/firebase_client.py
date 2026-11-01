import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from django.conf import settings

# Global Firestore DB client handle
_db = None
_bucket = None

def is_firebase_active() -> bool:
    """Returns True if Firebase is enabled and initialized in settings."""
    return getattr(settings, "USE_FIREBASE", False)

def get_firestore_db():
    """Returns singleton Firestore client or None if in local mode."""
    global _db
    if not is_firebase_active():
        return None

    if _db is not None:
        return _db

    try:
        import firebase_admin
        from firebase_admin import firestore

        if firebase_admin._apps:
            _db = firestore.client()
            return _db
    except Exception as e:
        print(f"[WARNING] Error obtaining Firestore client: {e}")

    return None

def get_storage_bucket():
    """Returns singleton Firebase Storage bucket or None if in local mode."""
    global _bucket
    if not is_firebase_active():
        return None

    if _bucket is not None:
        return _bucket

    try:
        import firebase_admin
        from firebase_admin import storage

        if firebase_admin._apps:
            _bucket = storage.bucket()
            return _bucket
    except Exception as e:
        print(f"[WARNING] Error obtaining Storage bucket: {e}")

    return None

# ==============================================================================
# HYBRID PREDICTION HISTORY (Firestore in Cloud Mode, Local JSON in Offline Mode)
# ==============================================================================

def save_prediction_history(device_id: str, record: Dict[str, Any]) -> bool:
    """
    Saves a prediction entry.
    - Cloud Mode: Saves to Firestore `predictions/{device_id}/history/`.
    - Local Mode: Appends to local JSON file `{PRED_CACHE_DIR}/{device_id}.json`.
    """
    if not device_id:
        return False

    record["timestamp"] = time.time()

    if is_firebase_active():
        db = get_firestore_db()
        if db:
            try:
                from firebase_admin import firestore
                cloud_record = dict(record)
                cloud_record["created_at"] = firestore.SERVER_TIMESTAMP
                collection_ref = db.collection("predictions").document(device_id).collection("history")
                collection_ref.add(cloud_record)
                return True
            except Exception as e:
                print(f"[WARNING] Failed to save history to Firestore: {e}")

    # Fallback / Local Storage
    try:
        pred_dir = getattr(settings, "PRED_CACHE_DIR", settings.BASE_DIR / ".cache" / "predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        device_file = pred_dir / f"{device_id}.json"

        local_record = dict(record)
        local_record["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        history = []
        if device_file.exists():
            try:
                with open(device_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(local_record)
        # Cap local history at 50 most recent records per device
        if len(history) > 50:
            history = history[-50:]

        with open(device_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"[WARNING] Error saving local prediction history: {e}")
        return False

def get_prediction_history(device_id: str) -> List[Dict[str, Any]]:
    """Retrieves prediction history for the given device ID."""
    if not device_id:
        return []

    if is_firebase_active():
        db = get_firestore_db()
        if db:
            try:
                docs = (
                    db.collection("predictions")
                    .document(device_id)
                    .collection("history")
                    .order_by("created_at", direction="DESCENDING")
                    .limit(50)
                    .stream()
                )
                return [d.to_dict() for d in docs]
            except Exception as e:
                print(f"[WARNING] Error reading history from Firestore: {e}")

    # Local fallback
    try:
        pred_dir = getattr(settings, "PRED_CACHE_DIR", settings.BASE_DIR / ".cache" / "predictions")
        device_file = pred_dir / f"{device_id}.json"
        if device_file.exists():
            with open(device_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARNING] Error reading local history: {e}")

    return []

# ==============================================================================
# HYBRID BIOLOGICAL PROFILE CACHE
# ==============================================================================

def get_cached_bird_info(species_slug: str) -> Optional[Dict[str, Any]]:
    """Retrieves cached biological dossier from local NVMe cache or Firestore (<1ms latency)."""
    if not species_slug:
        return None

    # 1. Fast NVMe / local JSON cache (<1ms response)
    try:
        candidate_dirs = [
            getattr(settings, "BIRD_INFO_CACHE_DIR", settings.BASE_DIR / ".cache" / "bird_info"),
            settings.BASE_DIR / "myapp" / "data" / "bird_info",
        ]
        for info_dir in candidate_dirs:
            cached_file = info_dir / f"{species_slug}.json"
            if cached_file.exists():
                with open(cached_file, "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception as e:
        print(f"[WARNING] Error reading local bird_info cache: {e}")

    # 2. Firestore fallback if local cache missing
    if is_firebase_active():
        db = get_firestore_db()
        if db:
            try:
                doc = db.collection("bird_info_cache").document(species_slug).get()
                if doc.exists:
                    return doc.to_dict()
            except Exception as e:
                print(f"[WARNING] Error reading bird_info_cache from Firestore: {e}")

    return None

def cache_bird_info(species_slug: str, data: Dict[str, Any]) -> bool:
    """Stores biological dossier in shared Firestore cache and/or local cache."""
    if not species_slug:
        return False

    success = False
    payload = dict(data)
    if is_firebase_active():
        db = get_firestore_db()
        if db:
            try:
                from firebase_admin import firestore
                cloud_payload = dict(payload)
                cloud_payload["last_synced"] = firestore.SERVER_TIMESTAMP
                db.collection("bird_info_cache").document(species_slug).set(cloud_payload, merge=True)
                success = True
            except Exception as e:
                print(f"[WARNING] Error caching bird info in Firestore: {e}")

    # Always write to local cache as well
    try:
        info_dir = getattr(settings, "BIRD_INFO_CACHE_DIR", settings.BASE_DIR / ".cache" / "bird_info")
        info_dir.mkdir(parents=True, exist_ok=True)
        cached_file = info_dir / f"{species_slug}.json"
        with open(cached_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        success = True
    except Exception as e:
        print(f"[WARNING] Error caching bird info locally: {e}")

    return success


def clear_prediction_history(device_id: str) -> bool:
    """Deletes the prediction history for a device in Firestore and local cache."""
    if not device_id:
        return False
    cleared = False
    if is_firebase_active():
        db = get_firestore_db()
        if db:
            try:
                docs = (
                    db.collection("predictions")
                    .document(device_id)
                    .collection("history")
                    .stream()
                )
                batch = db.batch()
                count = 0
                for doc in docs:
                    batch.delete(doc.reference)
                    count += 1
                    if count % 400 == 0:
                        batch.commit()
                        batch = db.batch()
                if count % 400:
                    batch.commit()
                cleared = True
            except Exception as e:
                print(f"[WARNING] Error clearing Firestore history: {e}")

    try:
        pred_dir = getattr(settings, "PRED_CACHE_DIR", settings.BASE_DIR / ".cache" / "predictions")
        device_file = pred_dir / f"{device_id}.json"
        if device_file.exists():
            device_file.unlink()
        cleared = True
    except Exception as e:
        print(f"[WARNING] Error clearing local history: {e}")

    return cleared
