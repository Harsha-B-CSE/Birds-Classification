import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from django.conf import settings

def ensure_cache_structure() -> None:
    """Creates the standard cache directory tree if it does not already exist."""
    directories = [
        settings.CACHE_DIR,
        settings.MODELS_CACHE_DIR,
        settings.BIRD_INFO_CACHE_DIR,
        settings.PRED_CACHE_DIR,
        settings.LOGS_CACHE_DIR,
    ]
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[WARNING] Could not create cache directory {directory}: {e}")

def get_directory_size_mb(path: Path) -> float:
    """Calculates total size of all files inside a directory in Megabytes."""
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)

def get_cache_usage() -> Dict[str, float]:
    """Returns total and per-directory usage of the cache in Megabytes."""
    return {
        "models_mb": get_directory_size_mb(settings.MODELS_CACHE_DIR),
        "bird_info_mb": get_directory_size_mb(settings.BIRD_INFO_CACHE_DIR),
        "predictions_mb": get_directory_size_mb(settings.PRED_CACHE_DIR),
        "logs_mb": get_directory_size_mb(settings.LOGS_CACHE_DIR),
        "total_mb": get_directory_size_mb(settings.CACHE_DIR),
        "max_limit_mb": float(getattr(settings, "MAX_CACHE_SIZE_MB", 5000)),
    }

def evict_prediction_cache(max_size_mb: float = 200.0, eviction_ratio: float = 0.20) -> int:
    """
    LRU Eviction: If predictions/ exceeds max_size_mb, removes the oldest
    files (default: 20% of oldest entries) to restore headroom.
    Returns the count of deleted files.
    """
    pred_dir = settings.PRED_CACHE_DIR
    if not pred_dir.exists():
        return 0

    current_size = get_directory_size_mb(pred_dir)
    if current_size <= max_size_mb:
        return 0

    # Sort files by modification time (oldest first)
    entries = sorted(
        [f for f in pred_dir.glob("*.json") if f.is_file()],
        key=lambda f: f.stat().st_mtime
    )

    evicted_count = 0
    num_to_delete = max(1, int(len(entries) * eviction_ratio))

    for entry in entries[:num_to_delete]:
        try:
            entry.unlink()
            evicted_count += 1
        except Exception as e:
            print(f"[WARNING] Could not evict cached prediction {entry.name}: {e}")

    return evicted_count

def cleanup_expired_bird_info(ttl_days: int = 30) -> int:
    """Removes cached biological info JSON files older than ttl_days."""
    info_dir = settings.BIRD_INFO_CACHE_DIR
    if not info_dir.exists():
        return 0

    cutoff_time = time.time() - (ttl_days * 86400)
    expired_count = 0

    for entry in info_dir.glob("*.json"):
        if entry.is_file() and entry.stat().st_mtime < cutoff_time:
            try:
                entry.unlink()
                expired_count += 1
            except Exception as e:
                print(f"[WARNING] Could not remove expired bird info {entry.name}: {e}")

    return expired_count

def cleanup_before_training(
    max_cache_mb: float = 5000.0,
    pred_max_mb: float = 200.0,
    log_max_mb: float = 100.0,
    bird_info_ttl_days: int = 30,
    prediction_eviction_ratio: float = 0.25,
) -> Dict[str, Any]:
    """
    Comprehensive cache cleanup before training begins.
    Ensures strict adherence to the 5.0 GB local storage limit.
    
    Returns a summary dict of cleanup actions taken.
    """
    results = {
        "pre_cleanup_usage_mb": 0.0,
        "post_cleanup_usage_mb": 0.0,
        "freed_mb": 0.0,
        "actions": [],
    }
    
    # Initial usage
    usage_before = get_cache_usage()
    results["pre_cleanup_usage_mb"] = usage_before["total_mb"]
    
    # 1. Evict prediction cache if over limit
    evicted = evict_prediction_cache(max_size_mb=200.0, eviction_ratio=0.25)
    if evicted > 0:
        results["actions"].append(f"Evicted {evicted} prediction cache files (LRU)")
    
    # 2. Clean expired bird info cache
    expired_bird = cleanup_expired_bird_info(ttl_days=30)
    if expired_bird > 0:
        results["actions"].append(f"Removed {expired_bird} expired bird info entries (> 30 days)")
    
    # 3. Clean old log files (older than 7 days by default)
    log_dir = getattr(settings, "LOGS_CACHE_DIR", settings.CACHE_DIR / "logs")
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if log_dir.exists():
        cutoff = time.time() - (7 * 86400)
        removed_logs = 0
        for entry in log_dir.rglob("*.log"):
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                try:
                    entry.unlink()
                    removed_logs += 1
                except Exception as e:
                    print(f"[WARNING] Could not remove log file {entry}: {e}")
        if removed_logs > 0:
            results["actions"].append(f"Removed {removed_logs} old log files (> 7 days)")
    
    # 4. Clean temporary files in cache root (e.g., *.tmp, *.temp, core dumps)
    cache_root = Path(settings.CACHE_DIR)
    temp_patterns = ["*.tmp", "*.temp", "*.pid", "core.*", "*.crash"]
    temp_removed = 0
    for pattern in temp_patterns:
        for entry in cache_root.rglob(pattern):
            if entry.is_file():
                try:
                    entry.unlink()
                    temp_removed += 1
                except Exception:
                    pass
    if temp_removed > 0:
        results["actions"].append(f"Removed {temp_removed} temporary files")
    
    # 5. Check total cache size and enforce hard limit
    usage_after = get_cache_usage()
    max_cache_mb = 5000.0
    if usage_after["total_mb"] > max_cache_mb:
        # Aggressive cleanup: remove oldest prediction files until under limit
        pred_dir = Path(settings.PRED_CACHE_DIR)
        entries = sorted(
            [f for f in pred_dir.glob("*.json") if f.is_file()],
            key=lambda f: f.stat().st_mtime
        )
        removed = 0
        for entry in entries:
            if get_directory_size_mb(Path(settings.CACHE_DIR)) <= 5000.0:
                break
            try:
                entry.unlink()
                removed += 1
            except Exception:
                pass
        if removed > 0:
            results["actions"].append(f"Emergency eviction: removed {removed} prediction files to meet 5000 MB limit")
    
    # Final usage
    usage_after = get_cache_usage()
    results["pre_cleanup_usage_mb"] = round(get_cache_usage()["total_mb"], 2)  # will be overwritten below
    results["post_cleanup_usage_mb"] = usage_after["total_mb"]
    results["freed_mb"] = round(usage_before["total_mb"] - usage_after["total_mb"], 2)
    
    # Fix pre_cleanup_usage_mb
    results["pre_cleanup_usage_mb"] = round(usage_before["total_mb"], 2)
    
    return results


# Auto-ensure cache directory structure on module import
ensure_cache_structure()