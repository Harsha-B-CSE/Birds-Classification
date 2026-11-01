"""Small local instruct model that composes bird dossiers offline."""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from django.conf import settings

from myapp.services.dossier_schema import (
    SCHEMA,
    extract_json,
    normalize_profile,
    prose_fallback,
)

_model = None
_tokenizer = None
_device = "cpu"
_load_lock = threading.Lock()


def is_available() -> bool:
    """True when a local model is configured and its files exist."""
    return bool(settings.LOCAL_LLM_ENABLED and Path(settings.LOCAL_LLM_MODEL_PATH).is_dir())


def _resolve_device() -> str:
    import torch

    requested = settings.LOCAL_LLM_DEVICE
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _load_model():
    """Loads the tokenizer and model once per process."""
    global _model, _tokenizer, _device
    if _model is not None:
        return _tokenizer, _model, _device

    with _load_lock:
        if _model is not None:
            return _tokenizer, _model, _device
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(settings.LOCAL_LLM_MODEL_PATH)
        _device = _resolve_device()
        dtype = torch.float16 if _device == "cuda" else torch.float32
        _tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        _model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, local_files_only=True
        ).to(_device)
        _model.eval()
        print(f"[OK] Local dossier model loaded on {_device}: {path}")
    return _tokenizer, _model, _device


def _build_prompt(common_name: str, reference_facts: Dict[str, Any]) -> list:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful ornithologist. Return only valid JSON. "
                "Never contradict supplied reference facts. Use null when unsure."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Species: {common_name}\n"
                f"Reference facts: {json.dumps(reference_facts, ensure_ascii=False)}\n"
                f"JSON schema: {SCHEMA}\n"
                "In summary, describe appearance, behavior, and diet in 2-3 sentences; "
                "do not list the native range there. Give one habitat sentence, the native "
                "range, IUCN status, and typical wild and captive lifespan ranges in years. "
                "Do not use markdown."
            ),
        },
    ]
    return messages


def generate_profile(
    class_label: str,
    reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates a structured dossier with the local model."""
    import torch

    tokenizer, model, device = _load_model()
    reference = reference or {}
    reference_facts = {
        key: reference[key]
        for key in ("scientific_name", "taxonomy", "iucn_status", "native_range", "lifespan")
        if reference.get(key)
    }
    common_name = class_label.split(".", 1)[-1].replace("_", " ")
    prompt = tokenizer.apply_chat_template(
        _build_prompt(common_name, reference_facts),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt + "{", return_tensors="pt").to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=settings.LOCAL_LLM_MAX_NEW_TOKENS + 60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = "{" + tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    payload = extract_json(text)
    profile = (
        normalize_profile(payload, class_label, reference)
        if payload is not None
        else prose_fallback(text, class_label, reference)
    )
    profile["model_id"] = Path(settings.LOCAL_LLM_MODEL_PATH).name
    print(
        f"[OK] Local dossier for {common_name} in "
        f"{time.perf_counter() - started:.1f}s"
    )
    return profile
