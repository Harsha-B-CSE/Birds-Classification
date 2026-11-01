# 🧠 Project Memory & Architectural Decision Log (ADR)
## Project: Birds Classification & Avian Intelligence System (Feathered Insights)

---

## 1. Project Identity & Persistent Context

- **Repository**: `Birds-classification` (Harsha-B-CSE / HARSHA-HAPPY)
- **Primary Tech Stack**: Django 5.1 (Python 3.11), PyTorch / timm, ONNX Runtime (CPU), Firebase Hosting & Cloud Firestore.
- **Core ML Architecture**: Swin Transformer (`swin_tiny` baseline $\to$ `swin_base` target) trained on CUB-200-2011 (200 species) + 25 Indian regional species.
- **Primary Domain Goal**: Instant, public bird identification with biological profile enrichment (taxonomy, lifespan, genetics) in $< 2\text{ seconds}$ without authentication.

---

## 2. Architectural Decision Records (ADRs)

### ADR-001: Frictionless Access Model over Mandatory User Accounts
- **Date**: 2026-03
- **Context**: The legacy project included Django standard user authentication (`login/`, `signup/`, `logout/`). In field conditions, requiring users to log in severely throttled conversion and adoption.
- **Decision**: Remove the mandatory authentication wall for core prediction workflows. Adopt an anonymous, device-keyed approach using a persistent client UUID stored in `localStorage` and indexed in Firestore under `predictions/{device_id}/history`.
- **Consequences**: Zero signup barrier for field birders; users retain their prediction history automatically across browser sessions.

---

### ADR-002: Production Inference via ONNX Runtime CPU instead of PyTorch CUDA
- **Date**: 2026-03
- **Context**: Serving a PyTorch model in a cloud container requires massive Docker images (>4GB with CUDA toolkits) and costly GPU virtual machines ($50–$150/month).
- **Decision**: Export the finalized PyTorch `.pth` checkpoint to ONNX (`FP32`, optional `INT8`). Serve predictions using `onnxruntime` on multi-threaded Cloud Run CPU instances.
- **Consequences**:
  - Image size drops from 4.5 GB to ~350 MB.
  - Cold starts on Cloud Run drop from 25s to <2.5s.
  - CPU inference latency remains under 400ms for Swin-B, satisfying the sub-2-second total budget.
  - Enables scale-to-zero serverless hosting (effectively free for low-to-moderate field traffic).

---

### ADR-003: 3-Check Out-of-Distribution (OOD) Verification System
- **Date**: 2026-03
- **Context**: Deep neural networks exhibit catastrophic overconfidence on out-of-distribution inputs (e.g., predicting a photo of a dog or car as an *Indigo Bunting* with 98% softmax probability).
- **Decision**: Implement a 3-tier sequential validation gate:
  1. **Check 1 (Binary Bird Detector)**: Lightweight MobileNetV3 ONNX gate checking if an avian subject is present ($P \ge 0.60$).
  2. **Check 2 (Species Classification)**: Swin Transformer forward pass yielding class logits.
  3. **Check 3 (Energy-Based OOD Gating)**: Calculate Helmholtz free energy $E(\mathbf{x}) = -T \log \sum \exp(z_i / T)$. If $E(\mathbf{x}) > \tau_{\text{energy}}$, abstain from guessing and flag as an unknown species.
- **Consequences**: Eliminates false identifications on non-birds and uncataloged species; establishes high trust with scientific researchers.

---

### ADR-004: Firebase as the Primary Data & Edge Layer
- **Date**: 2026-03
- **Context**: Previously, predictions and media relied on local SQLite (`db.sqlite3`) and local file paths, preventing public multi-user deployments.
- **Decision**: Position Firebase at the forefront:
  - **Firebase Hosting**: Public CDN and dynamic rewrite gateway to Cloud Run.
  - **Cloud Firestore**: Holds device-keyed prediction histories and the globally shared `bird_info_cache`.
  - **Firebase Storage**: Hosts ONNX model binaries for automated startup synchronization.
- **Consequences**: High availability, instant public sharing (`https://birds-classification.web.app`), and shared global cache hits for species profiles across all users.

---

### ADR-005: Strict 5.0 GB Storage Cap for Local Training Workstation
- **Date**: 2026-03
- **Context**: Training Vision Transformers, logging checkpoints, and caching web scrapes can rapidly exhaust disk storage on the training PC.
- **Decision**: Enforce a strict **5.0 GB total budget** on `D:\BirdsCache\` with automated LRU (Least Recently Used) eviction:
  - `models/` $\le 500\text{ MB}$ (one active `.pth` / `.onnx` version at a time).
  - `bird_info/` $\le 50\text{ MB}$ (JSON backups with 30-day TTL).
  - `predictions/` $\le 200\text{ MB}$ (auto-delete oldest 20% on breach).
  - `logs/` $\le 300\text{ MB}$ (live logs overwritten per training run).
- **Consequences**: Prevents workstation disk saturation and guarantees clean, reproducible experiment artifacts.

---

### ADR-006: Progressive Augmentation Protocol Driven by Validation Loss
- **Date**: 2026-03
- **Context**: Uncontrolled augmentation often leads to unquantified performance degradation or overfitting to distorted imagery.
- **Decision**: Augmentations must be introduced in disciplined 10-epoch increments using `timm` preset configurations:
  - Step 1: Baseline resize + normalization.
  - Step 2: RandAugment (`rand-m9-n2-mstd0.5`).
  - Step 3: Mixup ($\alpha = 0.2$).
  - Step 4: CutMix (only if dataset $> 50\text{k}$ samples).
  - Rollback immediately if validation loss increases for 3 consecutive epochs.
- **Consequences**: Systematic, verifiable accuracy improvements; zero premature model exports.

---

## 3. Discovered Gotchas & Historical Bug Fixes

| Issue | Root Cause | Permanent Resolution |
|---|---|---|
| **Hardcoded Windows Paths** | `os.path.join("D:\\Bird_class\\...")` hardcoded in `myapp/views.py` and `settings.py` broke deployment on other systems and Cloud Run Linux. | Standardized on `pathlib.Path` with `settings.BASE_DIR / "myapp" / ...`. |
| **Duplicate Imports & Routes** | `myapp/views.py` had 4 redundant import blocks for `PIL`, `torch`, and `JsonResponse`; `urls.py` had duplicated wildcard imports. | Cleaned and consolidated imports into modular handlers under `myapp/ml/` and `myapp/views.py`. |
| **Credential Security Leak** | `serviceAccountKey.json` risked accidental git commit. | Added `serviceAccountKey.json`, `*.json` credentials, and `.env` to `.gitignore`. Enabled runtime loading via environment variable. |
| **Model In-Memory Duplication** | Re-instantiating the PyTorch model inside view calls caused rapid memory leaks and process crashes. | Implemented singleton model loading at application startup with thread-safe evaluation state. |
| **Expired Python Trust Store** | `urllib` rejected `en.wikipedia.org` with `CERTIFICATE_VERIFY_FAILED` while browsers and PowerShell succeeded. | Use the certifi CA bundle in a shared SSL context (`myapp/services/http_client.py`). |
| **OOD Threshold Drift** | The initial energy gate (`-4.5`) rejected 15% of valid CUB images although non-bird photos were already caught by the 0.40 confidence floor. | Calibrated energy threshold to `-3.90` (in-distribution p95); values live in settings/env. |
| **Ambiguous Common Names** | iNaturalist resolved "Peacock" to a butterfly because autocomplete ranked non-avian taxa first. | Restrict iNaturalist candidates to `iconic_taxon_name == "Aves"` before selecting a taxon. |
| **Unset CSRF Cookie** | Browser uploads could receive 403 because no template emitted `{% csrf_token %}`. | Include the token in the upload form and send it as `X-CSRFToken`. |
| **Duplicate Scripts & Root Keys** | Redundant training/export scripts in root and exposed Firebase service account key files. | Consolidated to `myapp/ml/`, removed legacy notebook, updated `.gitignore` for `*.json` keys, standardized `.env`. |

---

## 4. Current State & Immediate Evolution

- **Current Stage**: Transfer learning, 224-class Swin ONNX export, and zero-LLM biological dossier caching are operational.
- **Model Status**: `myapp/swin_b_birds.onnx` (108.30 MB, FP32, opset 18) is the active 224-class classifier. Numerical equivalence with PyTorch verified (`max_abs_diff = 0.000002 < 1e-3`). CPU forward pass latency: 42.55 ms average, well below the 400 ms threshold.
- **Gate Status**:
  - Check 1: `myapp/bird_detector.onnx` MobileNetV3 binary gate ($P(\text{bird}) \ge 0.60$). Returns `"🚫 No bird detected"` on failure.
  - Check 2: `swin_b_birds.onnx` forward pass producing 224 class logits.
  - Check 3: Helmholtz Free Energy gate ($E(\mathbf{x}) \le -3.90$) and confidence floor ($\text{Top-1} \ge 0.40$). Returns `"⚠️ Unclear or uncataloged bird image"` on failure.
- **Enrichment Status**: Local JSON and Firestore `bird_info_cache` populated for all 224 species (common name, scientific name, IUCN badge, habitat, diet, taxonomy tree, wild vs. captive lifespan). Cache hit latency: 0.389 ms (<1ms target) without LLM overhead.
- **UI Status**: Natural Metallic UI renders Top-5 probability distribution bars, Chart.js horizontal bar graph for Wild vs. Captive lifespan, and anonymous device-keyed `/history/` field log.
