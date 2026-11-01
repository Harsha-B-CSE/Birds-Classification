# 🛡️ Engineering Rules & AI Guardrails
## Project: Birds Classification & Avian Intelligence System (Feathered Insights)

> **Mandatory Notice for AI Agents & Developers:**
> This document defines absolute, non-negotiable engineering rules, coding standards, and architectural guardrails for the **Birds Classification** repository. Every code edit, refactoring step, or feature addition must strictly adhere to these constraints to prevent context drift and code rot.

---

## 1. Architectural & File Size Guardrails

- **RULE-01 (Max File Length)**: No single source file shall exceed **250 lines of code**. If a file approaches this threshold, decompose it into focused sub-modules (e.g., split `views.py` into `views/predict_view.py`, `views/history_view.py`).
- **RULE-02 (Separation of Concerns)**:
  - **Views** (`myapp/views/`): Exclusively handle HTTP requests, input validation, and JSON/Template responses.
  - **ML Subsystem** (`myapp/ml/`): Houses model initialization, image tensor transforms, inference sessions, and OOD calculations.
  - **Service Layer** (`myapp/services/`): Manages Firebase Firestore/Storage calls and external REST APIs (Wikipedia, NCBI).
- **RULE-03 (No Random Tech Stack Swaps)**: Never introduce alternative backend frameworks (e.g., FastAPI, Flask, Express) or CSS utility systems (e.g., Tailwind) without explicit user authorization. Maintain the established stack: **Django 5.1 + Vanilla CSS + ONNX Runtime + Firebase**.

---

## 2. Path & Configuration Rules

- **RULE-04 (Zero Hardcoded Absolute Paths)**:
  - ❌ **FORBIDDEN**: `os.path.join("D:\\Bird_class\\myapp\\swin_model_best.pth")`
  - ✅ **MANDATORY**: `settings.BASE_DIR / "myapp" / "swin_model_best.pth"` or configurable via `os.getenv()`.
- **RULE-05 (Credential Secrecy & Git Safety)**:
  - Never stage or commit `serviceAccountKey.json`, `.env`, `firebase_credentials.json`, or any file containing private keys.
  - If a service account file is created, immediately verify its presence in `.gitignore`.
- **RULE-06 (Environment Isolation)**: Provide a clean fallback mechanism when external services are disabled (e.g., `FIREBASE_ENABLED = False` allows offline local execution with mock responses).

---

## 3. Machine Learning & Inference Rules

- **RULE-07 (Inference Memory Safety)**:
  - All PyTorch inference must be strictly wrapped inside `with torch.no_grad():`.
  - In production, prefer `onnxruntime.InferenceSession` with CPU execution providers (`intra_op_num_threads=4`).
- **RULE-08 (Singleton Model Lifecycle)**:
  - Models must be loaded **once** at application initialization or lazy-loaded on the first request as a singleton.
  - **NEVER** instantiate or re-load model weights inside a view or request loop.
- **RULE-09 (Input Ingestion Sanitization)**:
  - All image uploads must be validated for:
    - Supported MIME types: `image/jpeg`, `image/png`, `image/webp`.
    - Maximum payload size: $\le 15.0\text{ MB}$.
    - Automatic conversion to 3-channel RGB (handling RGBA transparency, greyscale, or CMYK).
- **RULE-10 (Enforce the 3-Check Pipeline)**:
  - The production prediction endpoint (`/predict/`) must never bypass the 3-check validation cascade:
    1. **Check 1**: Binary Bird Presence Verification ($P(\text{bird}) \ge 0.60$).
    2. **Check 2**: Species Logit Generation.
    3. **Check 3**: Helmholtz Free Energy OOD Gate ($E \le \tau$) and Confidence Floor ($\text{Top-1} \ge 0.40$).
- **RULE-11 (Workstation Cache Ceiling)**:
  - Any caching script writing to `D:\BirdsCache\` must respect the **5.0 GB storage budget** and execute automated LRU eviction upon reaching 90% capacity.

---

## 4. UI, Styling & Accessibility Rules

- **RULE-12 (Strict Design Token Adherence)**:
  - All UI elements must use the **Natural Metallic** design tokens defined in `Design.md`:
    - Base: `--bg-deep-forest` (`hsl(150, 20%, 8%)`).
    - Accents: `--accent-copper` (`hsl(28, 65%, 45%)`), `--accent-gold` (`hsl(43, 70%, 52%)`), `--accent-teal` (`hsl(175, 50%, 45%)`).
    - Surfaces: Translucent moss glass (`--bg-card-glass`).
  - ❌ **FORBIDDEN**: Default browser alert boxes (`window.alert()`), unstyled buttons, or primary blue/red bootstrap colors.
- **RULE-13 (Testability & Semantic HTML)**:
  - Every interactive element (file input, dropzone, predict trigger, tabs, filter buttons) must possess a unique, descriptive HTML `id` attribute (e.g., `id="bird-file-input"`, `id="upload-dropzone"`, `id="prediction-result-card"`).
  - Use proper semantic tags (`<header>`, `<main>`, `<section>`, `<article>`, `<footer>`).
- **RULE-14 (Mobile First Responsiveness)**:
  - UI layouts must render cleanly without horizontal scrolling across all screen widths from **360px** (mobile) to **1920px+** (desktop).
  - All touch targets must be at least $44 \times 44\text{ px}$.

---

## 5. Code Quality & Formatting Rules

- **RULE-15 (Python Conventions & Type Annotations)**:
  - Adhere to PEP 8 standards.
  - Include type annotations for all public function signatures (e.g., `def predict_species(image_tensor: np.ndarray) -> Dict[str, Any]:`).
- **RULE-16 (Clean Imports)**:
  - ❌ **FORBIDDEN**: Star imports (`from myapp.views import *`).
  - ✅ **MANDATORY**: Explicit, grouped imports (Standard library $\to$ Third-party $\to$ Local app).
- **RULE-17 (Non-Destructive Refactoring)**:
  - Preserve existing working logic, docstrings, and comments during refactoring unless an explicit replacement is warranted.
  - When deprecating legacy routes (such as user login/signup), redirect gracefully to `/predict/` rather than throwing 404 errors.
