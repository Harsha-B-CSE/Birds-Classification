# 🦅 Birds Classification — Architecture & Roadmap (v3)

> **Goal:** Upload a bird image → Get species name + full scientific profile in <2 sec. Indian birds supported. Robust OOD. No login. Public access via Firebase link. Firebase is the primary backend — set up NOW.

---

## 🔥 PRIORITY CHANGE: Firebase First

> Firebase moves from "future optional" → **Phase 1 setup NOW**.
> It handles public access, prediction history sync, and model file hosting.
> The local ML pipeline is unchanged — Firebase slots in as the data/hosting layer only.

---

## 1. 🏗️ System Architecture (Firebase-integrated)

```
User (anyone with the link — college, tech people, anyone)
    │
    ▼
Firebase Hosting  ←──────────────────────────────────────────┐
    │ (rewrites dynamic requests)                             │
    ▼                                                         │
Cloud Run (Django + ONNX)                        Firebase Services
    │                                                  │
    ├── Check 1: Bird Detector (ONNX)           Firestore DB
    ├── Check 2: Species Classifier (ONNX)      └── predictions/{device_id}/
    ├── Check 3: Energy OOD + Abstain gate       Firebase Storage
    └── Info Aggregator (Wikipedia/GBIF/NCBI)   └── models/ (ONNX files)
              │
              ▼
         Result Page
   (history stored in Firestore, device-keyed)
```

### Stack

| Layer | Technology |
|-------|-----------|
| Hosting | Firebase Hosting |
| Backend | Django on Cloud Run (Docker) |
| ML Inference | ONNX Runtime (CPU, Cloud Run) |
| History / Data | Firestore (device-based, no account) |
| Model File Storage | Firebase Storage (or Google Drive for dev) |
| Local Cache | `D:\BirdsCache\` (≤5GB, training PC only) |
| Info APIs | Wikipedia + iNaturalist + GBIF + NCBI (free) |
| Auth | None — anonymous device ID only |

---

## 2. 🔥 Firebase Setup Guide (Do This Now)

### Step 1 — Create Firebase Project

```
1. Go to: https://console.firebase.google.com
2. Click "Add project" → Name: "birds-classification"
3. Enable Google Analytics → Continue
4. Project created ✅
```

### Step 2 — Enable Services in Console

```
Firestore Database:
  Build → Firestore Database → Create database
  → Start in TEST mode (we tighten rules later)
  → Region: asia-south1 (Mumbai — lowest latency for India)

Firebase Storage:
  Build → Storage → Get started
  → Start in TEST mode
  → Same region: asia-south1

Firebase Hosting:
  Build → Hosting → Get started
  → We configure this after Docker step
```

### Step 3 — Get Credentials (Service Account)

```
Project Settings (⚙️) → Service Accounts
→ "Generate new private key"
→ Save as: serviceAccountKey.json
→ Place at: Birds-classification/ (root, NEVER commit this)
```

Add to `.gitignore` immediately:
```
serviceAccountKey.json
.env
firebase_credentials.json
```

### Step 4 — Install Firebase Tools & Admin SDK

```bash
# Firebase CLI (one-time)
npm install -g firebase-tools
firebase login

# Python SDK
pip install firebase-admin
```

### Step 5 — Initialize Firebase in Django

```python
# Bird_class/settings.py — add this block
import firebase_admin
from firebase_admin import credentials, firestore, storage as fb_storage
from pathlib import Path

FIREBASE_ENABLED = True   # ← flip to False for purely local dev
FIREBASE_CRED_PATH = BASE_DIR / "serviceAccountKey.json"
FIREBASE_STORAGE_BUCKET = "birds-classification.appspot.com"

if FIREBASE_ENABLED and not firebase_admin._apps:
    cred = credentials.Certificate(str(FIREBASE_CRED_PATH))
    firebase_admin.initialize_app(cred, {
        "storageBucket": FIREBASE_STORAGE_BUCKET
    })

# Accessible anywhere via:
# from firebase_admin import firestore
# db = firestore.client()
```

### Step 6 — Firestore Data Schema

```
Collection: predictions
  Document: {device_id}          ← browser fingerprint, no account needed
    Sub-collection: history
      Document: {timestamp}
        - species: "Indian Peafowl"
        - scientific_name: "Pavo cristatus"
        - confidence: 0.943
        - top5: [{name, prob}, ...]
        - image_hash: "md5..."
        - timestamp: Firestore.SERVER_TIMESTAMP
        - check1_passed: true
        - energy_score: -12.4

Collection: bird_info_cache
  Document: {species_slug}        ← cached info, shared across all users
    - wiki_summary: "..."
    - habitat: "..."
    - lifespan_wild: "10-25 years"
    - ncbi_taxon_id: 9049
    - iucn_status: "Least Concern"
    - last_updated: timestamp
```

### Step 7 — Firebase Storage for Model Files

```python
# Upload ONNX models to Firebase Storage (one-time, from training PC)
from firebase_admin import storage

bucket = storage.bucket()

def upload_model(local_path, remote_name):
    blob = bucket.blob(f"models/{remote_name}")
    blob.upload_from_filename(local_path)
    print(f"Uploaded {remote_name} to Firebase Storage")

# Usage (run once after training):
upload_model("D:/BirdsCache/models/swin_b_birds.onnx", "swin_b_birds.onnx")
upload_model("D:/BirdsCache/models/bird_detector.onnx", "bird_detector.onnx")
```

```python
# Download model on first startup (Cloud Run or new machine)
def download_model_if_missing(remote_name, local_path):
    if not local_path.exists():
        blob = bucket.blob(f"models/{remote_name}")
        blob.download_to_filename(str(local_path))
        print(f"Downloaded {remote_name} from Firebase Storage")
```

### Step 8 — Dockerize Django for Cloud Run

**Dockerfile** (place in project root):
```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2",
     "--timeout", "120", "Bird_class.wsgi:application"]
```

**.dockerignore:**
```
venv/
__pycache__/
*.pyc
serviceAccountKey.json
.git
D:/BirdsCache/
CUB_200_2011/
*.pth
*.onnx
db.sqlite3
```

### Step 9 — Deploy to Cloud Run + Firebase Hosting

```bash
# 1. Deploy backend to Cloud Run
gcloud run deploy birds-classification \
  --source . \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 120

# 2. Get the Cloud Run URL (copy it)
# e.g. https://birds-classification-xxxx-el.a.run.app

# 3. Configure Firebase Hosting rewrite
# In firebase.json:
```

**firebase.json:**
```json
{
  "hosting": {
    "public": "public",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [
      {
        "source": "**",
        "run": {
          "serviceId": "birds-classification",
          "region": "asia-south1"
        }
      }
    ]
  },
  "firestore": {
    "rules": "firestore.rules",
    "indexes": "firestore.indexes.json"
  },
  "storage": {
    "rules": "storage.rules"
  }
}
```

```bash
# 4. Deploy hosting
firebase init hosting   # follow prompts
firebase deploy         # → gives you: https://birds-classification.web.app
```

> Your public link: **`https://birds-classification.web.app`**
> Anyone — college, tech people, anyone — can access it.

---

## 3. 🖥️ Training → Max Performance → ONNX Export

> [!IMPORTANT]
> Train and grind for max performance first. ONNX export happens **only after** validation confirms the model has peaked. Never commit to export prematurely.

### Training Pipeline (RTX 3060 12GB)

```
Step 1 → Train Swin-B (ImageNet-22K weights)
Step 2 → Validate — check Top-1, Top-5, per-class accuracy
Step 3 → Tune augmentations progressively (see Section 6)
Step 4 → Confirm performance plateau (early stopping triggered)
Step 5 → Export .onnx (FP32)
Step 6 → Benchmark on laptop CPU
Step 7 → Optional: INT8 quantization if latency > 2s
```

---

## 4. 🎨 UI/UX — Natural Metallic Theme

| Element | Color | HSL |
|---------|-------|-----|
| Background | Deep forest dark | `hsl(150, 20%, 8%)` |
| Primary | Copper / Bronze | `hsl(28, 65%, 45%)` |
| Secondary | Aged Gold | `hsl(43, 70%, 52%)` |
| Text | Warm ivory | `hsl(40, 30%, 90%)` |
| Cards | Dark moss glass | `rgba(30, 50, 35, 0.7)` |
| Borders | Oxidized copper | `hsl(28, 40%, 35%)` |
| Highlight | Feather teal | `hsl(175, 50%, 45%)` |

**Pages (no login, no signup):**
```
/           → Landing + upload (instant, no auth wall)
/predict/   → Upload + rich result page  ← MAIN PAGE
/history/   → Prediction history (from Firestore, device-keyed)
/about/     → Model info & tech specs
```

---

## 5. ⚡ Response Time < 2 Seconds

| Step | Target | Strategy |
|------|--------|----------|
| Image upload | ~50ms | Client compress to <1MB |
| Check 1: Bird detector | ~10ms | MobileNetV3 ONNX |
| Check 2: Species classifier | ~400ms | Swin-B ONNX, CPU |
| Check 3: Energy OOD | ~1ms | Numpy, no model call |
| Firestore history write | async | Non-blocking, after response |
| Bird info (Firestore cache hit) | ~80ms | Shared cache across all users |
| Bird info (cold API fetch) | ~600ms | Wikipedia + GBIF |
| **Total (cache hit)** | **~540ms** | ✅ Under 2s |
| **Total (cold)** | **~1.1s** | ✅ Under 2s |

> Firestore `bird_info_cache` is shared globally — once one user fetches a species, all future users get it from Firestore instantly.

---

## 6. 🧪 Augmentation — Progressive Efficient Strategy

> Instead of manual trial-and-error, follow a **structured progressive schedule** using timm's battle-tested presets. One change at a time, tracked by validation loss.

### Progressive Schedule

```
Baseline (no augmentation beyond resize+normalize)
    → Train 10 epochs → record val_acc  [CHECKPOINT A]
    ↓
Add RandAugment (magnitude=9, num_ops=2)  ← timm default preset
    → Train 10 more epochs → record val_acc  [CHECKPOINT B]
    → If val_acc improves → keep, else → rollback to A
    ↓
Add MixUp (alpha=0.2)  ← start conservative
    → Train 10 more epochs → record val_acc  [CHECKPOINT C]
    → If val_acc improves → keep, else → rollback to B
    ↓
Optionally add CutMix (prob=0.3)  ← only if dataset > 50k images
    → Validate → keep or rollback
    ↓
Fine-tune magnitudes on best combination using validation curve
```

### timm Config (auto-handles presets — no manual tuning needed)

```python
from timm.data import create_transform
from timm.data.mixup import Mixup

# Train transform — start here (conservative)
train_transform = create_transform(
    input_size=224,
    is_training=True,
    auto_augment="rand-m9-n2-mstd0.5",  # RandAugment timm preset
    interpolation="bicubic",
    re_prob=0.25,          # Random erasing probability
    re_mode="pixel",
)

# MixUp/CutMix — add after RandAugment confirms improvement
mixup_fn = Mixup(
    mixup_alpha=0.2,       # Start conservative
    cutmix_alpha=0.0,      # Start CutMix disabled
    cutmix_minmax=None,
    prob=0.5,
    switch_prob=0.0,       # No CutMix yet
    mode="batch",
    label_smoothing=0.1,
    num_classes=num_classes,
)
```

### Verification: Stop Adding If This Happens

```
Watch for these signals → rollback immediately:
  ✗ val_loss increases for 3+ epochs after adding new aug
  ✗ val_acc drops more than 1% vs previous checkpoint
  ✗ train_acc >> val_acc gap widens (overfitting to augmentations)
```

---

## 7. 📊 Model Report — Finalized Model Only

> [!IMPORTANT]
> **Report files are generated ONLY from the finalized, best-checkpoint model.** During training, only lightweight verification logs are written (for monitoring). Final reports are a separate post-training step.

### During Training — Verification Logs (lightweight, real-time)

```
D:\BirdsCache\logs\training_live\
├── epoch_log.csv           ← epoch, train_loss, val_loss, val_acc (appended live)
├── lr_schedule.csv         ← LR per epoch
└── aug_ablation.csv        ← val_acc after each augmentation stage
```

### After Training — Final Report Generation (one-time, from best .pth)

```python
# Run this script ONLY on the finalized model checkpoint
# python generate_report.py --checkpoint D:\BirdsCache\models\swin_b_best.pth

Outputs → D:\BirdsCache\logs\final_report\
├── training_curves.png          ← Full loss + acc curves (all epochs)
├── confusion_matrix.png         ← Full class heatmap
├── per_class_accuracy.csv       ← Worst / best 20 species
├── ood_roc_curve.png            ← OOD detection at different thresholds
├── inference_benchmark.json     ← GPU .pth vs CPU ONNX FP32 vs INT8 latency
├── resolution_ablation.json     ← 224 vs 384 tradeoff
└── model_comparison.md          ← Swin-T baseline vs Swin-B + tweaks table
```

> No final report files are generated during intermediate training runs — only `epoch_log.csv` updates. This keeps D:\\ storage clean and prevents confusion between draft and final results.

---

## 8. 💾 Cache & Storage — D:\ Capped at 5GB

**Total budget: 5GB on D:\**

```
D:\BirdsCache\                                    TARGET SIZE
├── models\                                       ← ~500MB max
│   ├── swin_b_birds.onnx         ~350MB          Main classifier
│   ├── swin_b_int8.onnx          ~90MB           Optional quantized
│   └── bird_detector.onnx        ~8MB            Binary gate
│
├── bird_info\                                    ← ~50MB max
│   └── {species_slug}.json       ~5KB each       ~200 species = ~1MB
│   (TTL: 30 days, then refetch from Firestore)
│
├── predictions\                                  ← ~200MB cap (LRU eviction)
│   └── {md5_hash}.json           ~2KB each       LRU: delete oldest when >200MB
│
└── logs\                                         ← ~300MB total
    ├── training_live\            ~5MB            Epoch logs (live)
    └── final_report\             ~250MB          Generated once, final model
                                  ─────────────────────────
                    TOTAL BUDGET: ~1.05GB used / 5GB limit  ✅ Safe
```

### Storage Rules

| Folder | Limit | Action on Breach |
|--------|-------|-----------------|
| `models\` | 500MB | Manual — don't store multiple .onnx versions simultaneously |
| `bird_info\` | 50MB | Auto TTL 30 days → Firestore is the source of truth |
| `predictions\` | 200MB | LRU eviction: delete 20% oldest when limit hit |
| `logs\training_live\` | 10MB | Overwrite on new training run |
| `logs\final_report\` | 300MB | Keep — this is your report archive |

### Auto-cleanup Utility

```python
# myapp/cache_manager.py
import os, json
from pathlib import Path
from django.conf import settings

def evict_prediction_cache(max_size_mb=200):
    """LRU eviction: delete oldest cached predictions if folder exceeds limit."""
    cache_dir = settings.PRED_CACHE_DIR
    entries = sorted(cache_dir.glob("*.json"), key=os.path.getmtime)
    total_mb = sum(f.stat().st_size for f in entries) / 1e6

    while total_mb > max_size_mb and entries:
        oldest = entries.pop(0)
        total_mb -= oldest.stat().st_size / 1e6
        oldest.unlink()

def get_cache_usage_mb():
    """Returns current D:\BirdsCache usage in MB."""
    total = sum(
        f.stat().st_size
        for f in settings.CACHE_DIR.rglob("*")
        if f.is_file()
    )
    return total / 1e6
```

---

## 9. 🎯 Robust Output

```
┌─────────────────────────────────────────────────┐
│  🦜 Indian Peafowl          ✅ Bird Confirmed    │
│  Confidence: 94.3%                               │
│                                                  │
│  Top 5 ─────────────────────────────────────    │
│  1. Indian Peafowl       94.3% ████████████     │
│  2. Green Peafowl         3.1% ███              │
│  3. Congo Peafowl         1.2% █                │
│                                                  │
│  Scientific  ── Pavo cristatus ─────────────    │
│  Taxonomy    ── Aves → Galliformes → Phasianidae │
│  IUCN Status ── Least Concern 🟢                │
│                                                  │
│  Origin & Habitat ──────────────────────────    │
│  Native: Indian subcontinent                    │
│  Habitat: Tropical forests, grasslands          │
│                                                  │
│  Lifespan ───────────────────────────────────   │
│  Wild: 10–25 yrs  │  Captivity: up to 40 yrs   │
│  [Chart.js bar chart]                           │
│                                                  │
│  DNA & Genetics ─────────────────────────────   │
│  Genome: ~1.0 Gb  │  Chromosomes: 2n=76        │
│  NCBI Taxon ID: 9049  [→ GenBank]              │
└─────────────────────────────────────────────────┘
```

---

## 10. 🚫 Three-Check OOD System

| Check | Method | Trigger | Message |
|-------|--------|---------|---------|
| **1** | MobileNetV3 binary | Bird prob < 60% | "🚫 Not a bird." |
| **2** | Energy-based OOD | Energy > threshold | "🔍 Unknown species — not in training set." |
| **3** | Confidence gate | Top-1 < 40% | "⚠️ Insufficient evidence. Try clearer photo." |

```python
# Energy-based OOD (no extra model needed)
energy = -torch.log(torch.sum(torch.exp(logits))).item()
# Threshold tuned on held-out OOD images (cars, buildings, unseen birds)
```

---

## 11. 🇮🇳 Indian Birds Data

| Source | Species | Images | Priority |
|--------|---------|--------|----------|
| Kaggle: 25 Indian Species | 25 | ~22,600 | ✅ Start here |
| CUB-200-2011 | 200 | ~11,788 | ✅ Combine |
| iNaturalist API (India, research-grade) | 500+ | Millions | Phase 2 |
| BirdCLEF 2024 (Western Ghats) | 182 | Large | Phase 2 |

---

## 🗺️ Revised Development Phases

```
─────────────── PHASE 1: Firebase + Foundation ───────────────
  🔥 Create Firebase project (console.firebase.google.com)
  🔥 Enable Firestore (asia-south1), Storage, Hosting
  🔥 Download serviceAccountKey.json → add to .gitignore
  🔥 Install: firebase-admin, firebase-tools (npm)
  🔥 Integrate firebase-admin into settings.py + views.py
  🔥 Firestore schema: predictions/{device_id}/history/
  ✅ Remove login/signup pages
  ✅ Fix all hardcoded paths → settings.py
  ✅ Create D:\BirdsCache\ folder structure (auto on startup)
  ✅ Fix duplicate imports in views.py
  ✅ Cache manager with LRU eviction (≤5GB total)
  ✅ Redesign UI — natural metallic theme

─────────────── PHASE 2: Model Training (RTX 3060) ──────────
  🔲 Download: 25 Indian species (Kaggle) + CUB-200-2011
  🔲 Train Swin-B (ImageNet-22K) with progressive aug schedule
  🔲 Monitor via epoch_log.csv (verification only, not report)
  🔲 Ablation: 224 vs 384 resolution
  🔲 Confirm performance plateau → export .onnx
  🔲 Train MobileNetV3 binary bird detector
  🔲 Tune OOD energy threshold on held-out images
  🔲 Upload ONNX to Firebase Storage

─────────────── PHASE 3: Rich Info & Polish ──────────────────
  🔲 Wikipedia REST API → bird descriptions
  🔲 iNaturalist API → habitat / origin / range
  🔲 NCBI Taxonomy API → DNA / chromosomes
  🔲 Cache all API results in Firestore bird_info_cache
  🔲 Chart.js lifespan visualization
  🔲 Collapsible taxonomy tree
  🔲 Firestore-backed prediction history page

─────────────── PHASE 4: Deploy + Report ────────────────────
  🔲 Dockerize Django (Dockerfile + .dockerignore)
  🔲 Deploy to Cloud Run (asia-south1)
  🔲 Deploy Firebase Hosting → get public link
  🔲 Run generate_report.py on finalized model → D:\BirdsCache\logs\final_report\
  🔲 model_comparison.md, confusion matrix, OOD ROC, benchmark table
  🔲 Tighten Firestore + Storage security rules for production
```

---

> [!IMPORTANT]
> **Start Phase 1 now.** Firebase setup takes ~30 min and unblocks public sharing immediately. The ML pipeline is independent — you can train while the Firebase layer is already live.

> [!WARNING]
> **Never commit `serviceAccountKey.json` to GitHub.** It gives full admin access to your Firebase project. Keep it only in the project root locally and add it to `.gitignore` immediately.

> [!NOTE]
> **D:\BirdsCache\ stays on your training PC only.** Cloud Run uses Firebase Storage to fetch models at startup. The 5GB limit applies only to your local training machine cache.
