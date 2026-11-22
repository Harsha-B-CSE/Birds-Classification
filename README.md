# 🐦 Birds Classification - Feathered Insights

A Django web application for identifying bird species using a **Swin Tiny Transformer** model trained on the **CUB-200-2011** dataset (200 bird species).

---

## 🚀 Features

- Upload any bird image and get the predicted species instantly
- Swin Transformer (SOTA vision model) for accurate classification
- 200 bird species from the CUB-200-2011 benchmark dataset
- 3-Check verification cascade: bird detection, species classification, and energy-based OOD rejection
- Scientific dossiers from Wikipedia, iNaturalist, GBIF, and NCBI
- Anonymous device-keyed field log at `/history/`
- Animated, responsive UI

---

## 📁 Project Structure

```
Birds-classification/
├── Bird_class/          # Django project settings
├── myapp/               # Main Django app
│   ├── templates/       # HTML templates
│   ├── views.py         # App logic & ML inference
│   ├── urls.py          # URL routing
│   ├── cleaned_classes.json  # 200 bird class names
│   └── swin_model_best.pth   # ⚠️ Not in repo (see below)
├── manage.py
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

The model was trained on the **CUB-200-2011 (Caltech-UCSD Birds 200)** dataset. Additionally, **Indian_Birds_25** dataset is used for regional bird species evaluation.

> ⚠️ Datasets are **NOT included** in this repository due to their large size.

📥 **Download from Kaggle:**
👉 [CUB-200-2011 Dataset on Kaggle](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images)

📥 **Indian Birds Dataset:**
👉 [Indian_Birds_25 on Kaggle](https://www.kaggle.com/datasets/ichhadhari/indian-birds)

After downloading, place the extracted folders as:
```
Birds-classification/
├── CUB_200_2011/
│   ├── images/
│   ├── classes.txt
│   ├── image_class_labels.txt
│   └── ...
└── Indian_Birds_25/
    ├── images/
    └── ...
```

---

## 🧠 Model Architecture & Training Details

### Model Architecture
- **Backbone**: Swin Tiny Transformer (`swin_tiny_patch4_window7_224`) via `timm`
- **Input Resolution**: 224×224 pixels
- **Patch Size**: 4×4, Window Size: 7
- **Output Classes**: 224 (200 CUB-200 + 24 Indian Birds)
- **Dropout**: 0.3 (applied before classification head)
- **Framework**: PyTorch + ONNX Runtime for deployment

### Training Strategy: Two-Stage Transfer Learning

#### Stage 1: Frozen Backbone (Epochs 1–5)
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Only classification head trained** (224-class Linear layer)
- **Head initialization**: Truncated normal (std=0.02), zero bias
- **Weight transfer**: First 200 weights copied from pre-trained CUB-200 head
- **Loss**: CrossEntropy with label_smoothing=0.1

#### Stage 2: Full Fine-Tuning (Epochs 6–30)
- **Learning Rate**: 1e-4 with Cosine Annealing (T_max=25, eta_min=1e-6)
- **Optimizer**: AdamW (weight_decay=1e-4)
- **All layers unfrozen** — full model fine-tuned
- **Augmentation**: Progressive (MixUp α=0.2 in middle epochs)

### Hyperparameters Summary

| Parameter | Value |
|-----------|-------|
| Total Epochs | 30 |
| Warmup Epochs (Stage 1) | 5 |
| Batch Size | 32 |
| Stage 1 LR (Head only) | 1e-3 |
| Stage 2 LR (Full model) | 1e-4 |
| Weight Decay | 1e-4 |
| Image Size | 224 |
| Label Smoothing | 0.1 |
| MixUp Alpha | 0.2 (selective epochs) |
| Dropout | 0.3 |
| Optimizer | AdamW |
| Scheduler (Stage 2) | CosineAnnealingLR |
| Mixed Precision | AMP (GradScaler) |

### Data Pipeline
- **Combined Manifest**: Merges CUB-200-2011 (200 classes) + Indian_Birds_25 (24 classes) → 224 total
- **Class-Balanced Sampling**: WeightedRandomSampler ensures equal class representation per batch
- **Train Transforms**: RandomResizedCrop, HorizontalFlip, ColorJitter, RandAugment, MixUp
- **Eval Transforms**: Resize 256 → CenterCrop 224 → Normalize (ImageNet stats)

### 3-Check Inference Pipeline
The deployed model uses a cascaded verification for robust predictions:

1. **Check 1 — Bird Presence Detector** (EfficientNet-B0): Filters non-bird images (threshold: 0.60)
2. **Check 2 — Swin Transformer Classifier**: 224-class species prediction (ONNX Runtime, ~43ms CPU)
3. **Check 3 — Energy-Based OOD Gate**: Helmholtz free energy scoring to reject unseen/out-of-distribution samples
   - Energy threshold: -3.90
   - Confidence floor: 0.40
   - Temperature scaling: T=1.0

### Performance Metrics
- **Top-1 Accuracy**: 95.33% (on 300 held-out CUB-200 test images)
- **Top-5 Accuracy**: 98.33%
- **ONNX Runtime CPU Latency**: ~43 ms per inference
- **PyTorch/ONNX Numerical Parity**: Max difference 2e-6

---

## 🤖 Pre-trained Model

The trained Swin Transformer model (swin_model_best.pth, ~105 MB) is **NOT included** in the repo.

You can either:
- Train it yourself using the Jupyter notebook: Main model used in project.ipynb
- Or request the model file from the maintainer

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/Harsha-B-CSE/Birds-Classification.git]
cd Birds-classification
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/macOS
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place model files
Place `swin_model_best.pth` inside `myapp/`. Then export the deployment model:
```bash
python export_onnx.py
```
This writes `myapp/swin_b_birds.onnx` (and a copy to the model cache). Verified metrics on 300 held-out CUB-200 test images: **95.33% Top-1**, **98.33% Top-5**, **43 ms** ONNX Runtime CPU forward pass, maximum PyTorch/ONNX difference `0.000002`.

### 5. Configure inference gates (optional)
Copy `.env.example` to `.env` and override if needed:
```bash
BIRD_PRESENCE_THRESHOLD=0.60
OOD_ENERGY_THRESHOLD=-3.90
OOD_CONFIDENCE_FLOOR=0.40
```

### 6. Run migrations, tests & start the server
```bash
python manage.py migrate
python manage.py test
python manage.py runserver
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🛠️ Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Backend     | Django (Python)                     |
| ML Model    | Swin Tiny Transformer (	imm)      |
| Dataset     | CUB-200-2011 (200 species)          |
| Frontend    | HTML, CSS, JavaScript               |
| Enrichment  | Wikipedia, iNaturalist, GBIF, NCBI  |
| History     | Firestore or local device-keyed JSON |

---

## 📓 Notebook

Main model used in project.ipynb — Jupyter notebook used to train the Swin Transformer on CUB-200-2011.

---

## 📄 License

This project is for educational/research purposes.
