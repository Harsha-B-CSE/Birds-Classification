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
