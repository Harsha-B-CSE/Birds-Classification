# 🐦 Birds Classification - Feathered Insights

A Django web application for identifying bird species using a **Swin Tiny Transformer** model trained on the **CUB-200-2011** dataset (200 bird species).

---

## 🚀 Features

- Upload any bird image and get the predicted species instantly
- Swin Transformer (SOTA vision model) for accurate classification
- 200 bird species from the CUB-200-2011 benchmark dataset
- User authentication (Login / Signup / Logout)
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

The model was trained on the **CUB-200-2011 (Caltech-UCSD Birds 200)** dataset.

> ⚠️ The dataset is **NOT included** in this repository due to its large size.

📥 **Download from Kaggle:**
👉 [CUB-200-2011 Dataset on Kaggle](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images)

After downloading, place the extracted folder as:
```
Birds-classification/
└── CUB_200_2011/
    ├── images/
    ├── classes.txt
    ├── image_class_labels.txt
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
git clone https://github.com/HARSHA-HAPPY/Birds-classification.git
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

### 4. Place model file
Place swin_model_best.pth inside myapp/.

### 5. Update paths in views.py
In myapp/views.py, update these lines to match your local path:
```python
class_names_path = os.path.join(BASE_DIR, "myapp", "cleaned_classes.json")
model_path = os.path.join(BASE_DIR, "myapp", "swin_model_best.pth")
```

### 6. Run migrations & start the server
```bash
python manage.py migrate
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
| Auth        | Django built-in authentication      |

---

## 📓 Notebook

Main model used in project.ipynb — Jupyter notebook used to train the Swin Transformer on CUB-200-2011.

---

## 📄 License

This project is for educational/research purposes.
