# 🔬 OnchoScan — Breast Cancer AI Predictor

> **Real biological dataset · ANN + Logistic Regression · Flask web app · Full VS Code setup**

---

## 📋 Project Overview

This project trains two machine learning models on the **Breast Cancer Wisconsin Diagnostic Dataset** (UCI / sklearn) and deploys them as an interactive web application where you can enter biopsy measurements and get an instant malignant/benign prediction.

### Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository (built into sklearn)
- **Samples:** 569 patients
- **Features:** 30 numeric measurements from fine needle aspirate (FNA) biopsies
- **Target:** Malignant (0) or Benign (1)

### Models Trained
| Model | Architecture | Accuracy | AUC |
|-------|-------------|----------|-----|
| Logistic Regression | Linear + L2 reg | ~95% | ~0.99 |
| ANN (MLP) | 30 → 128 → 64 → 32 → 2 | ~97% | ~0.99 |

---

## 📁 Project Structure

```
breast_cancer_predictor/
├── train_model.py          ← Step 1: Train ML models
├── app.py                  ← Step 2: Run Flask web server
├── requirements.txt        ← Python dependencies
├── README.md
├── models/                 ← Created after training
│   ├── best_model.pkl
│   ├── ann_mlp.pkl
│   ├── logistic_regression.pkl
│   └── metadata.json
├── templates/
│   └── index.html          ← Web app frontend
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

---

## 🚀 Quick Start (VS Code)

### Prerequisites
- Python 3.8 or higher
- VS Code with Python extension installed

---

### Step 1: Open in VS Code

```bash
# Open the project folder in VS Code
code breast_cancer_predictor
```

Or: `File → Open Folder → select breast_cancer_predictor`

---

### Step 2: Create Virtual Environment

Open VS Code Terminal (`Ctrl+`` ` or `Terminal → New Terminal`) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: scikit-learn, flask, flask-cors, numpy, pandas, joblib.

> ⏱ Takes 1–3 minutes depending on your connection.

---

### Step 4: Train the Models

```bash
python train_model.py
```

Expected output:
```
============================================================
  BREAST CANCER WISCONSIN DATASET — TRAINING PIPELINE
============================================================

📊 Dataset Shape      : (569, 30)
   ...

MODEL A: Logistic Regression
  Accuracy     : 97.37%
  ROC-AUC      : 0.9985

MODEL B: ANN (Multi-Layer Perceptron)
  Accuracy     : 97.37%
  ROC-AUC      : 0.9979

🏆  Best Model: Logistic Regression (AUC=0.9985)
    Saved → models/best_model.pkl

✅  Metadata saved → models/metadata.json
```

This creates the `models/` folder with all saved model files.

---

### Step 5: Launch the Web App

```bash
python app.py
```

Expected output:
```
=======================================================
  🔬 BREAST CANCER PREDICTOR — Flask Server
=======================================================
  🌐 Open: http://localhost:5000
  📡 API : http://localhost:5000/api/metadata
  🛑 Stop: Ctrl+C
```

---

### Step 6: Open in Browser

Navigate to: **http://localhost:5000**

---

## 🌐 Web App Features

### Predictor Tab
- Enter all 30 biopsy feature values manually
- **Or** click "Malignant Case" / "Benign Case" to auto-fill with a real patient sample
- Choose model: ANN, Logistic Regression, or Best (auto)
- Click **Run Prediction** to see:
  - Malignant / Benign classification
  - Confidence percentage
  - Risk level (High / Moderate / Low)
  - Probability bar chart
  - 5 most influential features (with z-scores)

### Model Dashboard Tab
- Accuracy, AUC, cross-validation scores for both models
- Confusion matrices
- ROC Curve (interactive Chart.js)
- Top 10 most discriminative features

### About Tab
- Dataset description
- ANN architecture diagram
- Feature explanations
- Tech stack

---

## 🔌 REST API Endpoints

### `GET /api/metadata`
Returns all training metadata (dataset stats, model performance, feature info).

### `POST /api/predict`
```json
{
  "features": [17.99, 10.38, 122.8, 1001, 0.1184, ...],
  "model": "best"  // "best" | "ann" | "lr"
}
```
Returns:
```json
{
  "prediction": 0,
  "label": "Malignant",
  "confidence": 99.2,
  "malignant_prob": 99.2,
  "benign_prob": 0.8,
  "risk": "HIGH",
  "notable_features": [...]
}
```

### `GET /api/sample/0`
Returns a random malignant patient's feature values (for demo).

### `GET /api/sample/1`
Returns a random benign patient's feature values (for demo).

### `POST /api/batch_predict`
```json
{
  "samples": [[...30 features...], [...30 features...]],
  "model": "ann"
}
```

---

## 📊 The 30 Features (10 measurements × 3 statistics)

Each measurement is computed as **mean**, **standard error (SE)**, and **worst** (largest):

| # | Measurement | Description |
|---|-------------|-------------|
| 1 | Radius | Mean distance from center to perimeter |
| 2 | Texture | Standard deviation of gray-scale values |
| 3 | Perimeter | Perimeter of nucleus |
| 4 | Area | Area of nucleus |
| 5 | Smoothness | Local variation in radius lengths |
| 6 | Compactness | perimeter² / area − 1.0 |
| 7 | Concavity | Severity of concave portions |
| 8 | Concave points | Number of concave portions |
| 9 | Symmetry | Symmetry of nucleus |
| 10 | Fractal dimension | Coastline approximation − 1 |

---

## 🏗 ANN Architecture

```
Input Layer  (30 neurons)
     ↓  [StandardScaler normalization]
Hidden Layer 1 (128 neurons, ReLU, Dropout via L2)
     ↓
Hidden Layer 2 (64 neurons, ReLU)
     ↓
Hidden Layer 3 (32 neurons, ReLU)
     ↓
Output Layer  (2 neurons, Softmax → probabilities)
```

**Training Config:**
- Optimizer: Adam (lr=0.001, adaptive)
- Loss: Cross-Entropy
- L2 Regularization: α = 0.001
- Batch size: 32
- Early stopping: patience=20
- Validation split: 15%
- Max epochs: 500

---

## 🛠 VS Code Tips

### Recommended Extensions
- **Python** (Microsoft) — Linting, debugging
- **Pylance** — Type checking
- **REST Client** — Test API endpoints

### Debugging (Run & Debug)
Create `.vscode/launch.json`:
```json
{
  "configurations": [
    {
      "name": "Train Models",
      "type": "python",
      "request": "launch",
      "program": "train_model.py"
    },
    {
      "name": "Flask App",
      "type": "python",
      "request": "launch",
      "program": "app.py",
      "env": { "FLASK_ENV": "development" }
    }
  ]
}
```

### Tasks (Optional)
Create `.vscode/tasks.json`:
```json
{
  "tasks": [
    {
      "label": "Train & Run",
      "type": "shell",
      "command": "python train_model.py && python app.py",
      "group": "build"
    }
  ]
}
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in venv |
| `metadata.json not found` | Run `python train_model.py` first |
| Port 5000 already in use | Change port in `app.py`: `app.run(port=5001)` |
| `Address already in use` | Kill existing Flask: `pkill -f "python app.py"` |
| Chart.js not loading | Check internet connection (CDN dependency) |
| Fonts not loading | Check internet connection (Google Fonts) |

---

## 📚 References

- Wolberg, W.H., Street, W.N., Mangasarian, O.L. (1993). *Machine learning techniques to diagnose breast cancer from image-processed nuclear features of fine needle aspirates.* Cancer Letters, 77(2–3), 163–171.
- UCI Machine Learning Repository: [Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

*Built for educational and research purposes. Not a medical diagnostic tool.*
