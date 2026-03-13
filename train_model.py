"""
=============================================================
  BREAST CANCER WISCONSIN DATASET — ML TRAINING PIPELINE
  Models: Logistic Regression + ANN (Multi-layer Perceptron)
  Dataset: sklearn built-in (569 samples, 30 features)
=============================================================
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("  BREAST CANCER WISCONSIN DATASET — TRAINING PIPELINE")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0=malignant, 1=benign

print(f"\n📊 Dataset Shape      : {X.shape}")
print(f"   Features           : {X.shape[1]}")
print(f"   Samples            : {X.shape[0]}")
print(f"   Malignant (0)      : {(y == 0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"   Benign    (1)      : {(y == 1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"\n   Feature names:\n   {list(data.feature_names)}\n")

# Save feature stats for frontend display
feature_stats = {
    "features": list(data.feature_names),
    "means": X.mean().round(4).tolist(),
    "stds": X.std().round(4).tolist(),
    "mins": X.min().round(4).tolist(),
    "maxs": X.max().round(4).tolist(),
    "target_names": list(data.target_names),
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "class_counts": {"malignant": int((y==0).sum()), "benign": int((y==1).sum())}
}

# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✂  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. MODEL A — LOGISTIC REGRESSION
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("  MODEL A: Logistic Regression")
print("─" * 40)

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    ))
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
lr_acc   = accuracy_score(y_test, lr_preds)
lr_auc   = roc_auc_score(y_test, lr_proba)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=cv, scoring='accuracy')

print(f"  Accuracy     : {lr_acc*100:.2f}%")
print(f"  ROC-AUC      : {lr_auc:.4f}")
print(f"  CV Accuracy  : {lr_cv_scores.mean()*100:.2f}% ± {lr_cv_scores.std()*100:.2f}%")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, lr_preds, target_names=['Malignant','Benign']))

# ─────────────────────────────────────────────
# 4. MODEL B — ARTIFICIAL NEURAL NETWORK (MLP)
# ─────────────────────────────────────────────
print("─" * 40)
print("  MODEL B: ANN (Multi-Layer Perceptron)")
print("─" * 40)

ann_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),   # 3 hidden layers
        activation='relu',
        solver='adam',
        alpha=0.001,                        # L2 regularization
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    ))
])

ann_pipeline.fit(X_train, y_train)
ann_preds = ann_pipeline.predict(X_test)
ann_proba = ann_pipeline.predict_proba(X_test)[:, 1]
ann_acc   = accuracy_score(y_test, ann_preds)
ann_auc   = roc_auc_score(y_test, ann_proba)

ann_cv_scores = cross_val_score(ann_pipeline, X, y, cv=cv, scoring='accuracy')

print(f"  Accuracy     : {ann_acc*100:.2f}%")
print(f"  ROC-AUC      : {ann_auc:.4f}")
print(f"  CV Accuracy  : {ann_cv_scores.mean()*100:.2f}% ± {ann_cv_scores.std()*100:.2f}%")
print(f"  Layers       : Input(30) → 128 → 64 → 32 → Output(2)")
print(f"  Activation   : ReLU (hidden), Softmax (output)")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, ann_preds, target_names=['Malignant','Benign']))

# ─────────────────────────────────────────────
# 5. CHOOSE BEST MODEL → SAVE
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# Save both models
joblib.dump(lr_pipeline, "models/logistic_regression.pkl")
joblib.dump(ann_pipeline, "models/ann_mlp.pkl")

# Pick best by AUC
best_model_name = "ANN" if ann_auc >= lr_auc else "Logistic Regression"
best_pipeline   = ann_pipeline if ann_auc >= lr_auc else lr_pipeline
best_acc        = ann_acc if ann_auc >= lr_auc else lr_acc
best_auc        = ann_auc if ann_auc >= lr_auc else lr_auc

joblib.dump(best_pipeline, "models/best_model.pkl")
print(f"\n🏆  Best Model: {best_model_name} (AUC={best_auc:.4f})")
print(f"    Saved → models/best_model.pkl")

# ─────────────────────────────────────────────
# 6. CONFUSION MATRIX DATA
# ─────────────────────────────────────────────
cm_ann = confusion_matrix(y_test, ann_preds).tolist()
cm_lr  = confusion_matrix(y_test, lr_preds).tolist()

# ROC curve data (ANN)
fpr, tpr, _ = roc_curve(y_test, ann_proba)

# Feature importances (LR coefficients as proxy)
scaler     = lr_pipeline.named_steps['scaler']
lr_model   = lr_pipeline.named_steps['model']
coef_abs   = np.abs(lr_model.coef_[0])
top10_idx  = np.argsort(coef_abs)[::-1][:10]
top10_feat = [(data.feature_names[i], round(float(coef_abs[i]), 4)) for i in top10_idx]

# ─────────────────────────────────────────────
# 7. SAVE METADATA FOR FRONTEND
# ─────────────────────────────────────────────
metadata = {
    "dataset": {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository / sklearn",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features": list(data.feature_names),
        "class_names": ["Malignant", "Benign"],
        "class_counts": {"malignant": int((y==0).sum()), "benign": int((y==1).sum())},
        "means": X.mean().round(6).tolist(),
        "stds": X.std().round(6).tolist(),
        "mins": X.min().round(6).tolist(),
        "maxs": X.max().round(6).tolist()
    },
    "models": {
        "logistic_regression": {
            "accuracy": round(float(lr_acc), 4),
            "auc": round(float(lr_auc), 4),
            "cv_mean": round(float(lr_cv_scores.mean()), 4),
            "cv_std": round(float(lr_cv_scores.std()), 4),
            "confusion_matrix": cm_lr
        },
        "ann_mlp": {
            "accuracy": round(float(ann_acc), 4),
            "auc": round(float(ann_auc), 4),
            "cv_mean": round(float(ann_cv_scores.mean()), 4),
            "cv_std": round(float(ann_cv_scores.std()), 4),
            "confusion_matrix": cm_ann,
            "architecture": "30 → 128 → 64 → 32 → 2"
        }
    },
    "best_model": best_model_name,
    "top10_features": top10_feat,
    "roc_curve": {
        "fpr": [round(float(v), 4) for v in fpr[:50]],
        "tpr": [round(float(v), 4) for v in tpr[:50]],
        "auc": round(float(ann_auc), 4)
    }
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅  Metadata saved → models/metadata.json")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE — ALL MODELS SAVED")
print("=" * 60)
print("\n  Run the web app with:  python app.py")
print("  Then open: http://localhost:5000\n")
