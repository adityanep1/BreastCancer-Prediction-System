"""
=============================================================
  FLASK WEB APPLICATION — Breast Cancer Predictor
  Serves prediction API + frontend dashboard
=============================================================
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import json
import numpy as np
import os
import sys

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ─────────────────────────────────────────────
# Load models & metadata on startup
# ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_assets():
    """Load trained models and metadata."""
    assets = {}
    
    # Load metadata
    meta_path = os.path.join(MODELS_DIR, 'metadata.json')
    if not os.path.exists(meta_path):
        print("⚠  metadata.json not found. Run train_model.py first!")
        sys.exit(1)
    
    with open(meta_path) as f:
        assets['metadata'] = json.load(f)
    
    # Load models
    for name, fname in [
        ('best',  'best_model.pkl'),
        ('lr',    'logistic_regression.pkl'),
        ('ann',   'ann_mlp.pkl')
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            assets[name] = joblib.load(path)
            print(f"✅  Loaded model: {fname}")
        else:
            print(f"⚠  Model not found: {fname}")
    
    return assets

ASSETS = load_assets()
METADATA = ASSETS['metadata']
FEATURES  = METADATA['dataset']['features']

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', metadata=METADATA)


@app.route('/api/metadata')
def get_metadata():
    """Return full training metadata."""
    return jsonify(METADATA)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts JSON with 'features' (list of 30 floats)
    and optional 'model' ('ann' | 'lr' | 'best').
    Returns prediction + probability + explanation.
    """
    try:
        body       = request.get_json(force=True)
        values     = body.get('features', [])
        model_key  = body.get('model', 'best')

        if len(values) != 30:
            return jsonify({'error': f'Expected 30 features, got {len(values)}'}), 400

        X = np.array(values, dtype=float).reshape(1, -1)

        # Select model
        model = ASSETS.get(model_key, ASSETS['best'])

        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()

        # Map: sklearn target 0=malignant, 1=benign
        label     = METADATA['dataset']['class_names'][pred]
        confidence = round(proba[pred] * 100, 2)
        malignant_prob = round(proba[0] * 100, 2)
        benign_prob    = round(proba[1] * 100, 2)

        # Risk level
        if pred == 0:  # malignant
            risk = "HIGH"
        elif proba[0] > 0.35:
            risk = "MODERATE"
        else:
            risk = "LOW"

        # Feature contribution hints (using feature means as reference)
        means = METADATA['dataset']['means']
        stds  = METADATA['dataset']['stds']
        z_scores = [(values[i] - means[i]) / (stds[i] + 1e-9) for i in range(30)]
        top5_idx = sorted(range(30), key=lambda i: abs(z_scores[i]), reverse=True)[:5]
        notable_features = [
            {
                "name": FEATURES[i],
                "value": round(values[i], 4),
                "z_score": round(z_scores[i], 2),
                "direction": "above" if z_scores[i] > 0 else "below",
                "mean": round(means[i], 4)
            }
            for i in top5_idx
        ]

        return jsonify({
            'prediction': pred,
            'label': label,
            'confidence': confidence,
            'malignant_prob': malignant_prob,
            'benign_prob': benign_prob,
            'risk': risk,
            'model_used': model_key,
            'notable_features': notable_features
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample/<int:label>')
def get_sample(label):
    """
    Return a sample patient input.
    label=0 → malignant-ish values, label=1 → benign-ish values
    Based on dataset class means.
    """
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    data = load_breast_cancer()
    X    = pd.DataFrame(data.data, columns=data.feature_names)
    y    = data.target

    subset = X[y == label]
    sample = subset.sample(1, random_state=np.random.randint(0, 100))
    return jsonify({
        'features': sample.values[0].round(4).tolist(),
        'feature_names': list(data.feature_names),
        'true_label': int(label),
        'true_class': METADATA['dataset']['class_names'][label]
    })


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Accepts JSON with 'samples' (list of feature lists).
    Returns list of predictions.
    """
    try:
        body    = request.get_json(force=True)
        samples = body.get('samples', [])
        model   = ASSETS.get(body.get('model', 'best'), ASSETS['best'])

        X     = np.array(samples, dtype=float)
        preds = model.predict(X).tolist()
        probas = model.predict_proba(X).tolist()

        results = []
        for i, (p, pr) in enumerate(zip(preds, probas)):
            results.append({
                'index': i,
                'prediction': p,
                'label': METADATA['dataset']['class_names'][p],
                'confidence': round(max(pr) * 100, 2)
            })
        return jsonify({'results': results, 'count': len(results)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  🔬 BREAST CANCER PREDICTOR — Flask Server")
    print("=" * 55)
    print("  🌐 Open: http://localhost:5000")
    print("  📡 API : http://localhost:5000/api/metadata")
    print("  🛑 Stop: Ctrl+C\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
