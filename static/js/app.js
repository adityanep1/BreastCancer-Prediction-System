/* ═══════════════════════════════════════════════════════
   OnchoScan — Frontend Application Logic
   Handles: tabs, form generation, API calls, charts
═══════════════════════════════════════════════════════ */

// ─────────────────────────────────────────────
// GLOBAL STATE
// ─────────────────────────────────────────────
let METADATA = null;
let FEATURES  = [];
let rocChartInstance = null;

const FEATURE_HINTS = {
  'mean radius': 'µm',
  'mean texture': 'SD',
  'mean perimeter': 'µm',
  'mean area': 'µm²',
  'mean smoothness': '',
  'mean compactness': '',
  'mean concavity': '',
  'mean concave points': '',
  'mean symmetry': '',
  'mean fractal dimension': '',
};

// ─────────────────────────────────────────────
// TABS
// ─────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;

    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    btn.classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');

    if (tab === 'dashboard' && METADATA) renderDashboard();
  });
});

// ─────────────────────────────────────────────
// INIT — load metadata from API
// ─────────────────────────────────────────────
async function init() {
  try {
    const res = await fetch('/api/metadata');
    METADATA  = await res.json();
    FEATURES  = METADATA.dataset.features;

    // Update hero stats
    const ann = METADATA.models.ann_mlp;
    document.getElementById('heroAcc').textContent = (ann.accuracy * 100).toFixed(1) + '%';
    document.getElementById('heroAuc').textContent = ann.auc.toFixed(3);

    buildForm();
  } catch (e) {
    console.error('Failed to load metadata:', e);
    document.querySelector('.hero-sub').innerHTML += '<br><span style="color:#e05b4b">⚠ API not available. Run <code>python app.py</code> first.</span>';
  }
}

// ─────────────────────────────────────────────
// BUILD FORM FROM FEATURES
// ─────────────────────────────────────────────
function buildForm() {
  // Split into 3 groups of 10
  const groups = ['mean', 'se', 'worst'];
  const containers = {
    mean:  document.getElementById('group-mean'),
    se:    document.getElementById('group-se'),
    worst: document.getElementById('group-worst')
  };

  FEATURES.forEach((feat, idx) => {
    const groupKey = feat.includes(' se') || feat.endsWith(' se') || feat.includes('error')
      ? 'se'
      : (feat.includes('worst') ? 'worst' : 'mean');

    // Determine group by index (10 per group)
    const gKey = idx < 10 ? 'mean' : (idx < 20 ? 'se' : 'worst');
    const container = containers[gKey];

    const minVal = METADATA.dataset.mins[idx];
    const maxVal = METADATA.dataset.maxs[idx];
    const meanVal = METADATA.dataset.means[idx];

    const div = document.createElement('div');
    div.className = 'feature-field';
    div.innerHTML = `
      <div class="feat-label" title="${feat}">${feat}</div>
      <input
        type="number"
        class="feat-input"
        id="feat_${idx}"
        name="${feat}"
        step="any"
        placeholder="${meanVal.toFixed(4)}"
        data-idx="${idx}"
        oninput="onFeatInput(this)"
      />
      <div class="feat-range">Range: ${minVal.toFixed(3)} – ${maxVal.toFixed(3)}</div>
    `;
    container.appendChild(div);
  });
}

function onFeatInput(el) {
  el.classList.toggle('filled', el.value !== '');
}

// ─────────────────────────────────────────────
// LOAD SAMPLE PATIENT
// ─────────────────────────────────────────────
async function loadSample(label) {
  try {
    const res  = await fetch(`/api/sample/${label}`);
    const data = await res.json();

    data.features.forEach((val, idx) => {
      const input = document.getElementById(`feat_${idx}`);
      if (input) {
        input.value = val;
        input.classList.add('filled');
        input.style.borderColor = label === 0 ? '#e05b4b55' : '#4caf8255';
        setTimeout(() => input.style.borderColor = '', 800);
      }
    });

    // Flash feedback
    const bar = document.querySelector('.controls-bar');
    const msg = document.createElement('div');
    msg.style.cssText = 'font-size:12px;color:' + (label===0 ? '#e05b4b' : '#4caf82') + ';margin-left:auto;animation:fadeIn 0.3s ease';
    msg.textContent = `✓ Loaded ${label === 0 ? 'malignant' : 'benign'} sample`;
    bar.appendChild(msg);
    setTimeout(() => msg.remove(), 2000);
  } catch (e) {
    alert('Could not load sample. Make sure the Flask server is running.');
  }
}

// ─────────────────────────────────────────────
// CLEAR FORM
// ─────────────────────────────────────────────
function clearForm() {
  FEATURES.forEach((_, idx) => {
    const input = document.getElementById(`feat_${idx}`);
    if (input) { input.value = ''; input.classList.remove('filled'); input.style.borderColor = ''; }
  });
  document.getElementById('resultPanel').classList.add('hidden');
}

// ─────────────────────────────────────────────
// RUN PREDICTION
// ─────────────────────────────────────────────
async function runPrediction() {
  const features = [];
  let missing = 0;

  FEATURES.forEach((_, idx) => {
    const input = document.getElementById(`feat_${idx}`);
    const val = parseFloat(input?.value);
    if (isNaN(val)) {
      missing++;
      // Use dataset mean as default
      features.push(METADATA.dataset.means[idx]);
    } else {
      features.push(val);
    }
  });

  if (missing > 10) {
    showToast(`Please fill in at least 20 fields (${missing} are empty — using dataset means for rest)`);
  }

  const btn = document.getElementById('predictBtn');
  btn.classList.add('loading');
  btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation:spin 1s linear infinite"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Analysing…`;

  try {
    const modelKey = document.getElementById('modelSelect').value;
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, model: modelKey })
    });
    const result = await res.json();

    if (result.error) throw new Error(result.error);
    renderResult(result);

  } catch (e) {
    showToast('Prediction failed: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> Run Prediction`;
  }
}

// ─────────────────────────────────────────────
// RENDER RESULT
// ─────────────────────────────────────────────
function renderResult(r) {
  const panel  = document.getElementById('resultPanel');
  const left   = document.querySelector('.result-left');
  const isMal  = r.prediction === 0;

  document.getElementById('resultLabel').textContent = r.label.toUpperCase();
  const badge = document.getElementById('resultBadge');
  badge.textContent = r.label;
  badge.className = 'result-class-badge ' + (isMal ? 'malignant' : 'benign');
  document.getElementById('resultConf').textContent = r.confidence + '%';

  left.className = 'result-left ' + (isMal ? 'malignant-result' : 'benign-result');

  const riskEl = document.getElementById('riskValue');
  riskEl.textContent = r.risk;
  riskEl.className = `risk-value ${r.risk}`;

  // Probability bars (animated)
  setTimeout(() => {
    document.getElementById('malignantBar').style.width = r.malignant_prob + '%';
    document.getElementById('benignBar').style.width    = r.benign_prob + '%';
  }, 100);
  document.getElementById('malignantPct').textContent = r.malignant_prob + '%';
  document.getElementById('benignPct').textContent    = r.benign_prob + '%';

  // Notable features
  const list = document.getElementById('notableList');
  list.innerHTML = r.notable_features.map(f => `
    <div class="nf-item">
      <span class="nf-feat">${f.name}</span>
      <span class="nf-val">${f.value}</span>
      <span class="nf-z ${f.direction}">${f.direction === 'above' ? '+' : ''}${f.z_score}σ</span>
    </div>
  `).join('');

  panel.classList.remove('hidden');
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─────────────────────────────────────────────
// DASHBOARD
// ─────────────────────────────────────────────
function renderDashboard() {
  if (!METADATA) return;

  const lr  = METADATA.models.logistic_regression;
  const ann = METADATA.models.ann_mlp;

  // LR metrics
  document.getElementById('metricsLR').innerHTML = metricsHTML(lr);
  document.getElementById('metricsANN').innerHTML = metricsHTML(ann);

  // Confusion matrices
  renderCM('cmLR', lr.confusion_matrix);
  renderCM('cmANN', ann.confusion_matrix);

  // ROC Curve
  renderROC(METADATA.roc_curve);

  // Top features
  renderTopFeatures(METADATA.top10_features);
}

function metricsHTML(m) {
  return `
    <div class="mc-metrics">
      <div class="metric-box"><div class="m-label">Accuracy</div><div class="m-val">${(m.accuracy*100).toFixed(1)}%</div></div>
      <div class="metric-box"><div class="m-label">ROC-AUC</div><div class="m-val">${m.auc.toFixed(3)}</div></div>
      <div class="metric-box"><div class="m-label">CV Mean</div><div class="m-val">${(m.cv_mean*100).toFixed(1)}%</div></div>
      <div class="metric-box"><div class="m-label">CV Std</div><div class="m-val">±${(m.cv_std*100).toFixed(1)}%</div></div>
    </div>
  `;
}

function renderCM(containerId, cm) {
  const c = document.getElementById(containerId);
  // cm[0] = [TN, FP], cm[1] = [FN, TP]
  const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
  c.innerHTML = `
    <div class="cm-title">Confusion Matrix</div>
    <div class="cm-grid">
      <div></div><div class="cm-header">Pred Mal</div><div class="cm-header">Pred Ben</div>
      <div class="cm-label">Act Mal</div>
      <div class="cm-cell cm-tn">${tn}<br><small>TN</small></div>
      <div class="cm-cell cm-fp">${fp}<br><small>FP</small></div>
      <div class="cm-label">Act Ben</div>
      <div class="cm-cell cm-fn">${fn}<br><small>FN</small></div>
      <div class="cm-cell cm-tp">${tp}<br><small>TP</small></div>
    </div>
  `;
}

function renderROC(rocData) {
  const ctx = document.getElementById('rocChart').getContext('2d');
  if (rocChartInstance) rocChartInstance.destroy();

  rocChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rocData.fpr,
      datasets: [
        {
          label: `ANN (AUC = ${rocData.auc.toFixed(3)})`,
          data: rocData.tpr,
          borderColor: '#e05b4b',
          backgroundColor: 'rgba(224,91,75,0.08)',
          borderWidth: 2.5,
          fill: true,
          tension: 0.3,
          pointRadius: 0
        },
        {
          label: 'Random Classifier',
          data: rocData.fpr,
          borderColor: '#3a4050',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: { color: '#8f9bae', font: { family: 'DM Mono', size: 12 } }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'False Positive Rate', color: '#5a6475' },
          ticks: { color: '#5a6475', font: { family: 'DM Mono', size: 11 } },
          grid:  { color: '#1e2530' },
          min: 0, max: 1
        },
        y: {
          title: { display: true, text: 'True Positive Rate', color: '#5a6475' },
          ticks: { color: '#5a6475', font: { family: 'DM Mono', size: 11 } },
          grid:  { color: '#1e2530' },
          min: 0, max: 1
        }
      }
    }
  });
}

function renderTopFeatures(features) {
  const max = Math.max(...features.map(f => f[1]));
  const container = document.getElementById('topFeatures');
  container.innerHTML = features.map(([name, val]) => `
    <div class="tf-row">
      <div class="tf-name" title="${name}">${name}</div>
      <div class="tf-bar-track">
        <div class="tf-bar-fill" style="width:${(val/max)*100}%"></div>
      </div>
      <div class="tf-val">${val.toFixed(3)}</div>
    </div>
  `).join('');
}

// ─────────────────────────────────────────────
// UTILITY
// ─────────────────────────────────────────────
function showToast(msg) {
  const t = document.createElement('div');
  t.textContent = msg;
  t.style.cssText = `
    position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
    background:#1e2530;color:#e8edf5;padding:12px 24px;border-radius:8px;
    font-size:13px;z-index:9999;box-shadow:0 4px 20px rgba(0,0,0,0.4);
    border:1px solid #252d3a;animation:slideUp 0.3s ease;
  `;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// CSS for animations added dynamically
const style = document.createElement('style');
style.textContent = `
  @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
  @keyframes fadeIn { from{opacity:0} to{opacity:1} }
  @keyframes slideUp { from{opacity:0;transform:translateX(-50%) translateY(10px)} to{opacity:1;transform:translateX(-50%) translateY(0)} }
`;
document.head.appendChild(style);

// ─────────────────────────────────────────────
// BOOT
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
