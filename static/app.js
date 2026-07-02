/* ============================================================
   PeriodyX — App Logic
   Handles: mode switching, file upload, API calls, rendering
   ============================================================ */

'use strict';

// ── State ──────────────────────────────────────────────────
let currentMode = 'synthetic';

// Class metadata
const CLASS_META = {
  transit:          { label: 'Planet Transit',    icon: '🌍', color: 'var(--col-transit)',  cssClass: 'transit' },
  eclipsing_binary: { label: 'Eclipsing Binary',  icon: '⭐', color: 'var(--col-eb)',       cssClass: 'eclipsing_binary' },
  blend:            { label: 'Background Blend',  icon: '🌌', color: 'var(--col-blend)',    cssClass: 'blend' },
  other:            { label: 'Other / Noise',     icon: '🔭', color: 'var(--col-other)',    cssClass: 'other' },
};

const DIAG_LABELS = {
  bls_period_days:         'BLS Period (days)',
  tls_sde:                 'TLS SDE',
  tls_ran:                 'TLS Ran',
  detection_passes:        'Threshold Passed',
  depth_pct:               'Transit Depth (%)',
  depth_err_pct:           'Depth Uncertainty (%)',
  duration_hours:          'Duration (hours)',
  duration_err_hours:      'Duration Uncertainty',
  ingress_fraction:        'Ingress Fraction',
  depth_snr:               'Depth SNR',
  odd_even_diff:           'Odd-Even Diff',
  secondary_eclipse_depth: 'Secondary Eclipse Depth',
  period_alias_corrected:  'Period Alias Fixed',
  n_signals_detected:      'Signals Detected',
  single_transit_candidates: 'Single-Transit Flags',
};


// ── Mode Switching ─────────────────────────────────────────
function switchMode(mode) {
  currentMode = mode;
  document.getElementById('synthetic-section').classList.toggle('hidden', mode !== 'synthetic');
  document.getElementById('upload-section').classList.toggle('hidden', mode !== 'upload');
  document.getElementById('btn-synthetic').classList.toggle('active', mode === 'synthetic');
  document.getElementById('btn-upload').classList.toggle('active', mode === 'upload');
  document.getElementById('btn-synthetic').setAttribute('aria-selected', mode === 'synthetic');
  document.getElementById('btn-upload').setAttribute('aria-selected', mode === 'upload');
}


// ── File Upload Handling ───────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const fileInput  = document.getElementById('file-input');
const fileDisplay = document.getElementById('file-name-display');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) showFileName(fileInput.files[0].name);
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    // Create a DataTransfer to assign to the file input
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    fileInput.files = dt.files;
    showFileName(files[0].name);
  }
});

function showFileName(name) {
  fileDisplay.textContent = `📎 ${name}`;
  fileDisplay.classList.remove('hidden');
}


// ── Loading Animation ──────────────────────────────────────
const loadingSteps = ['ls-detrend', 'ls-bls', 'ls-fit', 'ls-classify'];
let stepTimer = null;

function startLoadingAnimation() {
  let idx = 0;
  loadingSteps.forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.classList.remove('active', 'done'); }
  });
  if (loadingSteps[0]) document.getElementById(loadingSteps[0])?.classList.add('active');

  stepTimer = setInterval(() => {
    const el = document.getElementById(loadingSteps[idx]);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
    idx++;
    if (idx < loadingSteps.length) {
      document.getElementById(loadingSteps[idx])?.classList.add('active');
    } else {
      clearInterval(stepTimer);
    }
  }, 900);
}

function stopLoadingAnimation() {
  if (stepTimer) { clearInterval(stepTimer); stepTimer = null; }
  loadingSteps.forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.classList.remove('active', 'done'); }
  });
}


// ── State Transitions ──────────────────────────────────────
function showState(state) {
  // state: 'idle' | 'loading' | 'result'
  document.getElementById('idle-state').classList.toggle('hidden', state !== 'idle');
  document.getElementById('loading-state').classList.toggle('hidden', state !== 'loading');
  document.getElementById('result-state').classList.toggle('hidden', state !== 'result');
}

function showError(msg) {
  const box = document.getElementById('error-box');
  document.getElementById('error-text').textContent = msg;
  box.classList.remove('hidden');
}

function clearError() {
  document.getElementById('error-box').classList.add('hidden');
}


// ── Main Pipeline Runner ───────────────────────────────────
async function runPipeline() {
  clearError();
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.querySelector('.run-btn-text').textContent = 'Running…';

  showState('loading');
  document.getElementById('plot-panel').classList.add('hidden');
  startLoadingAnimation();

  try {
    let data;

    if (currentMode === 'synthetic') {
      const caseVal = document.getElementById('case-select').value;
      const formData = new FormData();
      formData.append('case', caseVal);
      const res = await fetch('/api/run_synthetic', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Pipeline failed.');
      }
      data = await res.json();

    } else {
      const file = fileInput.files[0];
      if (!file) { throw new Error('Please select a CSV file first.'); }

      const formData = new FormData();
      formData.append('file', file);
      formData.append('koi_srad',                document.getElementById('koi_srad').value);
      formData.append('koi_steff',               document.getElementById('koi_steff').value);
      formData.append('koi_slogg',               document.getElementById('koi_slogg').value);
      formData.append('koi_kepmag',              document.getElementById('koi_kepmag').value);
      formData.append('centroid_offset_magnitude', document.getElementById('centroid_offset').value);

      const res = await fetch('/api/run_custom', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Pipeline failed.');
      }
      data = await res.json();
    }

    stopLoadingAnimation();
    renderResults(data);

  } catch (err) {
    stopLoadingAnimation();
    showState('idle');
    showError(err.message || 'An unexpected error occurred.');
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.querySelector('.run-btn-text').textContent = 'Run Full Pipeline';
  }
}


// ── Result Rendering ───────────────────────────────────────
function renderResults(data) {
  const { probabilities, diagnostics, plot_base64, target } = data;

  if (!probabilities || probabilities.error) {
    showState('idle');
    showError('Classifier not loaded. Please ensure the model file is present at models/exoplanet_classifier.joblib');
    return;
  }

  // Find top class
  const topClass = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0];
  const topKey   = topClass[0];
  const topProb  = topClass[1];
  const meta     = CLASS_META[topKey] || { label: topKey, icon: '🔬', color: 'var(--accent-blue)', cssClass: topKey };

  // Verdict banner
  const banner = document.getElementById('verdict-banner');
  banner.className = `verdict-banner ${meta.cssClass}`;
  document.getElementById('verdict-icon').textContent = meta.icon;
  document.getElementById('verdict-class').textContent = meta.label;
  document.getElementById('verdict-class').style.color = meta.color;
  document.getElementById('verdict-confidence').textContent = `${(topProb * 100).toFixed(1)}% confidence`;

  // Probability bars
  renderProbaBars(probabilities, topKey);

  // Diagnostics grid
  renderDiagnostics(diagnostics);

  // Show result pane
  showState('result');

  // Plot
  if (plot_base64) {
    const img = document.getElementById('result-plot');
    img.src = `data:image/png;base64,${plot_base64}`;
    img.style.opacity = 0;
    document.getElementById('plot-subtitle').textContent = `Target: ${target}`;
    document.getElementById('plot-panel').classList.remove('hidden');
    img.onload = () => { img.style.opacity = 1; };
  }
}

function renderProbaBars(probabilities, topKey) {
  const container = document.getElementById('proba-bars');
  container.innerHTML = '';

  // Sort by probability descending
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cls, prob]) => {
    const meta = CLASS_META[cls] || { label: cls, color: 'var(--accent-blue)' };
    const pct  = (prob * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'proba-row';

    row.innerHTML = `
      <span class="proba-label">${meta.label}</span>
      <div class="proba-track">
        <div class="proba-fill" data-pct="${prob * 100}" style="background: ${meta.color}; width: 0%;"></div>
      </div>
      <span class="proba-pct">${pct}%</span>
    `;
    container.appendChild(row);
  });

  // Animate bars after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll('.proba-fill').forEach(bar => {
        bar.style.width = `${bar.dataset.pct}%`;
      });
    });
  });
}

function renderDiagnostics(diagnostics) {
  const grid = document.getElementById('diag-grid');
  grid.innerHTML = '';

  const displayKeys = [
    'bls_period_days', 'tls_sde', 'detection_passes', 'depth_pct',
    'duration_hours', 'depth_snr', 'ingress_fraction', 'odd_even_diff',
    'secondary_eclipse_depth', 'period_alias_corrected',
    'n_signals_detected', 'single_transit_candidates',
  ];

  displayKeys.forEach(key => {
    if (!(key in diagnostics)) return;
    const raw = diagnostics[key];
    let display = raw;
    let valClass = '';

    if (typeof raw === 'boolean') {
      display = raw ? '✔ Yes' : '✘ No';
      if (key === 'detection_passes') valClass = raw ? 'pass' : 'fail';
      if (key === 'period_alias_corrected') valClass = raw ? 'warn' : '';
    } else if (typeof raw === 'number') {
      display = raw.toFixed(raw < 0.01 && raw > 0 ? 6 : 4);
    }

    const item = document.createElement('div');
    item.className = 'diag-item';
    item.innerHTML = `
      <span class="diag-key">${DIAG_LABELS[key] || key}</span>
      <span class="diag-val ${valClass}">${display}</span>
    `;
    grid.appendChild(item);
  });
}
