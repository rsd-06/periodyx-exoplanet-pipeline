/* ============================================================
   PeriodyX — App Logic
   Handles: mode switching, file upload, API calls, rendering
   ============================================================ */

'use strict';

// ── State ──────────────────────────────────────────────────
let currentMode = 'synthetic';
let batchResultsData = [];

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
  document.getElementById('batch-section').classList.toggle('hidden', mode !== 'batch');
  document.getElementById('btn-synthetic').classList.toggle('active', mode === 'synthetic');
  document.getElementById('btn-upload').classList.toggle('active', mode === 'upload');
  document.getElementById('btn-batch').classList.toggle('active', mode === 'batch');
  document.getElementById('btn-synthetic').setAttribute('aria-selected', mode === 'synthetic');
  document.getElementById('btn-upload').setAttribute('aria-selected', mode === 'upload');
  document.getElementById('btn-batch').setAttribute('aria-selected', mode === 'batch');

  // Show/hide the single-star result vs batch results panels
  const batchPanel = document.getElementById('batch-results-panel');
  if (mode === 'batch') {
    // Run button label changes
    document.querySelector('.run-btn-text').textContent = 'Run Batch Pipeline';
  } else {
    document.querySelector('.run-btn-text').textContent = 'Run Full Pipeline';
    batchPanel.classList.add('hidden');
  }
}

function autoAdjustSyntheticParams() {
  const caseVal = document.getElementById('case-select').value;
  const depthInput = document.getElementById('synth_depth_pct');
  const periodInput = document.getElementById('synth_period');
  const noiseInput = document.getElementById('synth_noise');
  const centroidInput = document.getElementById('synth_centroid');

  if (caseVal === 'transit') {
    depthInput.value = '0.15';
    periodInput.value = '4.3';
    noiseInput.value = '300';
    centroidInput.value = '0.0';
  } else if (caseVal === 'false_positive') {
    depthInput.value = '1.5';
    periodInput.value = '5.1';
    noiseInput.value = '300';
    centroidInput.value = '0.0';
  } else if (caseVal === 'blend') {
    depthInput.value = '0.5';
    periodInput.value = '3.5';
    noiseInput.value = '300';
    centroidInput.value = '0.8';
  } else if (caseVal === 'other') {
    depthInput.value = '0.0';
    periodInput.value = '4.3';
    noiseInput.value = '800';
    centroidInput.value = '0.0';
  }
}


// ── File Upload Handling ───────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const fileInput  = document.getElementById('file-input');
const fileDisplay = document.getElementById('file-name-display');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) showFileName(fileInput.files[0].name);
});

// ── Batch File List Display ────────────────────────────────
const batchFileInput = document.getElementById('batch-file-input');
const batchFileList  = document.getElementById('batch-file-list');
const batchZone      = document.getElementById('batch-upload-zone');

batchFileInput.addEventListener('change', () => {
  updateBatchFileList(batchFileInput.files);
});

batchZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  batchZone.classList.add('drag-over');
});
batchZone.addEventListener('dragleave', () => batchZone.classList.remove('drag-over'));
batchZone.addEventListener('drop', (e) => {
  e.preventDefault();
  batchZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) {
    const dt = new DataTransfer();
    for (const f of e.dataTransfer.files) dt.items.add(f);
    batchFileInput.files = dt.files;
    updateBatchFileList(dt.files);
  }
});

function updateBatchFileList(files) {
  if (!files || files.length === 0) {
    batchFileList.classList.add('hidden');
    return;
  }
  batchFileList.innerHTML = Array.from(files).map(f =>
    `<div class="batch-file-item"><span>${f.name}</span><span style="color:var(--text-muted)">${(f.size/1024).toFixed(1)} KB</span></div>`
  ).join('');
  batchFileList.classList.remove('hidden');
}

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
  const btn = document.getElementById('run-btn');
  btn.disabled = true;

  if (currentMode === 'batch') {
    await runBatchPipeline(btn);
    return;
  }

  clearError();
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
      formData.append('depth_pct', parseFloat(document.getElementById('synth_depth_pct').value) || 0);
      formData.append('period', parseFloat(document.getElementById('synth_period').value) || 4.3);
      formData.append('noise_ppm', parseFloat(document.getElementById('synth_noise').value) || 300);
      formData.append('centroid_offset', parseFloat(document.getElementById('synth_centroid').value) || 0);

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
    btn.querySelector('.run-btn-text').textContent =
      currentMode === 'batch' ? 'Run Batch Pipeline' : 'Run Full Pipeline';
  }
}


// ── Batch Pipeline Runner ──────────────────────────────────────────────────
async function runBatchPipeline(btn) {
  const batchInput = document.getElementById('batch-file-input');
  if (!batchInput.files || batchInput.files.length === 0) {
    showError('Please select at least one CSV file for batch processing.');
    btn.disabled = false;
    btn.querySelector('.run-btn-text').textContent = 'Run Batch Pipeline';
    return;
  }

  showState('loading');
  startLoadingAnimation();
  document.getElementById('batch-results-panel').classList.add('hidden');

  const formData = new FormData();
  const files = batchInput.files;
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }

  // Build global priors JSON (applied to all files)
  const priors = {
    koi_srad: parseFloat(document.getElementById('batch_koi_srad').value) || 1.0,
    koi_steff: parseFloat(document.getElementById('batch_koi_steff').value) || 5778.0,
    koi_slogg: parseFloat(document.getElementById('batch_koi_slogg').value) || 4.44,
    koi_kepmag: parseFloat(document.getElementById('batch_koi_kepmag').value) || 12.0,
    centroid_offset_magnitude: parseFloat(document.getElementById('batch_centroid').value) || 0.0,
  };
  // Map every uploaded filename to the same global priors
  const priorsMap = {};
  for (let i = 0; i < files.length; i++) {
    priorsMap[files[i].name] = priors;
  }
  formData.append('stellar_priors_json', JSON.stringify(priorsMap));

  try {
    const res = await fetch('/api/run_batch', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Batch pipeline failed.');
    }
    const data = await res.json();
    stopLoadingAnimation();
    showState('idle'); // reset single-star panel
    renderBatchResults(data);
  } catch (err) {
    stopLoadingAnimation();
    showState('idle');
    showError(err.message || 'Batch pipeline error.');
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.querySelector('.run-btn-text').textContent = 'Run Batch Pipeline';
  }
}


// ── Batch Results Renderer ─────────────────────────────────────────────────
function renderBatchResults(data) {
  batchResultsData = data.results || [];
  const errors     = data.errors   || [];

  const panel  = document.getElementById('batch-results-panel');
  const tbody  = document.getElementById('batch-results-tbody');
  const summary = document.getElementById('batch-summary-text');
  const errBox = document.getElementById('batch-errors-container');

  summary.textContent = `${data.n_processed} processed, ${data.n_errors} errors`;
  tbody.innerHTML = '';

  batchResultsData.forEach(row => {
    const meta = CLASS_META[row.prediction] || { label: row.prediction, icon: '🔬', color: 'var(--accent-blue)' };
    const confPct = row.confidence != null ? (row.confidence * 100).toFixed(1) : '—';
    const uncPct  = row.uncertainty != null ? (row.uncertainty * 100).toFixed(1) : '—';
    const singleFlag = row.n_single_transit_candidates > 0
      ? `<span style="color:#f0c040">⚠️ ${row.n_single_transit_candidates}</span>` : '—';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td title="${row.file}">${row.file.length > 22 ? row.file.slice(0, 22) + '…' : row.file}</td>
      <td><span style="color:${meta.color};font-weight:600">${meta.icon} ${meta.label}</span></td>
      <td>${confPct}%</td>
      <td style="color:#8b949e">${uncPct}%</td>
      <td>${row.period_days != null ? row.period_days.toFixed(4) : '—'}</td>
      <td>${row.depth_pct != null ? row.depth_pct.toFixed(4) : '—'}</td>
      <td>${row.duration_hours != null ? row.duration_hours.toFixed(3) : '—'}</td>
      <td>${row.tls_sde != null ? row.tls_sde.toFixed(2) : '—'}</td>
      <td>${singleFlag}</td>
    `;
    tbody.appendChild(tr);
  });

  // Error section
  if (errors.length > 0) {
    errBox.innerHTML = `<div class="alert-banner warning"><strong>⚠️ ${errors.length} file(s) failed</strong><ul>` +
      errors.map(e => `<li><code>${e.file}</code>: ${e.error.split('\n')[0]}</li>`).join('') +
      '</ul></div>';
    errBox.classList.remove('hidden');
  } else {
    errBox.classList.add('hidden');
  }

  panel.classList.remove('hidden');
}


// ── Batch CSV Download ─────────────────────────────────────────────────────
function downloadBatchCSV() {
  if (!batchResultsData.length) return;
  const headers = ['file','prediction','confidence_pct','uncertainty_pct',
                   'period_days','depth_pct','duration_hours','tls_sde','n_single_transit_candidates'];
  const rows = batchResultsData.map(r => [
    r.file, r.prediction,
    r.confidence != null ? (r.confidence * 100).toFixed(2) : '',
    r.uncertainty != null ? (r.uncertainty * 100).toFixed(2) : '',
    r.period_days != null ? r.period_days : '',
    r.depth_pct != null ? r.depth_pct : '',
    r.duration_hours != null ? r.duration_hours : '',
    r.tls_sde != null ? r.tls_sde : '',
    r.n_single_transit_candidates,
  ].join(','));
  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'periodyx_batch_results.csv';
  a.click();
  URL.revokeObjectURL(url);
}


// ── Result Rendering ───────────────────────────────────────
function renderResults(data) {
  const { probabilities, uncertainties, diagnostics, plot_base64, target, expected_label } = data;

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

  // Ground Truth Expected Label
  const expectedEl = document.getElementById('verdict-expected');
  if (expected_label && CLASS_META[expected_label]) {
    const expMeta = CLASS_META[expected_label];
    expectedEl.innerHTML = `Expected (Ground Truth): <strong>${expMeta.label}</strong>`;
    expectedEl.classList.remove('hidden');
  } else {
    expectedEl.classList.add('hidden');
    expectedEl.innerHTML = '';
  }

  // Probability bars
  renderProbaBars(probabilities, uncertainties, topKey);

  // Diagnostics grid
  renderDiagnostics(diagnostics);

  // Single Transit Alert
  const alertContainer = document.getElementById('single-transit-alert-container');
  if (alertContainer) {
    if (diagnostics.n_single_transit_candidates > 0) {
      let eventsHtml = diagnostics.single_transit_events.map((e, idx) => 
        `<li>Event ${idx+1}: Day ${e.time.toFixed(1)} (Depth: ${(e.depth*100).toFixed(2)}%, Sig: ${e.significance.toFixed(1)})</li>`
      ).join('');
      alertContainer.innerHTML = `<div class="alert-banner warning">
        <strong>⚠️ Single-Transit Candidates Found (${diagnostics.n_single_transit_candidates})</strong>
        <p>The sliding-window scan detected isolated, statistically significant dips that do not repeat periodically. These require manual follow-up as they cannot be confirmed by BLS/TLS.</p>
        <ul>${eventsHtml}</ul>
      </div>`;
      alertContainer.classList.remove('hidden');
    } else {
      alertContainer.innerHTML = '';
      alertContainer.classList.add('hidden');
    }
  }

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

function renderProbaBars(probabilities, uncertainties, topKey) {
  const container = document.getElementById('proba-bars');
  container.innerHTML = '';

  // Sort by probability descending
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cls, prob]) => {
    const meta = CLASS_META[cls] || { label: cls, color: 'var(--accent-blue)' };
    const pct  = (prob * 100).toFixed(1);
    const uncert = uncertainties && uncertainties[cls] ? (uncertainties[cls] * 100).toFixed(1) : "0.0";

    const row = document.createElement('div');
    row.className = 'proba-row';

    row.innerHTML = `
      <span class="proba-label">${meta.label}</span>
      <div class="proba-track">
        <div class="proba-fill" data-pct="${prob * 100}" style="background: ${meta.color}; width: 0%;"></div>
      </div>
      <span class="proba-pct">${pct}% <span style="font-size: 0.8em; color: #8b949e;">&plusmn;${uncert}</span></span>
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
    'n_signals_detected', 'n_single_transit_candidates',
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
