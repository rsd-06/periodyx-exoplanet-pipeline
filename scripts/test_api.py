import sys
sys.stdout.reconfigure(encoding='utf-8')
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("Starting API tests...")

# Test 1: transit case
resp = client.post('/api/run_synthetic', data={'case': 'transit'})
assert resp.status_code == 200, f'FAIL: {resp.text}'
d = resp.json()
assert 'probabilities' in d, 'Missing probabilities'
assert 'uncertainties' in d, 'Missing uncertainties'
assert 'diagnostics' in d, 'Missing diagnostics'
assert 'expected_label' in d, 'Missing expected_label'
print('Transit run: OK')
top_class = max(d["probabilities"], key=d["probabilities"].get)
print(f'  Top class: {top_class}')
print(f'  Stellar density ratio in diagnostics: {d["diagnostics"].get("stellar_density_ratio", "NOT FOUND")}')

# Test 2: batch endpoint with synthetic data
import io, numpy as np
t = np.linspace(0, 27, 1000)
flux = np.ones(1000) + np.random.randn(1000)*0.0003
csv_bytes = ('time,flux\n' + '\n'.join(f'{ti},{fi}' for ti,fi in zip(t, flux))).encode()

resp2 = client.post('/api/run_batch', files=[('files', ('test_star.csv', csv_bytes, 'text/csv'))], data={'stellar_priors_json': '{}'})
assert resp2.status_code == 200, f'Batch FAIL: {resp2.text}'
d2 = resp2.json()
print(f'Batch run: OK — {d2["n_processed"]} processed, {d2["n_errors"]} errors')
if d2['results']:
    r = d2['results'][0]
    print(f'  File: {r["file"]}, Prediction: {r["prediction"]}, Confidence: {r["confidence"]*100:.1f}%')
print('ALL API TESTS PASSED')
