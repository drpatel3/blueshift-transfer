"""Flowsheet prediction web app.

Slider for head grade with a Cu/Au commodity toggle. Shows predicted vs
ground-truth PFS comparison + per-doc metrics.

Cu range: 0.30–1.75 % (step 0.01)
Au range: 0.30–30.00 g/t (step 0.10)

Usage:
    python classification/app.py
    # Opens http://localhost:5000
"""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from flask import Flask, jsonify, send_file

app = Flask(__name__)
# Image cache: configurable via PFS_IMG_DIR (Lambda must point at /tmp).
_img_dir_env = os.environ.get("PFS_IMG_DIR")
if _img_dir_env:
    IMG_DIR = Path(_img_dir_env)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
else:
    IMG_DIR = Path(tempfile.mkdtemp(prefix="pfs_"))


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/predict/<commodity>/<float:grade>")
def predict_grade(commodity, grade):
    from predict import predict, visualize

    commodity = commodity.lower()
    if commodity not in {"cu", "au"}:
        return jsonify({"error": f"Unknown commodity {commodity!r}"}), 400

    try:
        result = predict(grade, commodity=commodity)
    except FileNotFoundError as e:
        # Au checkpoint may not exist yet (training in flight).
        return jsonify({"error": str(e)}), 503
    if "error" in result:
        return jsonify(result), 400

    img_name = f"pfs_{commodity}_{grade:.2f}.png"
    img_path = IMG_DIR / img_name
    visualize(result, output_file=str(img_path))

    return jsonify({
        "commodity": commodity,
        "grade": grade,
        "grade_unit": result.get("grade_unit"),
        "grade_label": result.get("grade_label"),
        "matched_grade": result.get("matched_grade"),
        "stages": result["stages"],
        "edges": result["edges"],
        "metrics": result.get("metrics"),
        "corpus_metrics": result.get("corpus_metrics"),
        "image": f"/img/{img_name}",
    })


# Backward-compat: existing /predict/<grade> defaults to Cu.
@app.route("/predict/<float:grade>")
def predict_grade_legacy(grade):
    return predict_grade("cu", grade)


@app.route("/img/<filename>")
def serve_image(filename):
    path = IMG_DIR / filename
    if not path.exists():
        return "Not found", 404
    return send_file(str(path), mimetype="image/png")


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flowsheet Predictor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f5f7fa;
    color: #1a1a2e;
  }
  .header {
    background: #1a1a2e;
    color: white;
    padding: 24px 40px;
  }
  .header h1 { font-size: 22px; font-weight: 600; }
  .header p { font-size: 14px; color: #8892a4; margin-top: 4px; }
  .controls {
    background: white;
    padding: 24px 40px;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
  }
  .commodity-toggle {
    display: inline-flex;
    background: #f0f4f8;
    border-radius: 999px;
    padding: 4px;
  }
  .commodity-toggle button {
    border: none;
    background: transparent;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 600;
    border-radius: 999px;
    cursor: pointer;
    color: #64748b;
    transition: all 0.15s;
  }
  .commodity-toggle button.active {
    background: #1a1a2e;
    color: white;
  }
  .slider-group {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .slider-group label {
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
  }
  input[type="range"] {
    width: 320px;
    accent-color: #1a1a2e;
  }
  .grade-display {
    font-size: 28px;
    font-weight: 700;
    color: #1a1a2e;
    min-width: 110px;
  }
  .metrics {
    display: flex;
    gap: 20px;
  }
  .metric-box {
    text-align: center;
    padding: 8px 16px;
    background: #f0f4f8;
    border-radius: 8px;
  }
  .metric-box .value {
    font-size: 22px;
    font-weight: 700;
  }
  .metric-box .label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .image-container {
    padding: 20px 40px;
    text-align: center;
  }
  .image-container img {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  }
  .loading {
    padding: 40px;
    text-align: center;
    color: #64748b;
    font-size: 16px;
  }
  .stages {
    padding: 0 40px 20px;
    font-size: 13px;
    color: #475569;
  }
</style>
</head>
<body>
  <div class="header">
    <h1>Process Flowsheet Predictor</h1>
    <p id="headerSubtitle">Predicted vs ground-truth flowsheet comparison from head grade</p>
  </div>
  <div class="controls">
    <div class="commodity-toggle">
      <button id="btnCu" class="active" data-commodity="cu">Copper</button>
      <button id="btnAu" data-commodity="au">Gold</button>
    </div>
    <div class="slider-group">
      <label id="sliderLabel">Cu Head Grade:</label>
      <input type="range" id="slider" min="0.30" max="1.75" step="0.01" value="1.25">
      <span class="grade-display" id="gradeLabel">1.25%</span>
    </div>
    <div class="metrics" id="metrics">
      <div class="metric-box">
        <div class="value" id="stageF1">—</div>
        <div class="label">Stage F1</div>
      </div>
      <div class="metric-box" title="Macro per-stage F1 over the validation set — same number reported at training">
        <div class="value" id="avgStageF1">—</div>
        <div class="label" id="avgStageF1Label">Val Macro F1</div>
      </div>
      <div class="metric-box">
        <div class="value" id="edgeF1">—</div>
        <div class="label">Edge F1</div>
      </div>
      <div class="metric-box">
        <div class="value" id="reachF1">—</div>
        <div class="label">Reach F1</div>
      </div>
    </div>
  </div>
  <div class="stages" id="stagesInfo"></div>
  <div class="image-container">
    <div class="loading" id="loading">Move the slider to generate a prediction</div>
    <img id="pfsImage" style="display:none" alt="PFS Comparison">
  </div>

<script>
const slider = document.getElementById('slider');
const gradeLabel = document.getElementById('gradeLabel');
const sliderLabel = document.getElementById('sliderLabel');
const pfsImage = document.getElementById('pfsImage');
const loading = document.getElementById('loading');
const stagesInfo = document.getElementById('stagesInfo');
const btnCu = document.getElementById('btnCu');
const btnAu = document.getElementById('btnAu');

const COMMODITY_CONFIG = {
  cu: { min: 0.30, max: 1.75, step: 0.01, value: 1.25,
        unit: '%', label: 'Cu Head Grade:', decimals: 2 },
  au: { min: 0.30, max: 30.00, step: 0.10, value: 5.00,
        unit: ' g/t', label: 'Au Head Grade:', decimals: 2 },
};

let currentCommodity = 'cu';
let debounceTimer = null;

function applyCommodity(commodity) {
  currentCommodity = commodity;
  const cfg = COMMODITY_CONFIG[commodity];
  slider.min = cfg.min;
  slider.max = cfg.max;
  slider.step = cfg.step;
  slider.value = cfg.value;
  sliderLabel.textContent = cfg.label;
  gradeLabel.textContent = cfg.value.toFixed(cfg.decimals) + cfg.unit;
  btnCu.classList.toggle('active', commodity === 'cu');
  btnAu.classList.toggle('active', commodity === 'au');
  fetchPrediction();
}

btnCu.addEventListener('click', () => applyCommodity('cu'));
btnAu.addEventListener('click', () => applyCommodity('au'));

slider.addEventListener('input', () => {
  const cfg = COMMODITY_CONFIG[currentCommodity];
  gradeLabel.textContent = parseFloat(slider.value).toFixed(cfg.decimals) + cfg.unit;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchPrediction, 300);
});

async function fetchPrediction() {
  const grade = parseFloat(slider.value);
  const cfg = COMMODITY_CONFIG[currentCommodity];
  loading.textContent = 'Generating prediction...';
  loading.style.display = 'block';
  pfsImage.style.display = 'none';

  try {
    const resp = await fetch('/predict/' + currentCommodity + '/' + grade.toFixed(cfg.decimals));
    const data = await resp.json();
    if (data.error) {
      loading.textContent = data.error;
      return;
    }

    pfsImage.src = data.image + '?t=' + Date.now();
    pfsImage.style.display = 'block';
    loading.style.display = 'none';

    const m = data.metrics || {};
    document.getElementById('stageF1').textContent =
      m.stage_f1 != null ? m.stage_f1.toFixed(3) : '—';
    document.getElementById('edgeF1').textContent =
      m.edge_f1 != null ? m.edge_f1.toFixed(3) : '—';
    document.getElementById('reachF1').textContent =
      m.reach_f1 != null ? m.reach_f1.toFixed(3) : '—';

    const cm = data.corpus_metrics || {};
    document.getElementById('avgStageF1').textContent =
      cm.stage_f1_val_mean != null ? cm.stage_f1_val_mean.toFixed(3) : '—';
    document.getElementById('avgStageF1Label').textContent =
      cm.n_val != null ? 'Val Macro F1 (n=' + cm.n_val + ')' : 'Val Macro F1';

    const matched = data.matched_grade != null
      ? data.matched_grade.toFixed(cfg.decimals)
      : '?';
    stagesInfo.textContent =
      'Matched doc: ' + matched + cfg.unit + '  |  ' +
      'Stages: ' + data.stages.join(', ');
  } catch (e) {
    loading.textContent = 'Error: ' + e.message;
  }
}

// Initial load
fetchPrediction();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Image cache: {IMG_DIR}")
    print(f"Open http://localhost:5000")
    app.run(debug=False, port=5000)
