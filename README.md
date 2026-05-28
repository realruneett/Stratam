# 🚦 Stratam — Traffic Demand Prediction

> Competition-winning ML pipeline for the **HackerEarth Flipkart Gridlock Hackathon 2.0**.

LightGBM + XGBoost + CatBoost ensemble with Ridge stacking meta-learner, GPU-accelerated, fully automated.

---

## Repository Structure

```
Stratam/
├── run.py                  ← Entry point — run this
├── config.py               ← Seeds, hyperparameters, paths
├── requirements.txt        ← Python dependencies
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── schema.py           ← Adaptive schema detection
│   ├── data.py             ← CSV loading, parsing, target transform
│   ├── spatial.py          ← Per-location demand statistics
│   ├── features.py         ← Feature engineering & memory optimization
│   ├── validation.py       ← Chronological split & time-based K-fold CV
│   ├── models.py           ← LightGBM / XGBoost / CatBoost training
│   ├── ensemble.py         ← Ridge stacking meta-learner
│   ├── postprocess.py      ← Inverse transform, clipping, submission
│   └── diagnostics.py      ← Feature importance, SHAP, MAPE, timing
│
├── data/
│   ├── .gitkeep
│   ├── train.csv           ← Place your data here
│   └── test.csv            ← Place your data here
│
└── master_prompt.txt       ← Original spec
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data
cp /path/to/train.csv data/
cp /path/to/test.csv  data/

# 3. Run
python run.py
```

## Outputs

| File                     | Description                          |
|--------------------------|--------------------------------------|
| `submission.csv`         | HackerEarth submission file          |
| `feature_importance.csv` | Full feature importance table        |
| `feature_importance.png` | Top-30 feature importance bar chart  |
| `shap_summary.png`       | SHAP summary plot (if shap installed)|

## Hardware Target

| Component | Spec |
|-----------|------|
| OS        | WSL2 on Windows |
| CPU       | Intel Core Ultra 9 275HX |
| RAM       | 32 GB |
| GPU       | NVIDIA RTX 5070 Ti (12 GB VRAM) |
| CUDA      | 12.x |
| Python    | 3.10+ |

## Pipeline Architecture

```
train.csv ──→ Schema Detection ──→ Target Transform (log1p)
                    │
                    ├──→ Spatial Stats (per-location)
                    │
                    ├──→ Feature Engineering
                    │      • Temporal (12 features)
                    │      • Cyclical encodings (8 features)
                    │      • Lag features (8 lags)
                    │      • Rolling stats (5 windows × 4 aggs)
                    │      • EWM, interactions, haversine
                    │
                    ├──→ Time-Based K-Fold CV (5 folds)
                    │      ├── LightGBM  (OOF + final)
                    │      ├── XGBoost   (OOF + final)
                    │      └── CatBoost  (OOF + final)
                    │
                    ├──→ Ridge Stacking (meta-learner on OOF)
                    │
                    └──→ Post-Processing
                           • expm1 inverse
                           • clip ≥ 0
                           • integer rounding (conditional)
                           • sanity assertions
                                    │
                                    ▼
                            submission.csv
```

## Key Design Decisions

- **No hardcoded column names** — adaptive schema detection makes the pipeline dataset-agnostic.
- **GPU with fallback** — every model tries GPU first, falls back to CPU automatically.
- **Time-based CV** — splits on unique timestamps, never leaks future data.
- **Lag safety** — test lags are computed via a combined history-tail + test frame.
- **Memory optimization** — automatic downcasting saves ~40–60% RAM.

## License

MIT
