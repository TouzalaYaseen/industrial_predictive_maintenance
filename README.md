# Industrial Telemetry & Predictive Maintenance Engine

An end-to-end Machine Learning and IoT telemetry system for early detection of milling machine component failures, cost-sensitive threshold calibration, and real-time inference serving.

---

## 📌 Problem Formulation
In industrial manufacturing, catastrophic equipment failure causes severe operational downtime, safety hazards, and exorbitant repair costs. Conversely, over-frequent manual inspection wastes labor and reduces throughput.

This project builds an automated fault-detection engine using telemetry sensor streams to predict failures before they happen, identify root causes, and provide explainable alerts for plant maintenance technicians.

---

## 🏗️ Architecture & Pipeline Overview

```
Raw Telemetry Data (10k records, 5 sensor streams)
   │
   ▼
Physics-Grounded Feature Engineering (ΔT, Power = τ * ω, Tool Wear Stress)
   │
   ▼
Modular Scikit-Learn Preprocessing & ColumnTransformer
   │
   ▼
Cost-Sensitive XGBoost Classifier (Optimized for PR-AUC & Recall)
   │
   ▼
Optimal Threshold Calibration (Downtime Cost vs. Inspection Cost Matrix)
   │
   ▼
FastAPI Microservice (/health, /predict, /explain, /simulate-stream)
   │
   ▼
Interactive Telemetry Monitoring Dashboard
```

---

## 📊 Dataset: AI4I 2020 Predictive Maintenance
* **Source:** UCI Machine Learning Repository / Kaggle (Matzka, 2020)
* **Scale:** 10,000 operational records, 14 attributes
* **Sensors:**
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`
* **Target:** `Machine failure` (Binary: 0 = Normal, 1 = Failure)
* **Failure Modes (Diagnostic post-mortem labels):**
  - `TWF`: Tool Wear Failure
  - `HDF`: Heat Dissipation Failure
  - `PWF`: Power Failure
  - `OSF`: Overstrain Failure
  - `RNF`: Random Failure

---

## 📂 Project Structure

```
predictive-maintenance/
├── api/                  # Production FastAPI serving layer
│   ├── __init__.py
│   ├── schemas.py        # Pydantic data contracts (TelemetryInput, Prediction, Explanation)
│   └── main.py           # REST endpoints (/health, /predict, /explain, /stream)
├── static/               # Zero-build shadcn/ui telemetry monitoring cockpit
│   ├── index.html        # High-density industrial HUD (Tailwind + Lucide + Chart.js)
│   └── app.js            # Streaming state, rolling strip-charts, TreeSHAP & sandbox
├── data/
│   ├── raw/              # Original immutable dataset (ai4i2020.csv)
│   └── processed/        # Cleaned, train/test split data (train.csv, test.csv)
├── models/               # Serialized pipelines (pipeline.joblib)
├── notebooks/            # Exploratory and prototyping notebooks
│   ├── 01_eda_initial_inspection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_imbalance_and_baselines.ipynb
│   ├── 04_pipeline_and_tuning.ipynb
│   └── 05_cost_calibration_and_shap.ipynb
├── reports/
│   └── figures/          # Evaluation plots and diagnostic figures
├── src/                  # Production-grade modular Python code
│   ├── features.py       # Physics domain feature engineering & sanitization
│   └── pipeline.py       # Scikit-learn Pipeline & ColumnTransformer
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## ⚡ API & Docker Quickstart

### Option A: Local Python Environment
Launch the FastAPI microservice directly:
```bash
uvicorn api.main:app --reload --port 8000
```

### Option B: Production Docker Container (Recommended)
Build the lightweight container image:
```bash
docker build -t predictive-maintenance:v1 .
```

Run the containerized microservice with port forwarding:
```bash
docker run -d --name pdm-engine -p 8000:8000 --rm predictive-maintenance:v1
```

Access services:
- **Interactive Telemetry Cockpit:** `http://localhost:8000/`
- **Swagger API Documentation:** `http://localhost:8000/docs`
- **Container Health Probing:** `http://localhost:8000/health`

Stop the container:
```bash
docker stop pdm-engine
```

---

## 🚀 Roadmap (10-Day Tactical Sprint)
- [x] **Day 1:** Project scaffold, dataset acquisition, initial EDA & target leakage audit.
- [x] **Day 2:** Physics feature engineering ($\Delta T$, Mechanical Power, Strain Index).
- [x] **Day 3:** Class imbalance handling (`scale_pos_weight`, PR-AUC baseline).
- [x] **Day 4:** Scikit-learn Pipeline & XGBoost model tuning.
- [x] **Day 5:** Cost matrix threshold calibration ($t^* = 0.020$) + TreeSHAP explainability.
- [x] **Day 6:** FastAPI microservice (`/health`, `/predict`, `/explain`, `/stream/tick`, `/stream/window`).
- [x] **Day 7:** Production Telemetry Dashboard & rolling Chart.js strip-charts.
- [x] **Day 8:** What-If Physics Sandbox, TreeSHAP root-cause attribution drawer & UI/UX polish.
- [x] **Day 9:** Production Docker containerization (`Dockerfile`, `.dockerignore`, build & local run).
- [ ] **Day 10:** Standalone GitHub repo graduation (`git subtree split`), live deployment (Hugging Face Spaces), & portfolio showcase.
