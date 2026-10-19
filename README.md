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
├── api/                  # FastAPI serving layer (Pydantic schemas, endpoints)
├── data/
│   ├── raw/              # Original immutable dataset (ai4i2020.csv)
│   └── processed/        # Cleaned, train/test split data
├── models/               # Serialized pipelines (joblib artifacts)
├── notebooks/            # Exploratory and prototyping notebooks
│   └── 01_eda_initial_inspection.ipynb
├── src/                  # Production-grade modular Python code
│   ├── features.py       # Physics domain feature engineering
│   ├── pipeline.py       # Scikit-learn Pipeline & ColumnTransformer
│   └── train.py          # Training, cross-validation, and threshold tuning
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## 🚀 Roadmap (2-Week Sprint)
- [x] **Day 1:** Project scaffold, dataset acquisition, initial EDA & target leakage audit.
- [ ] **Day 2:** Physics feature engineering ($\Delta T$, Mechanical Power, Strain Index).
- [ ] **Day 3:** Class imbalance handling (scale_pos_weight, PR-AUC baseline).
- [ ] **Day 4:** Scikit-learn Pipeline & XGBoost model tuning.
- [ ] **Day 5:** Cost matrix threshold calibration + SHAP model explainability.
- [ ] **Day 6–7:** FastAPI microservice (`/predict`, `/explain`, `/simulate-stream`).
- [ ] **Day 8–9:** Interactive telemetry simulator frontend dashboard.
- [ ] **Day 10–12:** Docker containerization, standalone repo publication & live deployment.
