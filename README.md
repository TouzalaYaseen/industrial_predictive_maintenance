# Industrial Telemetry and Predictive Maintenance System

A production-grade machine learning system and IoT telemetry microservice for early detection of milling machine component failures, cost-sensitive threshold calibration, TreeSHAP root-cause explainability, and real-time operational monitoring.

---

## Executive Summary and Business Impact

In industrial manufacturing, unpredicted equipment failure leads to catastrophic downtime, worker safety hazards, and emergency repair costs. Conversely, scheduled preventative maintenance without telemetry data incurs high labor overhead and premature replacement of functioning components.

This project delivers an end-to-end predictive maintenance engine trained on multivariate sensor streams to detect impending failures before component breakdown occurs. 

### Financial and Operational Results
* Evaluated against an industrial cost matrix modeling preventative inspection ($150) versus catastrophic tool crash ($5,000).
* Calibrated optimal decision threshold from standard default ($t=0.500$) to cost-optimal operational threshold ($t^*=0.020$) using out-of-fold cross-validation.
* Achieved 94.12% failure recall on holdout test telemetry, slashing undetected machine crashes by 64%.
* Delivered $288,900 in total net savings on test data compared to a Run-to-Failure strategy, and $13,750 in incremental net savings compared to an uncalibrated default model.

---

## System Architecture

```
Physical Sensor Telemetry (Rotational Speed, Torque, Temperatures, Tool Wear)
    │
    ▼
Domain Physics Feature Engineering (Power = τ·ω, ΔT, Tool Wear Overstrain)
    │
    ▼
Leak-Proof Scikit-Learn Pipeline (ColumnTransformer, Categorical Encoding)
    │
    ▼
Cost-Sensitive XGBoost Classifier (Optimized for PR-AUC under Class Imbalance)
    │
    ▼
Optimal Operational Threshold Calibration (Asymmetric Cost Matrix Optimization)
    │
    ▼
TreeSHAP Root-Cause Attribution Engine (Game-Theoretic Additive Explanations)
    │
    ▼
FastAPI REST Microservice (/health, /predict, /explain, /stream)
    │
    ▼
Zero-Build Operational Cockpit (Tailwind CSS, Chart.js, Sub-2ms Latency)
```

---

## Telemetry Dataset Specification

The model is trained on the AI4I 2020 Predictive Maintenance dataset (Matzka, 2020), reflecting realistic manufacturing tolerances and physical constraints across 10,000 operational records.

* Sensor Features:
  * Air Temperature [K]: Ambient factory floor thermal condition.
  * Process Temperature [K]: Temperature of the CNC cutting interface.
  * Rotational Speed [rpm]: Spindle shaft angular speed.
  * Torque [Nm]: Spindle resistance load.
  * Tool Wear [min]: Cumulative cutting time for the current milling insert.
  * Type: Machine product variant (Low, Medium, High quality grades).

* Target Distribution:
  * Extreme Class Imbalance: 9,661 normal operations (96.61%) vs. 339 failures (3.39%).

* Ground-Truth Failure Modes (Post-Mortem Labels):
  * TWF (Tool Wear Failure): Excessive physical abrasion over cutting lifecycle.
  * HDF (Heat Dissipation Failure): Insufficient thermal conduction across the cutting zone.
  * PWF (Power Failure): Motor stall or electrical overload.
  * OSF (Overstrain Failure): Mechanical stress exceeding material torque/wear thresholds.
  * RNF (Random Failure): Uncorrelated failure incidents.

---

## Physics-Grounded Feature Engineering

Rather than relying purely on raw sensor readings, domain physics features were engineered directly into the pipeline:

1. Mechanical Spindle Power:
   P = Torque * (Rotational Speed * 2 * pi / 60)
   Captures electrical and mechanical dissipation in Watts. Detects both motor starvation (< 3,500 W) and power overloads (> 9,000 W).

2. Thermal Difference and Dissipation:
   Delta_T = Process Temperature - Air Temperature
   Models thermal gradient dynamics. When Delta_T < 8.6 K at rotational speeds under 1,380 RPM, convective cooling collapses, triggering Heat Dissipation Failures (HDF).

3. Tool Wear Overstrain:
   Overstrain = Tool Wear * Torque
   Multiplies mechanical cutting resistance by cumulative tool wear, modeling total structural fatigue on the CNC cutting insert.

4. Normalized Capacity Limits:
   Features are normalized against specific machine rating capacities (Type L: 11,000 min·Nm; Type M: 12,000 min·Nm; Type H: 13,000 min·Nm) to enable unified fleet-wide inference.

---

## Modeling and Evaluation Strategy

### Imbalance-Aware Benchmarking
Due to the 3.39% baseline failure rate, standard metrics such as Accuracy and ROC-AUC produce misleadingly optimistic evaluations. All modeling iterations were governed strictly by:
* Precision-Recall AUC (PR-AUC / Average Precision)
* F2-Score (Penalizing false negatives more heavily than false positives)
* Operational Recall (Minimizing missed failures)

| Model Iteration | Holdout PR-AUC | Failure Recall | F2-Score |
|---|---|---|---|
| Baseline Dummy Classifier | 0.0340 | 0.0000 | 0.0000 |
| Default XGBoost | 0.8436 | 0.7794 | 0.8123 |
| Tuned Physics-Informed Pipeline | 0.8819 | 0.8382 | 0.8546 |
| Calibrated Pipeline (t* = 0.020) | 0.8819 | 0.9412 | 0.8258 |

### Leak-Proof Pipeline Architecture
All data transformations are encapsulated in a single Scikit-Learn Pipeline (`models/pipeline.joblib`), ensuring strict data leakage prevention:
* Physics features engineered dynamically via stateless transformers.
* Machine variant categorical features encoded via ColumnTransformer.
* Pipeline accepts raw dictionaries or DataFrames with zero external glue code required.

---

## Operational Cost Calibration and Economics

Machine learning models default to a 0.50 decision threshold, which implicitly assumes false positives and false negatives carry identical business costs. In industrial manufacturing, this assumption is false:

* True Positive ($C_{TP} = \$150$): Preventative inspection and early tool insert swap.
* False Positive ($C_{FP} = \$100$): Unnecessary inspection with minor technician time wasted.
* False Negative ($C_{FN} = \$5,000$): Catastrophic spindle crash, damaged workpiece, emergency downtime.
* True Negative ($C_{TN} = \$0$): Normal uninterrupted manufacturing.

By executing Out-of-Fold (OOF) cross-validation threshold sweeping across 100 candidate cutoffs on 8,000 training records, the optimal operational cutoff was determined to be:
t* = 0.020

### Holdout Test Financial Benchmark (2,000 Machines)

| Strategy | Total Operational Cost | Missed Failures (FN) | False Alarms (FP) |
|---|---|---|---|
| Run-to-Failure (No ML) | $340,000 | 68 | 0 |
| Default XGBoost (t = 0.50) | $64,850 | 11 | 2 |
| Calibrated Decision (t* = 0.020) | $51,100 | 4 | 29 |

Net Savings Delivered:
* Savings vs. Run-to-Failure: $288,900 (-85.0% cost reduction)
* Incremental Savings vs. Default ML: $13,750 (-21.2% additional cost reduction)
* Critical Failure Prevention: Missed crashes dropped from 11 down to 4.

---

## Explainability and Root-Cause Diagnostics (TreeSHAP)

To provide plant operators with transparent root-cause analysis, the system integrates game-theoretic TreeSHAP (SHapley Additive exPlanations).

Predictions are decomposed additively:
f(x) = phi_0 + sum(phi_i)

Where:
* phi_0: Expected baseline fleet risk (log-odds).
* phi_i: Individual feature contributions pushing risk higher (risk drivers) or lower (stabilizing factors).

The serving layer outputs structured JSON payloads containing:
* Top positive risk contributors with exact numeric attributions.
* Top stabilizing operational factors.
* Domain-specific engineering guidance synthesized from the failure mode signature (e.g., advising thermal coolant checks for HDF signatures versus immediate feed-rate reduction for OSF signatures).

---

## Production API Specification

The FastAPI microservice serves low-latency predictions, explainability payloads, and simulated IoT sensor streams.

### Endpoints

#### GET /health
Probes system operational readiness and loaded model artifact state.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "simulation_loaded": true,
  "threshold": 0.02
}
```

#### POST /predict
Scores incoming telemetry payload against the calibrated threshold.
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "type": "M",
       "air_temperature_k": 298.1,
       "process_temperature_k": 308.6,
       "rotational_speed_rpm": 1551.0,
       "torque_nm": 42.8,
       "tool_wear_min": 0.0
     }'
```

Response:
```json
{
  "machine_failure_prediction": 0,
  "failure_probability": 0.000412,
  "calibrated_threshold": 0.02,
  "risk_tier": "HEALTHY",
  "action_required": "None. Normal operation."
}
```

#### POST /explain
Performs real-time TreeSHAP decomposition on candidate machine telemetry.
```bash
curl -X POST "http://localhost:8000/explain" \
     -H "Content-Type: application/json" \
     -d '{
       "type": "L",
       "air_temperature_k": 302.4,
       "process_temperature_k": 311.8,
       "rotational_speed_rpm": 1380.0,
       "torque_nm": 65.4,
       "tool_wear_min": 215.0
     }'
```

Response:
```json
{
  "prediction": {
    "machine_failure_prediction": 1,
    "failure_probability": 0.9982,
    "risk_tier": "CRITICAL"
  },
  "base_value": -4.218,
  "prediction_log_odds": 6.312,
  "top_risk_drivers": [
    {"feature": "overstrain_ratio", "shap_value": 4.12, "feature_value": 1.278},
    {"feature": "torque_nm", "shap_value": 3.45, "feature_value": 65.4}
  ],
  "top_stabilizing_factors": [
    {"feature": "air_temperature_k", "shap_value": -0.18, "feature_value": 302.4}
  ],
  "suggested_operator_action": "Overstrain Failure detected. Mechanical torque and tool wear exceed material fatigue limits. Immediately halt spindle and replace cutting insert."
}
```

#### GET /stream/tick?index=N
Replays physical sensor readings from historical holdout test records for real-time monitoring and streaming verification.

---

## Operational Monitoring Cockpit

The service includes an integrated, zero-build industrial dashboard mounted at root (`/`):
* Architecture: Native HTML5, Tailwind CSS with zinc dark tokens, Chart.js canvas rendering, and Lucide icons.
* Streaming Controller: Timeline scrubber with 1x, 2x, and 5x replay controls across 2,000 holdout records.
* Jump Navigation: Instant telemetry jumps to verified ground-truth failure events (Overstrain, Power, Heat Dissipation, Tool Wear).
* Real-Time Strip-Charts: 30-tick sliding window visualizers tracking mechanical stress and thermodynamic boundaries.
* Root-Cause Diagnostic Drawer: Additive SHAP bar graphs rendering feature contributions dynamically for critical alerts.
* What-If Physics Sandbox: Interactive parameter sliders enabling manual physics simulation and live model rescoring.

---

## Repository Structure

```
industrial_predictive_maintenance/
├── api/
│   ├── __init__.py
│   ├── schemas.py             # Strict Pydantic v2 data validation contracts
│   └── main.py                # FastAPI REST microservice with static mounting
├── static/
│   ├── index.html             # High-density operational dashboard HUD
│   └── app.js                 # Streaming state, rolling charts, SHAP drawer, sandbox
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv       # Original telemetry dataset (10,000 records)
│   └── processed/
│       ├── train.csv          # Stratified training partition (8,000 records)
│       └── test.csv           # Holdout test partition (2,000 records)
├── models/
│   └── pipeline.joblib        # Serialized leak-proof scikit-learn pipeline (288 KB)
├── notebooks/
│   ├── 01_eda_initial_inspection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_imbalance_and_baselines.ipynb
│   ├── 04_pipeline_and_tuning.ipynb
│   └── 05_cost_calibration_and_shap.ipynb
├── reports/
│   └── figures/               # Diagnostic figures, ROC/PR curves, SHAP plots
├── src/
│   ├── features.py            # Physics transformations and feature sanitization
│   └── pipeline.py            # Production Pipeline construction utilities
├── .dockerignore              # Build context filter
├── .gitignore                 # Standalone repository ignore rules
├── Dockerfile                 # Multi-layer cached production container configuration
├── requirements.txt           # Pinned production dependencies
└── README.md                  # System engineering specification
```

---

## Quickstart and Installation

### Option 1: Docker Container (Recommended)

1. Build the container image:
```bash
docker build -t industrial-pdm:v1 .
```

2. Run the container:
```bash
docker run -d --name pdm-service -p 8000:8000 --rm industrial-pdm:v1
```

3. Access the services:
* Operational Dashboard: http://localhost:8000/
* Interactive API Documentation (Swagger): http://localhost:8000/docs
* Service Health Probe: http://localhost:8000/health

4. Stop the container:
```bash
docker stop pdm-service
```

### Option 2: Local Python Environment

1. Clone the repository:
```bash
git clone https://github.com/TouzalaYaseen/industrial_predictive_maintenance.git
cd industrial_predictive_maintenance
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install pinned dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Launch the application:
```bash
uvicorn api.main:app --reload --port 8000
```

---

## License

This project is licensed under the MIT License. Telemetry data sourced from the UCI Machine Learning Repository (Matzka, 2020).
