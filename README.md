# Industrial Telemetry and Predictive Maintenance System

A deployed demo machine learning system and IoT telemetry microservice for early detection of milling machine component failures, cost-sensitive threshold calibration, TreeSHAP root-cause explainability, and simulated real-time monitoring.

* Live Operational Cockpit: https://industrial-predictive-maintenance-yn32.onrender.com
* Interactive API Documentation (Swagger): https://industrial-predictive-maintenance-yn32.onrender.com/docs
* Live Health Probe: https://industrial-predictive-maintenance-yn32.onrender.com/health
* Note on Cloud Cold Starts: Hosted on Render's free tier. The container spins down after 15 minutes of inactivity; initial page loads and curl commands take approximately 40 seconds to wake the service. Subsequent requests respond immediately.

---

## Summary and Simulated Impact

In industrial manufacturing, unpredicted equipment failure leads to catastrophic downtime, worker safety hazards, and emergency repair costs. Conversely, scheduled preventative maintenance without telemetry data incurs high labor overhead and premature replacement of functioning components.

This project delivers an end-to-end predictive maintenance engine trained on multivariate sensor streams to detect impending failures before component breakdown occurs. 

### Financial and Operational Results (Simulated Under Assumed Cost Matrix)
* Evaluated against an assumed industrial cost matrix modeling preventative inspection ($150) versus catastrophic tool crash ($5,000).
* Calibrated optimal decision threshold from standard default ($t=0.500$) to cost-optimal operational threshold ($t^*=0.020$) using out-of-fold cross-validation on the training set.
* Reduced missed machine failures from 11 to 4 versus the default threshold (achieving 94.12% failure recall on holdout test telemetry).
* Delivered simulated net savings of $288,900 on the holdout test partition compared to a Run-to-Failure strategy, and $13,750 in incremental net savings compared to an uncalibrated default model (under the assumed cost model).

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
Zero-Build Operational Cockpit (Tailwind CSS, Chart.js, Sub-2ms Model Inference)
```

---

## Telemetry Dataset Specification

The model is trained on the AI4I 2020 Predictive Maintenance dataset (Matzka, 2020), a synthetic benchmark dataset generated to reflect manufacturing tolerances and physical constraints across 10,000 operational records.

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
  * TWF (Tool Wear Failure): Blade abrasion exceeding wear lifecycle.
  * HDF (Heat Dissipation Failure): Insufficient heat transfer across the cutting zone.
  * PWF (Power Failure): Motor stall or electrical overload.
  * OSF (Overstrain Failure): Mechanical stress exceeding material torque/wear thresholds.
  * RNF (Random Failure): Uncorrelated failure incidents.

Note on Data Generating Process: Ground-truth failure labels in the AI4I synthetic dataset were constructed from mathematical rules over these same sensor variables (e.g., power limits, temperature differences, torque-wear products). Consequently, strong model discriminative scores are partly expected because the feature pipeline directly models the equations governing the data generation.

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
   Features are normalized against the dataset's generating failure thresholds for Overstrain Failure (Type L: 11,000 min·Nm; Type M: 12,000 min·Nm; Type H: 13,000 min·Nm) as defined in Matzka (2020).

---

## Modeling and Evaluation Strategy

### Data Partitioning and Hyperparameter Optimization Guardrails
To prevent data leakage, evaluation was structured around strict validation boundaries:
* Stratified Partitioning: An 80/20 stratified split (`random_state=42`) separated 8,000 training records from 2,000 holdout test records, preserving the 3.39% failure incidence across both partitions.
* Systematic Hyperparameter Tuning: Conducted using `RandomizedSearchCV` across 25 parameter configurations over 5-fold stratified cross-validation (125 total fits) scored strictly on `average_precision` (PR-AUC). Tuned parameters: `max_depth = 5`, `learning_rate = 0.05`, `subsample = 0.8`, `colsample_bytree = 0.8`, and `min_child_weight = 2`.
* Single Holdout Evaluation: The 2,000 holdout records were locked away during feature exploration and tuning, and scored only once at the conclusion of model selection.

### Imbalance-Aware Benchmarking
Due to the 3.39% baseline failure rate, standard metrics such as Accuracy and ROC-AUC produce misleadingly optimistic evaluations. All modeling iterations were governed strictly by:
* Precision-Recall AUC (PR-AUC / Average Precision)
* Operational Recall (Minimizing catastrophic missed failures)
* Financial Cost under the Asymmetric Cost Matrix

| Model Iteration | Holdout PR-AUC | Failure Recall | F2-Score |
|---|---|---|---|
| Baseline Dummy Classifier | 0.0340 | 0.00% | 0.000 |
| Default XGBoost Pipeline (Untuned) | 0.8659 | 82.35% | 0.828 |
| Tuned Physics Pipeline (t = 0.50) | 0.8819 | 83.82% | 0.833 |
| Calibrated Pipeline (t* = 0.020) | 0.8819 | 94.12% | 0.581 |

Note: F2 is not the optimization target; the threshold minimizes simulated operational cost, so F2 drops as precision falls.

### Feature Ablation Study
During pipeline development on the 8,000-sample training partition (5-fold cross-validation):
* Raw 6 Features: PR-AUC = 0.8517 +/- 0.0412
* Physics-Enriched 11 Features: PR-AUC = 0.8556 +/- 0.0343

The numerical PR-AUC gain (+0.0039) is within cross-validation variance. However, physics features were retained for two architectural reasons:
1. Variance Reduction: Standard deviation across folds decreased by approximately 17%, indicating more stable generalization.
2. Domain Interpretability: Raw speeds and torques cannot indicate why a machine failed; explicit power and overstrain ratios allow TreeSHAP to attribute motor stall versus tool fatigue directly.

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

### Threshold Stability Basin
Out-of-fold threshold sweeping reveals that the optimal cutoff is not a brittle, overfitted point, but a resilient operating basin. Across thresholds between $t = 0.015$ and $t = 0.035$, total simulated operational cost remains stably within 5% of the absolute minimum ($164,700 +/- $8,000 on training OOF predictions). This allows factory operators to tune sensitivity within this range without significant economic degradation.

The full Out-of-Fold sweep curve is plotted in:
`reports/figures/day5_cost_vs_threshold_curve.png`

### Holdout Test Financial Benchmark (2,000 Machines | 68 True Failures)

Full confusion matrix and simulated costs under the assumed cost matrix:

| Strategy | Total Cost | TP | FP | FN | TN | Failure Recall |
|---|---|---|---|---|---|---|
| Run-to-Failure (No ML) | $340,000 | 0 | 0 | 68 | 1,932 | 0.00% |
| Default XGBoost (t = 0.50) | $64,850 | 57 | 13 | 11 | 1,919 | 83.82% |
| Calibrated Decision (t* = 0.020) | $51,100 | 64 | 215 | 4 | 1,717 | 94.12% |

Simulated Financial Savings:
* Savings vs. Run-to-Failure: $288,900 (-85.0% cost reduction)
* Incremental Savings vs. Default ML: $13,750 (-21.2% additional cost reduction)
* Critical Failure Prevention: Missed crashes dropped from 11 down to 4.

---

## Explainability and Root-Cause Diagnostics (TreeSHAP)

To provide plant operators with transparent root-cause analysis, the system integrates game-theoretic TreeSHAP (SHapley Additive exPlanations).

Predictions are decomposed additively in margin (log-odds) space:
Margin(x) = phi_0 + sum(phi_i)

Where:
* phi_0: Expected baseline leaf margin across the tree ensemble, equal to 0.7154. In TreeSHAP, this represents the cover-weighted mean margin, where scale_pos_weight = 28.52 inflates the positive failure instances' Hessian weights in tree covers, shifting the expected leaf margin upward into positive log-odds.
* phi_i: Individual feature contributions pushing risk higher (risk drivers) or lower (stabilizing factors).
* Probability mapping: Calculated through the logistic link function P(failure) = 1 / (1 + exp(-Margin(x))).

---

## API Specification

The FastAPI microservice serves low-latency predictions, explainability payloads, and simulated IoT sensor streams. Model inference executes on CPU in under 2ms per record (excluding network propagation latency).

### Endpoints

#### GET /health
Probes system operational readiness and loaded model artifact state.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "threshold": 0.02,
  "version": "1.0.0"
}
```

#### POST /predict
Scores incoming telemetry payload against the calibrated threshold.
```bash
# Note: If the container is sleeping on Render's free tier, curl requests may take ~40 seconds on cold start.
curl -X POST "https://industrial-predictive-maintenance-yn32.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "machine_type": "M",
       "air_temperature_k": 298.1,
       "process_temperature_k": 308.6,
       "rotational_speed_rpm": 1551.0,
       "torque_nm": 42.8,
       "tool_wear_min": 0.0
     }'
```

Actual Response:
```json
{
  "machine_type": "M",
  "failure_predicted": false,
  "failure_probability": 0.00095,
  "decision_threshold": 0.02,
  "risk_tier": "HEALTHY",
  "recommended_action": "HEALTHY: Operating within nominal physical parameters. Continue standard operation."
}
```

#### POST /explain
Performs real-time TreeSHAP decomposition on candidate machine telemetry.
```bash
# Note: If the container is sleeping on Render's free tier, curl requests may take ~40 seconds on cold start.
curl -X POST "https://industrial-predictive-maintenance-yn32.onrender.com/explain" \
     -H "Content-Type: application/json" \
     -d '{
       "machine_type": "L",
       "air_temperature_k": 302.4,
       "process_temperature_k": 311.8,
       "rotational_speed_rpm": 1380.0,
       "torque_nm": 65.4,
       "tool_wear_min": 215.0
     }'
```

Actual Response:
```json
{
  "machine_type": "L",
  "failure_predicted": true,
  "failure_probability": 0.99975,
  "decision_threshold": 0.02,
  "base_value": 0.7154,
  "top_risk_drivers": [
    {
      "feature": "overstrain_ratio",
      "value": 1.278,
      "shap_value": 2.9375,
      "impact": "INCREASES_FAILURE_RISK"
    },
    {
      "feature": "power_w",
      "value": 9451.167,
      "shap_value": 2.1285,
      "impact": "INCREASES_FAILURE_RISK"
    },
    {
      "feature": "torque_nm",
      "value": 65.4,
      "shap_value": 0.8559,
      "impact": "INCREASES_FAILURE_RISK"
    },
    {
      "feature": "rotational_speed_rpm",
      "value": 1380.0,
      "shap_value": 0.7391,
      "impact": "INCREASES_FAILURE_RISK"
    },
    {
      "feature": "tool_wear_min",
      "value": 215.0,
      "shap_value": 0.6703,
      "impact": "INCREASES_FAILURE_RISK"
    }
  ],
  "primary_concern": "Critical Overstrain: Excessive mechanical torque on worn cutting tool. Inspect and replace tool blade."
}
```

#### GET /stream/tick?index=N
Replays physical sensor readings from historical holdout test records for simulated real-time monitoring and streaming verification.

---

## Operational Monitoring Cockpit

The service includes an integrated, zero-build industrial dashboard mounted at root (`/`):
* Architecture: Native HTML5, Tailwind CSS with zinc dark tokens, Chart.js canvas rendering, and Lucide icons.
* Streaming Controller: Timeline scrubber with 1x, 2x, and 5x replay controls across 2,000 holdout records.
* Jump Navigation: Instant telemetry jumps to verified ground-truth failure events (Overstrain, Power, Heat Dissipation, Tool Wear).
* Real-Time Strip-Charts: 30-tick sliding window visualizers tracking mechanical stress and thermodynamic boundaries.
* Root-Cause Diagnostic Drawer: Additive SHAP bar graphs rendering feature contributions dynamically for critical alerts.
* What-If Physics Sandbox: Interactive parameter sliders enabling manual physics simulation and live model rescoring.
* Performance: Sub-2ms server-side inference allows instantaneous chart and metric updates.

---

## Project Limitations

1. Synthetic Benchmark Data: The AI4I 2020 dataset is synthetically generated using physical rules rather than collected from live industrial factory telemetry.
2. Assumed Operational Cost Matrix: The economic parameters ($150 inspection, $5,000 breakdown, $100 false alarm) are stylized assumptions for demonstration; real factory savings require plant-specific accounting.
3. Holdout Sample Size: The holdout test set contains 68 failure events, so absolute failure count differences are subject to small-sample variance.
4. No Empirical Bootstrapping: Confidence intervals on cost savings were not generated via bootstrapping.

---

## Reproducing the Benchmark Results

A dedicated evaluation script is included to reproduce all model progression benchmarks, confusion matrices, and financial costs directly:

```bash
python src/evaluate.py
```

Expected output:
```text
======================================================================================
1. MODEL PROGRESSION BENCHMARK (HOLDOUT TEST: 2,000 SAMPLES | 68 FAILURES)
======================================================================================
Model Iteration                      | Holdout PR-AUC | Failure Recall | F2-Score
--------------------------------------------------------------------------------------
Baseline Dummy Classifier            | 0.0340         | 0.00%          | 0.000   
Default XGBoost Pipeline (Untuned)   | 0.8659         | 82.35%         | 0.828   
Tuned Physics Pipeline (t = 0.50)    | 0.8819         | 83.82%         | 0.833   
Calibrated Pipeline (t* = 0.020)     | 0.8819         | 94.12%         | 0.581   
======================================================================================
Note: F2 is not the optimization target; the threshold minimizes simulated operational
cost, so F2 drops as precision falls.

======================================================================================
2. HOLDOUT FINANCIAL BENCHMARK (ASSUMED COST MATRIX: $5k FN / $150 TP / $100 FP)
======================================================================================
Strategy                         | Total Cost  | TP   | FP    | FN   | TN    | Recall 
--------------------------------------------------------------------------------------
Run-to-Failure (No ML)           | $340,000    | 0    | 0     | 68   | 1932  | 0.00%
Default XGBoost (t = 0.50)       | $64,850     | 57   | 13    | 11   | 1919  | 83.82%
Calibrated Decision (t* = 0.020) | $51,100     | 64   | 215   | 4    | 1717  | 94.12%
======================================================================================
SIMULATED FINANCIAL SAVINGS (UNDER ASSUMED COST MATRIX):
  * Savings vs. Run-to-Failure: $288,900 (-85.0%)
  * Savings vs. Default ML:     $13,750 (-21.2%)
  * Critical Failure Reduction: Missed crashes dropped from 11 down to 4.
======================================================================================
```

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
│   ├── pipeline.py            # Production Pipeline construction utilities
│   └── evaluate.py            # Automated evaluation script reproducing all tables
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

## License & Attribution

* Codebase License: The source code, API services, and front-end dashboard are licensed under the MIT License.
* Dataset Attribution & License: The AI4I 2020 Predictive Maintenance Dataset is sourced from the UCI Machine Learning Repository (Matzka, 2020) and is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
  * Citation: Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
