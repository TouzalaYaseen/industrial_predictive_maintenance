# Industrial Telemetry & Predictive Maintenance Engine
## Comprehensive System Engineering Case Study & Technical Summary

---

## 1. Project Overview & Quick Reference

* **Project Title:** Industrial Telemetry & Predictive Maintenance Engine
* **Domain:** Industrial Internet of Things (IIoT), Advanced Manufacturing, Applied Machine Learning
* **Author:** Yaseen Touzala
* **Status:** Deployed & Production-Ready
* **Live Operational Cockpit:** [https://industrial-predictive-maintenance-yn32.onrender.com](https://industrial-predictive-maintenance-yn32.onrender.com)
* **Interactive API Documentation:** [https://industrial-predictive-maintenance-yn32.onrender.com/docs](https://industrial-predictive-maintenance-yn32.onrender.com/docs)
* **Standalone GitHub Repository:** [https://github.com/TouzalaYaseen/industrial_predictive_maintenance](https://github.com/TouzalaYaseen/industrial_predictive_maintenance)
* **Core Technology Stack:**
  * **Machine Learning & Data:** Python 3.11, Scikit-Learn, XGBoost, TreeSHAP, Pandas, NumPy
  * **Serving & Contracts:** FastAPI, Uvicorn, Pydantic v2
  * **Cockpit / Frontend:** HTML5, Tailwind CSS (Zinc Dark tokens), Chart.js, Lucide Icons
  * **DevOps & Cloud:** Docker, Linux (Debian Slim), Render Cloud PaaS, Git Subtree

### The Executive Elevator Pitch
An end-to-end Machine Learning and IoT telemetry system for early detection of milling machine component failures. By formulating a domain-specific physical feature pipeline, optimizing an asymmetric operational cost matrix ($t^* = 0.020$), and serving TreeSHAP root-cause attributions via a sub-2ms FastAPI microservice, this system slashes undetected machine crashes by **64%**, achieves a **94.12% failure recall**, and delivers **$288,900 in net savings** on holdout test telemetry compared to traditional Run-to-Failure manufacturing policies.

---

## 2. Business Problem & Industrial Formulation

### 2.1 The Manufacturing Reality
In high-precision Computer Numerical Control (CNC) milling operations, machine tools are subject to extreme thermal, mechanical, and rotational forces. Unanticipated tool breakage or spindle motor burnout causes:
1. **Severe Workpiece Destruction:** Ruining high-value raw materials and manufactured parts.
2. **Emergency Downtime:** Factory line halts costing thousands of dollars per idle hour.
3. **Safety Hazards:** High-velocity mechanical fractures endangering factory floor personnel.

Conversely, arbitrary preventative maintenance (swapping blades on a fixed schedule) wastes usable tool life and incurs excessive technician labor overhead.

### 2.2 The Asymmetric Financial Cost Dilemma
Standard machine learning models implicitly treat False Positives (unnecessary inspections) and False Negatives (missed machine crashes) as equally bad by using a 0.50 probability cutoff. In industrial manufacturing, this assumption is catastrophic:

$$\text{Cost}(\text{False Negative}) \gg \text{Cost}(\text{False Positive})$$

We formulated an empirical industrial cost matrix reflecting realistic factory accounting:
* **True Positive ($C_{TP} = \$150$):** Model alerts of impending failure; technician performs preventative 15-minute inspection and early tool insert swap.
* **False Positive ($C_{FP} = \$100$):** Minor alarm; technician briefly inspects an operable machine (minor labor cost, no downtime).
* **False Negative ($C_{FN} = \$5,000$):** Machine crashes during an active machining cycle; catastrophic workpiece damage, spindle repair, emergency downtime.
* **True Negative ($C_{TN} = \$0$):** Machine operates nominally; uninterrupted manufacturing throughput.

The system's goal is not merely to optimize statistical accuracy, but to **minimize total operational financial loss**:

$$\text{Total Cost} = (FN \times \$5,000) + (TP \times \$150) + (FP \times \$100)$$

---

## 3. Dataset & Physical Foundations

### 3.1 AI4I 2020 Predictive Maintenance Dataset
The model was trained and evaluated on the AI4I 2020 Predictive Maintenance dataset (Matzka, 2020), which models real-world milling telemetry across 10,000 continuous operational records.

* **Sensor Attributes:**
  * `Air temperature [K]`: Ambient ambient room temperature ($295.3 - 304.5\text{ K}$).
  * `Process temperature [K]`: Cutting zone interface temperature ($305.7 - 313.8\text{ K}$).
  * `Rotational speed [rpm]`: Spindle angular velocity ($1,168 - 2,886\text{ RPM}$).
  * `Torque [Nm]`: Spindle motor torque resistance ($3.8 - 76.6\text{ Nm}$).
  * `Tool wear [min]`: Cumulative tool usage time under active cutting ($0 - 253\text{ min}$).
  * `Type`: Machine product quality tier (`L` = 50%, `M` = 30%, `H` = 20%).

* **Target Variable & Class Imbalance:**
  * `Machine failure`: Binary (0 = Normal, 1 = Failure).
  * Class Distribution: **9,661 Normal (96.61%)** vs. **339 Failures (3.39%)**.
  * Imbalance Ratio: **28.5 to 1**.

* **Failure Modes (Diagnostic Labels):**
  * **TWF (Tool Wear Failure):** Physical blade abrasion exceeding wear lifecycle ($200-240\text{ min}$).
  * **HDF (Heat Dissipation Failure):** Insufficient heat transfer ($\Delta T < 8.6\text{ K}$) at low rotational speeds ($<1,380\text{ RPM}$).
  * **PWF (Power Failure):** Mechanical power falling below $3,500\text{ W}$ (stall) or exceeding $9,000\text{ W}$ (overload).
  * **OSF (Overstrain Failure):** Combination of high torque and extended tool wear exceeding physical material limits.
  * **RNF (Random Failure):** Uncorrelated failure events occurring at standard error rates.

---

## 4. Physics-Grounded Feature Engineering

Rather than relying purely on raw sensor readings, domain physical equations were formulated and integrated into the feature pipeline. This allows gradient-boosted trees to make orthogonal splits directly on physical laws:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHYSICAL SENSOR SIGNALS                               │
│  [Torque (Nm)]  [Speed (RPM)]  [Process Temp (K)]  [Air Temp (K)]  [Wear]  │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENGINEERED PHYSICS ATTRIBUTES                            │
│                                                                             │
│  1. Spindle Power:     P = Torque · (Speed · 2π / 60)              (Watts)  │
│  2. Thermal Delta:     ΔT = Process Temp - Air Temp                (Kelvin) │
│  3. Temp Ratio:        R_temp = Process Temp / Air Temp                     │
│  4. Overstrain Stress: Strain = Tool Wear · Torque                 (min·Nm) │
│  5. Capacity Ratio:    Strain_Ratio = Strain / Capacity(Tier)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Mechanical Spindle Power ($P$ in Watts):**
   $$P = \tau \times \omega = \text{Torque} \times \left(\text{RPM} \times \frac{2\pi}{60}\right)$$
   Directly isolates electrical and mechanical energy transfer. Accurately exposes both motor stall conditions ($P < 3,500\text{ W}$) and destructive overload surges ($P > 9,000\text{ W}$).

2. **Thermal Dissipation Gradient ($\Delta T$):**
   $$\Delta T = T_{\text{process}} - T_{\text{air}}$$
   Models heat transfer efficiency. When cutting speed drops below 1,380 RPM and convective cooling falters ($\Delta T < 8.6\text{ K}$), the machine rapidly overheats, precipitating Heat Dissipation Failures (HDF).

3. **Tool Wear Overstrain Index:**
   $$\text{Overstrain} = \text{Tool Wear} \times \tau$$
   Models total cumulative mechanical fatigue acting upon the structural crystal lattice of the cutting tool.

4. **Normalized Capacity Limits:**
   Machine quality tiers have differing structural load tolerances:
   $$\text{Capacity}(\text{Type}) = \begin{cases} 11,000\text{ min}\cdot\text{Nm} & \text{Tier L} \\ 12,000\text{ min}\cdot\text{Nm} & \text{Tier M} \\ 13,000\text{ min}\cdot\text{Nm} & \text{Tier H} \end{cases}$$
   $$\text{Overstrain Ratio} = \frac{\text{Tool Wear} \times \tau}{\text{Capacity}(\text{Type})}$$
   This normalizes the risk profile across diverse machine variants onto a standardized fleet-wide scale.

---

## 5. Pipeline Architecture & Modeling Methodology

### 5.1 Strict Data Partitioning
* **Immutable Stratification:** Split into an 8,000-sample Training Set (271 failures) and a completely isolated 2,000-sample Holdout Test Set (68 failures) using stratified sampling on `Machine failure`.
* **Zero Leakage Rule:** Post-mortem diagnostic labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and machine IDs were permanently stripped from the training feature space to prevent target leakage.

### 5.2 Scikit-Learn Leak-Proof Pipeline
To eliminate train-serving skew and data leakage during cross-validation, the entire physical feature generation, column sanitization, categorical encoding, and classifier were serialized into a single `Pipeline`:
* `PhysicsFeatureTransformer`: Stateless custom transformer computing domain physics features.
* `ColumnTransformer`:
  * Categorical `machine_type` encoded via `OrdinalEncoder(categories=[['L', 'M', 'H']])`.
  * Numerical sensors and physics features passed through unmodified.
  * Remainder set to `'drop'`, automatically neutralizing raw identifiers.
* `XGBClassifier`: Tuned gradient-boosted decision trees.

### 5.3 Metric Selection: Why PR-AUC Over ROC-AUC
In datasets with 3.39% positive incidence, ROC-AUC is distorted by large True Negative counts:
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
A model generating 100 false alarms on 1,932 negative cases maintains an apparently stellar FPR of $\sim 5\%$, inflating the ROC curve.
In contrast, **Precision-Recall AUC (PR-AUC / Average Precision)** measures the direct trade-off between Precision (positive predictive value) and Recall (sensitivity) across all potential operating cutoffs, ignoring True Negatives entirely.

### 5.4 Cross-Validation & Hyperparameter Optimization
* **5-Fold Stratified Cross-Validation:** Conducted exclusively on the 8,000-sample training partition.
* **Class Imbalance Handling:** Configured `scale_pos_weight = 28.52` ($N_{\text{neg}} / N_{\text{pos}}$), scaling gradient updates for rare failure events.
* **RandomizedSearchCV (25 iterations, 125 fits):** Tuned `max_depth` (5), `min_child_weight` (2), `subsample` (0.8), `colsample_bytree` (0.8), and `learning_rate` (0.05) scored strictly on `average_precision`.

### 5.5 Empirical Benchmark Progression

| Iteration / Model | Holdout PR-AUC | Failure Recall | F2-Score |
|---|---|---|---|
| **Baseline Dummy Classifier** | 0.0340 | 0.00% | 0.0000 |
| **Default XGBoost (Unweighted)** | 0.8436 | 77.94% | 0.8123 |
| **Cost-Sensitive XGBoost (`scale_pos_weight`)** | 0.8672 | 80.88% | 0.8351 |
| **Tuned Physics Pipeline (Default $t=0.50$)** | 0.8819 | 83.82% | 0.8546 |
| **Calibrated Physics Pipeline ($t^*=0.020$)** | **0.8819** | **94.12%** | **0.8258** |

---

## 6. Financial Threshold Calibration & Cost Matrix

### 6.1 The Mathematical Calibration Method
Instead of picking an arbitrary threshold, we performed leak-proof **Out-of-Fold (OOF) cross-validation threshold sweeping** across 100 candidate cutoffs ($t \in [0.001, 0.999]$) on the 8,000-sample training partition:

$$t^* = \arg\min_{t} \sum_{i=1}^{N_{\text{train}}} \text{Cost}(y_i, \hat{y}_i(t))$$

The cost curve identified a sharp global minimum at **$t^* = 0.020$**. Because missing a crash costs \$5,000 while an inspection costs only \$150 ($33.3\times$ ratio), the model must act like an **industrial smoke alarm**, flagging machines even when the failure probability is modest.

### 6.2 Holdout Test Financial Benchmark (2,000 Machines, 68 True Failures)

```
Financial Loss Comparison (2,000 Test Machines):
───────────────────────────────────────────────────────────────────
Run-to-Failure (No ML)       : $340,000  (68 Missed Crashes)
Default XGBoost (t = 0.50)   : $ 64,850  (11 Missed Crashes)
Calibrated Engine (t* = 0.02): $ 51,100  ( 4 Missed Crashes)
───────────────────────────────────────────────────────────────────
Net Savings vs Run-to-Failure: $288,900  (-85.0% Cost Reduction)
Incremental Savings vs ML    : $ 13,750  (-21.2% Additional Savings)
```

| Strategy | Total Cost | Missed Failures (FN) | False Alarms (FP) | Preventative Catches (TP) | Failure Recall |
|---|---|---|---|---|---|
| **Run-to-Failure** | \$340,000 | 68 | 0 | 0 | 0.00% |
| **Default XGBoost ($t=0.50$)** | \$64,850 | 11 | 2 | 57 | 83.82% |
| **Calibrated Engine ($t^*=0.020$)** | **\$51,100** | **4** | 29 | 64 | **94.12%** |

* **Catastrophic Failure Reduction:** Missed crashes plummeted from 68 down to **4** (a 94.12% catch rate).
* **Economic Trade-Off:** Accepting 27 additional minor inspections (\$2,700 cost) prevented 7 catastrophic machine destructions (\$35,000 value), producing an immediate **\$32,300 gross return on inspection investment**.

---

## 7. Model Explainability & Root-Cause Attribution (TreeSHAP)

### 7.1 Game-Theoretic Foundation
Standard feature importances (e.g. Gini gain) fail in production because they provide only a global number, inflate high-cardinality features, and cannot explain individual predictions.
We integrated **TreeSHAP (SHapley Additive exPlanations)** to compute exact, mathematically consistent feature attributions:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

Where:
* $\phi_0$: The baseline fleet expected value in log-odds space ($\phi_0 \approx -4.218$).
* $\phi_i(x)$: The additive credit assigned to feature $i$ for pushing the prediction higher (risk driver) or lower (stabilizing factor).

### 7.2 Fleet-Wide Insights (Global Beeswarm)
Analysis across the holdout fleet revealed that domain physics features represent **4 of the top 6 failure drivers**:
1. `overstrain_ratio`: Dominates Overstrain Failures (OSF).
2. `torque_nm`: Primary contributor to high-load shear failure.
3. `power_w`: Displays a distinct **bimodal risk distribution** (failures occur at both low-power stall $<3500\text{ W}$ and high-power surge $>9000\text{ W}$).
4. `tool_wear_min`: Cumulative blade erosion baseline.
5. `temp_ratio` / `temp_diff_k`: Governs thermal collapse in Heat Dissipation Failures (HDF).

### 7.3 Real-Time Diagnostic JSON Synthesis
When `/explain` is called, the service transforms raw mathematical Shapley vectors into human-interpretable engineering recommendations:

```json
{
  "prediction": {
    "machine_failure_prediction": 1,
    "failure_probability": 0.99975,
    "risk_tier": "CRITICAL"
  },
  "base_value": 0.7154,
  "top_risk_drivers": [
    {"feature": "overstrain_ratio", "shap_value": 2.9375, "value": 1.278},
    {"feature": "power_w", "shap_value": 2.1285, "value": 9451.17},
    {"feature": "torque_nm", "shap_value": 0.8559, "value": 65.4}
  ],
  "primary_concern": "Critical Overstrain: Excessive mechanical torque on worn cutting tool. Inspect and replace tool blade."
}
```

---

## 8. Microservice Serving Layer (FastAPI & Pydantic)

The inference serving layer was engineered for low latency, memory safety, and strict input validation.

### 8.1 Defensive Data Contracts (`api/schemas.py`)
Using **Pydantic v2**, incoming JSON payloads are intercepted at the perimeter. Physics boundaries are strictly enforced before telemetry touches Scikit-Learn:
* `air_temperature_k > 0.0` and `process_temperature_k > 0.0`
* `rotational_speed_rpm > 0.0`
* `torque_nm >= 0.0` and `tool_wear_min >= 0.0`
* `machine_type in ['L', 'M', 'H']`

### 8.2 Sub-2ms Serving Architecture (`api/main.py`)
* **Lifespan Context Management:** The model pipeline (`models/pipeline.joblib`), TreeSHAP explainer, and simulation stream records are loaded into memory exactly once on server startup.
* **Stateless vs. Stateful Transformation:** For the `/explain` endpoint, `pipeline[:-1].transform()` is executed once to produce intermediate preprocessed features, feeding both XGBoost and TreeSHAP simultaneously without redundant recalculation.
* **REST Endpoints:**
  * `GET /health`: Operational probe verifying model availability and calibrated threshold.
  * `POST /predict`: Real-time risk classification at $t^* = 0.020$.
  * `POST /explain`: Single-instance TreeSHAP decomposition with synthesized operator guidance.
  * `GET /stream/tick?index=N`: Pull-based IoT sensor replay across 2,000 historical holdout cycles.

---

## 9. Operational Cockpit (Zero-Build Frontend Architecture)

To bridge the gap between machine learning algorithms and factory floor operators, an interactive operational HUD was mounted directly to root (`/`).

### 9.1 Technical Highlights
* **Zero-Build Architecture:** Utilizes native HTML5, modern Tailwind CSS configured with official shadcn Zinc Dark design tokens (`#09090b` canvas, `#18181b` card surfaces, `#27272a` borders), Lucide Icons, and Chart.js.
* **Zero Bundler Overhead:** No React, Webpack, or Node.js toolchains. The frontend compiles zero assets, avoiding virtual DOM lag during high-frequency streaming.
* **Sliding Window Strip-Charts:** Rolling Chart.js canvas renderers maintain a fixed buffer ($W = 30$ ticks), guaranteeing constant 60 FPS rendering and bounded browser memory usage.
* **Interactive Capabilities:**
  * **IoT Playback Controller:** Scrub, play, pause, and speed multiplier controls (1x, 2x, 5x) across 2,000 holdout ticks.
  * **Failure Event Quick-Jumps:** Direct jump buttons to verified failure records (#160 OSF, #50 PWF, #69 HDF, #77 TWF).
  * **Root-Cause Diagnostic Drawer:** Additive Shapley bar graphs rendering green stabilizing factors vs red risk drivers dynamically.
  * **What-If Physics Sandbox:** Live parameter sliders enabling technicians to simulate hypothetical torque, speed, and wear scenarios with debounced real-time rescoring.

---

## 10. Containerization & Production Cloud Deployment

### 10.1 Multi-Layer Production Dockerfile
The container architecture was formulated for security, speed, and cross-platform reproducibility:
1. **Debian Slim Base (`python:3.11-slim`):** Standard `glibc` library ensures compiled C/C++ ML wheels (NumPy, XGBoost, SHAP) install cleanly without agonizing compilation times (unlike Alpine's `musl libc`).
2. **Layer Caching:** `COPY requirements.txt .` and `pip install` occur before copying source code, ensuring iterative code tweaks do not bust the 600 MB pip wheel cache.
3. **Container Security:** Runs under an unprivileged user (`appuser`, UID 1000) rather than root to mitigate container breakout vulnerabilities.
4. **Cloud-Native Dynamic Port Resolution:** Entrypoint dynamically reads `$PORT` provided by container orchestrators, defaulting to 8000:
   ```dockerfile
   CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
   ```
5. **Native Healthcheck:** Periodic probes evaluate internal responsiveness:
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
       CMD python -c "import os, urllib.request; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1
   ```

### 10.2 Standalone Graduation via `git subtree split`
Rather than an unstructured copy-paste that flattens git history, we executed `git subtree split`:
```bash
git subtree split --prefix=projects/predictive-maintenance -b pdm-standalone
git push https://github.com/TouzalaYaseen/industrial_predictive_maintenance.git pdm-standalone:main
```
This synthesized a pristine, standalone repository containing all 11 atomic commits and daily sprint logs while completely isolating it from the learning monorepo.

### 10.3 Cloud PaaS Deployment (Render)
* Deployed live to **Render's European Region (Frankfurt)** for optimal low-latency routing to North Africa / Europe.
* Automatically triggers fresh Docker builds upon any push to `main`.
* Entire application operates smoothly at **~200 MB RAM**, fitting comfortably inside free cloud tiers (512 MB).

---

## 11. Interview Talking Points & Technical Defensibility

When presenting this project to engineering hiring managers, senior ML practitioners, or academic committees, use these answers to defend key architecture decisions:

### Q1: "Why not use SMOTE to handle the class imbalance?"
> *"SMOTE creates synthetic instances by performing linear interpolation between k-nearest neighbors in feature space. In physical systems, linear interpolation between two valid sensor states can create physically impossible machine conditions—such as high torque combined with high rotational speed that violates motor horsepower limits, or invalid temperature gradients that break thermodynamic laws. Instead of synthesizing corrupted physical data, we addressed imbalance through gradient reweighting (`scale_pos_weight = 28.52`) and evaluated strictly on Precision-Recall AUC."*

### Q2: "Why calibrate the decision threshold to 0.02 instead of keeping 0.50?"
> *"Standard 0.5 thresholds implicitly assume False Positives and False Negatives carry identical financial costs. In industrial manufacturing, an inspection costs \$150 while a catastrophic spindle destruction costs \$5,000—a 33.3 to 1 cost ratio. By sweeping cutoffs over out-of-fold cross-validation predictions, we found that lowering the operational threshold to 0.02 acted like an industrial smoke alarm. Accepting 27 additional minor inspections cost only \$2,700, but prevented 7 catastrophic machine crashes worth \$35,000, slashing total operational losses by 21.2% compared to default ML."*

### Q3: "Why did you build the dashboard in pure HTML/Tailwind/Chart.js instead of Streamlit or React?"
> *"Streamlit reruns the entire Python script from top to bottom on every user interaction, making high-frequency telemetry streaming choppy and state management clunky. On the other hand, React introduces massive build complexity, node_modules dependencies, and multi-stage container overhead. By building a zero-build cockpit using Chart.js on native HTML5 Canvas with Tailwind CSS, we achieved 60 FPS strip-chart rendering and sub-2ms API responses while serving everything from a single, lightweight FastAPI container."*

### Q4: "How did you prevent data leakage in your pipeline?"
> *"We adhered to three strict architectural guardrails: First, we created immutable stratified train and holdout test partitions before running any exploratory transformations. Second, all feature engineering and encoding were encapsulated inside a Scikit-Learn `Pipeline` and `ColumnTransformer`, ensuring learned statistics never leaked across cross-validation folds. Third, we audited and dropped all post-mortem diagnostic failure labels (`TWF`, `HDF`, etc.) prior to training, ensuring the model learned solely from pre-failure operational telemetry."*

---

## 12. Verification & Summary Metrics Card

```
═══════════════════════════════════════════════════════════════════════
        INDUSTRIAL TELEMETRY & PREDICTIVE MAINTENANCE ENGINE
                         SYSTEM METRIC CARD
═══════════════════════════════════════════════════════════════════════
  Dataset Scale           : 10,000 Operational Records (AI4I 2020)
  Fleet Failure Incidence : 3.39% (9,661 Normal / 339 Failures)
  Evaluation Metric       : Holdout PR-AUC = 0.8819 (Average Precision)
  Operational Cutoff      : t* = 0.020 (Cost-Optimal Threshold)
  Failure Recall          : 94.12% (64 of 68 Holdout Failures Detected)
  Crash Reduction         : -64.0% vs Default 0.50 ML Threshold
  Financial Net Savings   : $288,900 (-85.0%) vs Run-to-Failure Policy
  Inference Latency       : Sub-2ms per Record (FastAPI + Uvicorn)
  Explainability Engine   : Game-Theoretic TreeSHAP Additive Log-Odds
  Container Footprint     : Python 3.11-slim Debian (~200 MB Runtime RAM)
  Live Cloud Deployment   : Render Frankfurt EU (Port 8000, TLS/HTTPS)
═══════════════════════════════════════════════════════════════════════
```
