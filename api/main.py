"""
main.py - FastAPI Production Microservice for Industrial Predictive Maintenance
================================================================================
Exposes high-performance REST endpoints for CNC machine telemetry validation,
calibrated failure risk inference, TreeSHAP diagnostics, and continuous
IoT telemetry stream simulation.
"""

import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import shap

# Ensure 'src' is resolvable for joblib unpickling of custom FunctionTransformers
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODEL_PATH = BASE_DIR / "models" / "pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "raw" / "ai4i2020.csv"
STATIC_DIR = BASE_DIR / "static"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.schemas import (
    TelemetryInput,
    PredictionResponse,
    HealthResponse,
    ExplanationResponse,
    FeatureContribution,
    GroundTruthLabel,
    StreamTickResponse,
    StreamWindowResponse
)

# Application state container for cached global ML resources
ml_resources: Dict[str, Any] = {
    "pipeline": None,
    "classifier": None,
    "explainer": None,
    "feature_names": [],
    "simulation_data": None,
    "threshold": 0.020,
    "model_loaded": False,
    "version": "1.0.0"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads the serialized ML pipeline, initializes
    TreeSHAP explainer, and loads simulation telemetry once on server startup.
    """
    print(f"[STARTUP] Loading Scikit-Learn Pipeline from {MODEL_PATH}...")
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        
        pipeline = joblib.load(MODEL_PATH)
        ml_resources["pipeline"] = pipeline
        
        # Extract underlying classifier and preprocessor metadata
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        raw_names = preprocessor.get_feature_names_out().tolist()
        
        # Clean feature names (remove transformer prefixes like 'numerical_sensors__')
        clean_names = [name.split("__")[-1] for name in raw_names]
        ml_resources["classifier"] = classifier
        ml_resources["feature_names"] = clean_names
        
        # Initialize TreeSHAP explainer once at startup
        print("[STARTUP] Initializing TreeSHAP explainer engine...")
        ml_resources["explainer"] = shap.TreeExplainer(classifier)
        
        # Load continuous simulation dataset
        if DATA_PATH.exists():
            ml_resources["simulation_data"] = pd.read_csv(DATA_PATH)
            print(f"[STARTUP] Simulation dataset loaded ({len(ml_resources['simulation_data'])} records).")
        
        ml_resources["model_loaded"] = True
        print("[STARTUP] Pipeline, TreeSHAP, and Simulation Engine ready in memory.")
    except Exception as exc:
        print(f"[STARTUP ERROR] Failed to initialize ML resources: {exc}")
        ml_resources["model_loaded"] = False
    
    yield
    
    print("[SHUTDOWN] Releasing server resources...")
    ml_resources["pipeline"] = None
    ml_resources["classifier"] = None
    ml_resources["explainer"] = None
    ml_resources["simulation_data"] = None
    ml_resources["model_loaded"] = False


app = FastAPI(
    title="Industrial Predictive Maintenance Engine",
    description="Cost-calibrated telemetry monitoring, failure inference, TreeSHAP diagnostics, and IoT stream simulation for CNC milling machines.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for local development and dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files directory for telemetry dashboard
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """
    Serves the production industrial telemetry cockpit dashboard.
    Falls back to Swagger documentation if frontend files are missing.
    """
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/docs")


def _build_tick_response(index: int) -> StreamTickResponse:
    """
    Internal helper that constructs a single stream tick response,
    combining raw telemetry, real-time AI risk evaluation, and historical ground truth.
    """
    sim_data = ml_resources["simulation_data"]
    pipeline = ml_resources["pipeline"]

    if sim_data is None or len(sim_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Simulation dataset not loaded."
        )
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML pipeline not loaded."
        )

    total = len(sim_data)
    idx = index % total  # wrap around so stream loops indefinitely
    row = sim_data.iloc[idx]

    telemetry = TelemetryInput(
        machine_type=row["Type"],
        air_temperature_k=float(row["Air temperature [K]"]),
        process_temperature_k=float(row["Process temperature [K]"]),
        rotational_speed_rpm=float(row["Rotational speed [rpm]"]),
        torque_nm=float(row["Torque [Nm]"]),
        tool_wear_min=float(row["Tool wear [min]"])
    )

    df_raw = telemetry.to_raw_dataframe()
    proba = float(pipeline.predict_proba(df_raw)[0, 1])
    threshold = ml_resources["threshold"]
    failure_predicted = bool(proba >= threshold)

    if proba >= 0.50:
        risk_tier = "CRITICAL"
        action = "CRITICAL: Severe failure probability. Halt machine cycle immediately for overhaul."
    elif proba >= threshold:
        risk_tier = "WARNING"
        action = "WARNING: Elevated risk detected. Schedule preventative inspection before next shift."
    else:
        risk_tier = "HEALTHY"
        action = "HEALTHY: Operating within nominal physical parameters. Continue standard operation."

    prediction = PredictionResponse(
        machine_type=telemetry.machine_type,
        failure_predicted=failure_predicted,
        failure_probability=round(proba, 5),
        decision_threshold=threshold,
        risk_tier=risk_tier,
        recommended_action=action
    )

    active_modes = [
        col for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]
        if col in row and int(row[col]) == 1
    ]

    ground_truth = GroundTruthLabel(
        machine_failure=bool(int(row.get("Machine failure", 0)) == 1),
        active_failure_modes=active_modes
    )

    return StreamTickResponse(
        index=idx,
        total_ticks=total,
        telemetry=telemetry,
        prediction=prediction,
        ground_truth=ground_truth
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="Probe API health and model readiness"
)
async def health_check():
    """
    Returns system status and confirms whether the ML pipeline is loaded in memory.
    """
    is_ready = ml_resources["model_loaded"] and ml_resources["pipeline"] is not None
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        threshold=ml_resources["threshold"],
        version=ml_resources["version"]
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Evaluate machine failure risk against calibrated threshold t*=0.020"
)
async def predict_failure(telemetry: TelemetryInput):
    """
    Processes a raw telemetry packet:
    1. Validates physical sensor ranges via Pydantic.
    2. Packages data into raw DataFrame format.
    3. Runs pipeline feature engineering and XGBoost scoring.
    4. Evaluates against cost-optimal threshold (t* = 0.020).
    5. Returns categorized risk tier and actionable maintenance directive.
    """
    pipeline = ml_resources["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML pipeline is not loaded. System is degraded."
        )

    try:
        df_raw = telemetry.to_raw_dataframe()
        proba = float(pipeline.predict_proba(df_raw)[0, 1])
        threshold = ml_resources["threshold"]
        failure_predicted = bool(proba >= threshold)

        if proba >= 0.50:
            risk_tier = "CRITICAL"
            recommended_action = "CRITICAL: Severe failure probability. Halt machine cycle immediately for overhaul."
        elif proba >= threshold:
            risk_tier = "WARNING"
            recommended_action = "WARNING: Elevated risk detected. Schedule preventative inspection before next shift."
        else:
            risk_tier = "HEALTHY"
            recommended_action = "HEALTHY: Operating within nominal physical parameters. Continue standard operation."

        return PredictionResponse(
            machine_type=telemetry.machine_type,
            failure_predicted=failure_predicted,
            failure_probability=round(proba, 5),
            decision_threshold=threshold,
            risk_tier=risk_tier,
            recommended_action=recommended_action
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    summary="Generate real-time TreeSHAP root-cause feature attributions"
)
async def explain_failure(telemetry: TelemetryInput):
    """
    Generates granular root-cause diagnostics using TreeSHAP:
    1. Transforms raw telemetry once through pipeline preprocessing steps.
    2. Scores transformed features directly via cached XGBoost estimator.
    3. Extracts Shapley additive values to measure exact feature credit/fault.
    4. Automatically diagnoses the primary mechanical failure driver.
    """
    pipeline = ml_resources["pipeline"]
    classifier = ml_resources["classifier"]
    explainer = ml_resources["explainer"]
    feature_names = ml_resources["feature_names"]

    if any(res is None for res in [pipeline, classifier, explainer]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML explainability components are not loaded. System is degraded."
        )

    try:
        # Step 1: Optimized single-pass transformation
        df_raw = telemetry.to_raw_dataframe()
        X_transformed = pipeline[:-1].transform(df_raw)

        # Step 2: Inference on preprocessed data
        proba = float(classifier.predict_proba(X_transformed)[0, 1])
        threshold = ml_resources["threshold"]
        failure_predicted = bool(proba >= threshold)

        # Step 3: Compute TreeSHAP values for the 11 engineered features
        shap_vals = explainer.shap_values(X_transformed)[0]
        base_val = float(np.atleast_1d(explainer.expected_value)[0])

        # Step 4: Map values to feature names and determine directional impact
        contributions: List[FeatureContribution] = []
        for i, f_name in enumerate(feature_names):
            val = float(X_transformed[0, i])
            s_val = float(shap_vals[i])
            if s_val > 0.05:
                impact = "INCREASES_FAILURE_RISK"
            elif s_val < -0.05:
                impact = "DECREASES_FAILURE_RISK"
            else:
                impact = "NEUTRAL"

            contributions.append(FeatureContribution(
                feature=f_name,
                value=round(val, 3),
                shap_value=round(s_val, 4),
                impact=impact
            ))

        # Sort contributions by absolute SHAP impact (descending)
        contributions.sort(key=lambda item: abs(item.shap_value), reverse=True)

        # Step 5: Formulate automated root-cause diagnosis
        if failure_predicted:
            top_positive = next((c for c in contributions if c.shap_value > 0), None)
            if top_positive:
                top_f = top_positive.feature
                if top_f in ["overstrain_ratio", "overstrain", "tool_wear_min"]:
                    primary_concern = "Critical Overstrain: Excessive mechanical torque on worn cutting tool. Inspect and replace tool blade."
                elif top_f in ["power_w", "rotational_speed_rpm", "torque_nm"]:
                    primary_concern = "Electrical / Motor Power Anomaly: Spindle motor overload or stall condition. Inspect drive motor."
                elif top_f in ["temp_diff", "temp_ratio", "process_temperature_k"]:
                    primary_concern = "Heat Dissipation Failure: Thermal cooling gradient insufficient. Check coolant fluid and ventilation."
                else:
                    primary_concern = f"Elevated Risk: Primary failure driver is abnormal '{top_f}'."
            else:
                primary_concern = "Elevated Risk: Combined diffuse sensor anomalies."
        else:
            primary_concern = "Nominal Operation: Sensor telemetry indicates machine is operating safely within physical tolerances."

        return ExplanationResponse(
            machine_type=telemetry.machine_type,
            failure_predicted=failure_predicted,
            failure_probability=round(proba, 5),
            decision_threshold=threshold,
            base_value=round(base_val, 4),
            top_risk_drivers=contributions,
            primary_concern=primary_concern
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability execution failed: {str(exc)}"
        )


@app.get(
    "/stream/tick",
    response_model=StreamTickResponse,
    tags=["Telemetry Streaming"],
    summary="Fetch a single sequential telemetry tick with real-time AI risk scoring"
)
async def get_stream_tick(
    index: int = Query(0, ge=0, description="Sequential tick index in simulation dataset")
):
    """
    Returns a single operational tick, combining raw sensors, AI risk scoring, and ground truth.
    Allows clients full control over play, pause, step-forward, and scrub actions.
    """
    return _build_tick_response(index)


@app.get(
    "/stream/window",
    response_model=StreamWindowResponse,
    tags=["Telemetry Streaming"],
    summary="Prefetch a window of sequential telemetry ticks for timeline visualization"
)
async def get_stream_window(
    start: int = Query(0, ge=0, description="Starting tick index"),
    limit: int = Query(50, ge=1, le=200, description="Number of ticks to return (1-200)")
):
    """
    Returns a batch slice of consecutive ticks, ideal for initializing dashboard timeline charts.
    """
    sim_data = ml_resources["simulation_data"]
    if sim_data is None or len(sim_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Simulation dataset not loaded."
        )

    total = len(sim_data)
    ticks = [_build_tick_response(start + i) for i in range(limit)]
    return StreamWindowResponse(
        start_index=start,
        count=len(ticks),
        total_ticks=total,
        ticks=ticks
    )


@app.get(
    "/stream/live",
    tags=["Telemetry Streaming"],
    summary="Open a real-time Server-Sent Events (SSE) telemetry stream"
)
async def stream_live_telemetry(
    interval_sec: float = Query(1.0, ge=0.1, le=10.0, description="Delay between ticks in seconds"),
    start_index: int = Query(0, ge=0, description="Starting tick index")
):
    """
    Streams telemetry ticks sequentially as Server-Sent Events (SSE) at the requested interval.
    """
    async def event_generator():
        curr_idx = start_index
        while True:
            tick = _build_tick_response(curr_idx)
            yield f"data: {tick.model_dump_json()}\n\n"
            curr_idx += 1
            await asyncio.sleep(interval_sec)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
