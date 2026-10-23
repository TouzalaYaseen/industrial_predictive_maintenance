"""
schemas.py - Pydantic Data Contracts for Predictive Maintenance API
===================================================================
Defines runtime validation models, strict type safety, boundary checks,
and serialization schemas for incoming sensor telemetry and outgoing predictions.
"""

from typing import List, Literal
from pydantic import BaseModel, Field
import pandas as pd


class TelemetryInput(BaseModel):
    """
    Data contract for raw telemetry readings sent from CNC milling machines.
    Validates physical feasibility bounds to reject corrupt/noisy sensor data.
    """
    machine_type: Literal['L', 'M', 'H'] = Field(
        ...,
        description="Machine quality tier: 'L' (Low 50%), 'M' (Medium 30%), 'H' (High 20%)",
        examples=['M']
    )
    air_temperature_k: float = Field(
        ...,
        gt=0.0,
        description="Ambient air temperature in Kelvin (must be > 0 K)",
        examples=[300.0]
    )
    process_temperature_k: float = Field(
        ...,
        gt=0.0,
        description="Process operational temperature in Kelvin (must be > 0 K)",
        examples=[310.0]
    )
    rotational_speed_rpm: float = Field(
        ...,
        gt=0.0,
        description="Spindle rotational speed in RPM (must be > 0)",
        examples=[1538.0]
    )
    torque_nm: float = Field(
        ...,
        ge=0.0,
        description="Motor resistance torque in Newton-meters (must be >= 0)",
        examples=[41.2]
    )
    tool_wear_min: float = Field(
        ...,
        ge=0.0,
        description="Cumulative tool cutting time in minutes (must be >= 0)",
        examples=[120.0]
    )

    def to_raw_dataframe(self) -> pd.DataFrame:
        """
        Converts validated telemetry into a single-row Pandas DataFrame matching
        the exact column headers expected by the Scikit-Learn training pipeline.
        """
        return pd.DataFrame([{
            'Type': self.machine_type,
            'Air temperature [K]': float(self.air_temperature_k),
            'Process temperature [K]': float(self.process_temperature_k),
            'Rotational speed [rpm]': float(self.rotational_speed_rpm),
            'Torque [Nm]': float(self.torque_nm),
            'Tool wear [min]': float(self.tool_wear_min)
        }])


class PredictionResponse(BaseModel):
    """
    Data contract for model inference output, calibrated risk evaluation,
    and actionable industrial recommendations.
    """
    machine_type: Literal['L', 'M', 'H']
    failure_predicted: bool = Field(
        ...,
        description="True if calibrated failure probability >= threshold t* (0.020)"
    )
    failure_probability: float = Field(
        ...,
        description="Raw predicted failure probability from XGBoost [0.0 - 1.0]"
    )
    decision_threshold: float = Field(
        default=0.020,
        description="Cost-calibrated decision threshold (t* = 0.020)"
    )
    risk_tier: Literal['HEALTHY', 'WARNING', 'CRITICAL'] = Field(
        ...,
        description="Categorized operational risk level"
    )
    recommended_action: str = Field(
        ...,
        description="Directive for plant maintenance technicians"
    )


class FeatureContribution(BaseModel):
    """
    Represents an individual feature's TreeSHAP attribution score.
    """
    feature: str = Field(..., description="Feature identifier (e.g., 'power_w')")
    value: float = Field(..., description="Observed feature value for this machine")
    shap_value: float = Field(..., description="TreeSHAP attribution in log-odds impact")
    impact: Literal['INCREASES_FAILURE_RISK', 'DECREASES_FAILURE_RISK', 'NEUTRAL'] = Field(
        ...,
        description="Directional effect on failure likelihood"
    )


class ExplanationResponse(BaseModel):
    """
    Data contract for root-cause explainability responses.
    """
    machine_type: Literal['L', 'M', 'H']
    failure_predicted: bool
    failure_probability: float
    decision_threshold: float
    base_value: float = Field(..., description="Expected model log-odds across background fleet")
    top_risk_drivers: List[FeatureContribution] = Field(
        ...,
        description="Top features exerting upward or downward pressure on failure risk"
    )
    primary_concern: str = Field(
        ...,
        description="Automated root cause summary diagnosis"
    )


class HealthResponse(BaseModel):
    """
    Data contract for API health and model readiness probe.
    """
    status: Literal['healthy', 'degraded']
    model_loaded: bool
    threshold: float
    version: str


class GroundTruthLabel(BaseModel):
    """
    Ground truth post-mortem data from historical dataset for benchmarking.
    """
    machine_failure: bool = Field(..., description="True if machine experienced an actual failure")
    active_failure_modes: List[str] = Field(
        ...,
        description="List of active failure modes: TWF, HDF, PWF, OSF, RNF"
    )


class StreamTickResponse(BaseModel):
    """
    Data contract for a single sequential telemetry tick in simulation mode.
    """
    index: int = Field(..., description="Sequential operational tick index")
    total_ticks: int = Field(..., description="Total ticks available in simulator buffer")
    telemetry: TelemetryInput = Field(..., description="Raw sensor inputs")
    prediction: PredictionResponse = Field(..., description="Real-time model prediction")
    ground_truth: GroundTruthLabel = Field(..., description="Actual historical outcome")


class StreamWindowResponse(BaseModel):
    """
    Data contract for a historical slice/window of telemetry ticks.
    """
    start_index: int = Field(..., description="Starting tick index")
    count: int = Field(..., description="Number of ticks returned")
    total_ticks: int = Field(..., description="Total ticks available in simulator buffer")
    ticks: List[StreamTickResponse] = Field(..., description="List of sequential telemetry ticks")

