"""
pipeline.py - Production Scikit-Learn Pipeline Module
=====================================================
Builds an end-to-end, leak-proof Machine Learning Pipeline for the AI4I 2020
Predictive Maintenance system.

Components:
-----------
1. Physics Feature Engineering (FunctionTransformer):
   Computes domain thermal, power, and overstrain features.
2. Column Sanitization (FunctionTransformer):
   Cleans column names to lowercase, bracket-free identifiers.
3. Column Routing & Transformation (ColumnTransformer):
   - Ordinal encoding for machine quality tier ('L': 0, 'M': 1, 'H': 2).
   - Numerical sensor & physics feature passthrough.
   - Automatic dropping of arbitrary identifiers (UDI, Product ID) and target leakage labels.
4. Estimator:
   Cost-sensitive XGBoost classifier tuned for imbalanced industrial telemetry.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

from features import engineer_features, clean_column_names


# Raw numerical sensor streams
RAW_NUMERIC_FEATURES: List[str] = [
    'air_temperature_k',
    'process_temperature_k',
    'rotational_speed_rpm',
    'torque_nm',
    'tool_wear_min'
]

# Primary 3 physics-derived features (compact set)
PRIMARY_PHYSICS_FEATURES: List[str] = [
    'temp_diff',
    'power_w',
    'overstrain_ratio'
]

# Supplementary physics features (full set adds ratio and absolute overstrain)
SUPPLEMENTARY_PHYSICS_FEATURES: List[str] = [
    'temp_ratio',
    'overstrain'
]


def build_pipeline(
    scale_pos_weight: float = 28.52,
    use_compact_features: bool = False,
    random_state: int = 42,
    **xgb_params
) -> Pipeline:
    """
    Factory function that constructs a unified, production-ready Scikit-Learn Pipeline.

    Parameters:
    -----------
    scale_pos_weight : float, default=28.52
        Gradient balancing weight for minority failure class (~3.4% prevalence).
    use_compact_features : bool, default=False
        If True, only uses the 3 primary physics features (9 total features).
        If False, uses all 5 physics features (11 total features).
    random_state : int, default=42
        Reproducibility seed for the XGBoost estimator.
    **xgb_params :
        Arbitrary keyword arguments passed directly to xgb.XGBClassifier.

    Returns:
    --------
    Pipeline :
        Configured Scikit-learn Pipeline instance.
    """
    # 1. Physics Feature Transformer
    physics_transformer = FunctionTransformer(
        engineer_features,
        validate=False
    )

    # 2. Column Name Sanitizer
    sanitizer_transformer = FunctionTransformer(
        clean_column_names,
        validate=False
    )

    # 3. Determine numerical feature list based on feature selection strategy
    if use_compact_features:
        numeric_cols = RAW_NUMERIC_FEATURES + PRIMARY_PHYSICS_FEATURES
    else:
        numeric_cols = RAW_NUMERIC_FEATURES + PRIMARY_PHYSICS_FEATURES + SUPPLEMENTARY_PHYSICS_FEATURES

    # 4. ColumnTransformer router
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'categorical_tier',
                OrdinalEncoder(categories=[['L', 'M', 'H']]),
                ['type']
            ),
            (
                'numerical_sensors',
                'passthrough',
                numeric_cols
            )
        ],
        remainder='drop'  # Automatically drops UDI, Product ID, and any leakage columns
    )

    # 5. Cost-sensitive XGBoost estimator
    estimator = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        **xgb_params
    )

    # 6. Assemble into end-to-end Pipeline
    pipeline = Pipeline(steps=[
        ('physics_features', physics_transformer),
        ('column_sanitizer', sanitizer_transformer),
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])

    return pipeline


if __name__ == '__main__':
    # Smoke test on a raw mock observation
    raw_sample = pd.DataFrame([{
        'UDI': 10001,
        'Product ID': 'M14860',
        'Type': 'M',
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1551,
        'Torque [Nm]': 42.8,
        'Tool wear [min]': 0,
        'TWF': 0,
        'HDF': 0,
        'PWF': 0,
        'OSF': 0,
        'RNF': 0
    }])

    print("Building full pipeline (11 features)...")
    pipe_full = build_pipeline()
    print("Building compact pipeline (9 features)...")
    pipe_compact = build_pipeline(use_compact_features=True)

    print("Pipeline architecture verified successfully!")
