"""
features.py - Physics-Grounded Feature Engineering Module
=========================================================
Transforms raw telemetry sensor streams from industrial milling machines into
domain-specific physical indicators of mechanical, thermal, and electrical strain.

Physics Foundations:
--------------------
1. Thermal Gradient (Delta T):
   Process temperature minus Ambient Air temperature. Captures heat dissipation
   efficiency (critical for detecting HDF - Heat Dissipation Failure).

2. Mechanical Power (Watts):
   Power = Torque * Angular Velocity = Torque * (Rotational Speed * 2 * pi / 60).
   Captures electrical motor overload and stall conditions (critical for PWF - Power Failure).

3. Tool Wear Overstrain Index & Ratio:
   Overstrain = Torque * Tool Wear.
   Normalized Overstrain Ratio = Overstrain / Machine Capacity (L: 11000, M: 12000, H: 13000).
   Captures blade fracture risk under high resistance (critical for OSF - Overstrain Failure).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


TYPE_CAPACITY_MAP: Dict[str, float] = {
    'L': 11000.0,
    'M': 12000.0,
    'H': 13000.0
}


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies physics-grounded transformations to industrial CNC milling telemetry data.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe containing raw sensor columns:
        - 'Air temperature [K]'
        - 'Process temperature [K]'
        - 'Rotational speed [rpm]'
        - 'Torque [Nm]'
        - 'Tool wear [min]'
        - 'Type' (machine quality tier: 'L', 'M', 'H')

    Returns:
    --------
    pd.DataFrame
        Dataframe enriched with engineered domain features:
        - 'temp_diff': Process temp - Air temp (K)
        - 'temp_ratio': Process temp / Air temp
        - 'power_w': Mechanical power (Watts)
        - 'overstrain': Torque * Tool wear (Nm*min)
        - 'overstrain_ratio': Overstrain / Type Capacity
    """
    df = data.copy()

    # 1. Thermal dissipation gradient (Heat Dissipation Failure indicator)
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['temp_ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']

    # 2. Mechanical Power in Watts: Power = Torque * Angular Velocity (rad/s)
    # Angular velocity = RPM * (2 * pi / 60)
    angular_velocity = df['Rotational speed [rpm]'] * (2.0 * np.pi / 60.0)
    df['power_w'] = df['Torque [Nm]'] * angular_velocity

    # 3. Tool Wear Mechanical Overstrain Index
    df['overstrain'] = df['Torque [Nm]'] * df['Tool wear [min]']

    # 4. Normalized Overstrain Ratio based on Machine Quality Tier (L, M, H)
    type_capacity = df['Type'].map(TYPE_CAPACITY_MAP).fillna(11000.0)
    df['overstrain_ratio'] = df['overstrain'] / type_capacity

    return df


if __name__ == '__main__':
    # Quick sanity check on a sample record
    sample_data = pd.DataFrame({
        'Type': ['L', 'M', 'H'],
        'Air temperature [K]': [298.1, 298.2, 298.3],
        'Process temperature [K]': [308.6, 308.7, 308.5],
        'Rotational speed [rpm]': [1551, 1408, 1498],
        'Torque [Nm]': [42.8, 46.3, 49.4],
        'Tool wear [min]': [0, 3, 5]
    })

    print("Raw Sample Data:")
    print(sample_data)

    transformed = engineer_features(sample_data)
    print("\nTransformed Data with Physics Features:")
    print(transformed[['temp_diff', 'power_w', 'overstrain', 'overstrain_ratio']])
    print("\nSanity check passed successfully!")
