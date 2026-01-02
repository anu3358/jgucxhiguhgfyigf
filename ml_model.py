# ml_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st

@st.cache_resource
def train_heat_predictor():
    """Train a simple regressor for temperature prediction."""
    X = np.array([
        [0.05, 0.05, 0.90],
        [0.10, 0.10, 0.80],
        [0.30, 0.20, 0.60],
        [0.50, 0.30, 0.40],
        [0.70, 0.40, 0.25],
        [0.85, 0.50, 0.15],
    ])
    y = np.array([46, 42, 38, 34, 30, 27])
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model


def apply_intervention_to_dataframe(df, green_increase_pct, refl_increase_pct):
    """
    Apply intervention to a city's thermal dataframe.
    
    Logic:
    - Pixels with high NDBI (built-up) get more temperature reduction from reflectivity
    - Vegetation intervention helps more in dense areas
    """
    df_modified = df.copy()
    
    # Vegetation intervention: trees cool nearby areas
    # Effect: -0.15°C per 1% vegetation increase, more in built-up areas
    veg_effect = (green_increase_pct / 100.0) * 0.15 * (df['ndbi'] / (df['ndbi'].max() + 0.01))
    
    # Reflectivity intervention: white roofs
    # Effect: -0.20°C per 1% reflectivity increase in built-up areas
    refl_effect = (refl_increase_pct / 100.0) * 0.20 * (df['ndbi'] / (df['ndbi'].max() + 0.01))
    
    # Combined effect (capped at -3°C max reduction per pixel)
    total_reduction = np.minimum(veg_effect + refl_effect, 3.0)
    
    df_modified['temperature_c'] = df['temperature_c'] - total_reduction
    df_modified['temperature_reduction'] = total_reduction
    
    return df_modified


def get_intervention_summary(df_original, df_modified):
    """Compare before/after temperatures."""
    original_mean = df_original['temperature_c'].mean()
    modified_mean = df_modified['temperature_c'].mean()
    reduction = original_mean - modified_mean
    
    return {
        'base_temp': original_mean,
        'new_temp': modified_mean,
        'reduction': reduction,
        'max_reduction': df_modified['temperature_reduction'].max(),
    }
