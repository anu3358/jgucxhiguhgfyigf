# ml_model.py - COMPLETE VERSION WITH ALL FUNCTIONS
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


# ============ NEW FUNCTIONS FOR app.py ============

def estimate_trees_required(area_km2: float, target_temp_drop_c: float) -> int:
    """
    Estimate number of trees needed to achieve target temperature reduction.
    
    Based on research:
    - 1 mature tree = ~1.5-2.0°C cooling effect (local area ~100m²)
    - Urban tree density = ~500-1000 trees per km² (healthy cities)
    - Each 1°C reduction city-wide = ~1000 trees per km² of priority area
    
    Args:
        area_km2: Area of city (km²) where intervention applies
        target_temp_drop_c: Target temperature reduction (°C)
    
    Returns:
        Estimated number of trees required
    """
    if target_temp_drop_c <= 0:
        return 0
    
    # Heuristic: ~1000 trees per km² per 1°C reduction
    # (accounts for mature canopy coverage, local clustering effects)
    trees_per_km2_per_degree = 1000
    
    estimated_trees = int(area_km2 * target_temp_drop_c * trees_per_km2_per_degree)
    
    return estimated_trees


def estimate_paint_layers_and_color(target_temp_drop_c: float) -> tuple:
    """
    Estimate cool-roof paint layers and color recommendation based on cooling target.
    
    Based on research:
    - High-albedo white paint (95% reflectance) = ~0.8-1.2°C cooling per layer
    - Multiple coats improve durability and effectiveness
    - Color progression: Pure White → Off-White → Light Gray (based on intensity needed)
    
    Args:
        target_temp_drop_c: Target temperature reduction (°C)
    
    Returns:
        Tuple of (num_layers: int, color_name: str)
    """
    if target_temp_drop_c <= 0:
        return 0, "No coating required"
    
    # Cooling per coat: ~0.8°C per layer
    cooling_per_layer = 0.8
    
    # Calculate layers needed
    layers_needed = max(0, int(np.ceil(target_temp_drop_c / cooling_per_layer)))
    
    # Cap at 3 layers (diminishing returns + cost)
    layers_needed = min(layers_needed, 3)
    
    # Color recommendation based on intensity
    color_map = {
        0: "No coating required",
        1: "High-Albedo White (95% reflectance, Solar Reflectance Index ~105)",
        2: "Off-White (90% reflectance, SRI ~95) with primer coat",
        3: "Pure White + Double Layer (98% reflectance, SRI ~115+)",
    }
    
    color_name = color_map.get(layers_needed, "Consult engineer")
    
    return layers_needed, color_name
