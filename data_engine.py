# data_engine.py
import pandas as pd
import streamlit as st

@st.cache_data
def load_thermal_data():
    """Load pre-generated satellite thermal data from CSV."""
    return pd.read_csv('satellite_thermal_data.csv')

def get_city_data(city_name):
    """Get all pixels for a specific city."""
    df = load_thermal_data()
    return df[df['city'] == city_name].copy()

def get_baseline_stats(city_name):
    """Get mean temperature and hotspot percentage for a city."""
    df = get_city_data(city_name)
    
    mean_temp = df['temperature_c'].mean()
    max_temp = df['temperature_c'].max()
    
    # "Hotspot" = pixels in top 25% by temperature
    hotspot_threshold = df['temperature_c'].quantile(0.75)
    hotspot_pct = 100 * (df['temperature_c'] >= hotspot_threshold).sum() / len(df)
    
    # Urban density = mean NDBI
    urban_density = df['ndbi'].mean()
    
    return {
        'mean_temp': mean_temp,
        'max_temp': max_temp,
        'hotspot_pct': hotspot_pct,
        'urban_density': urban_density,
    }
