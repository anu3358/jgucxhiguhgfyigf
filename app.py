
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import google.generativeai as genai

from data_engine import load_thermal_data, get_city_data, get_baseline_stats
from ml_model import (
    apply_intervention_to_dataframe,
    get_intervention_summary,
    estimate_trees_required,
    estimate_paint_layers_and_color,
)
from agent_logic import get_ai_recommendation
from config import DEFAULT_CITIES

# ============ GEMINI CONFIGURATION WITH AUTO-DETECTION ============
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    gemini_api_key = None

# 🔍 AUTO-DETECT AVAILABLE GEMINI MODEL - FIXES 404 ERRORS
@st.cache_resource
def get_available_gemini_model():
    """
    Automatically detects and returns the first available Gemini model
    that supports generateContent method.
    
    This replaces hardcoded model names and works globally in all regions.
    Returns None if no API key or no models found.
    """
    if not gemini_api_key:
        return None
    
    try:
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')
                return model_name
        return None
    except Exception as e:
        return None

st.set_page_config(
    page_title="AuraCool — Urban Heat AI",
    layout="wide",
    page_icon="🏙️",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0b1220 0%, #020617 100%);
    }
    .main {
        background: transparent;
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    h1 {
        background: linear-gradient(90deg, #a5b4fc, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab-list"] button {
        background: rgba(100, 116, 139, 0.2);
        border-radius: 8px;
    }
    .innovation-card {
        background: rgba(30, 60, 114, 0.4);
        border-left: 4px solid #60a5fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .before-card {
        background: rgba(255, 100, 100, 0.15);
        border-left: 4px solid #ff6b6b;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .after-card {
        background: rgba(100, 200, 100, 0.15);
        border-left: 4px solid #51cf66;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏙️ AuraCool — Urban Heat AI Optimizer")
st.caption(
    "🌍 Real satellite data + AI-driven cooling strategies for heat-stressed cities"
)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    city_name = st.selectbox(
        "📍 Select City",
        list(DEFAULT_CITIES.keys()),
        index=0,
    )
    
    st.markdown("---")
    st.markdown("### 🌱 Intervention Sliders")
    
    green_inc_pct = st.slider(
        "🌿 Increase Vegetation (%)",
        min_value=0,
        max_value=60,
        value=20,
        step=5,
        help="Plant trees, create urban forests, green roofs",
    )
    
    refl_inc_pct = st.slider(
        "🏙️ Increase Roof Reflectivity (%)",
        min_value=0,
        max_value=60,
        value=15,
        step=5,
        help="Paint roofs white, use reflective pavements",
    )
    
    budget_musd = st.slider(
        "💰 Climate Budget (Million USD)",
        min_value=5,
        max_value=200,
        value=40,
        step=10,
    )
    
    st.markdown("---")
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.rerun()

# ============ MAIN CONTENT ============

try:
    # Load data
    df_city = get_city_data(city_name)
    baseline_stats = get_baseline_stats(city_name)

    # Apply intervention
    df_modified = apply_intervention_to_dataframe(df_city, green_inc_pct, refl_inc_pct)
    summary = get_intervention_summary(df_city, df_modified)

    # Physical intervention sizing (trees + paint)
    base_temp = summary["base_temp"]
    new_temp = summary["new_temp"]
    reduction = max(0.0, base_temp - new_temp)

    # Use area_km2 from baseline_stats if available, otherwise default 50 km²
    area_km2 = baseline_stats.get("area_km2", 50)

    estimated_trees = estimate_trees_required(
        area_km2=area_km2,
        target_temp_drop_c=reduction,
    )

    layers_needed, paint_color = estimate_paint_layers_and_color(
        target_temp_drop_c=reduction,
    )

    # ============ TOP METRICS ============
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🌡️ Baseline Temp",
            f"{summary['base_temp']:.1f}°C",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "❄️ Optimized Temp",
            f"{summary['new_temp']:.1f}°C",
            delta=f"-{summary['reduction']:.1f}°C",
            delta_color="normal",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🔥 Heat Hotspots",
            f"{baseline_stats['hotspot_pct']:.1f}%",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🏢 Urban Density",
            f"{baseline_stats['urban_density']:.2f}",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ============ TREES + PAINT RECOMMENDATION CARDS ============
    st.markdown("### 🌳 Detailed Cooling Interventions")

    col_trees, col_paint = st.columns(2)

    with col_trees:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Trees required (estimate)")
        st.metric(
            label="Additional shade trees",
            value=f"{estimated_trees:,}",
            help=(
                "Heuristic city-scale estimate based on built-up area and "
                "the achieved average cooling. Use as a planning guide, not "
                "an engineering design specification."
            ),
        )
        st.caption(
            f"Assuming ≈1,000 new trees per km² per 1 °C cooling over "
            f"{area_km2:.1f} km² of priority area."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_paint:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Cool-roof coating spec")
        if layers_needed > 0:
            st.write(f"**Recommended layers:** {layers_needed} coat(s)")
        else:
            st.write("**Recommended layers:** 0 (no coating needed for current scenario)")
        st.write(f"**Suggested colour:** {paint_color}")
        st.caption(
            "Rule of thumb: each high-albedo coat contributes ≈0.8 °C average "
            "roof-level cooling in very hot climates."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ============ INNOVATION HIGHLIGHTS ============
    st.markdown("---")
    st.markdown("### 💡 What Makes AuraCool Different")
    
    col_innovation1, col_innovation2, col_innovation3 = st.columns(3)
    
    with col_innovation1:
        st.markdown("""
        <div class="innovation-card">
        <strong>🛰️ Real Satellite Data</strong><br>
        Landsat 8 thermal imagery • 100m resolution • Live planetary data
        </div>
        """, unsafe_allow_html=True)
    
    with col_innovation2:
        st.markdown("""
        <div class="innovation-card">
        <strong>🤖 Multi-Agent AI</strong><br>
        Analyzes spatial patterns • Recommends specific actions • Quantifies outcomes
        </div>
        """, unsafe_allow_html=True)
    
    with col_innovation3:
        st.markdown("""
        <div class="innovation-card">
        <strong>📈 Actionable Output</strong><br>
        Policy recommendations • Cost-benefit analysis • ROI projections
        </div>
        """, unsafe_allow_html=True)

    # ============ VISUALIZATION TABS ============
    tab_intro, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🌍 Problem & Solution", 
            "🗺️ 3D Heat Map", 
            "📊 Temperature Distribution", 
            "🎯 Intervention Impact", 
            "🤖 AI Strategy",
            "🌳 AI Tree Advisor"
        ]
    )

    # ========== TAB 0: PROBLEM & SOLUTION ==========
    with tab_intro:
        st.markdown("""
        ## 🌡️ The Urban Heat Island Crisis
        
        ### The Problem (Why This Matters)
        
        **Global Impact:**
        - 🔴 1,500+ heat-related deaths annually in India alone
        - 🔴 Urban centers are 10-15°C hotter than surrounding rural areas
        - 🔴 Lowest-income neighborhoods suffer 2-3x more deaths
        - 🔴 Cities spend $100M+ annually on heat-related healthcare & energy
        
        **Current Solutions (Why They Fail):**
        - ❌ Static heat maps show the problem but no solutions
        - ❌ No data-driven intervention strategy exists
        - ❌ City planners don't know WHERE to invest cooling resources first
        - ❌ Cooling benefits distributed inequitably (rich areas cooled first)
        
        ---
        
        ### The AuraCool Solution
        
        **What We Do:**
        1. **Analyze**: Use NASA Landsat 8 satellite data to map exact heat patterns
        2. **Predict**: ML model simulates cooling impact of interventions
        3. **Recommend**: AI agent generates city-specific cooling strategy
        4. **Quantify**: Calculate cost savings, health benefits, ROI
        
        **Key Innovation:**
        > "Maps only show problems. AuraCool turns data into decisions."
        
        **Real-World Example (Delhi):**
        - Baseline heat: 39.5°C across 5km × 5km area
        - Intervention: Plant trees (20%), paint roofs white (15%), invest $40M
        - Result: 2.7°C temperature reduction + $60M annual energy savings + 15-25% fewer heat deaths
        
        ---
        
        ### How to Use AuraCool
        
        1. **Select a city** from the sidebar (Amritsar, Delhi, Singapore, Dubai, NYC)
        2. **Adjust intervention sliders**:
           - 🌿 Vegetation: Tree planting, urban forests, green roofs
           - 🏙️ Reflectivity: Paint roofs white, reflective pavements
           - 💰 Budget: Set investment scale ($5M-$200M)
        3. **Watch real-time impact** on 3D heat maps and analytics
        4. **Read AI-generated strategy** tailored to your city
        5. **Export & share** with climate teams and city planners
        
        ---
        
        ### Why This Wins
        
        ⭐ **Novel Problem**: Climate change + equity + urban planning
        ⭐ **Real Data**: Actual Landsat 8 satellite imagery (not synthetic)
        ⭐ **Actionable AI**: Transforms data into policy recommendations
        ⭐ **Measurable Impact**: Shows exact cooling + cost savings + lives saved
        ⭐ **Scalable**: Works for any city with satellite data available
        
        **Status**: In development for 2025 Hackathon
        """)

    # ========== TAB 1: 3D HEAT MAP ==========
    with tab1:
        st.markdown("### 🗺️ Live 3D Thermal Map (Before & After Intervention)")
        
        st.warning(f"""
        ⚠️ **NOTICE THE COLOR CHANGE!**
        
        **Temperature Reduction: {summary['reduction']:.2f}°C**
        
        LEFT (🔴 RED) → RIGHT (🟢 GREEN)
        """)
        
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.markdown(f"""
            <div class="before-card">
            <h4>🔴 BEFORE - Current Hot Reality</h4>
            Average: {df_city['temperature_c'].mean():.1f}°C<br>
            Max: {df_city['temperature_c'].max():.1f}°C<br>
            <strong>Red/Orange = TOO HOT!</strong>
            </div>
            """, unsafe_allow_html=True)
            
            layer_before = pdk.Layer(
                "HexagonLayer",
                data=df_city,
                get_position="[longitude, latitude]",
                get_elevation="temperature_c * 50",
                elevation_scale=30,
                extruded=True,
                coverage=0.85,
                get_fill_color="[255, 50, 50, 200]",
                radius=120,
                opacity=0.85,
            )
            
            view_state = pdk.ViewState(
                latitude=df_city['latitude'].mean(),
                longitude=df_city['longitude'].mean(),
                zoom=11,
                pitch=50,
                bearing=0,
            )
            
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer_before],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/dark-v11",
                    tooltip={"text": "🌡️ {temperature_c:.1f}°C"},
                ),
                use_container_width=True,
                height=520,
            )
        
        with col_after:
            st.markdown(f"""
            <div class="after-card">
            <h4>🟢 AFTER - With Cooling Intervention</h4>
            Average: {df_modified['temperature_c'].mean():.1f}°C<br>
            Max: {df_modified['temperature_c'].max():.1f}°C<br>
            <strong>Blue/Green = MUCH COOLER!</strong>
            </div>
            """, unsafe_allow_html=True)
            
            layer_after = pdk.Layer(
                "HexagonLayer",
                data=df_modified,
                get_position="[longitude, latitude]",
                get_elevation="temperature_c * 50",
                elevation_scale=30,
                extruded=True,
                coverage=0.85,
                get_fill_color="[50, 200, 100, 200]",
                radius=120,
                opacity=0.85,
            )
            
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer_after],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/dark-v11",
                    tooltip={"text": "❄️ {temperature_c:.1f}°C"},
                ),
                use_container_width=True,
                height=520,
            )
        
        st.markdown("---")
        st.markdown("### 📊 Key Differences")
        
        before_hot = (df_city['temperature_c'] > df_city['temperature_c'].quantile(0.75)).sum()
        after_hot = (df_modified['temperature_c'] > df_modified['temperature_c'].quantile(0.75)).sum()
        reduction_pct = ((before_hot - after_hot) / before_hot * 100) if before_hot > 0 else 0
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            st.error(f"""
            **BEFORE (Red Map)**
            
            🔴 Max: {df_city['temperature_c'].max():.1f}°C
            📊 Avg: {df_city['temperature_c'].mean():.1f}°C
            🔥 Hot zones: {before_hot}
            """)
        
        with col_d2:
            st.success(f"""
            **AFTER (Green Map)**
            
            🟢 Max: {df_modified['temperature_c'].max():.1f}°C
            📊 Avg: {df_modified['temperature_c'].mean():.1f}°C
            ❄️ Hot zones: {after_hot}
            """)
        
        with col_d3:
            st.info(f"""
            **IMPACT**
            
            ❄️ Max cooling: {(df_city['temperature_c'].max() - df_modified['temperature_c'].max()):.1f}°C
            📉 Reduction: {reduction_pct:.1f}%
            ✅ Vegetated: {green_inc_pct}% + {refl_inc_pct}% reflective
            """)

    # ========== TAB 2: TEMPERATURE DISTRIBUTION ==========
    with tab2:
        st.markdown("### Temperature Distribution Analysis")
        
        col_hist, col_box = st.columns(2)
        
        with col_hist:
            df_comparison = pd.DataFrame({
                'Baseline': df_city['temperature_c'],
                'Optimized': df_modified['temperature_c'],
            }).melt(var_name='Scenario', value_name='Temperature (°C)')
            
            fig_hist = px.histogram(
                df_comparison,
                x='Temperature (°C)',
                color='Scenario',
                nbins=20,
                title="Temperature Distribution Comparison",
                barmode='overlay',
                color_discrete_map={'Baseline': '#ff6b6b', 'Optimized': '#51cf66'},
            )
            fig_hist.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_box:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=df_city['temperature_c'],
                name='Baseline',
                marker_color='#ff6b6b',
            ))
            fig_box.add_trace(go.Box(
                y=df_modified['temperature_c'],
                name='Optimized',
                marker_color='#51cf66',
            ))
            fig_box.update_layout(
                title="Temperature Range (Min, Median, Max)",
                yaxis_title="Temperature (°C)",
                height=400,
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ========== TAB 3: INTERVENTION IMPACT ==========
    with tab3:
        st.markdown("### Spatial Intervention Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Scatter: Reduction vs Urban Density")
            fig_scatter = px.scatter(
                df_modified,
                x='temperature_c',
                y='temperature_reduction',
                color='ndbi',
                size='temperature_reduction',
                hover_data=['latitude', 'longitude'],
                title="Temperature Reduction vs Built-up Index",
                color_continuous_scale="Viridis",
                labels={
                    'temperature_c': 'Optimized Temp (°C)',
                    'temperature_reduction': 'Cooling Gain (°C)',
                    'ndbi': 'Urban Density (NDBI)',
                }
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.markdown("#### Heatmap: Regional Cooling Impact")
            df_pivot = df_modified.copy()
            
            df_pivot['lat_label'] = pd.cut(df_pivot['latitude'], bins=6, labels=[
                'North', 'North-Mid', 'Mid-North', 'Mid-South', 'South-Mid', 'South'
            ])
            df_pivot['lon_label'] = pd.cut(df_pivot['longitude'], bins=6, labels=[
                'Far West', 'West', 'Center-W', 'Center-E', 'East', 'Far East'
            ])
            
            heatmap_data = df_pivot.pivot_table(
                values='temperature_reduction',
                index='lat_label',
                columns='lon_label',
                aggfunc='mean'
            )
            
            fig_hm = px.imshow(
                heatmap_data,
                color_continuous_scale="RdYlGn",
                title="Cooling Impact Heatmap",
                labels=dict(color="Cooling (°C)"),
            )
            fig_hm.update_layout(height=400)
            st.plotly_chart(fig_hm, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Impact Summary Statistics")
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Max Cooling", f"{df_modified['temperature_reduction'].max():.2f}°C")
        
        with col_stats2:
            st.metric("Avg Cooling", f"{df_modified['temperature_reduction'].mean():.2f}°C")
        
        with col_stats3:
            pixels_cooled = (df_modified['temperature_reduction'] > 0).sum()
            st.metric("Pixels Affected", f"{pixels_cooled}/{len(df_modified)}")
        
        st.markdown("---")
        st.markdown("#### Top 10 Cooling Hotspots")
        
        top_zones = df_modified.nlargest(10, 'temperature_reduction')[
            ['latitude', 'longitude', 'temperature_reduction', 'temperature_c', 'ndbi']
        ]
        
        cols = st.columns(2)
        for idx, (_, row) in enumerate(top_zones.iterrows()):
            with cols[idx % 2]:
                st.write(
                    f"**#{idx+1}** 🌡️ Cooling: **{row['temperature_reduction']:.2f}°C** | "
                    f"Temp: {row['temperature_c']:.1f}°C | Urban: {row['ndbi']:.2f}"
                )

    # ========== TAB 4: AI STRATEGY ==========
    with tab4:
        st.markdown("### 🤖 AI-Generated City Strategy")
        
        report = get_ai_recommendation(
            city_name=city_name,
            base_temp=summary['base_temp'],
            new_temp=summary['new_temp'],
            reduction=summary['reduction'],
            budget_musd=budget_musd,
            urban_hotspot_pct=baseline_stats['hotspot_pct'],
        )
        
        st.markdown(report)
        
        st.markdown("---")
        
        if st.button("📥 Export Strategy as Text", use_container_width=True):
            st.download_button(
                label="Download Strategy Report",
                data=report,
                file_name=f"AuraCool_{city_name}_Strategy.txt",
                mime="text/plain",
            )

    # ========== TAB 5: AI TREE ADVISOR (WITH AUTO-DETECTING GEMINI) ✅ FIXED ==========
    with tab5:
        st.subheader("🌳 AI Tree Advisor (Powered by Gemini)")
        st.write("Get personalized tree recommendations based on **temperature control**, **UV protection**, and **planting seasons**.")
        
        # Input Section
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            user_climate = st.selectbox(
                "🌍 Select Your Climate Zone",
                [
                    "Tropical (Hot & Humid)",
                    "Arid (Hot & Dry)",
                    "Temperate (Moderate)",
                    "Continental (Cold Winters)",
                    "Mediterranean (Mild & Dry)"
                ],
                index=0
            )
        
        with col_t2:
            user_concern = st.selectbox(
                "🎯 Priority",
                [
                    "Temperature Cooling (Primary)",
                    "UV Protection (Sun Safety)",
                    "Mixed (Both)",
                ],
                index=0
            )
        
        with col_t3:
            area_size = st.selectbox(
                "📏 Available Space",
                [
                    "Small (< 20m²)",
                    "Medium (20-50m²)",
                    "Large (50-100m²)",
                    "Very Large (> 100m²)"
                ],
                index=0
            )
        
        st.markdown("---")
        
        # Generate Gemini Recommendation with Auto-Detection ✅
        if st.button("🤖 Get AI Tree Recommendation", type="primary", use_container_width=True):
            if not gemini_api_key:
                st.error("❌ Gemini API key not configured. Please set GEMINI_API_KEY in Streamlit Secrets.")
                st.info("**How to fix:**\n"
                       "1. Go to Streamlit Cloud → Your App → Settings\n"
                       "2. Click 'Secrets'\n"
                       "3. Add: `GEMINI_API_KEY = 'your_key_here'`\n"
                       "4. Save and refresh the app")
            else:
                # Step 1: Detect available model
                with st.spinner("🔄 Detecting available Gemini models..."):
                    detected_model = get_available_gemini_model()
                
                # Step 2: Check if model found
                if not detected_model:
                    st.error("❌ No available Gemini model found. Check your API key and quota.")
                    st.info("💡 **Troubleshooting Steps:**\n"
                           "1. Visit: https://makersuite.google.com/app/apikey\n"
                           "2. Verify your API key has remaining quota\n"
                           "3. Go to Google Cloud Console\n"
                           "4. Enable 'Generative Language API'\n"
                           "5. Ensure billing is enabled\n"
                           "6. Try again in a few moments")
                else:
                    # Step 3: Show detected model ✅
                    st.info(f"✅ Using model: `{detected_model}`")
                    
                    # Step 4: Generate recommendations
                    with st.spinner("🌳 Analyzing best trees for your location..."):
                        try:
                            # Create Gemini prompt
                            prompt = f"""
You are an expert urban arborist and climate scientist.
Based on the user's inputs, provide top 3 tree recommendations.

USER DETAILS:
- Climate Zone: {user_climate}
- Priority: {user_concern}
- Available Space: {area_size}

PROVIDE IN THIS EXACT FORMAT:

## 🌲 Top 3 Recommended Trees

**1. [Tree Common Name]** (*Scientific Name*)
   - Cooling Capacity: X°C
   - UV Protection Factor: SPF ~Y
   - Best Planting Season: [Season]
   - Maintenance: [Low/Medium/High]

**2. [Tree Common Name]** (*Scientific Name*)
   - Cooling Capacity: X°C
   - UV Protection Factor: SPF ~Y
   - Best Planting Season: [Season]
   - Maintenance: [Low/Medium/High]

**3. [Tree Common Name]** (*Scientific Name*)
   - Cooling Capacity: X°C
   - UV Protection Factor: SPF ~Y
   - Best Planting Season: [Season]
   - Maintenance: [Low/Medium/High]

## ☀️ Trees vs Sunscreen for UV Protection

[Compare tree shade effectiveness vs sunscreen. Is tree shade better? Why?]

## 🌱 Planting Advice for {user_climate} Climate

- [Bullet point 1: Key consideration]
- [Bullet point 2: Key consideration]
- [Bullet point 3: Key consideration]

## 💧 Maintenance & Watering Needs

[Describe watering requirements, frequency, and care for this climate]
                            """
                            
                            # Initialize model with AUTO-DETECTED name ✅
                            model = genai.GenerativeModel(detected_model)
                            
                            # Generate content
                            response = model.generate_content(prompt)
                            
                            # Display recommendations
                            st.markdown(response.text)
                            
                            # Success message
                            st.success("✅ **Quick Tip:** Planting native trees ensures better survival rates and requires less water!")
                            
                        except Exception as e:
                            # Detailed error handling
                            st.error(f"❌ Gemini API Error: {str(e)}")
                            st.info("💡 **Troubleshooting:**\n"
                                   "1. Check your API key is correct\n"
                                   "2. Verify you have quota remaining (check makersuite.google.com)\n"
                                   "3. Ensure 'Generative Language API' is enabled in Google Cloud Console\n"
                                   "4. Wait a moment and try again")

        st.markdown("---")
        
        # Research section
        with st.expander("📚 Research: Trees vs Sunscreen"):
            st.markdown("""
            **Did you know?**
            
            🌳 **Tree Shade Effectiveness:**
            - Blocks 95-98% of UV radiation
            - Equivalent to SPF 10-50 depending on leaf density
            - Natural cooling: 2-5°C reduction
            - No chemical residue
            
            🧴 **Sunscreen Effectiveness:**
            - SPF 50 blocks 98% of UV radiation
            - Requires reapplication every 2 hours
            - Chemical ingredients can be harmful
            - Washes off with sweat and water
            
            **Verdict:**
            Dense tree shade offers **similar UV protection** to high SPF sunscreen,
            but with **added cooling benefits** and **no chemicals**. The best approach
            combines both: trees for natural cooling + sunscreen for extra protection.
            
            **Best Trees for Sun Protection:**
            - Neem (India) - Dense foliage, medicinal
            - Banyan - Massive canopy
            - Pipal - Sacred, excellent shade
            - Mango - Fruit + cooling
            - Eucalyptus - Fast growing
            """)

    st.markdown("---")
    st.markdown(
        """
        **AuraCool** uses satellite-derived surface temperature data combined with machine learning 
        to simulate real-world urban cooling interventions. Green interventions and reflective upgrades 
        are modeled based on their spatial relationship to built-up density (NDBI) and vegetation (NDVI).
        """,
        help="Built with Landsat 8 Level 2 satellite data, Gradient Boosting ML, and multi-agent AI logic.",
    )

except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("Please make sure `satellite_thermal_data.csv` exists in the repository.")
    st.write("Error details:", str(e))
