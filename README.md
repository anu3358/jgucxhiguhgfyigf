# 🏙️ AuraCool — AI Urban Heat Optimizer

**An AI-powered climate technology platform that uses satellite thermal data and machine learning to recommend city-scale cooling interventions.**

---

## 🌍 Problem Statement

Urban Heat Island (UHI) effect kills thousands annually in cities worldwide. Current approaches rely on static heat maps and manual decision-making. **AuraCool** automates the process: real satellite data → AI prediction → actionable city policy.

---

## 🚀 Key Features

✅ **Real Satellite Thermal Data** — Landsat 8 Level 2 surface temperature imagery  
✅ **3D Heat Map Visualization** — PyDeck hexagon layer with before/after comparison  
✅ **Interactive Intervention Sliders** — Simulate green roofs, reflective pavements, tree planting  
✅ **Spatial Impact Analysis** — Understand which neighborhoods benefit most  
✅ **AI Strategy Agent** — Automatic generation of city cooling policy briefs  
✅ **Export Reports** — Download actionable strategies as PDF/text  

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit (Python web framework) |
| **Data Visualization** | PyDeck, Plotly |
| **ML/Prediction** | Scikit-learn (Gradient Boosting) |
| **Geospatial Data** | Pandas, NumPy |
| **Deployment** | Streamlit Community Cloud (Free) |

---

## 📊 How It Works

### Step 1: Load Satellite Data
- Pre-computed Landsat 8 thermal imagery for 5 major cities
- Each city: 900 pixels covering ~5km × 5km area
- Temperature range: 25–47°C (realistic urban heat island gradient)

### Step 2: Define Intervention Scenario
- **Vegetation Increase (%)**: Simulate tree planting, green roofs, urban forests
- **Roof Reflectivity (%)**: Simulate white roofs, cool pavements
- **Budget (Million USD)**: Set investment level

### Step 3: AI Prediction
- ML model predicts new temperature for each pixel
- Calculates energy savings and health impact
- Identifies highest-ROI neighborhoods

### Step 4: Agentic Recommendation
- Multi-agent logic analyzes threat level (Extreme/Severe/High/Moderate)
- Recommends specific actions (green roofs, mobile cooling buses, etc.)
- Estimates annual cost savings and payback period

### Step 5: Export & Share
- Download strategy report as text
- Share insights with city planners and climate teams

---

## 📦 Installation & Local Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Clone & Install

