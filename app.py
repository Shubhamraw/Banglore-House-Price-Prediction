# =============================================================================
# app.py
# Bengaluru House Price Prediction — Streamlit Production Frontend Dashboard
# =============================================================================

import pickle
import pandas as pd
import streamlit as st

# --- STREAMLIT GRAPHIC CONFIGURATION ---
st.set_page_config(
    page_title="Bengaluru Real Estate Valuation Hub",
    page_icon="🏠",
    layout="centered"
)

# --- SERIALIZATION BINDING LOADER ---
@st.cache_resource
def load_production_artifacts(path="bengaluru_price_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    artifact = load_production_artifacts()
except FileNotFoundError:
    st.error("⚠️ Server Error: 'bengaluru_price_model.pkl' not detected in current directory hierarchy.")
    st.stop()

# Unpacking components cleanly
pipeline = artifact["pipeline"]
dropdown_locations = artifact["locations"]
model_name = artifact["model_name"]
feature_columns = artifact["feature_columns"]

# Precompute quick-map indexes for one-hot mapping arrays defensively
location_dummy_cols = [c for c in feature_columns if str(c).startswith("location_")]
location_to_column = {col.replace("location_", ""): col for col in location_dummy_cols}

# --- SCREEN TEMPLATE GRAPHICS ---
st.title("🏠 Bengaluru Residential Valuation Hub")
st.markdown("Generate rapid property evaluations using advanced regression mapping across the local housing market.")

st.subheader("Property Dimensions Configuration")
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Target Neighborhood", options=dropdown_locations, index=0)
    bhk = st.slider("🛏️ Total Bedrooms (BHK)", min_value=1, max_value=8, value=2, step=1)

with col2:
    total_sqft = st.number_input("📐 Total Built-up Area (Sqft)", min_value=200.0, max_value=10000.0, value=1200.0, step=50.0)
    bath = st.slider("🚿 Total Bathrooms", min_value=1, max_value=10, value=2, step=1)

# Dynamic Client-Side Integrity Warnings
if bath > bhk + 2:
    st.warning(f"⚠️ Notice: The plumbing ratio ({bath} bathrooms for {bhk} BHK) is historically uncommon.")
if total_sqft / bhk < 300:
    st.warning(f"⚠️ Structural Warning: Floor plan allocates under 300 sqft per bedroom. Valuation may lean conservative.")

st.divider()

# --- RUNNING REAL-TIME SINGLE ROW INFERENCE ---
if st.button("🔮 Compute Property Value", type="primary", use_container_width=True):
    
    # 1. Build a raw matrix placeholder containing exact matching zeroes
    input_row = pd.DataFrame([[0] * len(feature_columns)], columns=feature_columns)
    
    # 2. Inject core parameters
    input_row["total_sqft"] = total_sqft
    input_row["bath"] = bath
    input_row["bhk"] = bhk
    
    # 3. Handle structural flag mapping for location dummy vector
    matched_target_col = location_to_column.get(location)
    if matched_target_col in input_row.columns:
        input_row[matched_target_col] = 1
    
    # Enforce safe execution sorting alignment
    input_row = input_row[feature_columns]
    
    # 4. Generate prediction
    predicted_value = pipeline.predict(input_row)[0]
    predicted_value = max(predicted_value, 0) # Non-negative boundary cap
    
    # Display Result Cards
    st.success("Analysis calculation finalized.")
    st.metric(label="Estimated Valuation Market Cost", value=f"₹ {predicted_value:,.2f} Lakhs")
    
    if predicted_value >= 100:
        st.caption(f"≈ ₹ {predicted_value / 100:,.2f} Crores INR")
        
    with st.sidebar:
        st.header("📊 Model Metrics")
        st.write(f"**Algorithm:** {model_name}")
        st.write(f"**Test $R^2$ Score:** {artifact['test_r2']:.4f}")
        st.write(f"**Test MAE:** ₹{artifact['test_mae']:.2f} Lakhs")