import streamlit as st  # First import streamlit
st.set_page_config(page_title="Bike Rental Prediction", layout="wide")  # Must be first Streamlit command

import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_data
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -----------------------------
# Page title and description
# -----------------------------
st.title("🚲 Bike Rental Demand Prediction")
st.markdown("Predict the estimated number of bike rentals using the same features as in the original Jupyter code.")

# -----------------------------
# Sidebar Inputs (match Jupyter features)
# -----------------------------
st.sidebar.header("Input Features")

# Numeric inputs
temp = st.sidebar.number_input("Temperature (temp, 0-1)", 0.0, 1.0, 0.5)
atemp = st.sidebar.number_input("Feels-like Temperature (atemp, 0-1)", 0.0, 1.0, 0.5)
hum = st.sidebar.number_input("Humidity (hum, 0-1)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.number_input("Windspeed (windspeed, 0-1)", 0.0, 1.0, 0.1)
yr = st.sidebar.selectbox("Year (yr, 0=2011, 1=2012)", [0, 1])
mnth = st.sidebar.selectbox("Month (mnth, 1-12)", list(range(1, 13)))
hr = st.sidebar.selectbox("Hour (hr, 0-23)", list(range(0, 24)))
weekday = st.sidebar.selectbox("Weekday (weekday, 0=Sun, 6=Sat)", list(range(0, 7)))

# Categorical inputs
season = st.sidebar.selectbox("Season (season, 1=Spring, 2=Summer, 3=Fall, 4=Winter)", [1, 2, 3, 4])
holiday = st.sidebar.selectbox("Holiday (holiday, 0=No, 1=Yes)", [0, 1])
workingday = st.sidebar.selectbox("Working Day (workingday, 0=No, 1=Yes)", [0, 1])
weathersit = st.sidebar.selectbox(
    "Weather Situation (weathersit, 1=Clear, 2=Mist, 3=Light Snow/Rain, 4=Heavy Rain/Snow)",
    [1, 2, 3, 4]
)

# -----------------------------
# Prepare input dataframe
# -----------------------------
input_df = pd.DataFrame({
    'temp': [temp],
    'atemp': [atemp],
    'hum': [hum],
    'windspeed': [windspeed],
    'yr': [yr],
    'mnth': [mnth],
    'hr': [hr],
    'weekday': [weekday],
    'season': [season],
    'holiday': [holiday],
    'workingday': [workingday],
    'weathersit': [weathersit]
})

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Bike Rentals"):
    prediction = model.predict(input_df)[0]
    st.success(f"🚲 Estimated Bike Rentals: {int(prediction)}")

# -----------------------------
# Show input data for reference
# -----------------------------
with st.expander("See Input Data"):
    st.write(input_df)
