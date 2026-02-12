# app.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bike Rental Demand Prediction")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

st.title("🚲 Bike Rental Demand Prediction")

st.sidebar.header("Input Parameters")

# ---------------- NUMERIC INPUTS ----------------
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 12)
weekday = st.sidebar.slider("Weekday (0=Sun)", 0, 6, 3)

temp = st.sidebar.slider("Temperature (0-1 scaled)", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temperature (0-1 scaled)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity (0-1 scaled)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed (0-1 scaled)", 0.0, 1.0, 0.5)

casual = st.sidebar.number_input("Casual Users", min_value=0, value=100)
registered = st.sidebar.number_input("Registered Users", min_value=0, value=200)

# ---------------- CATEGORICAL INPUTS ----------------
season = st.sidebar.selectbox("Season", ["springer", "summer", "fall", "winter"])
workingday = st.sidebar.selectbox("Working Day", ["No work", "work"])
holiday = st.sidebar.selectbox("Holiday", ["NO", "yes"])
weather = st.sidebar.selectbox(
    "Weather",
    ["Clear", "Mist", "heavy rain", "lightsnow"]
)

# ---------------- CATEGORY MAPPING ----------------
season_map = {"springer": 1, "summer": 2, "fall": 3, "winter": 4}
workingday_map = {"No work": 0, "work": 1}
holiday_map = {"NO": 0, "yes": 1}
weather_map = {"Clear": 1, "Mist": 2, "heavy rain": 3, "lightsnow": 4}

# ---------------- BUILD INPUT DATA ----------------
input_data = pd.DataFrame({
    "season": [season_map[season]],
    "mnth": [mnth],
    "hr": [hr],
    "holiday": [holiday_map[holiday]],
    "weekday": [weekday],
    "workingday": [workingday_map[workingday]],
    "weathersit": [weather_map[weather]],   # IMPORTANT: must match training column name
    "temp": [temp],
    "atemp": [atemp],
    "hum": [hum],
    "windspeed": [windspeed],
    "casual": [casual],
    "registered": [registered]
})

# Ensure column order matches model
input_data = input_data[model.feature_names_in_]

# Optional debug (remove later)
# st.write("Model expects:", model.feature_names_in_)
# st.write("Input sent:", input_data)

# ---------------- PREDICTION ----------------
if st.button("Predict Bike Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Bike Rentals: {int(prediction)}")
