import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Bike Rental Demand Prediction")

# Load model
model = joblib.load("model.pkl")

st.title("🚲 Bike Rental Demand Prediction")
st.sidebar.header("Input Parameters")

# -------- INPUTS --------
season = st.sidebar.selectbox("Season (1-4)", [1, 2, 3, 4])
yr = st.sidebar.selectbox("Year (0=2011, 1=2012)", [0, 1])
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 12)
holiday = st.sidebar.selectbox("Holiday (0/1)", [0, 1])
weekday = st.sidebar.slider("Weekday (0=Sun)", 0, 6, 3)
workingday = st.sidebar.selectbox("Working Day (0/1)", [0, 1])
weathersit = st.sidebar.selectbox("Weather Situation (1-4)", [1, 2, 3, 4])

temp = st.sidebar.slider("Temperature (0-1 scaled)", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

# -------- BUILD INPUT ARRAY (IMPORTANT ORDER) --------
input_data = np.array([[
    season,
    yr,
    mnth,
    hr,
    holiday,
    weekday,
    workingday,
    weathersit,
    temp,
    atemp,
    hum,
    windspeed
]])

# -------- PREDICTION --------
if st.button("Predict Bike Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Bike Rentals: {int(prediction)}")
