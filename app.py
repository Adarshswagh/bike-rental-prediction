import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bike Rental Demand Prediction")

# Load model
model = joblib.load("model.pkl")

st.title("🚲 Bike Rental Demand Prediction")

# Expected feature columns
feature_cols = model.feature_names_in_

st.sidebar.header("Input Parameters")

# ---------------- NUMERIC INPUTS ----------------
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 12)
weekday = st.sidebar.slider("Weekday (0=Sun)", 0, 6, 3)

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

casual = st.sidebar.number_input("Casual Users", min_value=0, value=100)
registered = st.sidebar.number_input("Registered Users", min_value=0, value=200)

# ---------------- CATEGORICAL INPUTS ----------------
season = st.sidebar.selectbox("Season", ["springer", "summer", "fall", "winter"])
workingday = st.sidebar.selectbox("Working Day", ["work", "No work"])
holiday = st.sidebar.selectbox("Holiday", ["NO", "yes"])
weather = st.sidebar.selectbox(
    "Weather",
    ["Clear", "Mist", "heavy rain", "lightsnow"]
)

# ---------------- BUILD INPUT DATA ----------------
input_data = pd.DataFrame(0, index=[0], columns=feature_cols)

# numeric fills
input_data["mnth"] = mnth
input_data["hr"] = hr
input_data["weekday"] = weekday
input_data["temp"] = temp
input_data["atemp"] = atemp
input_data["hum"] = hum
input_data["windspeed"] = windspeed
input_data["casual"] = casual
input_data["registered"] = registered

# categorical (already encoded)
input_data[season] = 1
input_data[workingday] = 1
input_data[holiday] = 1
input_data[weather] = 1

# optional debug (remove later)
st.write("Model Input Sent:", input_data)

# ---------------- PREDICTION ----------------
if st.button("Predict Bike Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Bike Rentals: {int(prediction)}")
