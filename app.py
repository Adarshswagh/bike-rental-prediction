import streamlit as st
st.set_page_config(page_title="Bike Rental Prediction", layout="wide")

import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model safely
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🚲 Bike Rental Demand Prediction")
st.write("Enter feature values to predict bike rental count")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Features")

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels-like Temperature", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.1)

yr = st.sidebar.selectbox("Year", [0, 1])
mnth = st.sidebar.selectbox("Month", list(range(1,13)))
hr = st.sidebar.selectbox("Hour", list(range(0,24)))
weekday = st.sidebar.selectbox("Weekday", list(range(0,7)))

season = st.sidebar.selectbox("Season", ["springer", "summer", "fall", "winter"])
holiday = st.sidebar.selectbox("Holiday", ["NO", "yes"])
workingday = st.sidebar.selectbox("Working Day", ["No work", "work"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Mist", "heavy rain", "lightsnow"])

casual = st.sidebar.number_input("Casual Users", min_value=0, value=100)
registered = st.sidebar.number_input("Registered Users", min_value=0, value=200)

# -----------------------------
# Create full encoded dataframe
# -----------------------------
feature_cols = model.feature_names_in_

input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

# numeric
input_df["temp"] = temp
input_df["atemp"] = atemp
input_df["hum"] = hum
input_df["windspeed"] = windspeed
input_df["yr"] = yr
input_df["mnth"] = mnth
input_df["hr"] = hr
input_df["weekday"] = weekday
input_df["casual"] = casual
input_df["registered"] = registered

# encoded categorical
input_df[season] = 1
input_df[holiday] = 1
input_df[workingday] = 1
input_df[weather] = 1

# instant column (dummy safe value)
if "instant" in input_df.columns:
    input_df["instant"] = 1

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Bike Rentals"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Bike Rentals: {int(prediction)}")

# -----------------------------
# Debug view
# -----------------------------
with st.expander("View Model Input"):
    st.write(input_df)
