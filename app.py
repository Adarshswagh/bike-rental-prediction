import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bike Rental Demand Prediction")

model = joblib.load("model.pkl")

st.title("🚲 Bike Rental Demand Prediction")
st.sidebar.header("Input Parameters")

# Inputs
yr = st.sidebar.selectbox("Year (0=2011, 1=2012)", [0, 1])
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 12)
weekday = st.sidebar.slider("Weekday (0=Sun)", 0, 6, 3)

temp = st.sidebar.slider("Temperature (0-1 scaled)", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

season = st.sidebar.selectbox("Season (1-4)", [1, 2, 3, 4])
holiday = st.sidebar.selectbox("Holiday (0/1)", [0, 1])
workingday = st.sidebar.selectbox("Working Day (0/1)", [0, 1])
weathersit = st.sidebar.selectbox("Weather Situation (1-4)", [1, 2, 3, 4])

# Build dictionary EXACTLY in model order
input_dict = {}

for col in model.feature_names_in_:
    if col == "season":
        input_dict[col] = season
    elif col == "yr":
        input_dict[col] = yr
    elif col == "mnth":
        input_dict[col] = mnth
    elif col == "hr":
        input_dict[col] = hr
    elif col == "holiday":
        input_dict[col] = holiday
    elif col == "weekday":
        input_dict[col] = weekday
    elif col == "workingday":
        input_dict[col] = workingday
    elif col == "weathersit":
        input_dict[col] = weathersit
    elif col == "temp":
        input_dict[col] = temp
    elif col == "atemp":
        input_dict[col] = atemp
    elif col == "hum":
        input_dict[col] = hum
    elif col == "windspeed":
        input_dict[col] = windspeed

input_data = pd.DataFrame([input_dict])

if st.button("Predict Bike Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Bike Rentals: {int(prediction)}")
