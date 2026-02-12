import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
feature_cols = [
    "yr",
    "mnth","hr","weekday","temp","atemp","hum","windspeed",
    "casual","registered",
    "springer","summer","fall","winter",
    "work","No work","NO","yes",
    "Clear","Mist","heavy rain","lightsnow"
]


# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #2E86C1;
        }
        .sub-text {
            text-align: center;
            font-size: 18px;
            color: grey;
        }
        .result-card {
            background-color: #1f2937;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">🚲 Bike Rental Demand Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict total bike rentals based on weather and time conditions</div>', unsafe_allow_html=True)
st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Input Parameters")

# Numeric Inputs
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 12)
weekday = st.sidebar.slider("Weekday (0=Sun)", 0, 6, 3)

temp = st.sidebar.slider("Temperature (0-1)", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temperature (0-1)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity (0-1)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed (0-1)", 0.0, 1.0, 0.5)

casual = st.sidebar.number_input("Casual Users", min_value=0, value=100)
registered = st.sidebar.number_input("Registered Users", min_value=0, value=200)

# Categorical Inputs
season = st.sidebar.selectbox("Season", ["springer", "summer", "fall", "winter"])
workingday = st.sidebar.selectbox("Working Day", ["work", "No work"])
holiday = st.sidebar.selectbox("Holiday", ["NO", "yes"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Mist", "heavy rain", "lightsnow"])

# ---------------- BUILD INPUT DATA ----------------
input_data = pd.DataFrame(0, index=[0], columns=feature_cols)

# Fill numeric values
input_data["mnth"] = mnth
input_data["hr"] = hr
input_data["weekday"] = weekday
input_data["temp"] = temp
input_data["atemp"] = atemp
input_data["hum"] = hum
input_data["windspeed"] = windspeed
input_data["casual"] = casual
input_data["registered"] = registered

# Safe categorical encoding
for col in [season, workingday, holiday, weather]:
    if col in input_data.columns:
        input_data[col] = 1

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Selected Input Summary")
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("🔮 Prediction Result")

    if st.button("🚀 Predict Bike Demand", use_container_width=True):
        prediction = model.predict(input_data)[0]

        st.markdown(
            f'<div class="result-card">Estimated Bike Rentals<br>🚲 {int(prediction)}</div>',
            unsafe_allow_html=True
        )
