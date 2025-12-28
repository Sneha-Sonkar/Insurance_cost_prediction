import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("insurance_model.pkl")

model = load_model()

# -----------------------------
# App UI
# -----------------------------
st.title("Insurance Cost Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=5, value=0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])

    # Model predicts log(charges)
    log_pred = model.predict(input_df)[0]

    # Convert back to original scale
    prediction = np.exp(log_pred)

    st.success(f"Estimated Insurance Cost: ${prediction:,.2f}")
