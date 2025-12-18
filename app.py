import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction")

# -----------------------------
# Load Model (REQUIRED)
# -----------------------------
@st.cache_resource
def load_model():
    with open("car_price_prediction_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
st.success("✅ Model loaded successfully")

# -----------------------------
# Try loading dataset (OPTIONAL)
# -----------------------------
csv_file = "car_price_prediction_.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.info("📄 Dataset loaded")
    feature_names = [c for c in df.columns if c != "price"]
else:
    st.warning("⚠ Dataset not found. Using model features only.")
    feature_names = list(model.feature_names_in_)

# -----------------------------
# User Input
# -----------------------------
st.header("Enter Car Details")

user_input = {}

for feature in feature_names:
    user_input[feature] = st.number_input(
        feature,
        value=0.0
    )

input_df = pd.DataFrame([user_input])

# Align input with model features
input_df = input_df.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: ₹ {prediction:,.2f}")
