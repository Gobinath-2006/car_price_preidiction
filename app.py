import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")

# Load dataset (only for column reference)
@st.cache_data
def load_data():
    return pd.read_csv("car_price_prediction_.csv")

# Load trained model
@st.cache_resource
def load_model():
    with open("car_price_prediction_model.pkl", "rb") as file:
        return pickle.load(file)

df = load_data()
model = load_model()

st.success("Model and dataset loaded successfully!")

# Get feature columns (exclude target column)
target_column = "price"  # change ONLY if your target column has a different name
feature_columns = [col for col in df.columns if col != target_column]

st.header("Enter Car Details")

user_input = {}

for col in feature_columns:
    if df[col].dtype in ["int64", "float64"]:
        user_input[col] = st.number_input(
            f"{col}",
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )
    else:
        user_input[col] = st.selectbox(
            f"{col}",
            df[col].unique()
        )

input_df = pd.DataFrame([user_input])

# Encode categorical columns if needed
input_df = pd.get_dummies(input_df)
model_features = model.feature_names_in_
input_df = input_df.reindex(columns=model_features, fill_value=0)

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: ₹ {prediction:,.2f}")
