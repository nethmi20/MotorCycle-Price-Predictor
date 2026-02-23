import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


st.set_page_config(page_title="Motorcycle Price Predictor", page_icon="🏍️", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load('xgb_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return model, encoders, feature_names

@st.cache_data
def load_data():
    return pd.read_csv('ikman_bikes_cleaned.csv')

try:
    model, encoders, feature_names = load_model()
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Run `python train_model.py` first.")
    st.stop()


st.title("🏍️ Motorcycle Market Value Predictor")
st.write("Predict the market value of a motorcycle in Sri Lanka using our trained XGBoost model.")

st.divider()


st.sidebar.header("🔧 Enter Specifications")

make = st.sidebar.selectbox("Brand", sorted(df['make'].unique()))


available_models = sorted(df[df['make'] == make]['model'].unique())
model_name = st.sidebar.selectbox("Model", available_models)

location = st.sidebar.selectbox("Location", sorted(df['location'].unique()))

# Numeric inputs
yom = st.sidebar.slider("Year of Manufacture", 1990, 2026, 2018)
mileage = st.sidebar.slider("Mileage (km)", 0, 200000, 20000, step=1000)
engine_cc = st.sidebar.number_input("Engine Capacity (cc)", min_value=50, max_value=1500, value=150)

# --- Prediction ---
if st.sidebar.button("🔍 Predict Price", type="primary", use_container_width=True):
    with st.spinner("Analyzing market data..."):
        
        age = 2026 - yom

        # Encode categorical inputs using saved label encoders
        try:
            make_encoded = encoders['make'].transform([make])[0]
        except ValueError:
            st.error(f"Brand '{make}' not found in training data.")
            st.stop()

        try:
            model_encoded = encoders['model'].transform([model_name])[0]
        except ValueError:
            st.error(f"Model '{model_name}' not found in training data.")
            st.stop()

        try:
            location_encoded = encoders['location'].transform([location])[0]
        except ValueError:
            st.error(f"Location '{location}' not found in training data.")
            st.stop()

        
        input_data = pd.DataFrame({
            'make': [make_encoded],
            'model': [model_encoded],
            'mileage': [mileage],
            'engine_cc': [engine_cc],
            'location': [location_encoded],
            'age': [age]
        })

        # Ensure column order matches training
        input_data = input_data[feature_names]

        # Predict
        prediction = model.predict(input_data)[0]

        # Display result
        st.success(f"### 💰 Estimated Value: Rs. {prediction:,.0f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Brand", make)
        col2.metric("Model", model_name)
        col3.metric("Age", f"{age} years")

        col4, col5, col6 = st.columns(3)
        col4.metric("Mileage", f"{mileage:,} km")
        col5.metric("Engine", f"{engine_cc} cc")
        col6.metric("Location", location)

        # --- SHAP Waterfall for this prediction ---
        st.divider()
        st.subheader("🧠 Prediction Explanation (SHAP)")
        st.write("This shows how each feature contributed to the predicted price:")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        shap_explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_data.iloc[0],
            feature_names=feature_names
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_explanation, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# --- Training Results Section ---
st.divider()
st.subheader("📊 Model Training Results")

tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Feature Importance", "SHAP Analysis"])

with tab1:
    try:
        st.image('actual_vs_predicted.png', caption='Actual vs Predicted Prices (Test Set)')
    except FileNotFoundError:
        st.info("Run train_model.py to generate this plot.")

with tab2:
    try:
        st.image('xgb_feature_importance.png', caption='XGBoost Feature Importance')
    except FileNotFoundError:
        st.info("Run train_model.py to generate this plot.")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image('shap_summary.png', caption='SHAP Summary Plot')
        except FileNotFoundError:
            st.info("Run train_model.py to generate this plot.")
    with col2:
        try:
            st.image('shap_feature_importance.png', caption='SHAP Feature Importance')
        except FileNotFoundError:
            st.info("Run train_model.py to generate this plot.")

st.divider()
st.caption("Built with XGBoost + SHAP | Data from ikman.lk")