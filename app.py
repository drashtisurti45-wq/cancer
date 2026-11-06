import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model, scaler, and PCA transformer
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_logistic_regression_model.sav')
    pca = joblib.load('pca_transformer.sav')
    scaler = joblib.load('scaler.sav')
    return model, pca, scaler

try:
    model, pca, scaler = load_artifacts()
    st.success("Model and transformers loaded successfully!")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")

st.title("🧬 Breast Cancer Prediction App")
st.write("Enter the mean feature values of the cell nucleus to classify it as Malignant or Benign.")

# Input fields for a subset of critical mean features to keep UI clean
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=40.0, value=14.0)
    texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=40.0, value=19.0)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=90.0)
    area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=650.0)
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.2, value=0.1, format="%.5f")

with col2:
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=0.5, value=0.1, format="%.5f")
    concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=0.5, value=0.08, format="%.5f")
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=0.2, value=0.05, format="%.5f")
    symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=0.5, value=0.18, format="%.5f")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=0.1, value=0.06, format="%.5f")

# Fill remaining features with default zeros if not input via UI
if st.button("Predict"):
    input_data = np.zeros((1, 30))
    input_data[0, :10] = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
    ]
    
    # Preprocess the input data
    scaled_input = scaler.transform(input_data)
    pca_input = pca.transform(scaled_input)
    
    # Perform prediction
    prediction = model.predict(pca_input)[0]
    prediction_proba = model.predict_proba(pca_input)[0]
    
    # Display results
    if prediction == 'M':
        st.error(f"🚨 Malignant (Probability: {prediction_proba[1]:.2%})")
    else:
        st.success(f"✅ Benign (Probability: {prediction_proba[0]:.2%})")
