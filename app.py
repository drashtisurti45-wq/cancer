import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Page Title
st.title("🩺 Breast Cancer Prediction App")

# Load dataset
data = load_breast_cancer()

# Train model
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.header("Enter Patient Details")

# User Inputs
radius_mean = st.number_input("Radius Mean", value=14.0)
texture_mean = st.number_input("Texture Mean", value=19.0)
perimeter_mean = st.number_input("Perimeter Mean", value=90.0)
area_mean = st.number_input("Area Mean", value=650.0)
smoothness_mean = st.number_input("Smoothness Mean", value=0.10)

# Create input row
input_data = X.mean().to_frame().T

input_data["mean radius"] = radius_mean
input_data["mean texture"] = texture_mean
input_data["mean perimeter"] = perimeter_mean
input_data["mean area"] = area_mean
input_data["mean smoothness"] = smoothness_mean

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Result")

    if prediction == 0:
        st.error("🔴 Malignant (Cancer Detected)")
    else:
        st.success("🟢 Benign (No Cancer Detected)")

    st.write(f"Confidence: {max(probability)*100:.2f}%")
