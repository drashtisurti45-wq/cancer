import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Breast Cancer Prediction App")

# Load dataset
df = pd.read_csv("breast-cancer-wisconsin-data.csv")

# Remove unwanted columns
if "id" in df.columns:
    df = df.drop("id", axis=1)

if "Unnamed: 32" in df.columns:
    df = df.drop("Unnamed: 32", axis=1)

# Convert diagnosis
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Prepare data
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.header("Enter Patient Details")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(
        col,
        value=float(X[col].mean())
    )

if st.button("Predict"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("Result")

    if prediction == 1:
        st.error("🔴 Malignant (Cancer Detected)")
    else:
        st.success("🟢 Benign (No Cancer Detected)")

    st.write(
        f"Prediction Confidence: {max(probability)*100:.2f}%"
    )
