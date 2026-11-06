import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Breast Cancer Prediction")

st.title("🩺 Breast Cancer Prediction App")

# Load dataset
df = pd.read_csv("breast-cancer-wisconsin-data.csv")

# Remove unnecessary columns if present
if "id" in df.columns:
    df = df.drop("id", axis=1)

if "Unnamed: 32" in df.columns:
    df = df.drop("Unnamed: 32", axis=1)

# Convert diagnosis to numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.header("Enter Tumor Details")

user_input = {}

for col in X.columns:
    user_input[col] = st.number_input(
        col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])

if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🔴 Malignant (Cancer Detected)")
    else:
        st.success("🟢 Benign (No Cancer Detected)")

    st.write(
        f"Confidence Score: {max(probability)*100:.2f}%"
    )
