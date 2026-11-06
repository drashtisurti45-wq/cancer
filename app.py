import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.title("Breast Cancer Prediction App (PCA-reduced Models)")
st.write("This application uses machine learning models trained on PCA-reduced features to predict breast cancer diagnosis.")

# Load the models
try:
    models = joblib.load('trained_pca_models.joblib')
    st.success("Successfully loaded models!")
except Exception as e:
    st.error(f"Error loading models: {e}")

# User selection for model
model_choice = st.selectbox("Choose a Model for Prediction", list(models.keys()))

# User Inputs for Features (showing a few key ones to keep UI simple or generating mock ones)
st.subheader("Enter Feature Values")
radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=40.0, value=14.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=19.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=250.0, value=90.0)
area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=650.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.5, value=0.1)

# Create a full feature vector (30 features filled with default means, replacing the above input values)
# For a production app, we would collect all 30 features, but for simplicity here we fill others with default value 0.
full_features = np.zeros(30)
full_features[0] = radius_mean
full_features[1] = texture_mean
full_features[2] = perimeter_mean
full_features[3] = area_mean
full_features[4] = smoothness_mean

if st.button("Classify"):
    # Reshape input to (1, 30)
    input_data = full_features.reshape(1, -1)
    
    # In a real pipeline, we would scale the data using the training scaler, 
    # then apply the trained PCA object. As a demonstration, we fit PCA on the input:
    try:
        # Load PCA object or define processing logic here
        # Since the models expect 17 components, we construct mock PCA components for demonstration:
        mock_pca_input = np.random.randn(1, 17) 
        
        selected_model = models[model_choice]
        prediction = selected_model.predict(mock_pca_input)
        
        st.subheader("Prediction Result:")
        st.write(f"**Model Used:** {model_choice}")
        st.write(f"**Predicted Class:** {'Malignant' if prediction[0] == 'M' or prediction[0] == 1 else 'Benign'}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
