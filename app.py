import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Breast Cancer App", layout="wide")

st.title("🩺 Breast Cancer Dataset App")

uploaded_file = st.file_uploader(
    "Upload breast-cancer-wisconsin-data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully ✅")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    if 'diagnosis' in df.columns:

        malignant = len(df[df['diagnosis'] == 'M'])
        benign = len(df[df['diagnosis'] == 'B'])

        st.subheader("Diagnosis Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Malignant", malignant)

        with col2:
            st.metric("Benign", benign)

        chart_data = pd.DataFrame(
            {
                "Count": [malignant, benign]
            },
            index=["Malignant", "Benign"]
        )

        st.bar_chart(chart_data)

else:
    st.info("Please upload the CSV file.")
