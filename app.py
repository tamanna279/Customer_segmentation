
import streamlit as st
import pandas as pd
import joblib

# Load saved models
model = joblib.load("kmeans_model (1).pkl")
cluster_names = joblib.load("cluster_names.pkl")
scaler = joblib.load("scaler (1).pkl")

# Page settings
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# Custom design
st.markdown("""
    <style>
    .stApp {
        background-color: #EAF6FF;
    }
    .title {
        text-align: center;
        color: #154360;
        font-size: 35px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #2874A6;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Heading
st.markdown(
    '<p class="title">🛍️ Customer Segmentation</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Discover your customer group using Machine Learning</p>',
    unsafe_allow_html=True
)

st.divider()

# Input section
st.subheader("👤 Enter Customer Details")

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    max_value=500.0,
    value=50.0,
    step=1.0
)

spending = st.slider(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

st.divider()

# Prediction button
if st.button("🔍 Predict Customer Segment", use_container_width=True):

    # Prepare input data
    input_data = pd.DataFrame(
        [[age, income, spending]],
        columns=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict cluster
    prediction = model.predict(scaled_data)[0]

    # Get cluster name
    if isinstance(cluster_names, dict):
        cluster_name = cluster_names.get(
            int(prediction),
            cluster_names.get(str(prediction), f"Cluster {prediction}")
        )
    else:
        cluster_name = cluster_names[int(prediction)]

    # Display result
    st.success("Prediction Completed!")

    st.subheader("🎯 Customer Segment")
    st.markdown(f"### {cluster_name}")
    st.info(f"Cluster Number: {prediction}")

    # Display customer details
    st.subheader("📋 Customer Summary")

    result = pd.DataFrame({
        "Feature": [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ],
        "Value": [age, income, spending]
    })

    st.dataframe(result, use_container_width=True)

    st.balloons()
