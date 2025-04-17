import streamlit as st
import requests
import os

API_URL = "http://localhost:8000"

st.title("📈 StockGAP Dashboard")

# -----------------------------
# Training
# -----------------------------
st.header("🧠 Train Model")
with st.form("train_form"):
    window_size = st.number_input("Window Size", value=7)
    num_windows = st.number_input("Number of Windows", value=30)
    submitted = st.form_submit_button("Train")

    if submitted:
        with st.spinner("Training model..."):
            res = requests.get(f"{API_URL}/train", params={"window_size": window_size, "num_windows": num_windows})
            if res.ok:
                st.success("Training complete!")
                st.json(res.json())
            else:
                st.error(f"Training failed: {res.text}")

# -----------------------------
# Testing
# -----------------------------
st.header("📊 Test Model")
with st.form("test_form"):
    test_window_size = st.number_input("Test Window Size", value=7, key="test_ws")
    test_num_windows = st.number_input("Test Number of Windows", value=30, key="test_nw")
    test_submit = st.form_submit_button("Test")

    if test_submit:
        with st.spinner("Testing model..."):
            res = requests.get(f"{API_URL}/test", params={"window_size": test_window_size, "num_windows": test_num_windows})
            if res.ok:
                st.success("Test complete!")
                st.json(res.json())
            else:
                st.error(f"Testing failed: {res.text}")

# -----------------------------
# Prediction
# -----------------------------
st.header("🔮 Predict from Headlines")
model_file = st.text_input("Model filename (e.g., model_7_30.pkl)")
headline_input = st.text_area("Enter one or more headlines (one per line)")

if st.button("Predict"):
    if not model_file or not headline_input:
        st.warning("Please provide a model filename and at least one headline.")
    else:
        headlines = [line.strip() for line in headline_input.strip().split("\n") if line.strip()]
        payload = {"headlines": headlines}
        with st.spinner("Predicting..."):
            res = requests.post(f"{API_URL}/predict", params={"model_name": model_file}, json=payload)
            if res.ok:
                st.success("Prediction complete!")
                st.json(res.json())
            else:
                st.error(f"Prediction failed: {res.text}")
