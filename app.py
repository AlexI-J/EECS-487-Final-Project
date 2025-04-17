import streamlit as st
import requests
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="StockGAP Dashboard", layout="wide")
st.title("📈 StockGAP Dashboard")

with st.sidebar:
    st.markdown("## 🧠 StockGAP")
    st.markdown("**Generalized Article Predictor for Stock Prices from Category-Based News**")
    st.markdown("by Andre Gala-Garza, Alexander Iekel-Johnson, and Winson Li")
    st.markdown("🛠️ [GitHub Repository](https://github.com/AlexI-J/EECS-487-Final-Project)")
    st.markdown("---")
    st.info("Train, test, and run predictions using models built on financial and news data.")

# Tabs for sections
tab1, tab2, tab3 = st.tabs(["🧠 Train Model", "📊 Test Model", "🔮 Predict"])

# --------------------------------------------
# 🧠 Train Model
# --------------------------------------------
with tab1:
    st.markdown("### Train a new model")
    st.markdown("You can train a new model to predict stock trends, based on the window size and the number of windows to train on.")
    with st.form("train_form"):
        col1, col2 = st.columns(2)

        with col1:
            window_size = st.number_input(
                "🕒 Window Size",
                value=7,
                help="The number of days in each time window, e.g., set this to 7 for a weekly window."
            )
        with col2:
            num_windows = st.number_input(
                "📊 Number of Windows",
                value=30,
                help="The number of non-overlapping windows to extract from the dataset."
            )

        submitted = st.form_submit_button("🚀 Train")

        if submitted:
            with st.spinner("Training model..."):
                try:
                    res = requests.get(f"{API_URL}/train", params={"window_size": window_size, "num_windows": num_windows})
                    
                    if res.ok:
                        st.success("✅ Training complete!")
                        with st.expander("View training results"):
                            st.json(res.json())
                    else:
                        st.error(f"❌ Training failed: {res.text}")
                except Exception as e:
                    st.error(f"❌ Exception occurred: {e}")

# --------------------------------------------
# 📊 Test Model
# --------------------------------------------
with tab2:
    st.markdown("### Evaluate an existing model")

    with st.form("test_form"):
        model_list = []
        model_map = {}

        # Fetch available models
        try:
            response = requests.get(f"{API_URL}/models")
            if response.ok:
                model_data = response.json()
                model_list = [m["readable_name"] for m in model_data]
                model_map = {m["readable_name"]: m["filename"] for m in model_data}
            else:
                st.warning("Could not load model list.")
        except Exception as e:
            st.error(f"Failed to fetch models: {e}")

        selected_model = st.selectbox("Select a trained model to test", model_list)

        test_submit = st.form_submit_button("📈 Test")

        if test_submit and selected_model:
            filename = model_map[selected_model]
            # Extract window size and num windows from filename
            parts = filename.replace(".pkl", "").split("_")
            if len(parts) == 4:
                ws, nw = int(parts[1]), int(parts[2])
                with st.spinner("Testing model..."):
                    res = requests.get(f"{API_URL}/test", params={"window_size": ws, "num_windows": nw})
                    if res.ok:
                        st.success("✅ Test complete!")
                        with st.expander("View test results"):
                            st.json(res.json())
                    else:
                        st.error(f"❌ Testing failed: {res.text}")

# --------------------------------------------
# 🔮 Predict from Headlines
# --------------------------------------------
with tab3:
    st.markdown("### Make predictions using news headlines")

    use_scraped = st.checkbox("Use today's top news headlines (scraped)", value=False)

    model_list = []
    model_map = {}

    # Fetch available models
    try:
        response = requests.get(f"{API_URL}/models")
        if response.ok:
            model_data = response.json()
            model_list = [m["readable_name"] for m in model_data]
            model_map = {m["readable_name"]: m["filename"] for m in model_data}
        else:
            st.warning("Could not load model list.")
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")

    selected_model = st.selectbox("Select a trained model to use for prediction", model_list, key="predict_model_select")

    if not use_scraped:
        headline_input = st.text_area("Enter one or more headlines (one per line)")

        if st.button("🔍 Predict"):
            if not selected_model or not headline_input:
                st.warning("⚠️ Please select a model and enter at least one headline.")
            else:
                headlines = [line.strip() for line in headline_input.strip().split("\n") if line.strip()]
                payload = {"headlines": headlines}
                model_filename = model_map[selected_model]
                with st.spinner("Making prediction..."):
                    res = requests.post(f"{API_URL}/predict", params={"model_name": model_filename, "use_scraped": False}, json=payload)
                    if res.ok:
                        st.success("✅ Prediction complete!")
                        with st.expander("View prediction results"):
                            st.json(res.json())
                    else:
                        st.error(f"❌ Prediction failed: {res.text}")
    else:
        if st.button("🔍 Predict with Scraped Headlines"):
            if not selected_model:
                st.warning("⚠️ Please select a model.")
            else:
                model_filename = model_map[selected_model]
                with st.spinner("Scraping headlines and making prediction..."):
                    res = requests.post(f"{API_URL}/predict", params={"model_name": model_filename, "use_scraped": True})
                    if res.ok:
                        st.success("✅ Prediction complete!")
                        with st.expander("View prediction results"):
                            st.json(res.json())
                    else:
                        st.error(f"❌ Prediction failed: {res.text}")