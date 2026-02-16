import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the realistic model and features
model = joblib.load('models/satisfaction_model.pkl')
features_list = joblib.load('models/feature_list.pkl')

st.set_page_config(page_title="Amazon Risk Detector", page_icon="🛍️")

st.title("🛍️ Amazon Product Risk AI")
st.markdown("""
This tool predicts if a product is **High Risk** (likely to face returns or poor feedback) 
based on market positioning (Price & Popularity).
""")

# Sidebar inputs
st.sidebar.header("Product Parameters")
actual_price = st.sidebar.number_input("Original Price (₹)", value=1000)
discounted_price = st.sidebar.number_input("Selling Price (₹)", value=800)
rating_count = st.sidebar.number_input("Total Reviews (Popularity)", value=100)

# Calculate derived features for the model
discount_ratio = discounted_price / actual_price
popularity_log = np.log1p(rating_count)

# Create input dataframe for the model
input_data = pd.DataFrame([[
    actual_price, discounted_price, discount_ratio, popularity_log
]], columns=features_list)

# Prediction Logic
if st.button("Run Risk Analysis"):
    prediction = model.predict(input_data)[0]
    # predict_proba gives us the percentage of confidence
    risk_prob = model.predict_proba(input_data)[0][1] 
    
    st.subheader("Results:")
    if prediction == 1:
        st.error(f"🚩 HIGH RISK (Confidence: {risk_prob:.2%})")
        st.write("Pricing or popularity patterns match previously flagged 'Low Satisfaction' products.")
    else:
        st.success(f"✅ LOW RISK (Risk Probability: {risk_prob:.2%})")
        st.write("This product matches the profile of stable, well-received items.")

st.divider()
st.info("💡 **Tech Stack:** Python (Pandas, Scikit-Learn), VADER Sentiment, Random Forest Classifier.")