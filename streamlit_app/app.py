import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

data_path = "../data/Telco-Customer-Churn.csv"

if os.path.exists(data_path):
    df_original = pd.read_csv(data_path)
    churn_counts = df_original["Churn"].value_counts()

    st.subheader("📊 Churn Distribution (Original Dataset)")
    fig1, ax1 = plt.subplots()
    ax1.pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%", startangle=90, colors=["#66bb6a", "#ef5350"])
    ax1.axis("equal")
    st.pyplot(fig1)
else:
    st.info("Churn dataset not found for pie chart.")


# ---------- Load Model & Feature Columns ----------
model = joblib.load("../model/churn_model.pkl")
features = joblib.load("../model/feature_columns.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------- App Title ----------
st.title("📉 Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to **churn** based on their service usage and demographics.")

# ---------- Sidebar Input ----------
st.sidebar.header("🔍 Customer Information")
# st.sidebar.markdown("---")
# st.sidebar.markdown("📓 [View Full Model Notebook](https://github.com/your-username/your-repo/blob/main/notebooks/01_preprocessing_and_modeling.ipynb)")

def user_input():
    user_data = {}
    
    # Basic details
    user_data["gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
    user_data["SeniorCitizen"] = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    user_data["Partner"] = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    user_data["Dependents"] = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

    # Service usage
    user_data["tenure"] = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    user_data["PhoneService"] = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    user_data["MultipleLines"] = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    user_data["InternetService"] = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    user_data["OnlineSecurity"] = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    user_data["OnlineBackup"] = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
    user_data["DeviceProtection"] = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
    user_data["TechSupport"] = st.sidebar.selectbox("Tech Support", ["No", "Yes"])

    # Billing
    user_data["Contract"] = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    user_data["PaperlessBilling"] = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    user_data["PaymentMethod"] = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Credit card (automatic)", "Bank transfer (automatic)"
    ])
    user_data["MonthlyCharges"] = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 75.0)
    user_data["TotalCharges"] = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2500.0)

    # Has streaming
    user_data["HasStreaming"] = st.sidebar.selectbox("Uses Streaming Services", ["No", "Yes"])

    # Tenure group one-hot (example for 3 groups)
    for group in ["tenure_group_13-24", "tenure_group_25-48", "tenure_group_49-60", "tenure_group_61-72"]:
        user_data[group] = 0

    selected_tenure = user_data["tenure"]
    if 13 <= selected_tenure <= 24:
        user_data["tenure_group_13-24"] = 1
    elif 25 <= selected_tenure <= 48:
        user_data["tenure_group_25-48"] = 1
    elif 49 <= selected_tenure <= 60:
        user_data["tenure_group_49-60"] = 1
    elif 61 <= selected_tenure <= 72:
        user_data["tenure_group_61-72"] = 1

    return pd.DataFrame([user_data])

# ---------- Prepare Input ----------
input_df = user_input()

# Encode categorical fields (you must match the model training logic!)
encoding_map = {
    "Yes": 1, "No": 0,
    "Male": 0, "Female": 1,
    "Month-to-month": 0, "One year": 1, "Two year": 2,
    "Electronic check": 0, "Mailed check": 1,
    "Credit card (automatic)": 2, "Bank transfer (automatic)": 3,
    "DSL": 1, "Fiber optic": 0, "No": 2,
    "No phone service": 2
}
for col in input_df.columns:
    if input_df[col].dtype == object:
        input_df[col] = input_df[col].map(encoding_map)

# ---------- Align Columns ----------
input_df = input_df.reindex(columns=features, fill_value=0)

# ---------- Predict ----------
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn. (Confidence: {proba:.2%})")
    else:
        st.success(f"✅ This customer is likely to stay. (Confidence: {1 - proba:.2%})")


import matplotlib.pyplot as plt

if st.checkbox("📌 Show Feature Importance (Model Explanation)"):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-10:]  # top 10 features

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Features Affecting Churn")
    st.pyplot(plt)

