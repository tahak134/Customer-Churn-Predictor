import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Set page config with dark theme support
st.set_page_config(
    page_title="🔮 Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Theme toggle
def toggle_theme():
    if 'dark_theme' not in st.session_state:
        st.session_state.dark_theme = True
    st.session_state.dark_theme = not st.session_state.dark_theme

# Main header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🔮 Customer Churn Predictor</h1>
    <p class="main-subtitle">AI-Powered Customer Retention Insights & Analytics</p>
</div>
""", unsafe_allow_html=True)

# Load and display churn distribution with enhanced visualization
data_path = "../data/Telco-Customer-Churn.csv"

# Create columns for layout
col1, col2, col3 = st.columns([2, 1, 2])

# Stats Dashboard
st.markdown("## 📊 Dashboard Overview")

# Mock stats (replace with your actual data)
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.markdown("""
    <div class="stat-item">
        <div class="stat-value">7,043</div>
        <div class="stat-label">👥 Total Customers</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col2:
    st.markdown("""
    <div class="stat-item">
        <div class="stat-value">26.6%</div>
        <div class="stat-label">📉 Churn Rate</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col3:
    st.markdown("""
    <div class="stat-item">
        <div class="stat-value">$64.76</div>
        <div class="stat-label">💰 Avg Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col4:
    st.markdown("""
    <div class="stat-item">
        <div class="stat-value">32</div>
        <div class="stat-label">📅 Avg Tenure (months)</div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Churn Distribution Chart
if os.path.exists(data_path):
    df_original = pd.read_csv(data_path)
    churn_counts = df_original["Churn"].value_counts()
    
    st.markdown("### Churn Distribution Analysis")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Create enhanced pie chart with Plotly
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Retained Customers', 'Churned Customers'],
            values=churn_counts.values,
            hole=0.4,
            marker_colors=['#4CAF50', '#ff6b6b'],
            textinfo='label+percent',
            textfont_size=12,
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )])
        
        fig_pie.update_layout(
            title={
                'text': "Customer Retention Overview",
                'x': 0.2,
                'font': {'size': 18, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Create a gauge chart for churn rate
        churn_rate = (churn_counts['Yes'] / churn_counts.sum()) * 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_rate,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Rate %", 'font': {'color': 'white'}},
            delta = {'reference': 25, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 50], 'tickcolor': 'white'},
                'bar': {'color': "#ff6b6b"},
                'steps': [
                    {'range': [0, 15], 'color': "#4CAF50"},
                    {'range': [15, 30], 'color': "#FFC107"},
                    {'range': [30, 50], 'color': "#ff6b6b"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'family': 'Inter'}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("📁 Churn dataset not found. Please check the data path.")

# Load model and features
try:
    model = joblib.load("../model/churn_model.pkl")
    features = joblib.load("../model/feature_columns.pkl")
    model_loaded = True
except:
    st.error("🚨 Model files not found. Please ensure model files are in the correct location.")
    model_loaded = False

# Enhanced Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background-color: #7f7d9c; border-radius: 15px; margin-bottom: 1rem;">
    <h1 style="color: #ffffff; margin: 0;">🔍 Customer Profile</h1>
    <p style="color: #ffffff; margin: 0.5rem 0 0 0;">Enter customer details below</p>
</div>
""", unsafe_allow_html=True)

def user_input():
    user_data = {}
    
    # Demographics Section
    st.sidebar.markdown("### 👤 Demographics")
    user_data["gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
    user_data["SeniorCitizen"] = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"], help="Is the customer a senior citizen?")
    user_data["Partner"] = st.sidebar.selectbox("Has Partner", ["No", "Yes"], help="Does the customer have a partner?")
    user_data["Dependents"] = st.sidebar.selectbox("Has Dependents", ["No", "Yes"], help="Does the customer have dependents?")
    
    st.sidebar.markdown("---")
    
    # Service Usage Section
    st.sidebar.markdown("### 📱 Service Usage")
    user_data["tenure"] = st.sidebar.slider("Tenure (months)", 0, 72, 12, help="How long has the customer been with the company?")
    user_data["PhoneService"] = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    user_data["MultipleLines"] = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    user_data["InternetService"] = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Additional services in expandable section
    with st.sidebar.expander("🛡️ Additional Services"):
        user_data["OnlineSecurity"] = st.sidebar.selectbox("Online Security", ["No", "Yes"])
        user_data["OnlineBackup"] = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
        user_data["DeviceProtection"] = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
        user_data["TechSupport"] = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        user_data["HasStreaming"] = st.sidebar.selectbox("Streaming Services", ["No", "Yes"])
    
    st.sidebar.markdown("---")
    
    # Billing Section
    st.sidebar.markdown("### 💳 Billing Information")
    user_data["Contract"] = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    user_data["PaperlessBilling"] = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    user_data["PaymentMethod"] = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Credit card (automatic)", "Bank transfer (automatic)"
    ])
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        user_data["MonthlyCharges"] = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 75.0, step=5.0)
    with col2:
        user_data["TotalCharges"] = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 2500.0, step=100.0)
    
    # Tenure grouping logic
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

# Get user input
input_df = user_input()

# Prediction Section
st.markdown("## 🔮 Churn Prediction                  ")

if model_loaded:
    # Encode categorical fields
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
    
    # Align columns
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Prediction button with enhanced styling
    predict_col1, predict_col2, predict_col3 = st.columns([2, 1, 2])
    
    with predict_col1:
        if st.button("🔮 Predict Customer Churn", key="predict_btn"):
            with st.spinner("🤖 AI is analyzing customer data..."):
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1]
                
                with st.container():
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-warning" style="text-align: left;">
                            <h2>⚠️ High Churn Risk Detected!</h2>
                            <p style="font-size: 1.2rem; margin: 1rem 0;">This customer is likely to churn</p>
                            <div style="font-size: 2rem; font-weight: bold; margin: 1rem 0;">
                                Confidence: {proba:.1%}
                            </div>
                            <p>💡 <strong>Recommendation:</strong> Immediate retention action required!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        with st.container():
                            st.markdown(f"""
                            <div class="prediction-success" style="text-align: left;">
                                <h2>✅ Low Churn Risk</h2> 
                                <p style="font-size: 1.2rem; margin: 1rem 0;">This customer is likely to stay</p>
                                <div style="font-size: 2rem; font-weight: bold; margin: 1rem 0;">
                                    Confidence: {1 - proba:.1%}
                                </div>
                                <p>💡 <strong>Status:</strong> Customer retention is stable</p>
                            </div>
                            """, unsafe_allow_html=True)
                

else:
    st.error("🚨 Model not available. Please check model files.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.6);">
    <p>🔮 Customer Churn Predictor | Powered by AI & Machine Learning</p>

</div>
""", unsafe_allow_html=True)