# 🔮 Customer Churn Predictor

An end-to-end machine learning application that predicts whether a customer is likely to churn — and does it in real time through a clean, interactive dashboard.

Built using Python, trained on real-world Telco data, and deployed with Streamlit Cloud. This project demonstrates the full lifecycle of a data-driven product — from preprocessing and model training to deployment and user interaction.

---

## Overview

- **Input:** Customer demographics and service usage details
- **Output:** Churn prediction (`Yes` or `No`) + confidence score
- **Bonus Features:** Visual dashboards, churn distribution, and top predictive features

---

## Tech Stack

| Layer            | Tools Used                              |
|------------------|------------------------------------------|
| Data Handling    | `pandas`, `NumPy`                        |
| Modeling         | `scikit-learn` (RandomForest, KNN)       |
| Visualization    | `Plotly`, `Matplotlib`                   |
| Deployment       | `Streamlit`, hosted on Streamlit Cloud   |
| Model Persistence| `joblib`                                 |

---

## Live Demo

**🔗 App Link:** [Customer Churn Predictor](https://customer-churn-predictor01.streamlit.app/)  

---
## Screenshots
<img width="1912" height="916" alt="image" src="https://github.com/user-attachments/assets/2c687c04-52ea-4acf-a873-e0025a50a01d" />
<img width="450" height="417" alt="image" src="https://github.com/user-attachments/assets/a88e5367-c936-4975-814c-52491f0f4fd3" />

---

## Key Features

- Intuitive dashboard with real-time predictions
- Custom user input forms
- Clean UI with dark-themed styling
- Model confidence score
- Feature importance visualization
- Churn distribution analysis with Plotly charts

---

## How to Run Locally

```bash
git clone https://github.com/tahak134/customer-churn-predictor
cd customer-churn-predictor/streamlit_app
pip install -r requirements.txt
streamlit run app.py
