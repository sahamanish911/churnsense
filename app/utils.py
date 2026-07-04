import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load("models/churn_model_v1.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def get_kpis(df):
    return {
        "customers": len(df),
        "churn_rate": (df["Churn"] == "Yes").mean() * 100,
        "avg_monthly": df["MonthlyCharges"].mean(),
        "avg_tenure": df["tenure"].mean()
    }