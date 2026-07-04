# app/streamlit_app.py  (Dashboard - main page)
# import streamlit as st
import pandas as pd, plotly.express as px, joblib

# st.set_page_config(page_title="ChurnSense",
#                    page_icon="🏦", layout="wide")

# @st.cache_resource   # Load once, cache in memory
# def load_model():
#     return joblib.load('../models/churn_model_v1.pkl')

# @st.cache_data       # Cache data processing
# def load_data():
#     return pd.read_csv('../data/WA_Fn-UseC_Telco.csv')

# model = load_model()
# df = load_data()

# st.title("🏦 ChurnSense — Customer Analytics")
# st.markdown("*AI-Powered Churn Prediction Dashboard*")

# # KPI Cards Row
# col1,col2,col3,col4 = st.columns(4)
# churn_rate = (df['Churn']=='Yes').mean()*100
# col1.metric("Total Customers", f"{len(df):,}")
# col2.metric("Churn Rate", f"{churn_rate:.1f}%",
#             delta="-1.2%", delta_color="inverse")
# col3.metric("Avg Monthly Charges",
#             "$64.76")
# col4.metric("Model AUC Score", "0.87")

#---------------------------------------------------------------------
import streamlit as st
from utils import load_model, load_data, get_kpis

st.set_page_config(
    page_title="ChurnSense",
    page_icon="🏦",
    layout="wide"
)

model = load_model()
df = load_data()

kpi = get_kpis(df)

st.title("🏦 ChurnSense")
st.markdown("### AI-Powered Customer Churn Prediction")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Customers",
    f"{kpi['customers']:,}"
)

c2.metric(
    "Churn Rate",
    f"{kpi['churn_rate']:.1f}%"
)

c3.metric(
    "Avg Monthly Charges",
    f"${kpi['avg_monthly']:.2f}"
)

c4.metric(
    "Avg Tenure",
    f"{kpi['avg_tenure']:.1f} months"
)