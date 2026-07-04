import streamlit as st
import plotly.express as px
from utils import load_data

st.title("📊 Customer Analytics")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Churn Distribution")

fig = px.pie(
    df,
    names="Churn",
    title="Customer Churn Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Contract Type vs Churn")

fig = px.histogram(
    df,
    x="Contract",
    color="Churn",
    barmode="group"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Monthly Charges")

fig = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn"
)

st.plotly_chart(fig, use_container_width=True)