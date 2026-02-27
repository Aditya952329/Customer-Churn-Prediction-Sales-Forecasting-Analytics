"""Simple Streamlit app for churn and sales analytics."""

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(page_title="Customer Churn & Sales Dashboard", layout="wide")
st.title("Customer Churn Prediction & Sales Forecasting Analytics")

processed = pd.read_csv(PROCESSED_DIR / "customer_analytics_processed.csv")
forecast = pd.read_csv(PROCESSED_DIR / "sales_forecast.csv") if (PROCESSED_DIR / "sales_forecast.csv").exists() else None

c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{len(processed):,}")
c2.metric("Churn Rate", f"{processed['churn'].mean():.1%}")
c3.metric("Avg Monthly Spend", f"${processed['monthly_spend'].mean():,.0f}")

st.subheader("Churn by Segment")
seg = processed.groupby("segment", as_index=False)["churn"].mean()
st.plotly_chart(px.bar(seg, x="segment", y="churn", color="segment"), use_container_width=True)

st.subheader("Spend vs Engagement")
st.plotly_chart(
    px.scatter(processed, x="monthly_spend", y="engagement_score", color="churn", hover_data=["region", "segment"]),
    use_container_width=True,
)

if forecast is not None:
    st.subheader("6-Month Sales Forecast")
    st.plotly_chart(px.line(forecast, x="month", y="forecast_sales", markers=True), use_container_width=True)
