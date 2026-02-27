# Customer Churn Prediction & Sales Forecasting Analytics

An end-to-end Data Analyst project that combines customer retention analytics and sales planning in one modular workflow.

## 1) Business Problem
Organizations with recurring revenue need to:
- Identify customers likely to churn before they leave.
- Understand behavioral and commercial drivers of churn.
- Forecast near-term sales to align inventory, staffing, and growth targets.

This project delivers a reproducible analytics pipeline that supports retention campaigns and financial planning decisions.

## 2) Project Objectives
1. Load raw customer sales and user behavior data.
2. Perform exploratory data analysis (EDA) with summary statistics and visualizations.
3. Train and evaluate a churn prediction model.
4. Forecast future monthly sales using time-series methods.
5. Save processed datasets, model artifacts, and charts.
6. Provide SQL for common business insight questions.
7. Provide dashboard outputs in both matplotlib and Plotly, plus a Streamlit sample app.

## 3) Repository Structure
```text
Customer-Churn-Prediction-Sales-Forecasting-Analytics/
├── data/
│   ├── raw/
│   │   ├── customer_sales.csv
│   │   ├── user_behavior.csv
│   │   └── monthly_sales.csv
│   └── processed/
├── notebooks/
├── src/
│   ├── analyze.py
│   ├── model.py
│   ├── visualize.py
│   └── app_streamlit.py
├── sql/
│   └── queries.sql
├── models/
├── figures/
├── requirements.txt
└── README.md
```

## 4) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5) How to Run
### Step 1: EDA + Processed Data
```bash
python src/analyze.py
```
Outputs:
- `data/processed/customer_analytics_processed.csv`
- `data/processed/monthly_sales_processed.csv`
- EDA charts in `figures/`

### Step 2: Modeling (Churn + Forecast)
```bash
python src/model.py
```
Outputs:
- `models/churn_model.joblib`
- `models/sales_forecast_model.joblib`
- `data/processed/churn_model_metrics.txt`
- `data/processed/sales_forecast.csv`
- model charts in `figures/`

### Step 3: Dashboard Visuals
```bash
python src/visualize.py
```
Outputs:
- `figures/dashboard_matplotlib.png`
- `figures/dashboard_plotly.html`

### Optional: Streamlit Dashboard
```bash
streamlit run src/app_streamlit.py
```

## 6) KPIs Tracked
- **Churn Rate (%)**
- **Model Accuracy & ROC-AUC** for churn classification
- **Average Monthly Spend**
- **Revenue by Region/Segment**
- **Engagement Score**
- **Month-over-Month Sales Growth**
- **Forecasted Sales (Next 6 Months)**

## 7) Analytical Approach
### Data Preparation
- Merge customer sales and behavior datasets on `customer_id`.
- Engineer `engagement_score` using email and session activity.
- Persist clean, analysis-ready datasets.

### Churn Modeling
- Train/test split with stratification.
- Pipeline preprocessing (imputation, scaling, one-hot encoding).
- Logistic Regression classifier.
- Performance measured with accuracy, classification report, and ROC-AUC.

### Sales Forecasting
- Monthly sales indexed as time series.
- Holt-Winters Exponential Smoothing with additive trend/seasonality.
- Forecast next 6 months.

## 8) Interpretation Framework (Example)
- **High churn in a segment** → prioritize retention programs and tailored outreach.
- **Low engagement + high support tickets** → trigger customer success interventions.
- **Strong forecasted growth** → scale sales coverage and operational capacity.
- **Weak/flat forecast** → emphasize upsell, win-back, and pricing experiments.

## 9) SQL Insights
Use `sql/queries.sql` to answer key business questions such as:
- What is current churn rate?
- Which segment has highest churn?
- Which regions drive most revenue?
- Who are high-risk high-value customers?
- What is monthly sales momentum (MoM growth)?

## 10) Notes
- Raw data provided in `data/raw/` is a sample synthetic dataset for demonstration.
- You can replace files in `data/raw/` with production extracts while keeping the same schema.
