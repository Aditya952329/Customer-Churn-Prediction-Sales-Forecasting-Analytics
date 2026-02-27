"""Visualization module for dashboard-style analytics outputs."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"


def create_matplotlib_dashboard(df: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """Generate static dashboard with key KPI charts."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    churn_by_segment = df.groupby("segment", as_index=False)["churn"].mean()
    sns.barplot(data=churn_by_segment, x="segment", y="churn", ax=axes[0, 0], palette="mako")
    axes[0, 0].set_title("Churn Rate by Segment")

    region_revenue = df.groupby("region", as_index=False)["monthly_spend"].sum()
    sns.barplot(data=region_revenue, x="region", y="monthly_spend", ax=axes[0, 1], palette="crest")
    axes[0, 1].set_title("Revenue by Region")

    sns.histplot(data=df, x="engagement_score", hue="churn", bins=20, ax=axes[1, 0], multiple="stack")
    axes[1, 0].set_title("Engagement Score by Churn")

    monthly["month"] = pd.to_datetime(monthly["month"])
    axes[1, 1].plot(monthly["month"], monthly["sales"], color="teal")
    axes[1, 1].set_title("Monthly Sales Trend")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "dashboard_matplotlib.png", dpi=140)
    plt.close()


def create_plotly_dashboard(df: pd.DataFrame) -> None:
    """Generate interactive Plotly chart sample."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = px.scatter(
        df,
        x="monthly_spend",
        y="engagement_score",
        color="churn",
        symbol="segment",
        hover_data=["region", "tenure_months"],
        title="Customer Engagement vs Spend (Churn Highlight)",
    )
    fig.write_html(FIG_DIR / "dashboard_plotly.html")


if __name__ == "__main__":
    processed_df = pd.read_csv(PROCESSED_DIR / "customer_analytics_processed.csv")
    monthly_df = pd.read_csv(PROCESSED_DIR / "monthly_sales_processed.csv")

    create_matplotlib_dashboard(processed_df, monthly_df)
    create_plotly_dashboard(processed_df)
    print("Dashboard files saved in figures/.")
