"""Data loading, cleaning, and exploratory analysis module."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw sales, behavior, and monthly sales datasets."""
    sales = pd.read_csv(RAW_DIR / "customer_sales.csv")
    behavior = pd.read_csv(RAW_DIR / "user_behavior.csv")
    monthly = pd.read_csv(RAW_DIR / "monthly_sales.csv", parse_dates=["month"])
    return sales, behavior, monthly


def preprocess_data(sales: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    """Merge and engineer features for churn analytics."""
    df = sales.merge(behavior, on="customer_id", how="inner")
    df["engagement_score"] = (
        0.4 * df["email_open_rate"]
        + 0.3 * (df["web_sessions_30d"] / df["web_sessions_30d"].max())
        + 0.3 * (df["mobile_sessions_30d"] / df["mobile_sessions_30d"].max())
    )
    return df


def run_eda(df: pd.DataFrame) -> dict:
    """Create summary statistics and visual outputs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "shape": df.shape,
        "churn_rate": df["churn"].mean(),
        "avg_monthly_spend": df["monthly_spend"].mean(),
        "segment_spend": df.groupby("segment")["monthly_spend"].mean().to_dict(),
        "region_churn": df.groupby("region")["churn"].mean().to_dict(),
    }

    df.describe(include="all").to_csv(PROCESSED_DIR / "summary_statistics.csv")

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="churn", palette="viridis")
    plt.title("Churn Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_distribution.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="churn", y="monthly_spend", palette="Set2")
    plt.title("Monthly Spend by Churn")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "monthly_spend_by_churn.png", dpi=140)
    plt.close()

    interactive = px.bar(
        df.groupby("segment", as_index=False)["monthly_spend"].mean(),
        x="segment",
        y="monthly_spend",
        title="Average Monthly Spend by Segment",
        color="segment",
    )
    interactive.write_html(FIG_DIR / "segment_spend_plotly.html")

    return summary


def save_processed(df: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """Persist processed datasets."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "customer_analytics_processed.csv", index=False)
    monthly.to_csv(PROCESSED_DIR / "monthly_sales_processed.csv", index=False)


if __name__ == "__main__":
    sales_df, behavior_df, monthly_sales_df = load_raw_data()
    processed_df = preprocess_data(sales_df, behavior_df)
    eda_summary = run_eda(processed_df)
    save_processed(processed_df, monthly_sales_df)
    print("EDA complete.")
    for key, value in eda_summary.items():
        print(f"{key}: {value}")
