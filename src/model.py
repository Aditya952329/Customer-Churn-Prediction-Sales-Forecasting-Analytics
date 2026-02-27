"""Model training for churn classification and sales forecasting."""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "figures"


def train_churn_model(df: pd.DataFrame) -> dict:
    """Train churn classifier and save artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    y = df["churn"]
    X = df.drop(columns=["churn", "customer_id"])

    categorical = ["region", "segment"]
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, proba)

    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("Churn Model ROC Curve")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_roc_curve.png", dpi=140)
    plt.close()

    joblib.dump(pipeline, MODEL_DIR / "churn_model.joblib")

    report = classification_report(y_test, pred, output_dict=False)
    with open(PROCESSED_DIR / "churn_model_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write(report)

    return {"accuracy": accuracy, "roc_auc": roc_auc}


def forecast_sales(monthly_df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """Forecast monthly sales and save model outputs."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    monthly_df = monthly_df.sort_values("month").copy()
    monthly_df["month"] = pd.to_datetime(monthly_df["month"])
    monthly_df = monthly_df.set_index("month")

    ts_model = ExponentialSmoothing(
        monthly_df["sales"], trend="add", seasonal="add", seasonal_periods=12
    ).fit()

    forecast = ts_model.forecast(periods)
    future_dates = pd.date_range(
        start=monthly_df.index.max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    forecast_df = pd.DataFrame({"month": future_dates, "forecast_sales": forecast.values})
    forecast_df.to_csv(PROCESSED_DIR / "sales_forecast.csv", index=False)
    joblib.dump(ts_model, MODEL_DIR / "sales_forecast_model.joblib")

    plt.figure(figsize=(10, 4))
    plt.plot(monthly_df.index, monthly_df["sales"], label="Actual Sales")
    plt.plot(forecast_df["month"], forecast_df["forecast_sales"], label="Forecast", linestyle="--")
    plt.title("Sales Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sales_forecast.png", dpi=140)
    plt.close()

    return forecast_df


if __name__ == "__main__":
    processed = pd.read_csv(PROCESSED_DIR / "customer_analytics_processed.csv")
    monthly = pd.read_csv(PROCESSED_DIR / "monthly_sales_processed.csv")

    metrics = train_churn_model(processed)
    forecast_df = forecast_sales(monthly, periods=6)

    print("Churn model metrics:", metrics)
    print("Sales forecast generated:")
    print(forecast_df)
