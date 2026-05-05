from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "traffic_weather_2023.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
SUMMARY_FILE = RESULTS_DIR / "ml_model_results.md"
PREDICTIONS_FILE = RESULTS_DIR / "ml_predictions_2023.csv"


TARGET = "average_traffic_index"
NUMERIC_FEATURES = [
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "humidity_mean_pct",
    "precipitation_sum_mm",
    "rain_sum_mm",
    "pressure_mean_hpa",
    "cloud_cover_mean_pct",
    "wind_speed_mean_kmh",
    "wind_gusts_max_kmh",
    "month",
]
CATEGORICAL_FEATURES = ["day_of_week", "is_weekend", "is_rainy", "is_heavy_rain"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # A chronological split avoids training on future dates and testing on earlier dates.
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def train_models(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    x_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train[TARGET]
    x_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[TARGET]

    baseline_prediction = pd.Series([y_train.mean()] * len(test), index=test.index)
    baseline_metrics = evaluate(y_test, baseline_prediction)

    models = {
        "Linear Regression": Pipeline(
            [
                ("preprocessor", make_preprocessor(scale_numeric=True)),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    metrics = {"Baseline Mean": baseline_metrics}
    predictions = pd.DataFrame(
        {
            "date": test["date"],
            "actual_average_traffic_index": y_test,
            "Baseline Mean": baseline_prediction,
        }
    )

    fitted_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics[name] = evaluate(y_test, pred)
        predictions[name] = pred
        fitted_models[name] = model

    return metrics, fitted_models, predictions


def get_random_forest_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    return importance_df


def save_prediction_plot(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        predictions["date"],
        predictions["actual_average_traffic_index"],
        label="Actual",
        color="#222222",
        linewidth=2,
    )
    ax.plot(
        predictions["date"],
        predictions["Linear Regression"],
        label="Linear Regression",
        alpha=0.8,
    )
    ax.plot(
        predictions["date"],
        predictions["Random Forest"],
        label="Random Forest",
        alpha=0.8,
    )
    ax.set_title("Traffic Index Predictions on Test Period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average traffic index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ml_predictions_test_period.png", dpi=160)
    plt.close(fig)


def save_feature_importance_plot(importance_df: pd.DataFrame) -> None:
    top_features = importance_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features["feature"], top_features["importance"], color="#4c78a8")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ml_feature_importance.png", dpi=160)
    plt.close(fig)


def write_summary(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
    importance_df: pd.DataFrame,
) -> None:
    metric_df = pd.DataFrame(metrics).T
    best_model = metric_df["rmse"].idxmin()
    top_features = importance_df.head(5)

    lines = [
        "# Machine Learning Summary",
        "",
        "## Goal",
        "This step applies machine learning methods to predict Istanbul's daily average traffic index from weather and calendar features.",
        "",
        "## Dataset",
        f"- Observations: {len(df)} daily rows",
        f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}",
        f"- Training period: {train['date'].min().date()} to {train['date'].max().date()} ({len(train)} rows)",
        f"- Test period: {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} rows)",
        "",
        "## Features Used",
        "- Weather: temperature, humidity, precipitation, rain, pressure, cloud cover, wind speed, and wind gusts.",
        "- Calendar: month, day of week, weekend flag, rainy-day flag, and heavy-rain flag.",
        "",
        "## Models",
        "- Baseline Mean: predicts the training-set average traffic index.",
        "- Linear Regression: a simple interpretable regression model.",
        "- Random Forest Regressor: a nonlinear tree-based model.",
        "",
        "## Test Results",
    ]

    for model_name, row in metric_df.iterrows():
        lines.append(
            f"- {model_name}: MAE = {row['mae']:.3f}, RMSE = {row['rmse']:.3f}, R2 = {row['r2']:.3f}"
        )

    lines.extend(
        [
        "",
        f"The best model by RMSE is **{best_model}**.",
        "",
        "## Random Forest Feature Importance",
        "The strongest features in the random forest model were:",
        ]
    )

    for _, row in top_features.iterrows():
        lines.append(f"- {row['feature']}: {row['importance']:.4f}")

    lines.extend(
        [
        "",
        "## Notes",
        "The model is trained only on 2023 daily data, so the results should be treated as an initial machine learning experiment rather than a final production-quality model.",
        "A stronger final report could use multiple years of matching weather data and add holiday or school-calendar variables.",
        "",
        "## Output Files",
        "- `results/ml_predictions_2023.csv`",
        "- `results/figures/ml_predictions_test_period.png`",
        "- `results/figures/ml_feature_importance.png`",
        ]
    )
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train, test = split_by_time(df)
    metrics, fitted_models, predictions = train_models(train, test)
    importance_df = get_random_forest_importance(fitted_models["Random Forest"])

    predictions.to_csv(PREDICTIONS_FILE, index=False)
    save_prediction_plot(predictions)
    save_feature_importance_plot(importance_df)
    write_summary(df, train, test, metrics, importance_df)

    print(f"Saved summary: {SUMMARY_FILE}")
    print(f"Saved predictions: {PREDICTIONS_FILE}")
    print(f"Saved ML figures in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
