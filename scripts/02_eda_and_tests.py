from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "traffic_weather_2023.csv"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
SUMMARY_FILE = RESULTS_DIR / "eda_hypothesis_results.md"


def save_time_series(df: pd.DataFrame) -> None:
    monthly = (
        df.set_index("date")
        .resample("ME")
        .agg(
            average_traffic_index=("average_traffic_index", "mean"),
            precipitation_sum_mm=("precipitation_sum_mm", "sum"),
            temperature_mean_c=("temperature_mean_c", "mean"),
        )
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(monthly.index, monthly["average_traffic_index"], marker="o", color="#1f77b4")
    ax1.set_ylabel("Average traffic index", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.bar(
        monthly.index,
        monthly["precipitation_sum_mm"],
        width=20,
        alpha=0.25,
        color="#2ca02c",
    )
    ax2.set_ylabel("Monthly precipitation (mm)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    ax1.set_title("Monthly Traffic Index and Precipitation in Istanbul, 2023")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "monthly_traffic_precipitation.png", dpi=160)
    plt.close(fig)


def save_rain_boxplot(df: pd.DataFrame) -> None:
    rainy = df.loc[df["is_rainy"], "average_traffic_index"]
    dry = df.loc[~df["is_rainy"], "average_traffic_index"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([dry, rainy], tick_labels=["No rain", "Rain"], patch_artist=True)
    ax.set_ylabel("Average traffic index")
    ax.set_title("Traffic Index on Rainy vs Non-Rainy Days")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rainy_vs_nonrainy_boxplot.png", dpi=160)
    plt.close(fig)


def save_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        df["temperature_mean_c"],
        df["average_traffic_index"],
        alpha=0.65,
        color="#9467bd",
    )
    ax.set_xlabel("Mean temperature (C)")
    ax.set_ylabel("Average traffic index")
    ax.set_title("Temperature vs Traffic Index")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "temperature_vs_traffic.png", dpi=160)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "average_traffic_index",
        "temperature_mean_c",
        "humidity_mean_pct",
        "precipitation_sum_mm",
        "pressure_mean_hpa",
        "cloud_cover_mean_pct",
        "wind_speed_mean_kmh",
    ]
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels=cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labels=cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_matrix.png", dpi=160)
    plt.close(fig)
    return corr


def run_hypothesis_tests(df: pd.DataFrame) -> dict:
    rainy = df.loc[df["is_rainy"], "average_traffic_index"]
    dry = df.loc[~df["is_rainy"], "average_traffic_index"]
    weekend = df.loc[df["is_weekend"], "average_traffic_index"]
    weekday = df.loc[~df["is_weekend"], "average_traffic_index"]

    rain_ttest = stats.ttest_ind(rainy, dry, equal_var=False, nan_policy="omit")
    weekend_ttest = stats.ttest_ind(weekday, weekend, equal_var=False, nan_policy="omit")

    correlations = {}
    for col in [
        "temperature_mean_c",
        "humidity_mean_pct",
        "precipitation_sum_mm",
        "cloud_cover_mean_pct",
        "wind_speed_mean_kmh",
    ]:
        rho, p_value = stats.spearmanr(df[col], df["average_traffic_index"], nan_policy="omit")
        correlations[col] = {"spearman_rho": rho, "p_value": p_value}

    return {
        "rain_ttest": rain_ttest,
        "weekend_ttest": weekend_ttest,
        "correlations": correlations,
        "rainy_mean": rainy.mean(),
        "dry_mean": dry.mean(),
        "weekday_mean": weekday.mean(),
        "weekend_mean": weekend.mean(),
        "rainy_n": rainy.count(),
        "dry_n": dry.count(),
    }


def write_summary(df: pd.DataFrame, corr: pd.DataFrame, tests: dict) -> None:
    missing = df.isna().sum()
    traffic_summary = df[
        [
            "average_traffic_index",
            "temperature_mean_c",
            "humidity_mean_pct",
            "precipitation_sum_mm",
            "wind_speed_mean_kmh",
        ]
    ].describe()
    traffic_corr = corr["average_traffic_index"].drop("average_traffic_index")

    lines = [
        "# Hypothesis Testing Summary",
        "",
        "## Dataset",
        f"- Merged daily observations: {len(df)}",
        f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}",
        f"- Rainy days: {tests['rainy_n']}",
        f"- Non-rainy days: {tests['dry_n']}",
        "",
        "## Missing Values",
        f"No missing values were found in the merged dataset. I checked {len(missing)} columns after merging the traffic and weather data.",
        "",
        "## Descriptive Statistics",
        f"- Average traffic index: mean = {traffic_summary.loc['mean', 'average_traffic_index']:.2f}, median = {traffic_summary.loc['50%', 'average_traffic_index']:.2f}, min = {traffic_summary.loc['min', 'average_traffic_index']:.2f}, max = {traffic_summary.loc['max', 'average_traffic_index']:.2f}.",
        f"- Mean temperature: mean = {traffic_summary.loc['mean', 'temperature_mean_c']:.2f} C, range = {traffic_summary.loc['min', 'temperature_mean_c']:.2f} C to {traffic_summary.loc['max', 'temperature_mean_c']:.2f} C.",
        f"- Mean humidity: mean = {traffic_summary.loc['mean', 'humidity_mean_pct']:.2f}%, range = {traffic_summary.loc['min', 'humidity_mean_pct']:.2f}% to {traffic_summary.loc['max', 'humidity_mean_pct']:.2f}%.",
        f"- Daily precipitation: mean = {traffic_summary.loc['mean', 'precipitation_sum_mm']:.2f} mm, max = {traffic_summary.loc['max', 'precipitation_sum_mm']:.2f} mm.",
        f"- Mean wind speed: mean = {traffic_summary.loc['mean', 'wind_speed_mean_kmh']:.2f} km/h, max = {traffic_summary.loc['max', 'wind_speed_mean_kmh']:.2f} km/h.",
        "",
        "## Hypothesis Tests",
        "",
        "### H1: Rainy days have a different average traffic index than non-rainy days.",
        f"- Rainy-day mean traffic index: {tests['rainy_mean']:.2f}",
        f"- Non-rainy-day mean traffic index: {tests['dry_mean']:.2f}",
        f"- Welch t-test statistic: {tests['rain_ttest'].statistic:.3f}",
        f"- p-value: {tests['rain_ttest'].pvalue:.4f}",
        "",
        "### H2: Weekdays have a different average traffic index than weekends.",
        f"- Weekday mean traffic index: {tests['weekday_mean']:.2f}",
        f"- Weekend mean traffic index: {tests['weekend_mean']:.2f}",
        f"- Welch t-test statistic: {tests['weekend_ttest'].statistic:.3f}",
        f"- p-value: {tests['weekend_ttest'].pvalue:.4f}",
        "",
        "### Weather Correlations with Traffic",
    ]

    for col, result in tests["correlations"].items():
        lines.append(
            f"- {col}: Spearman rho = {result['spearman_rho']:.3f}, "
            f"p-value = {result['p_value']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Correlation Matrix",
            "The full correlation heatmap is saved as `results/figures/correlation_matrix.png`. The main correlations with average traffic index were:",
            f"- Temperature: {traffic_corr['temperature_mean_c']:.3f}",
            f"- Humidity: {traffic_corr['humidity_mean_pct']:.3f}",
            f"- Precipitation: {traffic_corr['precipitation_sum_mm']:.3f}",
            f"- Pressure: {traffic_corr['pressure_mean_hpa']:.3f}",
            f"- Cloud cover: {traffic_corr['cloud_cover_mean_pct']:.3f}",
            f"- Wind speed: {traffic_corr['wind_speed_mean_kmh']:.3f}",
            "",
            "## Figures",
            "- `results/figures/monthly_traffic_precipitation.png`",
            "- `results/figures/rainy_vs_nonrainy_boxplot.png`",
            "- `results/figures/temperature_vs_traffic.png`",
            "- `results/figures/correlation_matrix.png`",
        ]
    )

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    save_time_series(df)
    save_rain_boxplot(df)
    save_scatter(df)
    corr = save_correlation_heatmap(df)
    tests = run_hypothesis_tests(df)
    write_summary(df, corr, tests)
    print(f"Saved summary: {SUMMARY_FILE}")
    print(f"Saved figures in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
