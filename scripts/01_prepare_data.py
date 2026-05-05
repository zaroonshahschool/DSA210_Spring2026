from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
CLEAN_DATA_DIR = PROJECT_DIR / "data" / "processed"
TRAFFIC_FILE = RAW_DATA_DIR / "traffic_index.csv"
WEATHER_FILE = RAW_DATA_DIR / "istanbul_weather_2023_open_meteo.csv"
MERGED_FILE = CLEAN_DATA_DIR / "traffic_weather_2023.csv"


def load_traffic() -> pd.DataFrame:
    traffic = pd.read_csv(TRAFFIC_FILE)
    traffic["date"] = pd.to_datetime(
        traffic["trafficindexdate"].str.slice(0, 10), errors="coerce"
    )

    numeric_cols = [
        "minimum_traffic_index",
        "maximum_traffic_index",
        "average_traffic_index",
    ]
    for col in numeric_cols:
        traffic[col] = pd.to_numeric(traffic[col], errors="coerce")

    traffic = traffic.dropna(subset=["date", "average_traffic_index"])
    traffic = traffic[(traffic["date"] >= "2023-01-01") & (traffic["date"] <= "2023-12-31")]
    return traffic[["date", *numeric_cols]].sort_values("date")


def load_weather() -> pd.DataFrame:
    # Open-Meteo CSV exports include metadata in the first three lines.
    weather = pd.read_csv(WEATHER_FILE, skiprows=3)
    weather = weather.rename(
        columns={
            "time": "datetime",
            "temperature_2m (Â°C)": "temperature_2m_c",
            "temperature_2m (°C)": "temperature_2m_c",
            "relative_humidity_2m (%)": "relative_humidity_2m_pct",
            "precipitation (mm)": "precipitation_mm",
            "rain (mm)": "rain_mm",
            "weather_code (wmo code)": "weather_code",
            "pressure_msl (hPa)": "pressure_msl_hpa",
            "cloud_cover (%)": "cloud_cover_pct",
            "wind_speed_10m (km/h)": "wind_speed_10m_kmh",
            "wind_gusts_10m (km/h)": "wind_gusts_10m_kmh",
            "wind_direction_10m (Â°)": "wind_direction_10m_deg",
            "wind_direction_10m (°)": "wind_direction_10m_deg",
        }
    )

    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
    weather["date"] = weather["datetime"].dt.floor("D")

    numeric_cols = [
        "temperature_2m_c",
        "relative_humidity_2m_pct",
        "precipitation_mm",
        "rain_mm",
        "pressure_msl_hpa",
        "cloud_cover_pct",
        "wind_speed_10m_kmh",
        "wind_gusts_10m_kmh",
    ]
    for col in numeric_cols:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    daily = (
        weather.groupby("date", as_index=False)
        .agg(
            temperature_mean_c=("temperature_2m_c", "mean"),
            temperature_min_c=("temperature_2m_c", "min"),
            temperature_max_c=("temperature_2m_c", "max"),
            humidity_mean_pct=("relative_humidity_2m_pct", "mean"),
            precipitation_sum_mm=("precipitation_mm", "sum"),
            rain_sum_mm=("rain_mm", "sum"),
            pressure_mean_hpa=("pressure_msl_hpa", "mean"),
            cloud_cover_mean_pct=("cloud_cover_pct", "mean"),
            wind_speed_mean_kmh=("wind_speed_10m_kmh", "mean"),
            wind_gusts_max_kmh=("wind_gusts_10m_kmh", "max"),
        )
        .sort_values("date")
    )
    return daily


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["is_rainy"] = df["precipitation_sum_mm"] > 0
    df["is_heavy_rain"] = df["precipitation_sum_mm"] >= 10
    return df


def main() -> None:
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic = load_traffic()
    weather = load_weather()
    merged = traffic.merge(weather, on="date", how="inner")
    merged = add_features(merged)
    merged.to_csv(MERGED_FILE, index=False)

    print(f"Traffic rows in 2023: {len(traffic)}")
    print(f"Weather daily rows in 2023: {len(weather)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Saved: {MERGED_FILE}")


if __name__ == "__main__":
    main()
