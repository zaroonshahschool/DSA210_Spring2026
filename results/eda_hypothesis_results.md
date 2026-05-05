# Hypothesis Testing Summary

## Dataset
- Merged daily observations: 365
- Date range: 2023-01-01 to 2023-12-31
- Rainy days: 179
- Non-rainy days: 186

## Missing Values
No missing values were found in the merged dataset. I checked 19 columns after merging the traffic and weather data.

## Descriptive Statistics
- Average traffic index: mean = 31.24, median = 32.05, min = 4.65, max = 58.19.
- Mean temperature: mean = 16.02 C, range = 1.56 C to 30.68 C.
- Mean humidity: mean = 75.04%, range = 40.04% to 94.46%.
- Daily precipitation: mean = 2.54 mm, max = 58.70 mm.
- Mean wind speed: mean = 16.18 km/h, max = 40.78 km/h.

## Hypothesis Tests

### H1: Rainy days have a different average traffic index than non-rainy days.
- Rainy-day mean traffic index: 31.94
- Non-rainy-day mean traffic index: 30.57
- Welch t-test statistic: 1.835
- p-value: 0.0674

### H2: Weekdays have a different average traffic index than weekends.
- Weekday mean traffic index: 33.46
- Weekend mean traffic index: 25.75
- Welch t-test statistic: 9.461
- p-value: 0.0000

### Weather Correlations with Traffic
- temperature_mean_c: Spearman rho = -0.108, p-value = 0.0393
- humidity_mean_pct: Spearman rho = 0.205, p-value = 0.0001
- precipitation_sum_mm: Spearman rho = 0.222, p-value = 0.0000
- cloud_cover_mean_pct: Spearman rho = 0.235, p-value = 0.0000
- wind_speed_mean_kmh: Spearman rho = 0.030, p-value = 0.5694

## Correlation Matrix
The full correlation heatmap is saved as `results/figures/correlation_matrix.png`. The main correlations with average traffic index were:
- Temperature: -0.091
- Humidity: 0.165
- Precipitation: 0.232
- Pressure: -0.011
- Cloud cover: 0.207
- Wind speed: 0.086

## Figures
- `results/figures/monthly_traffic_precipitation.png`
- `results/figures/rainy_vs_nonrainy_boxplot.png`
- `results/figures/temperature_vs_traffic.png`
- `results/figures/correlation_matrix.png`