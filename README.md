# Weather and Traffic in Istanbul

This project looks at whether daily weather conditions are related to Istanbul's traffic index. I used 2023 as the main analysis year because it gives a complete one-year overlap between the traffic file and the weather file I downloaded.

## Data

- `data/raw/traffic_index.csv`: Istanbul traffic index data. The original traffic source is the Istanbul Metropolitan Municipality traffic index history data/API (`https://api.ibb.gov.tr/tkmservices/api/TrafficData/v1/TrafficIndexHistory/{period}/{frequency}`). The file used here is daily traffic index data.
- `data/raw/istanbul_weather_2023_open_meteo.csv`: Hourly historical Istanbul weather data downloaded from Open-Meteo Historical Weather API.
- `data/processed/traffic_weather_2023.csv`: Daily merged dataset used for the EDA, hypothesis tests, and ML models.

Open-Meteo settings used for the weather file:

- Location: Istanbul, around `41.02N, 28.89E`
- Dates: `2023-01-01` to `2023-12-31`
- Hourly variables: temperature, relative humidity, precipitation, rain, weather code, sea-level pressure, cloud cover, wind speed, wind gusts, and wind direction
- Units: Celsius, km/h, millimeters

Open-Meteo documentation: `https://open-meteo.com/en/docs/historical-weather-api`

## Hypothesis Testing Summary

The data collection, EDA, and hypothesis testing files are:

- Data cleaning and merging: `scripts/01_prepare_data.py`
- EDA and hypothesis tests: `scripts/02_eda_and_tests.py`
- EDA/hypothesis summary: `results/eda_hypothesis_results.md`
- Figures: `results/figures/`

## Machine Learning Summary

The machine learning files are:

- Machine learning models: `scripts/03_ml_models.py`
- ML summary: `results/ml_model_results.md`
- Test-period predictions: `results/ml_predictions_2023.csv`
- ML visualizations: `results/figures/ml_predictions_test_period.png` and `results/figures/ml_feature_importance.png`

## How To Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the data cleaning script:

```bash
python scripts/01_prepare_data.py
```

Run the EDA and hypothesis testing script:

```bash
python scripts/02_eda_and_tests.py
```

Run the machine learning script:

```bash
python scripts/03_ml_models.py
```

## Hypotheses

The analysis tests:

- Whether rainy days have a different average traffic index than non-rainy days.
- Whether weekdays have a different average traffic index than weekends.
- Whether weather variables such as temperature, humidity, precipitation, cloud cover, and wind speed are correlated with average traffic index.

## Machine Learning

The machine learning analysis predicts daily average traffic index using weather and calendar features. I compare a baseline mean model, linear regression, and random forest regression with a chronological train/test split, so the test set comes after the training dates.

## AI Use Disclosure

I used AI as a coding helper while preparing the repository. My prompts were mainly about organizing the folder structure, writing Python scripts for cleaning/EDA/ML, fixing path errors, and improving the README wording. I reviewed the outputs and kept the project focused on my original topic: weather and traffic in Istanbul.
