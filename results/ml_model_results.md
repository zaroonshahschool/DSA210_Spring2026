# Machine Learning Summary

## Goal
This step applies machine learning methods to predict Istanbul's daily average traffic index from weather and calendar features.

## Dataset
- Observations: 365 daily rows
- Date range: 2023-01-01 to 2023-12-31
- Training period: 2023-01-01 to 2023-10-19 (292 rows)
- Test period: 2023-10-20 to 2023-12-31 (73 rows)

## Features Used
- Weather: temperature, humidity, precipitation, rain, pressure, cloud cover, wind speed, and wind gusts.
- Calendar: month, day of week, weekend flag, rainy-day flag, and heavy-rain flag.

## Models
- Baseline Mean: predicts the training-set average traffic index.
- Linear Regression: a simple interpretable regression model.
- Random Forest Regressor: a nonlinear tree-based model.

## Test Results
- Baseline Mean: MAE = 6.209, RMSE = 7.441, R2 = -0.209
- Linear Regression: MAE = 2.823, RMSE = 3.829, R2 = 0.680
- Random Forest: MAE = 3.117, RMSE = 3.898, R2 = 0.668

The best model by RMSE is **Linear Regression**.

## Random Forest Feature Importance
The strongest features in the random forest model were:
- categorical__day_of_week_Sunday: 0.4494
- numeric__cloud_cover_mean_pct: 0.0898
- numeric__pressure_mean_hpa: 0.0625
- numeric__wind_gusts_max_kmh: 0.0618
- numeric__humidity_mean_pct: 0.0615

## Notes
The model is trained only on 2023 daily data, so the results should be treated as an initial machine learning experiment rather than a final production-quality model.
A stronger final report could use multiple years of matching weather data and add holiday or school-calendar variables.

## Output Files
- `results/ml_predictions_2023.csv`
- `results/figures/ml_predictions_test_period.png`
- `results/figures/ml_feature_importance.png`