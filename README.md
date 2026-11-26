# Gap-Imputation-XGBoost-Forecasting

Quantitative analysis of imputation methods on XGBoost forecast performance by gap position and length.

This is the code for a study comparing four gap-filling methods on meteorological sensor data — Linear interpolation, Spline, LOCF (last observation carried forward), and Diagonal Climatology — and checking how each one affects downstream forecasting accuracy with XGBoost.

## Paper

**Quantitative Analysis of Imputation Methods on XGBoost Forecast Performance by Gap Position and Length**
Srinivasa Harshith Pataneni, T. Lakshanya, Kandula V. Subrahmanyam, M. Roja Raman

Accepted at the 2026 IEEE India Geoscience and Remote Sensing Symposium (InGARSS 2026), Hyderabad, India, 1–4 December 2026. Paper ID: 1571313102.

**Key findings**
- For gaps longer than 12 h, Diagonal Climatology recovers missing values best (RMSE 1.97 °C vs 3.35–4.86 °C for the other methods). Linear is best on short gaps.
- An XGBoost forecaster trained on Spline-filled data is the most accurate at every missingness level (RMSE 0.467–0.539 °C), even though Spline is the poorest reconstructor on long gaps.
- Reconstruction accuracy is a poor proxy for forecast skill. Imputers should be chosen based on gap characteristics and downstream performance.

## Dataset

High-resolution sensor data from Shadnagar, Telangana, India, recorded at 18 m above ground level through 2017, at 30-minute intervals. Fields used: air temperature (AirTC_18m, the target), relative humidity, wind speed, and wind direction.

Synthetic gaps were introduced at 5%, 10%, 15%, and 20% missingness, placed deliberately at local maxima, minima, and points of rapid change — the segments where missing data usually causes the most damage in practice.

## Method

1. Introduce synthetic gaps into the temperature series at varying lengths and positions.
2. Fill the gaps using each of the four interpolation methods.
3. Train an XGBoost regressor (with Optuna-tuned hyperparameters) on the gap-filled data to forecast temperature.
4. Compare both the recovery accuracy of each interpolation method and its downstream effect on forecast performance.

Train/validation/test split is chronological (80% / 10% / 10%) to avoid leakage between periods.

## Files

- `proj_exp3_final.py` — entry point; runs the full experiment end to end, calling into the two modules below
- `xgboost_model.py` — feature engineering, lag selection (justified using ACF/PACF/cross-correlation), model training, evaluation
- `interpolation_methods.py` — implementations of the four gap-filling methods
- `Proj data.csv` — raw dataset
- `outputs/` — generated figures, tables, and reports:
  - `gap_visualisations/` — gap distribution and recovery plots
  - `model_results/` — per-method forecast results and residuals
  - `comparison_reports/` — cross-method comparison metrics and plots
  - `extra_analysis/` — boosting curves, feature importance, Optuna diagnostics

## Running it

```bash
pip install xgboost scikit-learn pandas numpy matplotlib statsmodels optuna scipy
python proj_exp3_final.py
```

Output figures and reports get written to `outputs/`.

## Notes

This was built as part of an internship under the NICES program at the National Remote Sensing Centre (NRSC), ISRO, Hyderabad.
