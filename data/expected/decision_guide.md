# UK wheat yield forecasting — practitioner decision guide

Generated from the verified outputs of `scripts/05_Model.py` (see `data/expected/` for the verification gate).

## Best overall model: ARIMA+XGBoost

Mean RMSE across horizons 1-4 (t/ha):

| Model | Mean RMSE |
|---|---|
| ARIMA+XGBoost | 0.6936 |
| ARIMA | 0.7454 |
| Prophet | 0.7647 |
| SARIMA | 0.7840 |
| ARIMAX | 0.8191 |
| Persistence | 0.8245 |
| RandomForest | 0.9893 |
| XGBoost | 1.1923 |

### Best model per horizon

| Horizon | Model | RMSE (t/ha) |
|---|---|---|
| h=1 | Prophet | 0.6793 |
| h=2 | ARIMA+XGBoost | 0.6855 |
| h=3 | ARIMA+XGBoost | 0.6884 |
| h=4 | ARIMA+XGBoost | 0.6936 |

## Statistically significant pairwise differences (DM, 5%): 62 pairs

| Horizon | Pair | Loss | DM stat | p |
|---|---|---|---|---|
| h=2 | SARIMA vs XGBoost | MSE | -5.11 | 0.0000 |
| h=2 | SARIMA vs XGBoost | MAE | -5.69 | 0.0000 |
| h=3 | ARIMA vs RandomForest | MAE | -5.49 | 0.0000 |
| h=3 | ARIMA vs RandomForest | MSE | -9.25 | 0.0000 |
| h=4 | RandomForest vs SARIMA | MAE | 6.49 | 0.0000 |
| h=4 | ARIMAX vs XGBoost | MSE | -18.22 | 0.0000 |
| h=4 | ARIMA+XGBoost vs XGBoost | MAE | -10.12 | 0.0000 |
| h=4 | ARIMA+XGBoost vs XGBoost | MSE | -7.72 | 0.0000 |
| h=3 | SARIMA vs XGBoost | MAE | -11.30 | 0.0000 |
| h=4 | ARIMA vs SARIMA | MSE | -32.71 | 0.0000 |

## Prediction-interval coverage (nominal 95%)

| Model | Horizon | Coverage | Avg width | Interval score |
|---|---|---|---|---|
| ARIMA | h=1 | 79.2% | 2.11 | 3.974 |
| ARIMA | h=2 | 87.0% | 2.265 | 4.65 |
| ARIMA | h=3 | 95.5% | 2.47 | 3.75 |
| ARIMA | h=4 | 90.5% | 2.57 | 3.551 |
| SARIMA | h=1 | 83.3% | 2.3 | 3.323 |
| SARIMA | h=2 | 91.3% | 2.72 | 4.0 |
| SARIMA | h=3 | 95.5% | 2.869 | 3.908 |
| SARIMA | h=4 | 95.2% | 2.928 | 3.409 |
| ARIMAX | h=1 | 79.2% | 2.21 | 4.162 |
| ARIMAX | h=2 | 91.3% | 2.547 | 4.299 |
| ARIMAX | h=3 | 90.9% | 2.651 | 4.032 |
| ARIMAX | h=4 | 85.7% | 2.685 | 4.263 |
| Prophet | h=1 | 75.0% | 1.576 | 4.954 |
| Prophet | h=2 | 73.9% | 1.597 | 5.175 |
| Prophet | h=3 | 72.7% | 1.584 | 6.114 |
| Prophet | h=4 | 61.9% | 1.586 | 7.938 |

## Ceiling: perfect weather foresight (oracle experiment)

Largest benefit: XGBoost at h=4 (36.8% RMSE reduction with perfect weather instead of ARIMA(1,0,0) projections).

| Model | Horizon | Std RMSE | Oracle RMSE | Impr. % |
|---|---|---|---|---|
| XGBoost | h=4 | 1.2521 | 0.7915 | 36.8 |
| XGBoost | h=3 | 1.2401 | 0.7916 | 36.2 |
| XGBoost | h=1 | 1.1397 | 0.7788 | 31.7 |
| XGBoost | h=2 | 1.1372 | 0.8111 | 28.7 |
| RandomForest | h=4 | 1.0551 | 0.7539 | 28.5 |
| RandomForest | h=3 | 1.0185 | 0.7494 | 26.4 |
