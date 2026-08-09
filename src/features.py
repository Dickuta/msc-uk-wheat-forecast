"""
features.py

Feature engineering helpers shared by the model stage.

The only exogenous-feature engineering used in the corrected pipeline is the
multi-step-ahead forecast of every covariate with a plain ARIMA(1, 0, 0)
model fitted on the training window. This mirrors operational practice: to
forecast yield for year Y+h the model must first obtain the weather of that
year, so those values are projected from the covariates' own history.
"""

import warnings

import numpy as np


def forecast_exogenous(train_df, covariate_cols, horizon):
    """Forecast every covariate for ``horizon`` steps ahead with ARIMA(1,0,0).

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data (must contain ``covariate_cols``).
    covariate_cols : list of str
        Covariate columns to forecast.
    horizon : int
        Number of future steps.

    Returns
    -------
    dict
        Mapping ``column -> np.ndarray`` of length ``horizon``.
    """
    from statsmodels.tsa.arima.model import ARIMA

    forecasts = {}
    for col in covariate_cols:
        y = train_df[col].dropna().values
        if len(y) < 5:
            forecasts[col] = np.full(horizon, float(np.mean(y)))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(y, order=(1, 0, 0))
                fitted = model.fit()
            pred = fitted.forecast(steps=horizon)
            vals = pred.values if hasattr(pred, "values") else np.asarray(pred)
            forecasts[col] = vals.ravel()
        except Exception:
            # Degenerate series -> flat forecast at the training mean.
            forecasts[col] = np.full(horizon, float(np.mean(y)))
    return forecasts
