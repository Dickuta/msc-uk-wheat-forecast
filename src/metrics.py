"""
metrics.py

Forecast accuracy metrics and the Diebold-Mariano (DM) test.

The DM implementation is verbatim from the thesis `validation_pipeline.py`:
it uses the Harvey-Leybourne-Newbold (HLN) small-sample correction and a
one-sided... two-sided Student-t approximation for the p-value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def diebold_mariano(
    y_true: ArrayLike,
    y_pred1: ArrayLike,
    y_pred2: ArrayLike,
    loss: str = "MSE",
    h: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test between two forecast series.

    Parameters
    ----------
    y_true : array_like
        Observed values.
    y_pred1, y_pred2 : array_like
        Forecasts from model 1 and model 2.
    loss : {"MSE", "MAE"}
        Loss used to define the differential series d_t.
    h : int
        Forecast horizon used in the autocovariance correction.

    Returns
    -------
    dm_stat : float
        HLN-corrected DM statistic.
    p_val : float
        Two-sided p-value on a Student-t distribution with T-1 df.

    Notes
    -----
    A negative statistic favours model 1 (the first argument).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred1 = np.asarray(y_pred1, dtype=float)
    y_pred2 = np.asarray(y_pred2, dtype=float)
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    if loss == "MSE":
        d = e1**2 - e2**2
    elif loss == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    T = len(d)
    d_bar = np.mean(d)
    if np.isclose(d_bar, 0):
        return 0.0, 1.0

    # Sample autocovariances of the differential series (biased estimator).
    gamma = np.correlate(d - d_bar, d - d_bar, mode="full")
    gamma = gamma[T - 1 :] / T
    var_d = gamma[0] + 2 * np.sum(gamma[1:h]) if h > 1 else gamma[0]
    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(var_d / T)

    # Harvey-Leybourne-Newbold small-sample correction.
    hln_corr = (T + 1 - 2 * h + h * (h - 1) / T) / T
    dm_stat_corr = dm_stat * np.sqrt(hln_corr)
    p_val = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corr), df=T - 1))
    return float(dm_stat_corr), float(p_val)
