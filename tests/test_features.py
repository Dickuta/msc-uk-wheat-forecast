import numpy as np
import pandas as pd

from src.features import forecast_exogenous


def test_forecast_exogenous_short_series_uses_mean():
    train = pd.DataFrame({"year": range(2000, 2004), "x": [1.0, 2.0, 3.0, 4.0]})
    fcst = forecast_exogenous(train, ["x"], 3)["x"]
    assert fcst.shape == (3,)
    assert np.allclose(fcst, 2.5)


def test_forecast_exogenous_constant_series_returns_mean_fallback():
    train = pd.DataFrame({"year": range(2000, 2020), "x": [7.0] * 20})
    fcst = forecast_exogenous(train, ["x"], 4)["x"]
    assert fcst.shape == (4,)
    assert np.all(np.isfinite(fcst))
    # ARIMA(1,0,0) on a zero-variance series is degenerate -> the fallback
    # returns the training mean, so a stable target is fine to assert.
    assert np.allclose(fcst, 7.0, atol=1.0)


def test_forecast_exogenous_multi_column():
    train = pd.DataFrame(
        {
            "year": range(2000, 2020),
            "a": np.arange(20, dtype=float),
            "b": np.full(20, 3.0),
        }
    )
    fcst = forecast_exogenous(train, ["a", "b"], 2)
    assert set(fcst) == {"a", "b"}
    assert fcst["a"].shape == (2,)
    assert fcst["b"].shape == (2,)
    assert np.all(np.isfinite(fcst["a"]))
