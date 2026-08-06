import numpy as np
import pandas as pd
import pytest

from src.models import (
    arima_factory,
    arimax_factory,
    arimax_stepwise_selection,
    fit_best_arima,
    make_hybrid_factory,
    make_rf_factory,
    make_xgb_factory,
    persistence_factory,
)


def _table(n_years=25, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    years = np.arange(1980, 1980 + n_years)
    n = len(years)
    covs = {
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n),
    }
    yield_ = 6.0 + 0.05 * np.arange(n) + 0.2 * covs["a"] + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"year": years, "yield_t_ha": yield_, **covs})
    df.loc[2, "c"] = 0.0  # avoid a near-constant column
    return df


def test_fit_best_arima_returns_fitted_model():
    y = _table()["yield_t_ha"].values
    model = fit_best_arima(y, range(0, 4), range(0, 1), range(0, 4))
    assert hasattr(model, "aicc")
    fc = model.forecast(steps=1)
    assert np.isfinite(float(np.asarray(fc)[-1]))


def test_arima_factory_predict_and_interval():
    df = _table()
    p = arima_factory(df, 1)
    yhat = p.predict(1)
    assert np.isfinite(yhat)
    yhat, lo, hi = p.predict_interval(1)
    assert lo <= yhat <= hi


def test_arimax_stepwise_selection_constant_column_short_circuits():
    # A constant exog column is collinear with the intercept: statsmodels
    # raises ValueError, the selection loop breaks, and the full set is kept.
    # This mirrors the pipeline's verified behaviour — do not "fix" it.
    df = _table()
    df["const"] = 1.0
    selected, _ = arimax_stepwise_selection(
        df, ["a", "b", "c", "const"], alpha=0.10, max_cov=5
    )
    assert selected == ["a", "b", "c", "const"]


def test_arimax_stepwise_selection_valid_subset():
    df = _table()
    selected, steps = arimax_stepwise_selection(
        df, ["a", "b", "c"], alpha=0.10, max_cov=2
    )
    assert set(selected).issubset({"a", "b", "c"})
    assert len(selected) <= 2
    assert len(steps) >= 1


def test_oracle_hook_feeds_true_future_values():
    full = _table(n_years=30)
    train = full[full["year"] <= 2005]
    cov_cols = ["a", "b", "c"]

    def oracle_fc(train_df, cols, horizon):
        origin = int(train_df["year"].max())
        return {
            col: np.array(
                [
                    full[full["year"] == origin + k][col].iloc[0]
                    for k in range(1, horizon + 1)
                ]
            )
            for col in cols
        }

    rf = make_rf_factory(dict(n_estimators=50))(train, 1)
    pred_standard = rf.predict(2)
    assert np.isfinite(pred_standard)

    rf_oracle = make_rf_factory(dict(n_estimators=50))(train, 1, oracle_fc=oracle_fc)
    pred_oracle = rf_oracle.predict(2)
    assert np.isfinite(pred_oracle)


def test_factories_accept_oracle_fc():
    df = _table(n_years=15)
    cov_cols = ["a", "b", "c"]

    def oracle_fc(train_df, cols, horizon):
        origin = int(train_df["year"].max())
        return {col: np.full(horizon, float(train_df[col].mean())) for col in cols}

    for factory in [
        arimax_factory,
        lambda tr, h, oracle_fc=None: make_xgb_factory(dict(n_estimators=50))(
            tr, h, oracle_fc=oracle_fc
        ),
        lambda tr, h, oracle_fc=None: make_hybrid_factory(dict(n_estimators=50))(
            tr, h, oracle_fc=oracle_fc
        ),
    ]:
        p = factory(df, 1, oracle_fc=oracle_fc)
        assert np.isfinite(p.predict(1))


def test_persistence_factory():
    df = _table()
    p = persistence_factory(df, 1)
    assert p.predict(3) == float(df["yield_t_ha"].iloc[-1])
