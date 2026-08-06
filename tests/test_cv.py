import numpy as np
import pandas as pd
from functools import partial

from src.cv import ExpandingWindowCV, expanding_windows, evaluate_baseline
from src.models import persistence_factory


def _table():
    rng = np.random.default_rng(0)
    years = np.arange(1980, 2025)
    n = len(years)
    return pd.DataFrame(
        {
            "year": years,
            "yield_t_ha": 6.0 + 0.05 * np.arange(n) + rng.normal(0, 0.3, n),
            "x": rng.normal(0, 1, n),
        }
    )


def test_expanding_windows_count_and_years():
    df = _table()
    pairs = list(expanding_windows(df, 1, initial_train_end=2000))
    assert len(pairs) == 24  # origins 2000..2023
    train, test_year = pairs[0]
    assert train["year"].max() == 2000
    assert test_year == 2001
    assert len(list(expanding_windows(df, 4, initial_train_end=2000))) == 21


def test_evaluate_baseline_shapes():
    df = _table()
    y_true, y_pred = evaluate_baseline(
        df, 1, lambda tr: tr["yield_t_ha"].mean(), initial_train_end=2000
    )
    assert len(y_true) == len(y_pred) == 24
    assert np.all(np.isfinite(y_true))
    assert np.all(np.isfinite(y_pred))


def test_expanding_window_cv_with_persistence():
    df = _table()
    cv = ExpandingWindowCV(
        data=df,
        model_factory=persistence_factory,
        horizons=[1, 2],
        initial_train_end=2003,
    )
    summary = cv.evaluate()
    assert set(summary["horizon"]) == {1, 2}
    assert (summary["n_test"] > 0).all()
    assert (summary["rmse"] > 0).all()
    detail = pd.DataFrame(cv.results)
    assert len(detail) == summary["n_test"].sum()
    assert set(detail["horizon"]) == {1, 2}
    assert np.all(np.isfinite(detail[["y_true", "y_pred"]]).values)


def test_expanding_window_cv_with_partial_factory():
    df = _table()
    cv = ExpandingWindowCV(
        data=df,
        model_factory=partial(persistence_factory),
        model_name="Persistence-Partial",
        horizons=[1],
        initial_train_end=2003,
    )
    summary = cv.evaluate()
    detail = pd.DataFrame(cv.results)
    assert set(summary["model"]) == {"Persistence-Partial"}
    assert set(detail["model"]) == {"Persistence-Partial"}
    assert len(detail) > 0


def test_expanding_window_cv_records_skipped_folds():
    df = _table()

    class FlakyFactory:
        def __init__(self, train_df, horizon):
            pass

        def predict(self, h):
            raise RuntimeError("boom")

    cv = ExpandingWindowCV(
        data=df,
        model_factory=FlakyFactory,
        model_name="Flaky",
        horizons=[1],
        initial_train_end=2003,
    )
    summary = cv.evaluate()
    assert summary.empty
    assert len(cv.skipped_folds) > 0
    assert cv.skipped_folds[0]["horizon"] == 1
    assert "boom" in cv.skipped_folds[0]["error"]
