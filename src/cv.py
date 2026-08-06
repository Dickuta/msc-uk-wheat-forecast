"""
cv.py

Expanding-window cross-validation used for every model in the study.

Protocol (identical to the corrected Colab pipeline / validation_pipeline.py):
  * training window grows by one year at a time, starting at 1980;
  * the first forecast origin is 2000 (``INITIAL_TRAIN_END``), so the first
    out-of-sample test year is 2001;
  * for every origin the model is fit ONCE and shared across all four
    horizons (a per-origin model cache), so each horizon uses the same fit;
  * optional standardisation (not used in the corrected results);
  * each prediction also records training time and peak memory (tracemalloc).

Also contains the two trivial baselines: Climatology (training-window mean)
and Naive_RandomWalk (last observed yield).
"""

import time
import tracemalloc
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import mae, rmse


@dataclass
class ExpandingWindowCV:
    """Expanding-window cross-validation for a single model factory.

    Parameters
    ----------
    data : pandas.DataFrame
        Full modelling table (must contain ``year`` and ``yield_t_ha``).
    model_factory : callable
        ``factory(train_df, horizon) -> object`` exposing ``predict(h)``.
    model_name : str, optional
        Name stamped on every prediction row; falls back to the factory's
        ``__name__`` when unset (handles ``functools.partial`` factories).
    horizons : list of int
        Forecast horizons evaluated.
    initial_train_end : int
        First forecast origin (years <= this are the first training set).
    seed : int
        Random seed (used to seed the RNG before evaluation).
    standardise : bool
        Standardise covariates to zero mean / unit variance (default False,
        matching the corrected results).
    """

    data: pd.DataFrame
    model_factory: object
    model_name: str = None
    horizons: list = field(default_factory=lambda: [1, 2, 3, 4])
    initial_train_end: int = 2000
    seed: int = 42
    standardise: bool = False

    def __post_init__(self):
        self.data = self.data.sort_values("year").reset_index(drop=True)
        self.covariate_cols = [
            c for c in self.data.columns if c not in ("year", "yield_t_ha")
        ]
        self.model_name = self.model_name or getattr(
            self.model_factory, "__name__", type(self.model_factory).__name__
        )
        self.results = []
        self.skipped_folds = []

    def _get_train_test(self, train_end_year, horizon):
        train_mask = self.data["year"] <= train_end_year
        test_year = train_end_year + horizon
        if test_year > self.data["year"].max():
            return None, None, None
        test_mask = self.data["year"] == test_year
        train_df = self.data[train_mask].copy()
        test_row = self.data[test_mask]
        if len(test_row) == 0:
            return None, None, None
        return train_df, test_row.iloc[0], test_year

    def _standardise_train_test(self, train_df, test_row):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        train_scaled = train_df.copy()
        test_scaled = test_row.copy()
        train_scaled[self.covariate_cols] = scaler.fit_transform(
            train_df[self.covariate_cols]
        )
        test_scaled[self.covariate_cols] = scaler.transform(
            test_row[self.covariate_cols].values.reshape(1, -1)
        )
        return train_scaled, test_scaled

    def evaluate(self):
        """Run the CV for every horizon and return a per-horizon summary."""
        np.random.seed(self.seed)
        model_cache = {}
        skipped = 0
        for horizon in self.horizons:
            predictions = []
            for train_end in range(
                self.initial_train_end, int(self.data["year"].max())
            ):
                train_df, test_row, test_year = self._get_train_test(train_end, horizon)
                if train_df is None:
                    continue
                if train_end not in model_cache:
                    if self.standardise:
                        train_df_scaled, _ = self._standardise_train_test(
                            train_df, test_row
                        )
                        model_cache[train_end] = self.model_factory(
                            train_df_scaled, horizon
                        )
                    else:
                        model_cache[train_end] = self.model_factory(train_df, horizon)

                model = model_cache[train_end]

                if self.standardise:
                    _, test_row, _ = self._get_train_test(train_end, horizon)
                    _, test_row = self._standardise_train_test(
                        self.data[self.data["year"] <= train_end].copy(),
                        test_row,
                    )
                tracemalloc.start()
                t0 = time.time()
                try:
                    y_pred = model.predict(horizon)
                except Exception as exc:
                    tracemalloc.stop()
                    skipped += 1
                    self.skipped_folds.append(
                        {
                            "horizon": horizon,
                            "test_year": int(test_year),
                            "error": str(exc)[:200],
                        }
                    )
                    continue
                train_time_s = time.time() - t0
                _, peak_memory_mb = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                y_true = test_row["yield_t_ha"]
                predictions.append(
                    {
                        "model": self.model_name,
                        "horizon": horizon,
                        "test_year": int(test_year),
                        "y_true": float(y_true),
                        "y_pred": float(y_pred),
                        "train_time_s": train_time_s,
                        "peak_memory_mb": peak_memory_mb / 1e6,
                    }
                )
            if predictions:
                self.results.extend(predictions)
        if skipped:
            warnings.warn(
                f"{self.model_name}: {skipped} origin x horizon "
                "predictions failed and were skipped"
            )
        return self.summarise()

    def summarise(self):
        """Aggregate the per-fold predictions into per-horizon metrics."""
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame(self.results)
        summary = (
            df.groupby("horizon")
            .agg(
                rmse=(
                    "y_true",
                    lambda y: rmse(y.values, df.loc[y.index, "y_pred"].values),
                ),
                mae=(
                    "y_true",
                    lambda y: mae(y.values, df.loc[y.index, "y_pred"].values),
                ),
                n_test=("test_year", "count"),
                avg_train_time_s=("train_time_s", "mean"),
                avg_peak_memory_mb=("peak_memory_mb", "mean"),
            )
            .reset_index()
        )
        summary.insert(0, "model", self.model_name)
        return summary


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def expanding_windows(df, horizon, initial_train_end=2000):
    """Yield (train, test_year) pairs under the same expanding-window protocol."""
    max_origin = int(df["year"].max()) - horizon
    for origin in range(initial_train_end, max_origin + 1):
        train = df[df["year"] <= origin]
        test_year = origin + horizon
        yield train, int(test_year)


def evaluate_baseline(df, horizon, predictor, initial_train_end=2000):
    """Return (y_true, y_pred) arrays for a baseline predictor on a horizon."""
    y_true, y_pred = [], []
    for train, test_year in expanding_windows(df, horizon, initial_train_end):
        y_true.append(df[df["year"] == test_year]["yield_t_ha"].iloc[0])
        y_pred.append(predictor(train))
    return np.array(y_true), np.array(y_pred)
