"""
models.py

Model factories used by the expanding-window cross-validation.

Each factory has the signature ``factory(train_df, horizon)`` and returns an
object exposing ``predict(h)`` that returns a single scalar forecast for the
h-th step ahead. The specifications below are byte-for-byte identical to the
corrected Colab pipeline (`validation_pipeline.py` / `run_ml_models.py`):

  * Persistence     - last observed yield.
  * ARIMA           - AICc selection over p, q in 0..3 with d = 0.
  * SARIMA          - AICc selection over (p, 0, q) with seasonal (p, 0, q, 1);
                      fallback is plain ARIMA(1, 0, 0) (see note).
  * ARIMAX          - stepwise covariate selection (p > 0.10, cap 5) with an
                      ARIMA(1, 0, 0) base and ARIMA-forecast exog inputs.
  * Prophet         - changepoint prior selected by 1-year-ahead rolling CV
                      over {0.01, 0.05, 0.1}; all covariates as regressors.
  * RandomForest    - hyperparameters tuned once on the full series.
  * XGBoost         - hyperparameters tuned once on the full series.
  * ARIMA+XGBoost   - ARIMA (AICc, d in 0..1) residuals modelled by XGBoost.

``oracle_fc``
-------------
The exogenous-forecasting factories (ARIMAX, Prophet, RandomForest, XGBoost,
ARIMA+XGBoost) accept an optional ``oracle_fc`` callable with the same
signature as ``features.forecast_exogenous``. When given, it is used *instead
of* the ARIMA(1, 0, 0) projections — this is what stage 05's "oracle"
experiment (perfect knowledge of future weather) is built on, without
duplicating any model code.

``predict_interval(h)``
-----------------------
The ARIMA and Prophet factories also expose ``predict_interval(h)`` returning
``(yhat, lower, upper)`` at ``interval_width`` coverage. Stage 05 uses this for
the prediction-interval coverage experiment (Tables 4.5 / Figure 4.3).

SARIMA fallback note
--------------------
``validation_pipeline.py`` contained an intermediate fallback loop that could
never assign a model (an undefined variable raised a ``NameError`` which was
swallowed), so the fallback was always plain ``ARIMA(1, 0, 0)``. That behaviour
is written explicitly here so the results are identical.
"""

import warnings

import numpy as np
import pandas as pd

from .features import forecast_exogenous


def _fit_arima(model, **kwargs):
    """Fit an ARIMA model with statsmodels warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(method_kwargs={"disp": False}, **kwargs)


def fit_best_arima(y, p_range, d_range, q_range):
    """Fit ARIMA selecting (p, d, q) by AICc; fallback is ARIMA(1, 0, 0).

    The loop order (p, then d, then q) and the strict ``<`` comparison are
    intentional: they reproduce the AICc tie-breaking of the corrected Colab
    pipeline, so refactoring here cannot change any produced number.
    """
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = np.inf
    best_model = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    fitted = _fit_arima(ARIMA(y, order=(p, d, q)))
                    if fitted.aicc < best_aic:
                        best_aic = fitted.aicc
                        best_model = fitted
                except Exception:
                    continue
    if best_model is None:
        best_model = _fit_arima(ARIMA(y, order=(1, 0, 0)))
    return best_model


def arimax_stepwise_selection(train_df, covariate_cols, alpha=0.10, max_cov=5):
    """Stepwise ARIMAX covariate selection (p-values read by name).

    Fits an ARIMA(1,0,0) on ``train_df`` with the current covariate set and
    drops the covariate with the largest p-value whenever that p-value
    exceeds ``alpha``, iterating to convergence (at most one pass per
    candidate covariate). Returns ``(selected, steps)`` where ``steps`` holds
    the p-value series of each iteration.

    The exog matrix is passed as a DataFrame so statsmodels names the
    parameters after the covariates; p-values are then read *by name*.
    The old implementation sliced ``pvalues`` from the tail, which under
    the actual layout ``[const, x1..xk, ar.L1, sigma2]`` mistook ``ar.L1``
    and ``sigma2`` for covariates and never inspected the first exog
    parameter.

    Note: a covariate that is constant within the window is collinear with
    the intercept, so statsmodels raises ``ValueError`` and the loop breaks,
    keeping the full set. This short-circuit is part of the verified results
    and must not be "fixed".
    """
    from statsmodels.tsa.arima.model import ARIMA

    y = train_df["yield_t_ha"].values
    selected = list(covariate_cols)
    steps = []
    for _ in range(len(covariate_cols)):
        X = train_df[selected] if selected else None
        try:
            fitted = _fit_arima(ARIMA(y, exog=X, order=(1, 0, 0)))
        except Exception:
            break
        if X is None:
            break
        pv = pd.Series(fitted.pvalues, index=fitted.param_names)
        pvals = pv[selected].fillna(np.inf)
        steps.append(pvals.copy())
        if pvals.max() > alpha and len(selected) > 1:
            selected.remove(pvals.idxmax())
        else:
            break
    return selected[:max_cov], steps


# --------------------------------------------------------------------------- #
# Statistical models
# --------------------------------------------------------------------------- #
def persistence_factory(train_df, horizon):
    last_yield = float(train_df["yield_t_ha"].iloc[-1])

    class PersistencePredictor:
        def predict(self, h):
            return last_yield

    return PersistencePredictor()


def arima_factory(train_df, horizon):
    y = train_df["yield_t_ha"].values
    fitted = fit_best_arima(y, range(0, 4), range(0, 1), range(0, 4))

    class ARIMAPredictor:
        def __init__(self, fitted):
            self.fitted = fitted

        def predict(self, h):
            pred = self.fitted.forecast(steps=h)
            return float(pred.iloc[-1]) if hasattr(pred, "iloc") else float(pred[-1])

        def predict_interval(self, h):
            fc = self.fitted.get_forecast(steps=h)
            ci = np.asarray(fc.conf_int(alpha=0.05))
            yhat = float(np.asarray(fc.predicted_mean)[-1])
            return yhat, float(ci[-1, 0]), float(ci[-1, 1])

    return ARIMAPredictor(fitted)


def sarima_factory(train_df, horizon):
    from statsmodels.tsa.arima.model import ARIMA

    y = train_df["yield_t_ha"].values
    best_aic = np.inf
    best_model = None
    for p in range(0, 3):
        for q in range(0, 3):
            try:
                model = ARIMA(y, order=(p, 0, q), seasonal_order=(p, 0, q, 1))
                fitted = _fit_arima(model)
                if fitted.aicc < best_aic:
                    best_aic = fitted.aicc
                    best_model = fitted
            except Exception:
                continue
    if best_model is None:
        best_model = _fit_arima(ARIMA(y, order=(1, 0, 0)))

    class SARIMAPredictor:
        def __init__(self, fitted):
            self.fitted = fitted

        def predict(self, h):
            pred = self.fitted.forecast(steps=h)
            return float(pred.iloc[-1]) if hasattr(pred, "iloc") else float(pred[-1])

    return SARIMAPredictor(best_model)


def arimax_factory(train_df, horizon, oracle_fc=None):
    from statsmodels.tsa.arima.model import ARIMA

    cov_cols = [c for c in train_df.columns if c not in ("year", "yield_t_ha")]
    selected, _ = arimax_stepwise_selection(train_df, cov_cols)

    class ARIMAXPredictor:
        def __init__(self, train_df, selected, order=(1, 0, 0), oracle_fc=None):
            self.train_df = train_df
            self.selected = selected
            self.order = order
            self.oracle_fc = oracle_fc
            y = train_df["yield_t_ha"].values
            X = train_df[selected].values if selected else None
            self.model = _fit_arima(ARIMA(y, exog=X, order=order))

        def predict(self, h):
            if self.selected:
                fc = self.oracle_fc or forecast_exogenous
                exog_forecasts = fc(self.train_df, self.selected, h)
                X_future = np.column_stack([exog_forecasts[c] for c in self.selected])
            else:
                X_future = None
            pred = self.model.forecast(steps=h, exog=X_future)
            return float(pred.iloc[-1]) if hasattr(pred, "iloc") else float(pred[-1])

    return ARIMAXPredictor(train_df, selected, oracle_fc=oracle_fc)


# --------------------------------------------------------------------------- #
# Prophet helpers (shared by the CV factory and the PI experiment)
# --------------------------------------------------------------------------- #
def _prophet_df(train_df):
    pdf = train_df.copy()
    pdf["ds"] = pd_to_datetime_years(pdf["year"])
    pdf.rename(columns={"yield_t_ha": "y"}, inplace=True)
    return pdf


def _select_prophet_changepoint_scale(pdf, cov_cols, candidates=(0.01, 0.05, 0.1)):
    """Pick the changepoint prior scale by 1-year-ahead rolling CV."""
    from prophet import Prophet

    best_scale = 0.05
    best_cv_rmse = np.inf
    years = pdf["year"].values
    for scale in candidates:
        cv_errors = []
        for split_year in range(int(years[len(years) // 2]), int(years[-1])):
            train_cv = pdf[pdf["year"] <= split_year]
            test_cv = pdf[pdf["year"] == split_year + 1]
            if len(test_cv) == 0:
                continue
            try:
                m = Prophet(
                    changepoint_prior_scale=scale,
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                for c in cov_cols:
                    m.add_regressor(c)
                m.fit(train_cv[["ds", "y"] + cov_cols])
                future = m.make_future_dataframe(periods=1, freq="YE")
                for c in cov_cols:
                    future[c] = train_cv[c].values[-1]
                fcst = m.predict(future)
                cv_errors.append((fcst["yhat"].iloc[-1] - test_cv["y"].iloc[0]) ** 2)
            except Exception:
                continue
        if cv_errors:
            cv_rmse = np.sqrt(np.mean(cv_errors))
            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_scale = scale
    return best_scale


def _fill_future_covariates(future, train_df, cov_cols, exog_fcst):
    """Fill the covariate columns of a Prophet future frame.

    Rows beyond the training window take the exogenous forecast; rows inside
    it take the observed training value (with a training-mean fallback).
    """
    max_train_year = train_df["year"].max()
    fcst_idx = 0
    for i, row in future.iterrows():
        yr = row["ds"].year
        if yr > max_train_year:
            for c in cov_cols:
                future.at[i, c] = exog_fcst[c][fcst_idx]
            fcst_idx += 1
        else:
            for c in cov_cols:
                past_mask = train_df["year"] == yr
                if past_mask.any():
                    future.at[i, c] = train_df.loc[past_mask, c].iloc[0]
                else:
                    future.at[i, c] = train_df[c].mean()


def prophet_factory(train_df, horizon, oracle_fc=None, interval_width=0.95):
    cov_cols = [c for c in train_df.columns if c not in ("year", "yield_t_ha")]
    pdf = _prophet_df(train_df)
    best_scale = _select_prophet_changepoint_scale(pdf, cov_cols)

    class ProphetPredictor:
        def __init__(self, pdf, cov_cols, scale, oracle_fc=None, interval_width=0.95):
            from prophet import Prophet

            self.pdf = pdf
            self.train_df = train_df
            self.cov_cols = cov_cols
            self.oracle_fc = oracle_fc
            self.model = Prophet(
                changepoint_prior_scale=scale,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=interval_width,
            )
            for c in cov_cols:
                self.model.add_regressor(c)
            self.model.fit(pdf[["ds", "y"] + cov_cols])

        def _forecast_frame(self, h):
            fc = self.oracle_fc or forecast_exogenous
            exog_fcst = fc(self.train_df, self.cov_cols, h)
            future = self.model.make_future_dataframe(periods=h, freq="YE")
            _fill_future_covariates(future, self.train_df, self.cov_cols, exog_fcst)
            return self.model.predict(future)

        def predict(self, h):
            return float(self._forecast_frame(h)["yhat"].iloc[-1])

        def predict_interval(self, h):
            last = self._forecast_frame(h).iloc[-1]
            return (
                float(last["yhat"]),
                float(last["yhat_lower"]),
                float(last["yhat_upper"]),
            )

    return ProphetPredictor(pdf, cov_cols, best_scale, oracle_fc, interval_width)


def pd_to_datetime_years(years):
    import pandas as pd

    return pd.to_datetime(years, format="%Y")


# --------------------------------------------------------------------------- #
# Machine-learning models
# --------------------------------------------------------------------------- #
def tune_rf(X_full, y_full, tscv):
    """Grid-search RandomForest hyperparameters once on the full series."""
    from sklearn.ensemble import RandomForestRegressor

    best_score, best_params = np.inf, None
    for depth in [3, 5, None]:
        for leaf in [1, 4]:
            for split in [2, 10]:
                for mf in ["sqrt", 0.7]:
                    scores = []
                    for tr_idx, val_idx in tscv.split(X_full):
                        try:
                            rf = RandomForestRegressor(
                                n_estimators=500,
                                max_depth=depth,
                                min_samples_leaf=leaf,
                                min_samples_split=split,
                                max_features=mf,
                                random_state=42,
                            )
                            rf.fit(X_full[tr_idx], y_full[tr_idx])
                            pred = rf.predict(X_full[val_idx])
                            scores.append(
                                np.sqrt(np.mean((y_full[val_idx] - pred) ** 2))
                            )
                        except Exception:
                            continue
                    if scores and np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_params = dict(
                            n_estimators=500,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            min_samples_split=split,
                            max_features=mf,
                        )
    return best_params or dict(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
    )


def tune_xgb(X_full, y_full, tscv):
    """Grid-search XGBoost hyperparameters once on the full series."""
    import xgboost as xgb

    best_score, best_params = np.inf, None
    for depth in [3, 5]:
        for lr in [0.05, 0.1]:
            for subsample in [0.8, 1.0]:
                for colsample in [0.8, 1.0]:
                    scores = []
                    for tr_idx, val_idx in tscv.split(X_full):
                        try:
                            model = xgb.XGBRegressor(
                                n_estimators=500,
                                max_depth=depth,
                                learning_rate=lr,
                                subsample=subsample,
                                colsample_bytree=colsample,
                                early_stopping_rounds=50,
                                random_state=42,
                            )
                            model.fit(
                                X_full[tr_idx],
                                y_full[tr_idx],
                                eval_set=[(X_full[val_idx], y_full[val_idx])],
                                verbose=False,
                            )
                            pred = model.predict(X_full[val_idx])
                            scores.append(
                                np.sqrt(np.mean((y_full[val_idx] - pred) ** 2))
                            )
                        except Exception:
                            continue
                    if scores and np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_params = dict(
                            n_estimators=500,
                            max_depth=depth,
                            learning_rate=lr,
                            subsample=subsample,
                            colsample_bytree=colsample,
                        )
    return best_params or dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )


def make_rf_factory(rf_params):
    from sklearn.ensemble import RandomForestRegressor

    class RFPredictor:
        def __init__(self, train_df, horizon, oracle_fc=None):
            self.train_df = train_df
            self.cov_cols = [
                c for c in train_df.columns if c not in ("year", "yield_t_ha")
            ]
            self.oracle_fc = oracle_fc
            self.model = RandomForestRegressor(**rf_params, random_state=42)
            self.model.fit(
                train_df[self.cov_cols].values, train_df["yield_t_ha"].values
            )

        def predict(self, h):
            fc = self.oracle_fc or forecast_exogenous
            exog_fcst = fc(self.train_df, self.cov_cols, h)
            X_future = np.array([[exog_fcst[c][-1] for c in self.cov_cols]])
            return float(self.model.predict(X_future)[0])

    return RFPredictor


def make_xgb_factory(xgb_params):
    import xgboost as xgb

    class XGBPredictor:
        def __init__(self, train_df, horizon, oracle_fc=None):
            self.train_df = train_df
            self.cov_cols = [
                c for c in train_df.columns if c not in ("year", "yield_t_ha")
            ]
            self.oracle_fc = oracle_fc
            self.model = xgb.XGBRegressor(**xgb_params, random_state=42)
            self.model.fit(
                train_df[self.cov_cols].values, train_df["yield_t_ha"].values
            )

        def predict(self, h):
            fc = self.oracle_fc or forecast_exogenous
            exog_fcst = fc(self.train_df, self.cov_cols, h)
            X_future = np.array([[exog_fcst[c][-1] for c in self.cov_cols]])
            return float(self.model.predict(X_future)[0])

    return XGBPredictor


def make_hybrid_factory(xgb_params):
    import xgboost as xgb

    class HybridPredictor:
        def __init__(self, train_df, horizon, oracle_fc=None):
            self.train_df = train_df
            self.cov_cols = [
                c for c in train_df.columns if c not in ("year", "yield_t_ha")
            ]
            self.oracle_fc = oracle_fc
            y = train_df["yield_t_ha"].values
            best_arima = fit_best_arima(y, range(0, 4), range(0, 2), range(0, 4))
            self.arima = best_arima
            residuals = y - best_arima.fittedvalues
            X_s2 = train_df[self.cov_cols].values
            self.xgb = xgb.XGBRegressor(**xgb_params, random_state=42)
            self.xgb.fit(X_s2, residuals)

        def predict(self, h):
            arima_pred = self.arima.forecast(steps=h)
            arima_val = (
                float(arima_pred.iloc[-1])
                if hasattr(arima_pred, "iloc")
                else float(arima_pred[-1])
            )
            fc = self.oracle_fc or forecast_exogenous
            exog_fcst = fc(self.train_df, self.cov_cols, h)
            X_future = np.array([[exog_fcst[c][-1] for c in self.cov_cols]])
            correction = float(self.xgb.predict(X_future)[0])
            return arima_val + correction

    return HybridPredictor
