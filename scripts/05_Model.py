# %% [markdown]
# # 05 · Model — forecast comparison & statistical inference
#
# **Goal.** Reproduce the Results chapter of the thesis (Tables 4.1–4.7,
# Figure 4.3) under the **exact corrected protocol**:
#
# * expanding-window CV from 2000 (seed 42), horizons 1–4;
# * one model fit per training origin, shared across all horizons;
# * ARIMA selected by AICc over p,q ∈ 0–3 (d=0);
# * SARIMA seasonal order (p,0,q,1); ARIMAX stepwise (p > 0.10, cap 5);
# * Prophet changepoint scale from {0.01, 0.05, 0.1} by 1-year-ahead CV;
# * ML models tuned once on the full series with TimeSeriesSplit(5);
# * exogenous covariates forecast with ARIMA(1,0,0).
#
# **Input.** `data/processed/uk_wheat_modelling_table_1980_2024.csv`.
#
# **Outputs** (saved to `data/outputs/`):
#
# | Output | Thesis table |
# |---|---|
# | `model_comparison_results_corrected.csv` | Tables 4.1, 4.2, 4.6 |
# | `baseline_results.csv` | Table 4.3 |
# | `dm_test_results.csv` | Table 4.4 |
# | `pi_coverage_results_corrected.csv` | Table 4.5 |
# | `pi_detailed_results_corrected.csv` | Figure 4.3 |
# | `oracle_exogenous_results.csv` | Table 4.7 |
#
# The final section verifies every output against the canonical thesis numbers
# stored in `data/expected/`.
#
# Everything model-specific lives in `src/models.py`; this stage only
# orchestrates the comparison, inference and verification.

# %% [markdown]
# ## 5.1 Setup

# %%
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Resolve the pipeline root whether this file runs as a script or as a
# notebook cell (where `__file__` is not defined), independent of the CWD.
try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()
PROJECT_ROOT = _here if (_here / "config.py").exists() else _here.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src import plotting  # noqa: F401  (inline in notebooks, Agg when headless)
from src.cv import ExpandingWindowCV, evaluate_baseline
from src.metrics import rmse, mae, diebold_mariano
from src.guards import assert_balanced_test_sets
from src.models import (
    persistence_factory,
    arima_factory,
    sarima_factory,
    arimax_factory,
    prophet_factory,
    tune_rf,
    tune_xgb,
    make_rf_factory,
    make_xgb_factory,
    make_hybrid_factory,
)

warnings.filterwarnings("ignore")
# Silence verbose statsmodels / Prophet / CmdStanPy logging up front. Prophet
# and CmdStanPy re-*configure* their loggers on import (e.g. forecaster.py sets
# its level to INFO), so the loggers are *disabled* here before they exist; a
# subsequent import only returns these same (disabled) logger objects.
for _name in (
    "cmdstanpy",
    "prophet",
    "prophet.models",
    "prophet.logger",
    "prophet.plot",
    "prophet.forecaster",
    "fbprophet",
    "numexpr",
    "numexpr.utils",
):
    logging.getLogger(_name).disabled = True
logging.basicConfig(level=logging.INFO, format="%(message)s")

data = pd.read_csv(config.MODEL_TABLE_FILE)
data["year"] = data["year"].astype(int)
print(
    f"Loaded modelling table: {data.shape[0]} rows ({data['year'].min()}-{data['year'].max()})"
)

# %% [markdown]
# ## 5.2 The evaluation protocol
#
# ```text
#   origin 2000 .. 2023                     (expanding window)
#      |-> fit model ONCE on [1980 .. origin]   (shared across horizons)
#      |-> predict origin + 1, +2, +3, +4 years ahead
#   metrics: RMSE, MAE per horizon
#   DM test: MSE loss, HLN small-sample correction, two-sided t p-value
#   seed 42 everywhere
# ```
#
# `ExpandingWindowCV` (in `src/cv.py`) implements this identically to the
# corrected pipeline, including the per-origin model cache and per-prediction
# timing / memory tracking.


# %%
def run_cv(name, factory):
    """Run the expanding-window CV for one model and print its h=1 RMSE."""
    cv = ExpandingWindowCV(
        data=data,
        model_factory=factory,
        model_name=name,
        horizons=config.HORIZONS,
        seed=config.SEED,
    )
    t0 = time.time()
    summary = cv.evaluate()
    elapsed = time.time() - t0
    detail = pd.DataFrame(cv.results)
    h1 = summary[summary["horizon"] == 1]["rmse"].values[0]
    print(f"  {name:16s} RMSE h=1 = {h1:.4f}  ({elapsed:6.1f}s)")
    if cv.skipped_folds:
        per_h = {}
        for s in cv.skipped_folds:
            per_h[s["horizon"]] = per_h.get(s["horizon"], 0) + 1
        print(
            f"  {name:16s} WARNING: {len(cv.skipped_folds)} skipped fold(s) "
            f"by horizon {per_h} (n_test not comparable)"
        )
    return summary, detail


all_summaries, all_details = [], []

# %% [markdown]
# ## 5.3 Baselines (Table 4.3)
#
# Two trivial benchmarks under the same protocol: **Climatology** (training
# window mean) and **Naive_RandomWalk** (last observed yield). No model should
# be taken seriously if it cannot beat these.

# %%
baseline_rows = []
for horizon in config.HORIZONS:
    for bname, fn in [
        ("Climatology", lambda tr: tr["yield_t_ha"].mean()),
        ("Naive_RandomWalk", lambda tr: tr["yield_t_ha"].iloc[-1]),
    ]:
        y_true, y_pred = evaluate_baseline(
            data, horizon, fn, initial_train_end=config.INITIAL_TRAIN_END
        )
        baseline_rows.append(
            [
                bname,
                horizon,
                float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                float(np.mean(np.abs(y_true - y_pred))),
                len(y_true),
            ]
        )
baseline_df = pd.DataFrame(
    baseline_rows, columns=["model", "horizon", "rmse", "mae", "n_test"]
)
baseline_df.to_csv(config.OUTPUT_FILES["baselines"], index=False)
print(baseline_df.round(4).to_string(index=False))

# %% [markdown]
# ## 5.4 Statistical models (Tables 4.1 / 4.2 / 4.6, part 1)
#
# Persistence, ARIMA, SARIMA, ARIMAX and Prophet. This cell is the slowest of
# the statistical block because Prophet runs an internal 1-year-ahead
# changepoint-selection CV for every training window.

# %%
for name, factory in [
    ("Persistence", persistence_factory),
    ("ARIMA", arima_factory),
    ("SARIMA", sarima_factory),
    ("ARIMAX", arimax_factory),
    ("Prophet", prophet_factory),
]:
    s, d = run_cv(name, factory)
    all_summaries.append(s)
    all_details.append(d)

# %% [markdown]
# ## 5.5 Machine-learning models (Tables 4.1 / 4.2 / 4.6, part 2)
#
# Hyperparameters are tuned **once** on the full series using
# TimeSeriesSplit(5). The hybrid fits an ARIMA (AICc, d ∈ {0,1}) and then
# models the residuals with XGBoost.

# %%
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
import xgboost as xgb  # noqa: E402

X_full = data[config.COVARIATE_COLS].values
y_full = data["yield_t_ha"].values
tscv = TimeSeriesSplit(n_splits=config.TSCV_N_SPLITS)

print("Tuning RandomForest...")
rf_params = tune_rf(X_full, y_full, tscv)
print(f"  RF params: {rf_params}")
print("Tuning XGBoost...")
xgb_params = tune_xgb(X_full, y_full, tscv)
print(f"  XGB params: {xgb_params}")

# %%
for name, factory in [
    ("RandomForest", make_rf_factory(rf_params)),
    ("XGBoost", make_xgb_factory(xgb_params)),
    ("ARIMA+XGBoost", make_hybrid_factory(xgb_params)),
]:
    s, d = run_cv(name, factory)
    all_summaries.append(s)
    all_details.append(d)

# %% [markdown]
# ## 5.6 Results — RMSE comparison (Tables 4.1, 4.2, 4.6)
#
# All eight models, all four horizons, plus the per-fold detail file.

# %%
comparison = pd.concat(all_summaries, ignore_index=True)
details = pd.concat(all_details, ignore_index=True)
comparison.to_csv(config.OUTPUT_FILES["comparison"], index=False)
details.to_csv(config.OUTPUT_FILES["details"], index=False)
print(
    f"Saved {config.OUTPUT_FILES['comparison'].name} ({len(comparison)} rows) and "
    f"{config.OUTPUT_FILES['details'].name} ({len(details)} rows)"
)

pivot = comparison.pivot_table(index="model", columns="horizon", values="rmse")
model_order = [
    "Persistence",
    "ARIMA",
    "SARIMA",
    "ARIMAX",
    "Prophet",
    "RandomForest",
    "XGBoost",
    "ARIMA+XGBoost",
]
pivot = pivot.reindex(model_order)
print("\n=== RMSE by model and horizon (t/ha) ===")
print(pivot.round(4).to_string())

# %%
fig, ax = plt.subplots(figsize=(10.5, 4.6))
width = 0.09
colors = [
    "#7f7f7f",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#d62728",
]
x = np.arange(len(model_order))
for i, h in enumerate(config.HORIZONS):
    vals = pivot[h].values
    ax.bar(x + (i - 1.5) * width, vals, width, label=f"h = {h}", color=colors[i])
ax.set_xticks(x, model_order, rotation=20)
ax.set_ylabel("RMSE (t/ha)")
ax.set_title("Forecast accuracy by model and horizon (expanding-window CV)")
ax.legend(ncol=4)
fig.tight_layout()
plt.show()

# %%
print("RMSE by model and horizon, ranked (lower is better):")
pivot.rank(ascending=True).astype(int).round(0)

# %% [markdown]
# ## 5.7 Diebold–Mariano tests (Table 4.4)
#
# Pairwise DM tests with MSE loss and the HLN small-sample correction across
# the seven forecast models (Persistence excluded). A **negative** statistic
# favours the model listed first.

# %%
dm_no_baseline = details[details["model"] != "Persistence"]
dm_results = []
for h in config.HORIZONS:
    models = sorted(dm_no_baseline["model"].unique())
    for i, m1 in enumerate(models):
        for m2 in models[i + 1 :]:
            h1 = dm_no_baseline[
                (dm_no_baseline["model"] == m1) & (dm_no_baseline["horizon"] == h)
            ].sort_values("test_year")
            h2 = dm_no_baseline[
                (dm_no_baseline["model"] == m2) & (dm_no_baseline["horizon"] == h)
            ].sort_values("test_year")
            common = pd.merge(
                h1[["test_year", "y_true", "y_pred"]],
                h2[["test_year", "y_true", "y_pred"]],
                on="test_year",
                suffixes=("_1", "_2"),
            )
            if len(common) < 2:
                continue
            dm_stat, p_val = diebold_mariano(
                common["y_true_1"].values,
                common["y_pred_1"].values,
                common["y_pred_2"].values,
                loss="MSE",
                h=h,
            )
            e1 = common["y_true_1"].values - common["y_pred_1"].values
            e2 = common["y_true_1"].values - common["y_pred_2"].values
            dm_results.append(
                {
                    "horizon": h,
                    "model_1": m1,
                    "model_2": m2,
                    "loss": "MSE",
                    "dm_statistic": round(dm_stat, 4),
                    "p_value": round(p_val, 4),
                    "significant_005": bool(p_val < 0.05),
                    "significant_001": bool(p_val < 0.01),
                    "n_common": len(common),
                    "rmse_1": round(float(np.sqrt(np.mean(e1**2))), 4),
                    "rmse_2": round(float(np.sqrt(np.mean(e2**2))), 4),
                    "mae_1": round(float(np.mean(np.abs(e1))), 4),
                    "mae_2": round(float(np.mean(np.abs(e2))), 4),
                }
            )
dm_df = pd.DataFrame(dm_results)
dm_df.to_csv(config.OUTPUT_FILES["dm_tests"], index=False)
print(f"Saved {config.OUTPUT_FILES['dm_tests'].name} ({len(dm_df)} rows)")
for h in config.HORIZONS:
    sub = dm_df[dm_df["horizon"] == h]
    print(
        f"  h={h}: {len(sub)} pairs, significant_005={sub['significant_005'].sum()}, "
        f"significant_001={sub['significant_001'].sum()}"
    )

# %% [markdown]
# ### DM heatmap (Figure: significance at h = 1)
#
# The heatmap shows the DM statistic for every ordered pair at h = 1. Blue
# cells mean the row model outperforms the column model (negative statistic).

# %%
dm_h1 = dm_df[dm_df["horizon"] == 1]
models_h1 = sorted(dm_h1["model_1"].unique())
stat_mat = pd.DataFrame(0.0, index=models_h1, columns=models_h1)
for _, r in dm_h1.iterrows():
    stat_mat.loc[r["model_1"], r["model_2"]] = r["dm_statistic"]
    stat_mat.loc[r["model_2"], r["model_1"]] = -r["dm_statistic"]

fig, ax = plt.subplots(figsize=(7, 5.6))
im = ax.imshow(stat_mat.values, cmap="RdBu_r", vmin=-5, vmax=5)
ax.set_xticks(range(len(models_h1)), models_h1, rotation=40, ha="right")
ax.set_yticks(range(len(models_h1)), models_h1)
ax.set_title("DM statistics at h = 1 (negative favours row model)")
for i in range(len(models_h1)):
    for j in range(len(models_h1)):
        if i != j:
            ax.text(
                j,
                i,
                f"{stat_mat.values[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
fig.colorbar(im, shrink=0.8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5.8 Prediction-interval coverage (Table 4.5, Figure 4.3)
#
# 95% prediction intervals for ARIMA and Prophet under the corrected
# specifications (`interval_width = 0.95`, exogenous covariates forecast with
# ARIMA(1,0,0)). Coverage reports the fraction of out-of-sample years whose
# true yield fell inside the interval.
#
# Both predictors come straight from `src/models.py` — the same AICc selection
# and Prophet changepoint selection as the main comparison, plus a
# `predict_interval(h)` method — so this section can never drift from the
# models it measures.


# %%
def compute_arima_pi(data):
    rows = []
    for train_end in range(config.INITIAL_TRAIN_END, int(data["year"].max())):
        train_df = data[data["year"] <= train_end]
        predictor = arima_factory(train_df, None)
        for h in config.HORIZONS:
            test_year = train_end + h
            if test_year > data["year"].max():
                continue
            try:
                y_pred, lo, hi = predictor.predict_interval(h)
            except Exception:
                continue
            test_row = data[data["year"] == test_year]
            if len(test_row) == 0:
                continue
            y_true = test_row["yield_t_ha"].values[0]
            rows.append(
                {
                    "model": "ARIMA",
                    "horizon": h,
                    "test_year": int(test_year),
                    "y_true": float(y_true),
                    "y_pred": y_pred,
                    "pi_lower": lo,
                    "pi_upper": hi,
                    "pi_width": hi - lo,
                    "covered": bool(lo <= y_true <= hi),
                }
            )
    return pd.DataFrame(rows)


def compute_prophet_pi(data):
    rows = []
    for train_end in range(config.INITIAL_TRAIN_END, int(data["year"].max())):
        train_df = data[data["year"] <= train_end]
        # Prophet's interval sampling draws from the global RNG; seeding per
        # origin keeps the coverage experiment reproducible (same as the
        # corrected Colab pipeline).
        np.random.seed(config.SEED)
        predictor = prophet_factory(train_df, None)
        for h in config.HORIZONS:
            test_year = train_end + h
            if test_year > data["year"].max():
                continue
            try:
                y_pred, lo, hi = predictor.predict_interval(h)
            except Exception:
                continue
            test_row = data[data["year"] == test_year]
            if len(test_row) == 0:
                continue
            y_true = test_row["yield_t_ha"].values[0]
            rows.append(
                {
                    "model": "Prophet",
                    "horizon": h,
                    "test_year": int(test_year),
                    "y_true": float(y_true),
                    "y_pred": y_pred,
                    "pi_lower": lo,
                    "pi_upper": hi,
                    "pi_width": hi - lo,
                    "covered": bool(lo <= y_true <= hi),
                }
            )
    return pd.DataFrame(rows)


print("Computing ARIMA prediction intervals...")
arima_pi = compute_arima_pi(data)
print(f"  ARIMA rows: {len(arima_pi)}")
print("Computing Prophet prediction intervals...")
prophet_pi = compute_prophet_pi(data)
print(f"  Prophet rows: {len(prophet_pi)}")

# %%
pi_details = pd.concat([arima_pi, prophet_pi], ignore_index=True)
pi_summary = []
for model in ["ARIMA", "Prophet"]:
    mdf = pi_details[pi_details["model"] == model]
    for h in config.HORIZONS:
        sub = mdf[mdf["horizon"] == h]
        pi_summary.append(
            {
                "model": model,
                "horizon": h,
                "pi_coverage_95": f"{sub['covered'].mean():.1%}",
                "avg_pi_width": f"{sub['pi_width'].mean():.3f}",
                "n_test": len(sub),
            }
        )
pi_summary_df = pd.DataFrame(pi_summary)
pi_details.to_csv(config.OUTPUT_FILES["pi_details"], index=False)
pi_summary_df.to_csv(config.OUTPUT_FILES["pi_coverage"], index=False)
print("\nCoverage + width summary:")
print(pi_summary_df.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(8.5, 4))
for i, model in enumerate(["ARIMA", "Prophet"]):
    sub = pi_summary_df[pi_summary_df["model"] == model]
    cov = [float(v.strip("%")) / 100 for v in sub["pi_coverage_95"]]
    ax.plot(config.HORIZONS, cov, marker="o", label=model)
ax.axhline(0.95, color="green", linestyle="--", label="nominal 95%")
ax.set_ylim(0.5, 1.0)
ax.set_xticks(config.HORIZONS)
ax.set_xlabel("Horizon (years)")
ax.set_ylabel("Coverage rate")
ax.set_title("95% prediction-interval coverage by horizon")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5.9 Oracle exogenous experiment (Table 4.7)
#
# What if the models had **perfect knowledge** of the future weather instead of
# the ARIMA(1,0,0) projections? We re-run the covariate-driven models feeding
# the true covariate values (`oracle`) and report the RMSE improvement over the
# standard forecasts. ARIMA needs no exogenous inputs, so its oracle rows carry
# zero improvement (shown for context).
#
# Instead of re-implementing the models, the factories accept an ``oracle_fc``
# callable that replaces `features.forecast_exogenous` — so the oracle run uses
# the exact same model code as the main comparison.

# %%
from functools import partial


def oracle_forecast_factory(data):
    """Build an exogenous forecaster that returns the TRUE future values."""

    def oracle_fc(train_df, cov_cols, horizon):
        fc = {}
        origin = int(train_df["year"].max())
        for col in cov_cols:
            vals = []
            for k in range(1, horizon + 1):
                yr = origin + k
                row = data[data["year"] == yr]
                vals.append(
                    row[col].values[0] if len(row) > 0 else float(train_df[col].mean())
                )
            fc[col] = np.array(vals)
        return fc

    return oracle_fc


oracle_fc = oracle_forecast_factory(data)

oracle_summaries = []
for name, factory in [
    ("ARIMAX", partial(arimax_factory, oracle_fc=oracle_fc)),
    ("Prophet", partial(prophet_factory, oracle_fc=oracle_fc)),
    ("RandomForest", partial(make_rf_factory(rf_params), oracle_fc=oracle_fc)),
    ("XGBoost", partial(make_xgb_factory(xgb_params), oracle_fc=oracle_fc)),
    ("ARIMA+XGBoost", partial(make_hybrid_factory(xgb_params), oracle_fc=oracle_fc)),
]:
    summary, _ = run_cv(name, factory)
    oracle_summaries.append(summary)

oracle_combined = pd.concat(oracle_summaries, ignore_index=True).rename(
    columns={"rmse": "oracle_rmse"}
)[["model", "horizon", "oracle_rmse", "n_test"]]

# %%
standard = pd.read_csv(config.OUTPUT_FILES["comparison"])
standard = standard[standard["model"] != "Persistence"]
merged = standard.merge(
    oracle_combined[["model", "horizon", "oracle_rmse"]],
    on=["model", "horizon"],
    how="left",
)
merged["improvement_pct"] = (
    (merged["rmse"] - merged["oracle_rmse"]) / merged["rmse"] * 100
).round(1)
merged = merged.rename(columns={"rmse": "standard_rmse"})

for h in [2, 3, 4]:  # ARIMA needs no exog -> add context rows with zero improvement
    row = standard[(standard["model"] == "ARIMA") & (standard["horizon"] == h)]
    if len(row) > 0:
        r = row.iloc[0]
        merged = pd.concat(
            [
                merged,
                pd.DataFrame(
                    [
                        {
                            "model": "ARIMA",
                            "horizon": h,
                            "standard_rmse": r["rmse"],
                            "oracle_rmse": r["rmse"],
                            "improvement_pct": 0.0,
                            "n_test": r["n_test"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

merged = merged.sort_values(["model", "horizon"]).reset_index(drop=True)
cols = ["horizon", "model", "standard_rmse", "oracle_rmse", "improvement_pct", "n_test"]
merged = merged.dropna(subset=["oracle_rmse"]).reset_index(drop=True)
merged[cols].to_csv(config.OUTPUT_FILES["oracle"], index=False)
print(f"\nSaved {config.OUTPUT_FILES['oracle'].name} ({len(merged)} rows)")
print(merged[cols].round(4).to_string(index=False))

# %% [markdown]
# ## 5.10 Verification against the canonical thesis numbers
#
# Every output is compared with the values stored in `data/expected/` (extracted
# from the thesis Results chapter). Small tolerances allow for library-version
# drift; any "DIFF" means a thesis number was not reproduced.


# %%
def verify(fname, keys, vals, atol):
    e = pd.read_csv(config.EXPECTED_DIR / fname)
    p = pd.read_csv(config.OUTPUT_DIR / fname)
    key_e = set(map(tuple, e[keys].itertuples(index=False)))
    key_p = set(map(tuple, p[keys].itertuples(index=False)))
    if key_e != key_p:
        print(f"  {fname}: KEY MISMATCH (exp {len(key_e)} rows, got {len(key_p)})")
        return False
    m = e.merge(p, on=keys, suffixes=("_exp", "_got"))
    bad = []
    for v in vals:
        ec, gc = v + "_exp", v + "_got"
        if ec not in m or gc not in m:
            bad.append((v, "col missing"))
            continue
        col_exp, col_got = m[ec], m[gc]
        mask = ~(col_exp.isna() | col_got.isna())
        if pd.api.types.is_numeric_dtype(col_exp):
            diff = (col_exp[mask] - col_got[mask]).abs()
            n_bad = int((diff > atol).sum())
        else:
            n_bad = int((col_exp[mask] != col_got[mask]).sum())
        if n_bad:
            bad.append((v, n_bad))
    status = "OK" if not bad else f"DIFF {bad}"
    print(f"  {fname}: {status}")
    return not bad


all_ok = True
print("Verifying outputs against the canonical numbers in data/expected/...")

# F-4 guard: every model must be evaluated on the same test years per horizon.
# A model that silently failed to converge on some folds would drop test points
# and get an unfairly easy RMSE; this makes that visible instead of invisible.
assert_balanced_test_sets(details)
print("  test-year symmetry across models: OK")

all_ok &= verify(
    "model_comparison_results_corrected.csv",
    ["model", "horizon"],
    ["rmse", "mae", "n_test"],
    1e-4,
)
all_ok &= verify(
    "dm_test_results.csv",
    ["horizon", "model_1", "model_2"],
    ["dm_statistic", "p_value", "n_common", "rmse_1", "rmse_2", "mae_1", "mae_2"],
    1e-4,
)
all_ok &= verify(
    "baseline_results.csv", ["model", "horizon"], ["rmse", "mae", "n_test"], 1e-6
)
all_ok &= verify(
    "oracle_exogenous_results.csv",
    ["horizon", "model"],
    ["standard_rmse", "oracle_rmse", "improvement_pct", "n_test"],
    1e-4,
)
all_ok &= verify(
    "pi_detailed_results_corrected.csv",
    ["model", "horizon", "test_year"],
    ["y_true", "y_pred", "pi_lower", "pi_upper", "pi_width"],
    1e-3,
)

mc = pd.read_csv(config.EXPECTED_DIR / "pi_coverage_results_corrected.csv").merge(
    pd.read_csv(config.OUTPUT_DIR / "pi_coverage_results_corrected.csv"),
    on=["model", "horizon"],
    suffixes=("_exp", "_got"),
)
cov_ok = (
    (mc["pi_coverage_95_exp"] == mc["pi_coverage_95_got"]).all()
    and (mc["avg_pi_width_exp"] == mc["avg_pi_width_got"]).all()
    and (mc["n_test_exp"] == mc["n_test_got"]).all()
)
print("  pi_coverage_results_corrected.csv:", "OK" if cov_ok else "DIFF")
all_ok = all_ok and bool(cov_ok)

print()
if all_ok:
    print("ALL CHECKS PASSED - every canonical number was reproduced.")
else:
    print("SOME CHECKS FAILED - inspect the DIFF details above.")

# %% [markdown]
# ## 5.11 Practitioner decision guide (deliverable, FR-9)
#
# A compact, machine-generated summary for the practitioner. Every number below
# is read directly from the verified output CSVs — nothing is hand-written, so
# prose and data can never drift apart (NFR-10).

# %%
comparison_g = pd.read_csv(config.OUTPUT_FILES["comparison"])
dm_g = pd.read_csv(config.OUTPUT_FILES["dm_tests"])
pi_g = pd.read_csv(config.OUTPUT_FILES["pi_coverage"])
oracle_g = pd.read_csv(config.OUTPUT_FILES["oracle"])

lines = ["# UK wheat yield forecasting — practitioner decision guide"]
lines.append("")
lines.append(
    "Generated from the verified outputs of `scripts/05_Model.py` "
    "(see `data/expected/` for the verification gate)."
)
lines.append("")

overall = comparison_g.groupby("model")["rmse"].mean().sort_values()
lines.append(f"## Best overall model: {overall.index[0]}")
lines.append("")
lines.append("Mean RMSE across horizons 1-4 (t/ha):")
lines.append("")
lines.append("| Model | Mean RMSE |")
lines.append("|---|---|")
for model, v in overall.items():
    lines.append(f"| {model} | {v:.4f} |")
lines.append("")

lines.append("### Best model per horizon")
lines.append("")
lines.append("| Horizon | Model | RMSE (t/ha) |")
lines.append("|---|---|---|")
for h in config.HORIZONS:
    row = comparison_g[comparison_g["horizon"] == h].sort_values("rmse").iloc[0]
    lines.append(f"| h={h} | {row['model']} | {row['rmse']:.4f} |")
lines.append("")

wins = dm_g[dm_g["significant_005"]]
lines.append(
    f"## Statistically significant pairwise differences (DM, 5%): {len(wins)} pairs"
)
lines.append("")
lines.append("| Horizon | Pair | DM stat | p |")
lines.append("|---|---|---|---|")
for _, r in wins.sort_values("p_value").head(10).iterrows():
    lines.append(
        f"| h={r['horizon']} | {r['model_1']} vs {r['model_2']} "
        f"| {r['dm_statistic']:.2f} | {r['p_value']:.4f} |"
    )
lines.append("")

lines.append("## Prediction-interval coverage (nominal 95%)")
lines.append("")
lines.append("| Model | Horizon | Coverage | Avg width |")
lines.append("|---|---|---|---|")
for _, r in pi_g.iterrows():
    lines.append(
        f"| {r['model']} | h={r['horizon']} | {r['pi_coverage_95']} | {r['avg_pi_width']} |"
    )
lines.append("")

lines.append("## Ceiling: perfect weather foresight (oracle experiment)")
lines.append("")
best_oracle = oracle_g.loc[oracle_g["improvement_pct"].idxmax()]
lines.append(
    f"Largest benefit: {best_oracle['model']} at h={best_oracle['horizon']} "
    f"({best_oracle['improvement_pct']:.1f}% RMSE reduction with perfect "
    "weather instead of ARIMA(1,0,0) projections)."
)
lines.append("")
lines.append("| Model | Horizon | Std RMSE | Oracle RMSE | Impr. % |")
lines.append("|---|---|---|---|---|")
for _, r in oracle_g.sort_values("improvement_pct", ascending=False).head(6).iterrows():
    lines.append(
        f"| {r['model']} | h={r['horizon']} | {r['standard_rmse']:.4f} "
        f"| {r['oracle_rmse']:.4f} | {r['improvement_pct']:.1f} |"
    )

guide_path = config.OUTPUT_FILES["decision_guide"]
guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(
    f"Decision guide written to {guide_path.name} ({guide_path.stat().st_size:,} bytes)"
)

# %% [markdown]
# ## 5.12 Summary of saved outputs

# %%
print("Outputs written to data/outputs/:")
for key, path in config.OUTPUT_FILES.items():
    if path.exists():
        print(f"  {path.name:50s} {path.stat().st_size:>10,} bytes")

print("\nAll charts are rendered inline in this notebook (no PNG files are saved).")
