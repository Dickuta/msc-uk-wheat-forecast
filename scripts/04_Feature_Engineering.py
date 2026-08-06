# %% [markdown]
# # 04 · Feature Engineering
#
# **Goal.** Define and demonstrate every piece of feature construction the
# forecasting models rely on, so stage 05 is purely about fitting and
# comparing models.
#
# **Input.** `data/processed/uk_wheat_modelling_table_1980_2024.csv`.
#
# The corrected pipeline uses a deliberately small, interpretable feature set:
#
# | Feature | Used by | Construction |
# |---|---|---|
# | 8 weather covariates (raw) | ARIMAX, Prophet, RF, XGBoost, hybrid | raw values, **no transformation** |
# | 3 policy dummies | all covariate-driven models | one-hot events |
# | **Forecast weather** for year `Y+h` | ARIMAX, Prophet, RF, XGBoost, hybrid | each covariate forecast `h` steps ahead with **ARIMA(1,0,0)** on its own history |
# | ARIMAX covariate subset | ARIMAX | stepwise: drop highest p-value > 0.10, cap 5 covariates |
# | Residual target | hybrid | `y - ARIMA fitted values`, modelled by XGBoost |
#
# **Why forecast the weather?** To predict yield in year `Y+h` the model must
# know the weather of year `Y+h`, which is itself unknown at forecast time.
# The pipeline therefore *projects* every covariate forward with a simple
# ARIMA(1,0,0) — an honest, reproducible proxy for "what the weather is likely
# to be".
#
# **Scaling is not used** in the corrected results (`standardise=False` in the
# cross-validation): tree ensembles are scale-invariant and ARIMA-family
# models are fit on raw units.

# %% [markdown]
# ## 4.1 Setup

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

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
from src.features import forecast_exogenous

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

data = pd.read_csv(config.MODEL_TABLE_FILE)
data["year"] = data["year"].astype(int)
covariate_cols = config.COVARIATE_COLS
print(f"Loaded modelling table: {data.shape[0]} rows; {len(covariate_cols)} covariates")

# %% [markdown]
# ## 4.2 The model-ready feature matrix
#
# Stage 05 constructs, for every training window `[1980, origin]`, the feature
# matrix `X` = the covariate columns, and the target `y` = `yield_t_ha`.

# %%
X_demo = data[covariate_cols]
y_demo = data["yield_t_ha"]
feature_summary = pd.DataFrame(
    {
        "feature": covariate_cols,
        "dtype": X_demo.dtypes.astype(str).values,
        "mean": X_demo.mean().round(3).values,
        "std": X_demo.std().round(3).values,
        "min": X_demo.min().round(3).values,
        "max": X_demo.max().round(3).values,
        "n_missing": X_demo.isna().sum().values,
    }
)
feature_summary

# %% [markdown]
# ## 4.3 Exogenous forecasting — the core feature operation
#
# For a horizon `h`, every covariate is forecast `h` years ahead using
# `ARIMA(1,0,0)` fitted on the training window. Below we demonstrate this on
# `autumn_rain` (the covariate most correlated with yield) for `h = 1..4`,
# training on the 1980–2020 window.

# %%
train_demo = data[data["year"] <= 2020].reset_index(drop=True)
h = 4
fcst = forecast_exogenous(train_demo, ["autumn_rain"], h)["autumn_rain"]

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(
    train_demo["year"],
    train_demo["autumn_rain"],
    marker="o",
    markersize=3,
    linewidth=1.4,
    color="#1f77b4",
    label="autumn_rain (observed)",
)
future_years = np.arange(2021, 2021 + h)
ax.plot(
    future_years,
    fcst,
    marker="s",
    linewidth=1.6,
    color="#d62728",
    label="ARIMA(1,0,0) forecast",
)
ax.set_xlabel("Harvest year")
ax.set_ylabel("Autumn rainfall (mm)")
ax.set_title("Exogenous covariate forecasting with ARIMA(1,0,0)")
ax.legend()
fig.tight_layout()
plt.show()

# %%
pd.DataFrame({"year": future_years, "autumn_rain_forecast": fcst.round(2)})

# %% [markdown]
# ## 4.4 Forecast all covariates for the next 4 years
#
# This is exactly the table that an operational forecast for 2025–2028 would
# need. It is produced purely from data in this notebook — no external inputs.

# %%
all_fcst = forecast_exogenous(data, covariate_cols, 4)
forecast_table = pd.DataFrame(all_fcst)
forecast_table.index = [2025, 2026, 2027, 2028]
forecast_table.index.name = "year"
forecast_table.round(2)

# %% [markdown]
# ## 4.5 ARIMAX stepwise covariate selection
#
# The ARIMAX model prunes covariates by refitting an ARIMA(1,0,0) with all
# candidate covariates and iteratively removing the least significant one
# (largest p-value) while any p-value exceeds 0.10, keeping at most five
# covariates. Here is the selection for the full 1980–2024 sample.

# %%
from src.models import arimax_stepwise_selection

selected, steps = arimax_stepwise_selection(
    data,
    covariate_cols,
    alpha=config.ARIMAX_ALPHA,
    max_cov=config.ARIMAX_MAX_COVARIATES,
)
print(f"ARIMAX selected covariates on the full sample: {selected}")
print("p-values at each step (last step = final selection):")
steps[-1].round(4).to_frame("p-value")

# %% [markdown]
# **Note.** In the expanding-window protocol the selection runs *inside every
# training window*, so the chosen subset changes year to year. The full-sample
# result above is illustrative only — stage 05 reruns selection per origin.

# %% [markdown]
# ## 4.6 What stage 05 receives
#
# * **Feature matrix**: the 11 covariate columns (8 weather + 3 policy dummies).
# * **Exogenous forecasts**: ARIMA(1,0,0) projections for horizons 1–4
#   (used by ARIMAX, Prophet, RF, XGBoost and the hybrid).
# * **ARIMAX subset**: per-window stepwise selection (cap 5, p > 0.10).
# * **Hybrid target**: ARIMA residuals refit to the covariate matrix with
#   XGBoost.
#
# No data is hard-coded anywhere — every table and plot above is computed from
# `data/processed/uk_wheat_modelling_table_1980_2024.csv`.
