# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

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
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))

import matplotlib.pyplot as plt
import warnings

from src._bootstrap import init_script, common_imports

c = common_imports()
display = init_script()

from src import plotting  # noqa: F401  (inline in notebooks, Agg when headless)
from src.features import forecast_exogenous
from src.models import arimax_stepwise_selection

warnings.filterwarnings("ignore")


# %% [markdown]
# ## 4.2 The model-ready feature matrix
#
# Stage 05 constructs, for every training window `[1980, origin]`, the feature
# matrix `X` = the covariate columns, and the target `y` = `yield_t_ha`.


# %%
def load_data():
    """Load the modelling table and show a feature summary."""
    from src._bootstrap import load_modelling_table

    data = load_modelling_table()
    covariate_cols = c.config.COVARIATE_COLS
    print(
        f"Loaded modelling table: {data.shape[0]} rows; {len(covariate_cols)} covariates"
    )

    X_demo = data[covariate_cols]
    feature_summary = c.pd.DataFrame(
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
    display(feature_summary)
    return data, covariate_cols


# %% [markdown]
# ## 4.3 Exogenous forecasting — the core feature operation
#
# For a horizon `h`, every covariate is forecast `h` years ahead using
# `ARIMA(1,0,0)` fitted on the training window. Below we demonstrate this on
# `autumn_rain` (the covariate most correlated with yield) for `h = 1..4`,
# training on the 1980–2020 window.


# %%
def demo_exogenous_forecast(data):
    """Demonstrate exogenous forecasting for autumn_rain over a 4-year horizon."""
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
    future_years = c.np.arange(2021, 2021 + h)
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

    display(
        c.pd.DataFrame({"year": future_years, "autumn_rain_forecast": fcst.round(2)})
    )


# %% [markdown]
# ## 4.4 Forecast all covariates for the next 4 years
#
# This is exactly the table that an operational forecast for 2025–2028 would
# need. It is produced purely from data in this notebook — no external inputs.


# %%
def forecast_all(data, covariate_cols):
    """Forecast every covariate 4 years ahead and display the table."""
    all_fcst = forecast_exogenous(data, covariate_cols, 4)
    forecast_table = c.pd.DataFrame(all_fcst)
    forecast_table.index = [2025, 2026, 2027, 2028]
    forecast_table.index.name = "year"
    display(forecast_table.round(2))
    return forecast_table


# %% [markdown]
# ## 4.5 ARIMAX stepwise covariate selection
#
# The ARIMAX model prunes covariates by refitting an ARIMA(1,0,0) with all
# candidate covariates and iteratively removing the least significant one
# (largest p-value) while any p-value exceeds 0.10, keeping at most five
# covariates. Here is the selection for the full 1980–2024 sample.


# %%
def demo_stepwise(data, covariate_cols):
    """Demonstrate ARIMAX stepwise selection on the full sample."""
    selected, steps = arimax_stepwise_selection(
        data,
        covariate_cols,
        alpha=c.config.ARIMAX_ALPHA,
        max_cov=c.config.ARIMAX_MAX_COVARIATES,
    )
    print(f"ARIMAX selected covariates on the full sample: {selected}")
    print("p-values at each step (last step = final selection):")
    display(steps[-1].round(4).to_frame("p-value"))


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


# %%
def main():
    """Run the full stage 04: feature summaries, forecasts, and ARIMAX demo."""
    data, covariate_cols = load_data()
    demo_exogenous_forecast(data)
    forecast_all(data, covariate_cols)
    demo_stepwise(data, covariate_cols)


# %%
if __name__ == "__main__":
    main()
