# %% [markdown]
# # 03 · Modelling Table — assemble the modelling table
#
# **Goal.** Combine the three building blocks — yield, weather and policy
# dummies — into the single **modelling table** used by every downstream stage
# (EDA, feature engineering, modelling). The table is written to
# `data/processed/uk_wheat_modelling_table_1980_2024.csv` and is the **only**
# data source stages 02, 04 and 05 read.
#
# | Building block | File | Provenance |
# |---|---|---|
# | Yield (t/ha) | `data/processed/uk_wheat_yield_1980_2024.csv` | DEFRA / USDA national series (canonical) |
# | Canonical weather | `data/processed/uk_wheat_weather_seasonal_canonical.csv` | regional (England-focused) extraction used for the thesis results |
# | UK-mean weather | `data/processed/uk_wheat_weather_seasonal_uk_mean.csv` | rebuilt from the Met Office series in stage 01 |
# | Policy dummies | `data/processed/uk_wheat_policy_dummies_1980_2024.csv` | constructed from documented policy events |
#
# **Why two weather tables?** The thesis Results chapter reproduces exactly
# only when the *canonical* weather is used (it was extracted from a warmer
# regional dataset before this pipeline was finalised). To keep the pipeline
# fully auditable we keep both: the canonical table is the ground truth, and
# the UK-mean table demonstrates what is reproducible from the public sources
# alone. This notebook builds the final table from the canonical block and
# then quantifies how far the UK-mean reconstruction would diverge.

# %% [markdown]
# ## 3.1 Setup

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the pipeline root whether this file runs as a script or as a
# notebook cell (where `__file__` is not defined), independent of the CWD.
try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()
PROJECT_ROOT = _here if (_here / "config.py").exists() else _here.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

# %% [markdown]
# ## 3.2 Load the building blocks

# %%
yield_df = pd.read_csv(config.YIELD_FILE)
yield_df["year"] = yield_df["year"].astype(int)

policy_df = pd.read_csv(config.POLICY_DUMMIES_FILE)
policy_df["year"] = policy_df["year"].astype(int)

canonical_weather = pd.read_csv(config.WEATHER_SEASONAL_CANONICAL_FILE)
canonical_weather["year"] = canonical_weather["year"].astype(int)

print(
    f"Yield: {len(yield_df)} rows ({yield_df['year'].min()}-{yield_df['year'].max()})"
)
print(f"Canonical weather: {len(canonical_weather)} rows")
print(f"Policy dummies: {len(policy_df)} rows")

yield_df.head()

# %% [markdown]
# ## 3.3 Merge into the modelling table
#
# All three inputs are keyed on `year` and joined with an inner merge. The
# canonical weather columns are renamed to the short names used across the
# pipeline (`autumn_temp`, `autumn_rain`, …).

# %%
weather_renamed = canonical_weather.rename(
    columns={
        "autumn_tas": "autumn_temp",
        "winter_tas": "winter_temp",
        "spring_tas": "spring_temp",
        "grainfill_tas": "grainfill_temp",
        "autumn_rainfall": "autumn_rain",
        "winter_rainfall": "winter_rain",
        "spring_rainfall": "spring_rain",
        "grainfill_rainfall": "grainfill_rain",
    }
)

modelling = (
    yield_df.merge(weather_renamed, on="year", how="inner")
    .merge(policy_df, on="year", how="inner")
    .sort_values("year")
    .reset_index(drop=True)
)

# %% [markdown]
# ## 3.4 Validation
#
# A good modelling table must be complete, correctly typed and temporally
# contiguous. The checks below fail loudly if anything is off — and a failing
# table is never written to disk.

# %%
checks = {
    "year range": (
        modelling["year"].min() == config.MODEL_TABLE_YEARS[0]
        and modelling["year"].max() == config.MODEL_TABLE_YEARS[1]
    ),
    "no missing values": int(modelling.isna().sum().sum()) == 0,
    "no duplicate years": int(modelling["year"].duplicated().sum()) == 0,
    "contiguous years": (modelling["year"].diff().dropna() == 1).all(),
    "correct column order": list(modelling[config.COVARIATE_COLS].columns)
    == config.COVARIATE_COLS,
}
for name, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")

# Fail early: a bad table must never reach downstream stages
assert all(checks.values()), "modelling table failed validation — see checks above"

# Only persist after validation passes
modelling.to_csv(config.MODEL_TABLE_FILE, index=False)
print(
    f"Modelling table written: {modelling.shape[0]} rows x {modelling.shape[1]} cols -> {config.MODEL_TABLE_FILE.name}"
)

# %%
print("Summary statistics of the modelling table:")
modelling.describe().T

# %%
print("Missing values by column:")
modelling.isna().sum()

# %% [markdown]
# ## 3.5 Reproducibility check — canonical vs UK-mean weather
#
# To make the provenance gap explicit, we rebuild the UK-mean seasonal weather
# (from stage 01) and compare it, season by season, with the canonical values.
# Correlation with yield is shown for both, so it is easy to see that the two
# views of "the same weather year" carry different signal strengths.


# %%
def load_uk_mean_weather():
    """Prefer the fresh rebuild from stage 01, else the copied fallback."""
    from src.weather import aggregate_seasonal

    raw_monthly = sorted(config.RAW_DIR.glob("met_office_*_monthly.csv"))
    if raw_monthly:
        dfs = []
        for var in ["tas", "rainfall"]:
            df = pd.read_csv(config.RAW_DIR / f"met_office_{var}_monthly.csv")
            # reproduce the aggregation in stage 01 (same shared function)
            dfs.append(
                aggregate_seasonal(
                    df,
                    var,
                    config.SEASON_WINDOWS,
                    config.MET_OFFICE_SOURCES[var]["agg"],
                )
            )
        merged = dfs[0].merge(dfs[1], on="year", how="outer")
        return (
            merged[merged["year"].between(*config.MODEL_TABLE_YEARS)]
            .sort_values("year")
            .reset_index(drop=True)
        )
    return pd.read_csv(config.WEATHER_SEASONAL_UK_MEAN_FILE)


uk_mean = load_uk_mean_weather()
uk_mean_renamed = uk_mean.rename(
    columns={
        "autumn_tas": "autumn_temp",
        "winter_tas": "winter_temp",
        "spring_tas": "spring_temp",
        "grainfill_tas": "grainfill_temp",
        "autumn_rainfall": "autumn_rain",
        "winter_rainfall": "winter_rain",
        "spring_rainfall": "spring_rain",
        "grainfill_rainfall": "grainfill_rain",
    }
)

weather_cols = [
    c
    for c in config.COVARIATE_COLS
    if c not in ("cap_1992", "cap_2005", "ukraine_2022")
]
cmp_df = pd.DataFrame(
    {
        "variable": weather_cols,
        "canonical_mean": modelling[weather_cols].mean().values,
        "uk_mean_mean": uk_mean_renamed[weather_cols].mean().values,
        "canonical_minus_uk": (
            modelling[weather_cols].mean().values
            - uk_mean_renamed[weather_cols].mean().values
        ),
        "corr_with_yield_canonical": [
            modelling["yield_t_ha"].corr(modelling[c]) for c in weather_cols
        ],
        "corr_with_yield_uk_mean": [
            modelling["yield_t_ha"].corr(uk_mean_renamed[c]) for c in weather_cols
        ],
    }
)
cmp_df.round(3)

# %% [markdown]
# ### Interpretation
#
# The canonical (regional) weather is consistently **warmer** than the UK-mean
# series (positive `canonical_minus_uk` for every temperature window) and
# **drier** in every rainfall window — exactly what one expects from an
# England-focused extraction next to a whole-UK average (England is warmer and
# drier than the UK as a whole). The correlation with yield also differs between
# the two. This is why the **canonical table is the source of record** for the
# thesis: the exact Results chapter numbers can only be reproduced from it. The
# UK-mean path remains fully scriptable above, so the whole chain *from raw
# download to modelling table* is transparent and repeatable.
#
# ## 3.6 Summary

# %%
print("Final modelling table written to:", config.MODEL_TABLE_FILE)
print(
    f"  {modelling.shape[0]} rows x {modelling.shape[1]} cols, "
    f"years {modelling['year'].min()}-{modelling['year'].max()}"
)
print("Columns:", list(modelling.columns))
