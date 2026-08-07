# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,../notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

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
from src._bootstrap import init_script, common_imports
from src.logging_utils import (
    get_stage_logger,
    log_stage_start,
    log_stage_end,
    timed_block,
    log_artifact,
)

c = common_imports()
display = init_script(float_format=lambda v: f"{v:,.3f}")

log = get_stage_logger(__name__, "03")
log_stage_start(log, "03", "Modelling Table - assemble yield, weather, policy dummies")


# %% [markdown]
# ## 3.2 Load the building blocks


# %%
def load_inputs():
    """Load yield, canonical weather, and policy dummies; print summaries."""
    yield_df = c.pd.read_csv(c.config.YIELD_FILE)
    yield_df["year"] = yield_df["year"].astype(int)

    policy_df = c.pd.read_csv(c.config.POLICY_DUMMIES_FILE)
    policy_df["year"] = policy_df["year"].astype(int)

    canonical_weather = c.pd.read_csv(c.config.WEATHER_SEASONAL_CANONICAL_FILE)
    canonical_weather["year"] = canonical_weather["year"].astype(int)

    print(
        f"Yield: {len(yield_df)} rows ({yield_df['year'].min()}-{yield_df['year'].max()})"
    )
    print(f"Canonical weather: {len(canonical_weather)} rows")
    print(f"Policy dummies: {len(policy_df)} rows")
    display(yield_df.head())

    return (
        yield_df,
        canonical_weather.rename(columns=c.config.WEATHER_RENAME),
        policy_df,
    )


# %% [markdown]
# ## 3.3 Merge into the modelling table
#
# All three inputs are keyed on `year` and joined with an inner merge. The
# canonical weather columns are renamed to the short names used across the
# pipeline (`autumn_temp`, `autumn_rain`, …).


# %%
def merge_table(yield_df, weather_df, policy_df):
    """Inner-join the three building blocks on year, sorted chronologically."""
    return (
        yield_df.merge(weather_df, on="year", how="inner")
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
def validate(modelling):
    """Assert the modelling table passes all structural checks."""
    checks = {
        "year range": (
            modelling["year"].min() == c.config.MODEL_TABLE_YEARS[0]
            and modelling["year"].max() == c.config.MODEL_TABLE_YEARS[1]
        ),
        "no missing values": int(modelling.isna().sum().sum()) == 0,
        "no duplicate years": int(modelling["year"].duplicated().sum()) == 0,
        "contiguous years": (modelling["year"].diff().dropna() == 1).all(),
        "correct column order": list(modelling[c.config.COVARIATE_COLS].columns)
        == c.config.COVARIATE_COLS,
    }
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    # Fail early: a bad table must never reach downstream stages
    assert all(checks.values()), "modelling table failed validation — see checks above"
    return checks


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

    raw_monthly = sorted(c.config.RAW_DIR.glob("met_office_*_monthly.csv"))
    if raw_monthly:
        dfs = []
        for var in ["tas", "rainfall"]:
            df = c.pd.read_csv(c.config.RAW_DIR / f"met_office_{var}_monthly.csv")
            dfs.append(
                aggregate_seasonal(
                    df,
                    var,
                    c.config.SEASON_WINDOWS,
                    c.config.MET_OFFICE_SOURCES[var]["agg"],
                )
            )
        merged = dfs[0].merge(dfs[1], on="year", how="outer")
        return (
            merged[merged["year"].between(*c.config.MODEL_TABLE_YEARS)]
            .sort_values("year")
            .reset_index(drop=True)
        )
    return c.pd.read_csv(c.config.WEATHER_SEASONAL_UK_MEAN_FILE)


def compare_weather(modelling):
    """Quantify canonical-vs-UK-mean weather divergence."""
    uk_mean = load_uk_mean_weather().rename(columns=c.config.WEATHER_RENAME)
    weather_cols = [
        c
        for c in c.config.COVARIATE_COLS
        if c not in ("cap_1992", "cap_2005", "ukraine_2022")
    ]
    cmp_df = c.pd.DataFrame(
        {
            "variable": weather_cols,
            "canonical_mean": modelling[weather_cols].mean().values,
            "uk_mean_mean": uk_mean[weather_cols].mean().values,
            "canonical_minus_uk": (
                modelling[weather_cols].mean().values
                - uk_mean[weather_cols].mean().values
            ),
            "corr_with_yield_canonical": [
                modelling["yield_t_ha"].corr(modelling[c]) for c in weather_cols
            ],
            "corr_with_yield_uk_mean": [
                modelling["yield_t_ha"].corr(uk_mean[c]) for c in weather_cols
            ],
        }
    )
    display(cmp_df.round(3))


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
def main():
    """Run the full stage 03: load, merge, validate, persist, compare."""
    with timed_block(log, "load_inputs"):
        yield_df, weather_df, policy_df = load_inputs()

    with timed_block(log, "merge_table"):
        modelling = merge_table(yield_df, weather_df, policy_df)

    with timed_block(log, "validate"):
        validate(modelling)

    # Only persist after validation passes
    with timed_block(log, "write_modelling_table", path=str(c.config.MODEL_TABLE_FILE)):
        modelling.to_csv(c.config.MODEL_TABLE_FILE, index=False)

    log_artifact(log, c.config.MODEL_TABLE_FILE, "modelling table")
    log.info(
        "Modelling table written",
        extra={
            "stage": "03",
            "rows": modelling.shape[0],
            "cols": modelling.shape[1],
            "path": str(c.config.MODEL_TABLE_FILE),
        },
    )

    print("\nSummary statistics of the modelling table:")
    display(modelling.describe().T)

    print("Missing values by column:")
    display(modelling.isna().sum())

    with timed_block(log, "compare_weather"):
        compare_weather(modelling)

    log_stage_end(log, "03", success=True)


# %%
if __name__ == "__main__":
    main()
