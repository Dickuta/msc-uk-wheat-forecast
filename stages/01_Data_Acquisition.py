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
# # 01 · Data Acquisition — acquire the raw data
#
# **Goal.** Download the raw inputs from their **public data sources** into
# `data/raw/`, record a provenance manifest, and aggregate the weather series
# to the four phenological windows used in the study. No data is hard-coded in
# this notebook — everything is pulled from disk by the downstream stages.
#
# | Stage | Reads from | Writes to |
# |---|---|---|
# | 01 Data Acquisition | public data sources (internet) | `data/raw/` + manifest |
# | 02 EDA | `data/processed/uk_wheat_modelling_table_1980_2024.csv` | figures |
# | 03 Modelling Table | `data/raw/` + canonical building blocks | modelling table |
# | 04 Feature Engineering | modelling table | model-ready inputs |
# | 05 Model | modelling table | result CSVs + figures |
#
# ## Data sources used in this study
#
# | Variable | Source | Licence |
# |---|---|---|
# | Weather (temperature, rainfall) | Met Office UK Climate Series — areal values from HadUK-Grid 1 km | Open Government Licence |
# | Wheat yield | DEFRA / USDA FAS national series (canonical file, see `data_sources.md`) | Open Government Licence |
# | Policy dummies (CAP reform, Ukraine) | Constructed from documented policy events | — |
#
# The Met Office series are **area-weighted UK means** derived from the HadUK-Grid
# 1 km dataset. Two files are downloaded:
#
# * `Tmean` — monthly, seasonal and annual **mean air temperature** for the UK (degC)
# * `Rainfall` — monthly, seasonal and annual **total rainfall** for the UK (mm)
#
# **Temporal alignment.** For harvest year `Y`, weather is aligned as:
# autumn = Oct–Nov of `Y−1`, winter = Dec(`Y−1`)–Feb(`Y`), spring = Mar–May(`Y`),
# grain fill = Jun–Aug(`Y`). This guarantees **no forward-looking information**
# leaks into a training window.

# %% [markdown]
# ## 1.1 Setup

# %%
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src._bootstrap import init_script
from src.logging_utils import (
    get_stage_logger,
    log_stage_start,
    log_stage_end,
    log_artifact,
    timed_block,
)

display = init_script(width=140, max_columns=40)

log = get_stage_logger(__name__, "01")
log_stage_start(
    log, "01", "Data Acquisition - download Met Office series, parse, aggregate"
)

import config


# %% [markdown]
# ## 1.2 Download the Met Office monthly series
#
# Each series is saved verbatim as downloaded (`.txt`), then parsed into a tidy
# long-format table (`year`, `month`, value) and also stored as CSV. A SHA-256
# checksum of the raw file is recorded so the provenance is auditable.
#
# Once a raw file exists on disk it is **frozen**: subsequent runs reuse it
# without hitting the network, so a dead or reformatted source URL can never
# destroy reproducibility after the first successful acquisition.


# %%
def download_met_series(var_name, info):
    """Download one Met Office series, save the raw text, and parse to long format.

    Returns a dict with parsed DataFrame, file paths and a sha256 checksum.
    """
    url = info["url"]
    raw_path = config.RAW_DIR / info["raw_file"]
    if raw_path.exists():
        payload = raw_path.read_bytes()
        log.info(
            "Using frozen raw copy %s (sha256=%s...) - skipping download of %s",
            raw_path.name,
            hashlib.sha256(payload).hexdigest()[:12],
            url,
        )
    else:
        log.info("Downloading %s from %s", var_name, url)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download {var_name} from {url}: {exc}. "
                f"Place a frozen copy at {raw_path} to bypass the network."
            ) from exc
        payload = resp.content
        raw_path.write_bytes(payload)

    sha = hashlib.sha256(payload).hexdigest()
    text = payload.decode("utf-8")
    lines = text.strip().split("\n")
    columns = [c.strip().lower() for c in lines[5].strip().split()]
    month_cols = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    month_map = {m: i + 1 for i, m in enumerate(month_cols)}

    records = []
    for line in lines[6:]:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for col_name, month_num in month_map.items():
            col_idx = columns.index(col_name)
            if col_idx >= len(parts):
                continue
            raw = parts[col_idx].strip()
            if raw == "---" or raw == "":
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            records.append({"year": year, "month": month_num, var_name: val})

    df = pd.DataFrame(records)
    csv_path = config.RAW_DIR / f"met_office_{var_name}_monthly.csv"
    df.to_csv(csv_path, index=False)
    log.info(
        "Parsed %d monthly records -> %s (sha256=%s...)",
        len(df),
        csv_path.name,
        sha[:12],
    )
    return {
        "variable": var_name,
        "df": df,
        "raw_path": raw_path,
        "csv_path": csv_path,
        "sha256": sha,
    }


# %% [markdown]
# ## 1.3 Provenance manifest
#
# A single `manifest.csv` records, for every raw file, the source URL, the
# download timestamp and the SHA-256 checksum. This is the answer to the
# question *"where did this number come from?"*.


# %%
def build_manifest(downloads):
    """Write the provenance manifest CSV and display it."""
    manifest_rows = []
    for var_name, dl in downloads.items():
        manifest_rows.append(
            {
                "variable": dl["variable"],
                "source": config.MET_OFFICE_SOURCES[var_name]["description"],
                "url": config.MET_OFFICE_SOURCES[var_name]["url"],
                "raw_file": dl["raw_path"].name,
                "parsed_file": dl["csv_path"].name,
                "downloaded_at_utc": datetime.fromtimestamp(
                    dl["raw_path"].stat().st_mtime, timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "sha256": dl["sha256"],
                "n_monthly_rows": len(dl["df"]),
                "status": "OK",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = config.RAW_DIR / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    display(manifest)
    return manifest


# %% [markdown]
# ## 1.4 Aggregate monthly weather to phenological windows
#
# Temperature is averaged within each window; rainfall is summed. The result is
# the **UK-mean** seasonal weather table that 03 uses to reproduce the weather
# building block from first principles.

# %%
# The seasonal alignment (autumn = Oct-Nov of Y-1, winter = Dec(Y-1)-Feb(Y),
# spring = Mar-May(Y), grain fill = Jun-Aug(Y)) lives in `src/weather.py` and
# is shared with stage 03, so the two can never drift apart.
from src.weather import aggregate_seasonal
from src.guards import assert_alignment, assert_alignment_spanning_year


def aggregate_seasonal_weather(downloads):
    """Aggregate monthly series to seasonal windows, spot-check, and save."""
    seasonal_dfs = []
    for var_name, dl in downloads.items():
        seasonal_dfs.append(
            aggregate_seasonal(
                dl["df"],
                var_name,
                config.SEASON_WINDOWS,
                config.MET_OFFICE_SOURCES[var_name]["agg"],
            )
        )
    seasonal_uk_mean = seasonal_dfs[0]
    for df in seasonal_dfs[1:]:
        seasonal_uk_mean = seasonal_uk_mean.merge(df, on="year", how="outer")
    seasonal_uk_mean = (
        seasonal_uk_mean[seasonal_uk_mean["year"].between(*config.MODEL_TABLE_YEARS)]
        .sort_values("year")
        .reset_index(drop=True)
    )

    # Spot-check seasonal alignment with the guards (FR-3 / F-1)
    ref_year = int(seasonal_uk_mean["year"].iloc[len(seasonal_uk_mean) // 2])

    # autumn = mean(Oct, Nov of Y-1) -> year_offset=-1, months=[10, 11]
    assert_alignment(
        downloads["tas"]["df"],
        seasonal_uk_mean.loc[seasonal_uk_mean["year"] == ref_year, "autumn_tas"].iloc[
            0
        ],
        ref_year,
        months=[10, 11],
        year_offset=-1,
        value_col="tas",
        agg="mean",
    )
    # spring = mean(Mar, Apr, May of Y) -> year_offset=0, months=[3, 4, 5]
    assert_alignment(
        downloads["tas"]["df"],
        seasonal_uk_mean.loc[seasonal_uk_mean["year"] == ref_year, "spring_tas"].iloc[
            0
        ],
        ref_year,
        months=[3, 4, 5],
        year_offset=0,
        value_col="tas",
        agg="mean",
    )
    # grainfill = mean(Jun, Jul, Aug of Y) -> year_offset=0, months=[6, 7, 8]
    assert_alignment(
        downloads["tas"]["df"],
        seasonal_uk_mean.loc[
            seasonal_uk_mean["year"] == ref_year, "grainfill_tas"
        ].iloc[0],
        ref_year,
        months=[6, 7, 8],
        year_offset=0,
        value_col="tas",
        agg="mean",
    )
    # winter = Dec(Y-1) + Jan-Feb(Y) -> boundary-spanning, use the spanning guard
    assert_alignment_spanning_year(
        downloads["tas"]["df"],
        seasonal_uk_mean.loc[seasonal_uk_mean["year"] == ref_year, "winter_tas"].iloc[
            0
        ],
        ref_year,
        first_month=12,
        second_year_months=[1, 2],
        value_col="tas",
        agg="mean",
    )
    print("Seasonal alignment spot-check: OK")

    seasonal_uk_mean.to_csv(config.WEATHER_SEASONAL_UK_MEAN_FILE, index=False)
    print(f"UK-mean seasonal weather: {len(seasonal_uk_mean)} rows")
    display(seasonal_uk_mean.head())
    return seasonal_uk_mean


# %% [markdown]
# ## 1.5 What the raw download provides vs. what the thesis numbers need
#
# Two weather building blocks exist for this study:
#
# 1. **UK-mean weather** — reproducible from the Met Office series downloaded
#    above (this notebook).
# 2. **Canonical weather** — the values actually used in the thesis modelling
#    table, which come from a **warmer regional extraction** (England-focused)
#    made before this pipeline was finalised. They cannot be reconstructed from
#    the public UK-mean series alone.
#
# The pipeline therefore treats the **canonical modelling table** as ground
# truth for the Results chapter (so every thesis number reproduces exactly) and
# uses the downloaded UK-mean series to *demonstrate* the end-to-end
# reproducibility path. Stage 03 quantifies the difference between the two.
#
# ## 1.6 Summary


# %%
def main():
    """Run the full stage 01: download, manifest, aggregate, verify, summarise."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    downloads = {}
    for var_name, info in config.MET_OFFICE_SOURCES.items():
        with timed_block(log, f"download_{var_name}", variable=var_name):
            downloads[var_name] = download_met_series(var_name, info)

    with timed_block(log, "build_manifest", n_vars=len(downloads)):
        build_manifest(downloads)

    with timed_block(log, "aggregate_seasonal", ref_year=None):
        aggregate_seasonal_weather(downloads)

    log_stage_end(log, "01", success=True)
    log.info("Raw data now on disk", extra={"stage": "01"})
    for path in sorted(config.RAW_DIR.iterdir()):
        log_artifact(log, path, "raw file")
    for path in sorted(config.PROCESSED_DIR.iterdir()):
        log_artifact(log, path, "processed file")


# %%
if __name__ == "__main__":
    main()
