"""Aggregation of monthly weather to seasonal phenological windows.

Single source of truth for the seasonal alignment used by stage 01 (which
builds the UK-mean weather) and stage 02 (which rebuilds it for comparison
with the canonical weather). Keeping the logic in one place prevents the
two consumers from diverging.
"""

import numpy as np
import pandas as pd


def aggregate_seasonal(df, variable, season_windows, agg):
    """Aggregate monthly weather to phenological windows keyed by harvest year.

    For harvest year Y the windows are defined by ``season_windows`` (see
    ``config.SEASON_WINDOWS``): autumn = Oct-Nov(Y-1), winter = Dec(Y-1)-Feb(Y),
    spring = Mar-May(Y), grain fill = Jun-Aug(Y).

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy monthly series with columns ``year``, ``month`` and the variable.
    variable : str
        Column to aggregate (only used for naming the output columns).
    season_windows : dict
        Mapping season name -> (start_month, end_month, year_offset).
    agg : {"mean", "sum"}
        Reduction applied within each window; declared by the data source in
        ``config.MET_OFFICE_SOURCES``.

    Returns
    -------
    pandas.DataFrame
        One row per harvest year with columns ``year`` and ``<season>_<variable>``.
    """
    records = []
    for harvest_year in sorted(df["year"].unique()):
        row = {"year": int(harvest_year)}
        for season, (start, end, offset) in season_windows.items():
            if start <= end:
                vals = df[
                    (df["year"] == harvest_year + offset)
                    & df["month"].between(start, end)
                ][variable]
            else:  # window spans a calendar-year boundary (e.g. Dec(Y-1)-Feb(Y))
                dec = df[
                    (df["year"] == harvest_year + offset) & (df["month"] == start)
                ][variable]
                jan_feb = df[
                    (df["year"] == harvest_year) & df["month"].between(1, end)
                ][variable]
                vals = pd.concat([dec, jan_feb])
            if len(vals) == 0:
                row[f"{season}_{variable}"] = np.nan
            elif agg == "mean":
                row[f"{season}_{variable}"] = vals.mean()
            elif agg == "sum":
                row[f"{season}_{variable}"] = vals.sum()
            else:
                raise ValueError(f"unknown agg {agg!r} for variable {variable!r}")
        records.append(row)
    return pd.DataFrame(records)
