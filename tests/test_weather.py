import numpy as np
import pandas as pd
import pytest

from src.weather import aggregate_seasonal

SEASON_WINDOWS = {
    "autumn": (10, 11, -1),
    "winter": (12, 2, -1),
    "spring": (3, 5, 0),
    "grainfill": (6, 8, 0),
}


def _monthly_tidy():
    rows = []
    for year in (2010, 2011, 2012):
        for month in range(1, 13):
            rows.append({"year": year, "month": month, "tas": month})
    return pd.DataFrame(rows)


def test_aggregate_seasonal_mean_windows():
    df = _monthly_tidy()
    out = aggregate_seasonal(df, "tas", SEASON_WINDOWS, "mean")

    assert out["year"].tolist() == [2010, 2011, 2012]

    row_2011 = out[out["year"] == 2011].iloc[0]
    # autumn = Oct-Nov 2010 -> mean(10, 11) = 10.5
    assert row_2011["autumn_tas"] == 10.5
    # winter = Dec 2010 + Jan/Feb 2011 -> mean(12, 1, 2) = 5.0
    assert row_2011["winter_tas"] == 5.0
    # spring = Mar-May 2011 -> mean(3, 4, 5) = 4.0
    assert row_2011["spring_tas"] == 4.0
    # grain fill = Jun-Aug 2011 -> mean(6, 7, 8) = 7.0
    assert row_2011["grainfill_tas"] == 7.0


def test_aggregate_seasonal_sum():
    df = _monthly_tidy()
    out = aggregate_seasonal(df, "tas", SEASON_WINDOWS, "sum")
    row_2011 = out[out["year"] == 2011].iloc[0]
    assert row_2011["autumn_tas"] == 21.0  # 10 + 11
    assert row_2011["winter_tas"] == 15.0  # 12 + 1 + 2


def test_aggregate_seasonal_unknown_agg():
    df = _monthly_tidy()
    with pytest.raises(ValueError):
        aggregate_seasonal(df, "tas", SEASON_WINDOWS, "median")


def test_aggregate_seasonal_missing_month_gives_nan():
    df = _monthly_tidy()
    df = df[~((df["year"] == 2010) & (df["month"].isin([10, 11])))]
    out = aggregate_seasonal(df, "tas", SEASON_WINDOWS, "mean")
    assert np.isnan(out[out["year"] == 2011]["autumn_tas"].iloc[0])
