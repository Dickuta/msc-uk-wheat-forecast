import numpy as np
import pandas as pd
import pytest

from src.guards import (
    LeakageError,
    AlignmentError,
    assert_training_precedes_origin,
    assert_horizon_consistent,
    select_exog_pvalues,
    assert_alignment,
    assert_alignment_spanning_year,
    assert_balanced_test_sets,
    warn_on_swallowed_fits,
)


# --------------------------------------------------------------------------- #
# NFR-1 · Leakage-safety
# --------------------------------------------------------------------------- #
def test_assert_training_precedes_origin_pass():
    assert_training_precedes_origin([1980, 2000], 2000)
    assert_training_precedes_origin([2000], 2000)


def test_assert_training_precedes_origin_fail():
    with pytest.raises(LeakageError, match="leaked into the fit"):
        assert_training_precedes_origin([1980, 2001], 2000)


def test_assert_horizon_consistent_pass():
    assert_horizon_consistent(2000, 2001, 1)
    assert_horizon_consistent(2000, 2004, 4)


def test_assert_horizon_consistent_fail():
    with pytest.raises(LeakageError, match="horizon mismatch"):
        assert_horizon_consistent(2000, 2002, 1)


# --------------------------------------------------------------------------- #
# NFR-9 · Address by identity, not position
# --------------------------------------------------------------------------- #
def test_select_exog_pvalues_pass():
    pvalues = [0.01, 0.05, 0.10, 0.20]
    param_names = ["const", "x1", "x2", "ar.L1"]
    selected = ["x1", "x2"]
    out = select_exog_pvalues(pvalues, param_names, selected)
    assert out.tolist() == [0.05, 0.10]


def test_select_exog_pvalues_missing_raises():
    pvalues = [0.01, 0.05, 0.10]
    param_names = ["const", "x1", "x2"]
    selected = ["x1", "x3"]
    with pytest.raises(AssertionError, match="refusing to read a positional slice"):
        select_exog_pvalues(pvalues, param_names, selected)


def test_select_exog_pvalues_wrong_order():
    """Order must follow `selected`, not param_names."""
    pvalues = [0.01, 0.05, 0.10]
    param_names = ["const", "x2", "x1"]  # x2 before x1
    selected = ["x1", "x2"]
    out = select_exog_pvalues(pvalues, param_names, selected)
    assert out.tolist() == [0.10, 0.05]


# --------------------------------------------------------------------------- #
# FR-3 · Seasonal alignment spot-check
# --------------------------------------------------------------------------- #
def test_assert_alignment_pass_mean():
    df = pd.DataFrame(
        {
            "year": [2010, 2010, 2011, 2011],
            "month": [10, 11, 10, 11],
            "tas": [10.0, 12.0, 11.0, 13.0],
        }
    )
    # harvest year 2011, autumn = Oct-Nov 2010 -> mean = 11.0
    assert_alignment(
        df, 11.0, 2011, months=[10, 11], year_offset=-1, value_col="tas", agg="mean"
    )


def test_assert_alignment_pass_sum():
    df = pd.DataFrame(
        {
            "year": [2010, 2010, 2011],
            "month": [10, 11, 10],
            "rain": [100.0, 200.0, 50.0],
        }
    )
    assert_alignment(
        df, 300.0, 2011, months=[10, 11], year_offset=-1, value_col="rain", agg="sum"
    )


def test_assert_alignment_fail():
    df = pd.DataFrame({"year": [2010, 2010], "month": [10, 11], "tas": [10.0, 12.0]})
    with pytest.raises(AlignmentError, match="alignment mismatch"):
        assert_alignment(
            df, 99.0, 2011, months=[10, 11], year_offset=-1, value_col="tas", agg="mean"
        )


def test_assert_alignment_missing_rows():
    df = pd.DataFrame({"year": [2010], "month": [1], "tas": [5.0]})
    with pytest.raises(AlignmentError, match="no monthly rows"):
        assert_alignment(
            df, 10.0, 2011, months=[10, 11], year_offset=-1, value_col="tas", agg="mean"
        )


# --------------------------------------------------------------------------- #
# FR-3 · Seasonal alignment spot-check (boundary-spanning variant)
# --------------------------------------------------------------------------- #
def test_assert_alignment_spanning_year_pass():
    df = pd.DataFrame(
        {"year": [2010, 2010, 2011], "month": [12, 1, 2], "tas": [3.0, 5.0, 7.0]}
    )
    # winter of harvest 2011 = Dec(2010) + Jan-Feb(2011) -> mean = 5.0
    assert_alignment_spanning_year(
        df,
        5.0,
        2011,
        first_month=12,
        second_year_months=[1, 2],
        value_col="tas",
        agg="mean",
    )


def test_assert_alignment_spanning_year_fail():
    df = pd.DataFrame(
        {"year": [2010, 2010, 2011], "month": [12, 1, 2], "tas": [3.0, 5.0, 7.0]}
    )
    with pytest.raises(AlignmentError, match="alignment mismatch"):
        assert_alignment_spanning_year(
            df,
            99.0,
            2011,
            first_month=12,
            second_year_months=[1, 2],
            value_col="tas",
            agg="mean",
        )


# --------------------------------------------------------------------------- #
# NFR-2 · Protocol symmetry
# --------------------------------------------------------------------------- #
def test_assert_balanced_test_sets_pass():
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B"],
            "horizon": [1, 1, 1, 1],
            "test_year": [2001, 2002, 2001, 2002],
        }
    )
    assert_balanced_test_sets(df)


def test_assert_balanced_test_sets_fail():
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B"],
            "horizon": [1, 1, 1],
            "test_year": [2001, 2002, 2001],  # B missing 2002
        }
    )
    with pytest.raises(AssertionError, match="unbalanced test sets"):
        assert_balanced_test_sets(df)


def test_assert_balanced_test_sets_multiple_horizons():
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "horizon": [1, 1, 1, 1, 2, 2, 2, 2],
            "test_year": [2001, 2002, 2001, 2002, 2002, 2003, 2002, 2003],
        }
    )
    assert_balanced_test_sets(df)


def test_warn_on_swallowed_fits_emits_warning():
    with pytest.warns(UserWarning, match="swallowed"):
        warn_on_swallowed_fits(10, 7, "test_context")


def test_warn_on_swallowed_fits_no_warning_when_equal():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_swallowed_fits(10, 10, "test_context")
        assert len(w) == 0


def test_warn_on_swallowed_fits_no_warning_when_more_succeeded():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_swallowed_fits(10, 12, "test_context")
        assert len(w) == 0
