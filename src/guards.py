"""
Runtime invariant guards for the UK wheat yield forecasting pipeline.

Each function encodes ONE invariant that, if violated, produces a *silent-and-wrong*
result — the failure mode that actually threatens this project. The guards exist to
convert those silent failures into loud, specific ones, at the point of violation.

They are small, pure, and dependency-light (pandas/numpy only where a frame is the
natural input). Call them liberally; they are cheap relative to a wrong thesis number.

Mapping to the requirements (see REQUIREMENTS.md):
    assert_training_precedes_origin  -> NFR-1  (leakage-safety)      / F-2
    assert_horizon_consistent        -> NFR-1                         / F-2
    select_exog_pvalues              -> NFR-9  (address by name)      / F-1
    assert_alignment                 -> FR-3   (seasonal keying)      / F-1
    assert_balanced_test_sets        -> NFR-2  (protocol symmetry)    / F-4
    warn_on_swallowed_fits           -> NFR-2                         / F-4
"""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import pandas as pd


class LeakageError(AssertionError):
    """Raised when information from the test period reaches a training-time decision."""


class AlignmentError(AssertionError):
    """Raised when a seasonal value is not sourced from the expected months/year."""


# --------------------------------------------------------------------------- #
# NFR-1 · Leakage-safety
# --------------------------------------------------------------------------- #
def assert_training_precedes_origin(
    train_years: Iterable[int], origin_year: int
) -> None:
    """No training observation may fall on or after the forecast origin.

    Call inside every expanding-window fold, before fitting:

        assert_training_precedes_origin(train_df["year"], origin_year)
    """
    latest = max(train_years)
    if latest > origin_year:
        raise LeakageError(
            f"training data reaches year {latest}, which is after the forecast "
            f"origin {origin_year}: the future has leaked into the fit."
        )


def assert_horizon_consistent(origin_year: int, test_year: int, horizon: int) -> None:
    """The test year must be exactly `horizon` steps after the origin."""
    if test_year != origin_year + horizon:
        raise LeakageError(
            f"horizon mismatch: origin {origin_year} + h {horizon} should give "
            f"{origin_year + horizon}, but test_year is {test_year}."
        )


# --------------------------------------------------------------------------- #
# NFR-9 · Address by identity, not position  (prevents the pvalues[-k:] bug)
# --------------------------------------------------------------------------- #
def select_exog_pvalues(pvalues, param_names: Sequence[str], selected: Sequence[str]):
    """Return exog p-values selected BY NAME — never by tail position.

    Reproduces the corrected ARIMAX selection logic as a reusable guard. Refuses
    to guess: if a selected covariate is not among the fitted parameters, it
    raises rather than silently returning an AR term or sigma2.

    Returns a pandas Series aligned to `selected` (preserving the .max() interface).
    """
    lookup = dict(zip(param_names, pvalues))
    missing = [c for c in selected if c not in lookup]
    if missing:
        raise AssertionError(
            f"covariates {missing} are not in the fitted parameter names "
            f"{list(param_names)} — refusing to read a positional slice instead."
        )
    return pd.Series([lookup[c] for c in selected], index=list(selected))


# --------------------------------------------------------------------------- #
# FR-3 · Seasonal alignment spot-check
# --------------------------------------------------------------------------- #
def assert_alignment(
    monthly_df,
    seasonal_value: float,
    harvest_year: int,
    months: Sequence[int],
    year_offset: int,
    value_col: str,
    agg: str,
    year_col: str = "year",
    month_col: str = "month",
    atol: float = 1e-9,
) -> None:
    """Assert one seasonal cell was built from the intended months/year.

    Encodes the hand-computed check that proved the off-by-one fix, e.g. autumn of
    harvest year Y must equal the mean/sum of Oct+Nov of year Y-1:

        assert_alignment(monthly, seasonal.loc[Y, "autumn_tas"], Y,
                         months=[10, 11], year_offset=-1,
                         value_col="tas", agg="mean")

    Only covers windows that fall entirely within one calendar year; the
    boundary-spanning winter window (Dec(Y-1)-Feb(Y)) is out of scope here.
    Use ``assert_alignment_spanning_year`` for that case.
    """
    src_year = harvest_year + year_offset
    vals = monthly_df[
        (monthly_df[year_col] == src_year) & (monthly_df[month_col].isin(list(months)))
    ][value_col]
    if len(vals) == 0:
        raise AlignmentError(
            f"no monthly rows for year {src_year} months {list(months)} — "
            f"cannot verify {value_col} alignment for harvest year {harvest_year}."
        )
    expected = vals.mean() if agg == "mean" else vals.sum()
    if abs(expected - seasonal_value) > atol:
        raise AlignmentError(
            f"alignment mismatch for harvest year {harvest_year}: expected "
            f"{agg}({value_col}, {list(months)} of {src_year}) = {expected:.6f}, "
            f"got {seasonal_value:.6f}."
        )


def assert_alignment_spanning_year(
    monthly_df,
    seasonal_value: float,
    harvest_year: int,
    first_month: int,
    second_year_months: Sequence[int],
    value_col: str,
    agg: str,
    year_col: str = "year",
    month_col: str = "month",
    year_offset: int = -1,
    atol: float = 1e-9,
) -> None:
    """Assert a boundary-spanning seasonal window was built correctly.

    The winter window for harvest year Y spans the calendar-year boundary:
    Dec(Y-1) from ``year_offset`` year + months {first_month}, and Jan-Feb(Y)
    from ``harvest_year`` + ``second_year_months``.

    Example (config winter = (12, 2, -1)):
        assert_alignment_spanning_year(
            monthly, seasonal.loc[Y, "winter_tas"], Y,
            first_month=12, second_year_months=[1, 2],
            value_col="tas", agg="mean",
        )
    """
    prev_vals = monthly_df[
        (monthly_df[year_col] == harvest_year + year_offset)
        & (monthly_df[month_col] == first_month)
    ][value_col]
    later_vals = monthly_df[
        (monthly_df[year_col] == harvest_year)
        & (monthly_df[month_col].isin(list(second_year_months)))
    ][value_col]
    vals = pd.concat([prev_vals, later_vals])
    if len(vals) == 0:
        raise AlignmentError(
            f"no monthly rows for winter of harvest year {harvest_year}: "
            f"Dec({harvest_year + year_offset}) + Jan-Feb({harvest_year})."
        )
    expected = vals.mean() if agg == "mean" else vals.sum()
    if abs(expected - seasonal_value) > atol:
        raise AlignmentError(
            f"alignment mismatch for winter of harvest year {harvest_year}: "
            f"expected {agg} = {expected:.6f}, got {seasonal_value:.6f}."
        )


# --------------------------------------------------------------------------- #
# NFR-2 · Protocol symmetry  (catches the silent except:continue asymmetry)
# --------------------------------------------------------------------------- #
def assert_balanced_test_sets(
    details, model_col="model", horizon_col="horizon", test_col="test_year"
) -> None:
    """Every model must be scored on the SAME test years within each horizon.

    A model that silently fails on some windows (bare `except: continue`) ends up
    scored on a smaller, easier subset — an invisible apples-to-oranges. This
    raises with the specific offending (horizon, model) sets.
    """
    problems = []
    for h, grp in details.groupby(horizon_col):
        per_model = {m: frozenset(g[test_col]) for m, g in grp.groupby(model_col)}
        full = frozenset().union(*per_model.values()) if per_model else frozenset()
        for m, years in per_model.items():
            if years != full:
                problems.append((h, m, sorted(full - years)))
    if problems:
        lines = "; ".join(f"h={h} {m} missing {miss}" for h, m, miss in problems)
        raise AssertionError(f"unbalanced test sets across models: {lines}")


def warn_on_swallowed_fits(n_attempted: int, n_succeeded: int, context: str) -> None:
    """Emit a loud warning when a fit loop silently dropped windows."""
    if n_succeeded < n_attempted:
        warnings.warn(
            f"[{context}] {n_attempted - n_succeeded} of {n_attempted} fits were "
            f"swallowed — this model is scored on fewer points than its peers.",
            stacklevel=2,
        )
