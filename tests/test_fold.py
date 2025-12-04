"""Tests for collect_fold()."""

import polars as pl
from polars_numba import collect_fold


def add_columns(acc, *values):
    for v in values:
        acc += v
    return acc


def test_varying_number_of_columns():
    """
    Up to 9 columns can be processed.
    """
    df = pl.DataFrame({str(i): 10**i for i in range(1, 10)})
    for num_cols in range(1, 10):
        assert collect_fold(
            df, 7, add_columns, [str(col) for col in range(1, num_cols + 1)]
        ) == 7 + sum(10**j for j in range(1, num_cols + 1))


def test_lazy_and_not_lazy():
    """
    Both lazy and non-lazy Dataframes can be processed.
    """
    df = pl.DataFrame({"a": [1, 2], "b": [30, 50]})
    for test_df in [df, df.lazy()]:
        assert collect_fold(test_df, 0.5, add_columns, ["a", "b"]) == 83.5


def test_nulls_filtered_out():
    """
    Rows with nulls are filtered out, but only if the columns we care about
    have nulls!
    """
    df = pl.DataFrame(
        {
            "a": [1, 2, None, 3],
            "b": [30, None, 50, 100],
            "irrelevant": [9000, None, None, None],
        }
    )
    assert collect_fold(df, 0.5, add_columns, ["a", "b"]) == 0.5 + 1 + 30 + 3 + 100
