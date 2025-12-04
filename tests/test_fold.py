"""Tests for collect_fold()."""

import polars as pl
from polars_numba import collect_fold


def test_varying_number_of_columns():
    """
    Up to 9 columns can be processed.
    """

    def add_columns(acc, *values):
        for v in values:
            acc += v
        return acc

    df = pl.DataFrame({str(i): 10**i for i in range(1, 10)})
    for num_cols in range(1, 10):
        assert collect_fold(
            df, 7, add_columns, [str(col) for col in range(1, num_cols + 1)]
        ) == 7 + sum(10**j for j in range(1, num_cols + 1))
