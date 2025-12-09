"""Tests for collect_scan()."""

import polars as pl
from polars.testing import assert_series_equal
from polars_numba import collect_scan
from numba import float32
import numpy as np
import pytest

from .utils import measure_cpu_time


def add_columns(acc, *values):
    for v in values:
        acc += v
    return acc


def test_varying_number_of_columns():
    """
    Up to 9 columns can be processed.
    """
    df = pl.DataFrame({str(i): [10**i] for i in range(1, 10)})
    for num_cols in range(1, 10):
        assert collect_scan(
            df, 7, add_columns, pl.Int64, [str(col) for col in range(1, num_cols + 1)]
        ).to_list() == [7 + sum(10**j for j in range(1, num_cols + 1))]


def test_lazy_and_not_lazy():
    """
    Both lazy and non-lazy Dataframes can be processed.
    """
    df = pl.DataFrame({"a": [1, 2], "b": [30, 50]})
    for test_df in [df, df.lazy()]:
        assert collect_scan(
            test_df, 0.5, add_columns, pl.Float64, ["a", "b"]
        ).to_list() == [
            31.5,
            83.5,
        ]


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
    assert collect_scan(df, 0.5, add_columns, pl.Float64, ["a", "b"]).to_list() == [
        31.5,
        134.5,
    ]


def test_accumulator_type_casting():
    """
    If the accumulator type is different than the return type of the passed-in
    function, the correct type will be used for future calls.
    """

    def to_f32(x, a):
        result = float32(x + a)
        return result

    df = pl.DataFrame({"a": [1.5, 2.25]})
    # First input is explicitly an integer, but the result should be a float:
    result = collect_scan(df, np.int64(10), to_f32, pl.Float32, ["a"])
    assert_series_equal(result, pl.Series("scan", [11.5, 13.75], dtype=pl.Float32))

    # Result should be an int:
    result = collect_scan(df, np.int64(10), to_f32, pl.Int32, ["a"])
    assert_series_equal(result, pl.Series("scan", [11, 13], dtype=pl.Int32))


def test_generate_column_names():
    """
    If column names aren't given, use the argument names in the passed in
    function.
    """

    def operator(acc, b, a):
        return acc + 10 * b + a

    df = pl.DataFrame({"acc": [1], "a": [5], "b": [20]})
    assert collect_scan(df, 0.5, operator, pl.Float64).to_list() == [205.5]


def test_compiled_function_caching():
    """
    If collect_scan() is called with the same function and same types again, it
    uses a cached version of the Numba precompiled function.

    If the types change, the already-cached version is not used.
    """

    def multiply(acc, a):
        return acc * a

    # The first time we pass in a function with specific types, it needs to be
    # compiled so this will take some time.
    df = pl.DataFrame({"a": [3]}, schema={"a": pl.Int64()}).lazy()
    with measure_cpu_time() as elapsed:
        assert_series_equal(
            collect_scan(df, np.int64(2), multiply, pl.Int64),
            pl.Series("scan", [6], dtype=pl.Int64),
        )
    elapsed_with_compile = elapsed.time

    # The next few times we expect a cached version to be used, so it should be
    # much faster.
    rounds = 20
    with measure_cpu_time() as elapsed:
        for _ in range(rounds):
            assert_series_equal(
                collect_scan(df, np.uint64(4), multiply, pl.Int64),
                pl.Series("scan", [12], dtype=pl.Int64),
            )
    assert (elapsed.time / rounds) < (elapsed_with_compile / 10)

    # If we use a different type, it compiles a new version:
    df = pl.DataFrame({"a": [3]}, schema={"a": pl.UInt64()}).lazy()
    assert_series_equal(
        collect_scan(df, 0.5, multiply, pl.Float64),
        pl.Series("scan", [1.5], dtype=pl.Float64),
    )


CAPTURED_GLOBALS = 7000


def test_captured_variables_must_not_change():
    """
    If a function uses captured variables, they must not change across calls.
    """
    captured_var = 100

    def func(acc, a):
        return acc + a + captured_var + CAPTURED_GLOBALS

    df = pl.DataFrame({"a": [3]}, schema={"a": pl.UInt64()}).lazy()
    assert collect_scan(df, 20, func, pl.Int64).to_list() == [7123]

    # Change a local captured var:
    captured_var = 200
    with pytest.raises(RuntimeError, match="changed a captured variable"):
        collect_scan(df, 20, func, pl.Int64)

    # Restore local captured var:
    captured_var = 100
    assert collect_scan(df, 40, func, pl.Int64).to_list() == [7143]

    # Change global captured var:
    global CAPTURED_GLOBALS
    CAPTURED_GLOBALS = 6000
    with pytest.raises(RuntimeError, match="changed a captured variable"):
        collect_scan(df, 20, func, pl.Int64)


def test_two_dtype_variants():
    """
    Both instances and types can be used for return_dtype.
    """
    df = pl.DataFrame({"a": [1, 2]})
    for dtype in (pl.Int64(), pl.Int64):
        assert_series_equal(
            collect_scan(df, 0, lambda acc, a: acc + a, dtype),
            pl.Series("scan", [1, 3], pl.Int64),
        )
