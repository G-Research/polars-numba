"""Tests for plumba.scan."""

from typing import Sequence

import polars as pl
from polars.testing import assert_series_equal
import polars_numba  # noqa: F401
from numba import float32
import numpy as np
import pytest

from .utils import measure_cpu_time


def add_columns(acc, *values):
    for v in values:
        acc += v
    return acc


@pytest.mark.parametrize("num_cols", list(range(1, 10)))
@pytest.mark.parametrize("extra_args", [(), (0.5,), (0.25, 0.5)])
def test_varying_number_of_columns(num_cols: int, extra_args: Sequence[float]):
    """
    Up to 9 columns can be processed.
    """
    df = pl.DataFrame({str(i): [10.0**i] for i in range(1, num_cols + 1)})
    result = df.select(
        pl.struct(pl.all())
        .plumba.scan(add_columns, 7, pl.Int64, extra_args)
        .alias("result")
    )
    assert result.columns == ["result"]
    result["result"].to_list() == [
        7 + sum(10**j for j in range(1, num_cols + 1)) + sum(extra_args)
    ]


def test_nulls_kept_as_null():
    """
    Rows with nulls are not handed to the scan function, and instead converted
    to null in the result.  But only if the columns we care about have nulls!
    """
    df = pl.DataFrame(
        {
            "a": [1, 2, None, 3],
            "b": [30, None, 50, 100],
            "irrelevant": [9000, None, None, None],
        }
    )
    result = df.select(
        pl.struct(["a", "b"]).plumba.scan(add_columns, 0.5, pl.Float64).alias("scan")
    )["scan"].to_list()
    assert result == [
        31.5,
        None,
        None,
        134.5,
    ], result


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
    result = df.select(pl.col("a").plumba.scan(to_f32, np.int32(10), pl.Float32))["a"]
    assert_series_equal(result, pl.Series("a", [11.5, 13.75], dtype=pl.Float32))

    # Result should be an int:
    result = df.select(pl.col("a").plumba.scan(to_f32, np.int32(10), pl.Int32))["a"]
    assert_series_equal(result, pl.Series("a", [11, 13], dtype=pl.Int32))


def test_compiled_function_caching():
    """
    If plumba.scan is called with the same function and same types again, it
    uses a cached version of the Numba precompiled function.

    If the types change, the already-cached version is not used.
    """

    def multiply(acc, a):
        return acc * a

    # The first time we pass in a function with specific types, it needs to be
    # compiled so this will take some time.
    df = pl.DataFrame({"a": [3]}, schema={"a": pl.Int64()})
    with measure_cpu_time() as elapsed:
        assert_series_equal(
            df.select(pl.col("a").plumba.scan(multiply, 2, pl.Int64))["a"],
            pl.Series("a", [6], dtype=pl.Int64),
        )
    elapsed_with_compile = elapsed.time

    # The next few times we expect a cached version to be used, so it should be
    # much faster.
    rounds = 20
    with measure_cpu_time() as elapsed:
        for _ in range(rounds):
            assert_series_equal(
                df.select(pl.col("a").plumba.scan(multiply, 4, pl.Int64))["a"],
                pl.Series("a", [12], dtype=pl.Int64),
            )
    assert (elapsed.time / rounds) < (elapsed_with_compile / 10)

    # If we use a different type, it compiles a new version:
    assert_series_equal(
        df.select(pl.col("a").plumba.scan(multiply, 0.5, pl.Float64))["a"],
        pl.Series("a", [1.5], dtype=pl.Float64),
    )


CAPTURED_GLOBALS = 7000


def test_captured_variables_must_not_change():
    """
    If a function uses captured variables, they must not change across calls.
    """
    captured_var = 100

    def func(acc, a):
        return acc + a + captured_var + CAPTURED_GLOBALS

    df = pl.DataFrame({"a": [3]}, schema={"a": pl.UInt64()})
    assert df.select(pl.col("a").plumba.scan(func, 20, pl.Int64))["a"].to_list() == [
        7123
    ]

    # Change a local captured var:
    captured_var = 200
    with pytest.raises(RuntimeError, match="changed a captured variable"):
        df.select(pl.col("a").plumba.scan(func, 20, pl.Int64))

    # Restore local captured var:
    captured_var = 100
    assert df.select(pl.col("a").plumba.scan(func, 40, pl.Int64))["a"].to_list() == [
        7143
    ]

    # Change global captured var:
    global CAPTURED_GLOBALS
    CAPTURED_GLOBALS = 6000
    with pytest.raises(RuntimeError, match="changed a captured variable"):
        df.select(pl.col("a").plumba.scan(func, 20, pl.Int64))


def test_two_dtype_variants():
    """
    Both instances and types can be used for return_dtype.
    """
    df = pl.DataFrame({"a": [1, 2]})
    for dtype in (pl.Int64(), pl.Int64):
        assert_series_equal(
            df.select(pl.col("a").plumba.scan(lambda acc, a: acc + a, 0, dtype))["a"],
            pl.Series("a", [1, 3], pl.Int64),
        )


def test_multiple_outputs():
    """
    It's possible to return multiple outputs, which get converted to a
    pl.Array.
    """

    def cum_sum(acc, a, b):
        old_a, old_b = acc
        return (old_a + a, old_b + b)

    df = pl.DataFrame({"a": [1, 2, None, 6, 2], "b": [3, 2, 5, None, 1]})
    result = df.select(
        pl.struct(pl.all())
        .plumba.scan(cum_sum, (6, 9), pl.Array(pl.Int64, shape=2))
        .alias("result")
    )
    assert result["result"].to_list() == [
        [7, 12],
        [9, 14],
        None,
        None,
        [11, 15],
    ], result
