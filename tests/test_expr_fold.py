"""Tests for fold()."""

from typing import Sequence

import polars as pl
import polars_numba as _
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
    Up to 9 columns can be processed, with varying numbers of extra arguments.
    """
    df = pl.DataFrame({str(i): 10.0**i for i in range(1, num_cols + 1)})
    assert df.select(
        pl.struct(pl.all()).plumba.fold(7.0, add_columns, pl.Float64, extra_args)
    ).item() == 7 + sum(10**j for j in range(1, num_cols + 1)) + sum(extra_args)


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
    df.select(
        pl.struct("a", "b").plumba.fold(0.5, add_columns, pl.Float64)
    ).item() == 0.5 + 1 + 30 + 3 + 100


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
    result = df.select(pl.col("a").plumba.fold(np.int64(10), to_f32, pl.Float32)).item()
    assert result == 13.75


def test_compiled_function_caching():
    """
    If fold() is called with the same function and same types again, it
    uses a cached version of the Numba precompiled function.

    If the types change, the already-cached version is not used.
    """

    def multiply(acc, a):
        return acc * a

    # The first time we pass in a function with specific types, it needs to be
    # compiled so this will take some time.
    df = pl.DataFrame({"a": [3]}, schema={"a": pl.UInt64()})
    with measure_cpu_time() as elapsed:
        assert df.select(pl.col("a").plumba.fold(2, multiply, pl.UInt64)).item() == 6
    elapsed_with_compile = elapsed.time

    # The next few times we expect a cached version to be used, so it should be
    # much faster.
    rounds = 20
    with measure_cpu_time() as elapsed:
        for _ in range(rounds):
            assert (
                df.select(pl.col("a").plumba.fold(4, multiply, pl.UInt64)).item() == 12
            )
    assert (elapsed.time / rounds) < (elapsed_with_compile / 10)

    # If we use a different type, it compiles a new version:
    assert df.select(pl.col("a").plumba.fold(1.5, multiply, pl.Float64)).item() == 4.5


CAPTURED_GLOBALS = 7000


def test_captured_variables_must_not_change():
    """
    If a function uses captured variables, they must not change across calls.
    """
    captured_var = 100

    def func(acc, x):
        return acc + x + captured_var + CAPTURED_GLOBALS

    df = pl.DataFrame({"a": [3]}, schema={"a": pl.UInt64()})
    assert df.select(pl.col("a").plumba.fold(20, func, pl.UInt64)).item() == 7123

    # Change a local captured var:
    captured_var = 200
    with pytest.raises(RuntimeError, match="changed a captured variable"):
        df.select(pl.col("a").plumba.fold(20, func, pl.UInt64))

    # Restore local captured var:
    captured_var = 100
    assert df.select(pl.col("a").plumba.fold(40, func, pl.UInt64)).item() == 7143

    # Change global captured var:
    global CAPTURED_GLOBALS
    CAPTURED_GLOBALS = 6000
    with pytest.raises(RuntimeError, match="changed a captured variable"):
        df.select(pl.col("a").plumba.fold(20, func, pl.UInt64))
