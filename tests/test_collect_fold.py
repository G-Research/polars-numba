"""Tests for collect_fold()."""

from time import process_time
from contextlib import contextmanager
from dataclasses import dataclass

import polars as pl
from polars_numba import collect_fold
from numba import float32
import numpy as np
import pytest


def add_columns(acc, *values):
    for v in values:
        acc += v
    return acc


@dataclass
class Elapsed:
    time: float | None = None


@contextmanager
def measure_cpu_time():
    """
    Context manager that measures elapsed process CPU time.
    """
    elapsed = Elapsed()
    start = process_time()
    try:
        yield elapsed
    finally:
        elapsed.time = process_time() - start


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
    result = collect_fold(df, np.int64(10), to_f32, ["a"])
    assert result == 13.75


def test_generate_column_names():
    """
    If column names aren't given, use the argument names in the passed in
    function.
    """

    def operator(acc, b, a):
        return acc + 10 * b + a

    df = pl.DataFrame({"acc": [1, 2, 3], "a": [5, 6, 7], "b": [20, 30, 20]})
    assert collect_fold(df, 0.5, operator) == 718.5


def test_compiled_function_caching():
    """
    If collect_fold() is called with the same function and same types again, it
    uses a cached version of the Numba precompiled function.

    If the types change, the already-cached version is not used.
    """

    def multiply(acc, a):
        return acc * a

    # The first time we pass in a function with specific types, it needs to be
    # compiled so this will take some time.
    df = pl.DataFrame({"a": [3]}, schema={"a": pl.UInt64()}).lazy()
    with measure_cpu_time() as elapsed:
        assert collect_fold(df, np.uint64(2), multiply, ["a"]) == 6
    elapsed_with_compile = elapsed.time

    # The next few times we expect a cached version to be used, so it should be
    # much faster.
    rounds = 20
    with measure_cpu_time() as elapsed:
        for _ in range(rounds):
            assert collect_fold(df, np.uint64(4), multiply, ["a"]) == 12
    assert (elapsed.time / rounds) < (elapsed_with_compile / 10)

    # If we use a different type, it compiles a new version:
    df = pl.DataFrame({"a": [3]}, schema={"a": pl.UInt64()}).lazy()
    assert collect_fold(df, 0.5, multiply, ["a"]) == 1.5


CAPTURED_GLOBALS = 7000


def test_captured_variables_must_not_change():
    """
    If a function uses captured variables, they must not change across calls.
    """
    captured_var = 100

    def func(acc, x):
        return acc + x + captured_var + CAPTURED_GLOBALS

    df = pl.DataFrame({"a": [3]}, schema={"a": pl.UInt64()}).lazy()
    assert collect_fold(df, 20, func, ["a"]) == 7123

    # Change a local captured var:
    captured_var = 200
    with pytest.raises(RuntimeError, match="changed a captured variable"):
        collect_fold(df, 20, func, ["a"])

    # Restore local captured var:
    captured_var = 100
    assert collect_fold(df, 40, func, ["a"]) == 7143

    # Change global captured var:
    global CAPTURED_GLOBALS
    CAPTURED_GLOBALS = 6000
    with pytest.raises(RuntimeError, match="changed a captured variable"):
        collect_fold(df, 20, func, ["a"])
