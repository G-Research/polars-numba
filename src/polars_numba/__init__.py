"""
Higher-level programmable Polars APIs, using Numba.

TODO

    - Cached compiled version is faster, but it's still slower than I would
      expect, might be fixed overhead from Polars, investigate, write test that
      for large amounts of data it is actually fast, file/fix upstream if
      relevant.
"""

from __future__ import annotations

from functools import reduce
from inspect import signature, getclosurevars
from operator import or_
from types import FunctionType
from typing import (
    Any,
    Callable,
    Concatenate,
    TypeVar,
    ParamSpec,
    Sequence,
    TYPE_CHECKING,
)


import numpy as np
import polars as pl
from numba import jit
from numba.core.dispatcher import Dispatcher

if TYPE_CHECKING:
    from polars.datatypes import PolarsDataType

T = TypeVar("T")
P = ParamSpec("P")

__all__ = ["collect_fold", "collect_scan"]


@jit(nogil=True)
def _folder1(numba_function, acc, extra_args, arr1):
    """Loop and fold a 1-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, *extra_args, arr1[i])
    return acc


@jit(nogil=True)
def _folder2(numba_function, acc, extra_args, arr1, arr2):
    """Loop and fold a 2-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, *extra_args, arr1[i], arr2[i])
    return acc


@jit(nogil=True)
def _folder3(numba_function, acc, extra_args, arr1, arr2, arr3):
    """Loop and fold a 3-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, *extra_args, arr1[i], arr2[i], arr3[i])
    return acc


@jit(nogil=True)
def _folder4(numba_function, acc, extra_args, arr1, arr2, arr3, arr4):
    """Loop and fold a 4-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, *extra_args, arr1[i], arr2[i], arr3[i], arr4[i])
    return acc


@jit(nogil=True)
def _folder5(numba_function, acc, extra_args, arr1, arr2, arr3, arr4, arr5):
    """Loop and fold a 5-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(
            acc, *extra_args, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i]
        )
    return acc


@jit(nogil=True)
def _folder6(numba_function, acc, extra_args, arr1, arr2, arr3, arr4, arr5, arr6):
    """Loop and fold a 6-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(
            acc, *extra_args, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i]
        )
    return acc


@jit(nogil=True)
def _folder7(numba_function, acc, extra_args, arr1, arr2, arr3, arr4, arr5, arr6, arr7):
    """Loop and fold a 7-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(
            acc,
            *extra_args,
            arr1[i],
            arr2[i],
            arr3[i],
            arr4[i],
            arr5[i],
            arr6[i],
            arr7[i],
        )
    return acc


@jit(nogil=True)
def _folder8(
    numba_function, acc, extra_args, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8
):
    """Loop and fold a 8-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(
            acc,
            *extra_args,
            arr1[i],
            arr2[i],
            arr3[i],
            arr4[i],
            arr5[i],
            arr6[i],
            arr7[i],
            arr8[i],
        )
    return acc


@jit(nogil=True)
def _folder9(
    numba_function,
    acc,
    extra_args,
    arr1,
    arr2,
    arr3,
    arr4,
    arr5,
    arr6,
    arr7,
    arr8,
    arr9,
):
    """Loop and fold a 9-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(
            acc,
            *extra_args,
            arr1[i],
            arr2[i],
            arr3[i],
            arr4[i],
            arr5[i],
            arr6[i],
            arr7[i],
            arr8[i],
            arr9[i],
        )
    return acc


_NUMBA_CACHE = {}
_CAPTURED_VARS_HASHES: dict[FunctionType, int] = {}

_CAPTURED_VARS_CHANGED_MESSAGE = """\
You have changed a captured variable in a function passed to collect_fold().

Function {function} uses the following captured variables: {variables}.

If you are using a function repeatedly with collect_fold(), these captured\
 variables must not change, but one of them at least has changed since\
 the last call.
"""


def _ensure_captured_vars_are_unchanged(function: FunctionType) -> None:
    """
    Raise a RuntimeError if captured variables in the function have changed
    since the last time this was called.

    TODO Right now this is a heuristic and might not catch all cases.  Also
    will fail on mutable captured vars.  Maybe use DeepHash from DeepDiff?
    This would use cryptographic hash so would also reduce chance of false
    negative to be negligible.
    """
    closurevars = getclosurevars(function)
    captured = [
        (name, cell.cell_contents)
        for (name, cell) in zip(
            function.__code__.co_freevars or (), function.__closure__ or ()
        )
    ]
    captured.extend([(n, function.__globals__[n]) for n in closurevars.globals])
    captured.sort()
    vars_hash = hash(tuple(captured))
    if recorded_hash := _CAPTURED_VARS_HASHES.get(function):
        if recorded_hash != vars_hash:
            raise RuntimeError(
                _CAPTURED_VARS_CHANGED_MESSAGE.format(
                    variables=", ".join(name for (name, _) in captured),
                    function=function,
                )
            )
    else:
        _CAPTURED_VARS_HASHES[function] = vars_hash


def _compile_function(function: FunctionType) -> Dispatcher:
    """
    Wrap a Python function with ``@numba.jit``, cache, and return.
    """
    assert isinstance(function, FunctionType)
    _ensure_captured_vars_are_unchanged(function)

    if function in _NUMBA_CACHE:
        numba_function = _NUMBA_CACHE[function]
    else:
        numba_function = jit(nogil=True)(function)
        _NUMBA_CACHE[function] = numba_function
    return numba_function


def _get_column_names(
    function: FunctionType, column_names: None | list[str] = None
) -> list[str]:
    """
    Get the column names, extracted from the fold/scan function arguments if
    necessary.
    """
    if column_names is None:
        column_names = [p for p in signature(function).parameters.keys()][1:]
    return column_names


def _prep_for_df(
    df: pl.DataFrame | pl.LazyFrame,
    function: FunctionType,
    column_names: None | list[str] = None,
) -> tuple[pl.LazyFrame, Dispatcher, list[str]]:
    """
    Prepare arguments for use.

        1. Convert frame to ``LazyFrame``.

        2. Limit columns in the ``LazyFrame``, if necessary.

        3. Validate the function and (utilizing a cache) wrap it with
           ``numba.jit``.
    """
    numba_function = _compile_function(function)
    lazy_df = df.lazy()
    if column_names is not None:
        lazy_df = lazy_df.select_seq(*column_names)

    return (lazy_df, numba_function, column_names)


def _get_folder(num_args: int) -> Callable[Concatenate[T, P, T]]:
    """
    Return the folder function for the given number of arguments.
    """
    # As an alternative to doing dispatch here, we could do the dispatch inside
    # the Numba function, which would be less verbose and duplicative. However,
    # that means longer compilation time since the Numba function will be much
    # more complex.
    match num_args:
        case 0:
            raise ValueError("You must pass in at least one column name")

        case 1:
            folder = _folder1

        case 2:
            folder = _folder2

        case 3:
            folder = _folder3

        case 4:
            folder = _folder4

        case 5:
            folder = _folder5

        case 6:
            folder = _folder6

        case 7:
            folder = _folder7

        case 8:
            folder = _folder8

        case 9:
            folder = _folder9

        case _:
            raise RuntimeError(
                f"You passed in {num_args} columns, but currently "
                "only up to 9 columns are supported; if you need more, file "
                "an issue."
            )

    return folder


def collect_fold(
    df: pl.DataFrame | pl.LazyFrame,
    function: Callable[Concatenate[T, P], T],
    initial_accumulator: T,
    extra_args: Sequence[Any] = (),
    column_names: None | list[str] = None,
) -> T:
    """
    Collect a frame into a literal value by folding it using a function.

    Streaming is used to save memory.

    If column names are not given, all columns in the DataFrame will be passed
    to the given function.

    For each row, the accumulator will be passed in to the given function along
    with the values for respective columns.  The returned result will be a new
    accumulator used for the next row.  The final accumulator is the result of
    this function.

    Rows with nulls are filtered out before processing.

    The given function will be compiled with Numba.  If it captures variables,
    those variables must not change over time, since the function will only be
    compiled once.
    """
    (lazy_df, numba_function, column_names) = _prep_for_df(df, function, column_names)
    lazy_df = lazy_df.drop_nulls()
    extra_args = tuple(extra_args)
    acc = initial_accumulator
    folder = None

    for batch_df in lazy_df.collect_batches(chunk_size=50_000, lazy=True):
        if folder is None:
            if column_names is None:
                column_names = batch_df.columns
            folder = _get_folder(len(column_names))
        acc = folder(
            numba_function,
            acc,
            extra_args,
            *(s.to_numpy() for s in batch_df.get_columns()),
        )
    return acc


def fold(
    expr: pl.Expr,
    function: Callable[Concatenate[T, P], T],
    initial_accumulator: T,
    return_dtype: PolarsDataType,
    extra_args: Sequence[Any] = (),
) -> pl.Expr:
    """
    Collect an expression into a literal value by folding it using a function.

    Nulls will be dropped before the function is called.

    To support multiple arguments, you can use ``pl.struct()`` to combine
    multiple columns into a ``Struct``.  The struct columns should be in the
    order you wish to pass to the function.

    Streaming is NOT used, so memory usage may be high.

    ``**extra_args`` allows passing in additional constants to the called
    function, in the order they were passed in; they will be passed in at the
    end of the arguments.

    See ``collect_fold()`` for other details.
    """
    numba_function = _compile_function(function)

    def handle_data(series: pl.Series) -> T:
        if series.dtype == pl.Struct:
            df = series.struct.unnest()
            column_names = df.columns
        else:
            df = pl.DataFrame({"fold": series})
            column_names = ["fold"]
        df = df.drop_nulls()
        folder = _get_folder(len(column_names))
        return folder(
            numba_function,
            initial_accumulator,
            tuple(extra_args),
            *(df[n].to_numpy() for n in column_names),
        )

    return expr.map_batches(
        handle_data,
        is_elementwise=False,
        returns_scalar=True,
        return_dtype=return_dtype,
    )


_POLARS_DTYPE_TO_NUMPY = {
    pl.Datetime: np.datetime64,
    pl.Boolean: np.bool,
    pl.Float32: np.float32,
    pl.Float64: np.float64,
    pl.Int8: np.int8,
    pl.Int16: np.int16,
    pl.Int32: np.int32,
    pl.Int64: np.int64,
    pl.Duration: np.timedelta64,
    pl.UInt8: np.uint8,
    pl.UInt16: np.uint16,
    pl.UInt32: np.uint32,
    pl.UInt64: np.uint64,
}
if hasattr(pl, "Float16") and hasattr(np, "float16"):
    _POLARS_DTYPE_TO_NUMPY[pl.Float16] = np.float16


def _polars_dtype_to_numpy(dtype: PolarsDataType) -> np.dtype:
    """
    Convert a Polars dtype to a NumPy dtype.
    """
    # TODO test both paths
    if not isinstance(dtype, type):
        dtype = type(dtype)
    return _POLARS_DTYPE_TO_NUMPY[dtype]


@jit(nogil=True)
def _scanner1(numba_function, acc, extra_args, result, is_null, arr1):
    """Loop and fold a 1-argument function."""
    for i in range(len(arr1)):
        acc = acc if is_null[i] else numba_function(acc, *extra_args, arr1[i])
        result[i] = acc
    return acc, result


@jit(nogil=True)
def _scanner2(numba_function, acc, extra_args, result, is_null, arr1, arr2):
    """Loop and fold a 2-argument function."""
    for i in range(len(arr1)):
        acc = acc if is_null[i] else numba_function(acc, *extra_args, arr1[i], arr2[i])
        result[i] = acc
    return acc, result


@jit(nogil=True)
def _scanner3(numba_function, acc, extra_args, result, is_null, arr1, arr2, arr3):
    """Loop and fold a 3-argument function."""
    for i in range(len(arr1)):
        acc = (
            acc
            if is_null[i]
            else numba_function(acc, *extra_args, arr1[i], arr2[i], arr3[i])
        )
        result[i] = acc
    return acc, result


@jit(nogil=True)
def _scanner4(numba_function, acc, extra_args, result, is_null, arr1, arr2, arr3, arr4):
    """Loop and fold a 4-argument function."""
    for i in range(len(arr1)):
        acc = (
            acc
            if is_null[i]
            else numba_function(acc, *extra_args, arr1[i], arr2[i], arr3[i], arr4[i])
        )
        result[i] = acc
    return acc, result


@jit(nogil=True)
def _scanner5(
    numba_function, acc, extra_args, result, is_null, arr1, arr2, arr3, arr4, arr5
):
    """Loop and fold a 5-argument function."""
    for i in range(len(arr1)):
        acc = (
            acc
            if is_null[i]
            else numba_function(
                acc, *extra_args, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i]
            )
        )
        result[i] = acc
    return acc, result


@jit(nogil=True)
def _scanner6(
    numba_function, acc, extra_args, result, is_null, arr1, arr2, arr3, arr4, arr5, arr6
):
    """Loop and fold a 6-argument function."""
    for i in range(len(arr1)):
        acc = (
            acc
            if is_null[i]
            else numba_function(
                acc, *extra_args, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i]
            )
        )
        result[i] = acc
    return acc, result


@jit(nogil=True)
def _scanner7(
    numba_function,
    acc,
    extra_args,
    result,
    is_null,
    arr1,
    arr2,
    arr3,
    arr4,
    arr5,
    arr6,
    arr7,
):
    """Loop and fold a 7-argument function."""
    for i in range(len(arr1)):
        acc = (
            acc
            if is_null[i]
            else numba_function(
                acc,
                *extra_args,
                arr1[i],
                arr2[i],
                arr3[i],
                arr4[i],
                arr5[i],
                arr6[i],
                arr7[i],
            )
        )
        result[i] = acc
    return acc, result


@jit(nogil=True)
def _scanner8(
    numba_function,
    acc,
    extra_args,
    result,
    is_null,
    arr1,
    arr2,
    arr3,
    arr4,
    arr5,
    arr6,
    arr7,
    arr8,
):
    """Loop and fold a 8-argument function."""
    for i in range(len(arr1)):
        acc = (
            acc
            if is_null[i]
            else numba_function(
                acc,
                *extra_args,
                arr1[i],
                arr2[i],
                arr3[i],
                arr4[i],
                arr5[i],
                arr6[i],
                arr7[i],
                arr8[i],
            )
        )
        result[i] = acc
    return acc, result


@jit(nogil=True)
def _scanner9(
    numba_function,
    acc,
    extra_args,
    result,
    is_null,
    arr1,
    arr2,
    arr3,
    arr4,
    arr5,
    arr6,
    arr7,
    arr8,
    arr9,
):
    """Loop and fold a 9-argument function."""
    for i in range(len(arr1)):
        acc = (
            acc
            if is_null[i]
            else numba_function(
                acc,
                *extra_args,
                arr1[i],
                arr2[i],
                arr3[i],
                arr4[i],
                arr5[i],
                arr6[i],
                arr7[i],
                arr8[i],
                arr9[i],
            )
        )
        result[i] = acc
    return acc, result


def _get_scanner(num_args: int) -> Dispatcher:
    """
    Get the scanner function for the given number of arguments.
    """
    match num_args:
        case 0:
            raise ValueError("You must pass in at least one column name")

        case 1:
            scanner = _scanner1

        case 2:
            scanner = _scanner2

        case 3:
            scanner = _scanner3

        case 4:
            scanner = _scanner4

        case 5:
            scanner = _scanner5

        case 6:
            scanner = _scanner6

        case 7:
            scanner = _scanner7

        case 8:
            scanner = _scanner8

        case 9:
            scanner = _scanner9

        case _:
            raise RuntimeError(
                f"You passed in {num_args} columns, but currently "
                "only up to 9 columns are supported; if you need more, file "
                "an issue."
            )
    return scanner


def collect_scan(
    df: pl.DataFrame | pl.LazyFrame,
    function: Callable[Concatenate[T, P], T],
    initial_accumulator: T,
    result_dtype: PolarsDataType,
    extra_args: Sequence[Any] = (),
    column_names: None | list[str] = None,
) -> pl.Series:
    """
    Collect a frame into a ``Series`` by scanning it using a function.

    For each row, the accumulator will be passed in to the given function along
    with the values for respective columns.  The returned result is used both
    as the corresponding value for the final ``Series`` and as the accumulator
    for the next row.

    If any of the selected columns have nulls on a particular row, that
    particular row will be null in the output ``Series``, and the row will not
    be passed to the function.

    If no column names are given, all the columns in the DataFrame are passed
    to the function.
    """
    (lazy_df, numba_function, column_names) = _prep_for_df(df, function, column_names)
    np_dtype = _polars_dtype_to_numpy(result_dtype)
    extra_args = tuple(extra_args)
    scanner = None

    acc = initial_accumulator
    results = []
    for batch_df in lazy_df.collect_batches(chunk_size=50_000, lazy=True):
        if scanner is None:
            if column_names is None:
                column_names = batch_df.columns
            scanner = _get_scanner(len(column_names))
        batch_result = np.empty((len(batch_df),), dtype=np_dtype)
        is_null = reduce(or_, (s.is_null() for s in batch_df.get_columns()))

        # We can't have nulls in the dataframe; NumPy has no concept of nulls.
        # And so e.g. for an int Series, Polars will turn it into a float array
        # with nans. This is solvable with Numba Arrow support, maybe.
        batch_df = batch_df.fill_null(strategy="zero")

        scanner(
            numba_function,
            acc,
            extra_args,
            batch_result,
            is_null.to_numpy(),
            *(s.to_numpy() for s in batch_df.get_columns()),
        )
        results.append(
            pl.Series("scan", batch_result, dtype=result_dtype).set(is_null, None)
        )

    result = pl.concat(results)
    return result


def scan(
    expr: pl.Expr,
    function: Callable[Concatenate[T, P], T],
    initial_accumulator: T,
    return_dtype: PolarsDataType,
    extra_args: Sequence[Any] = (),
) -> pl.Expr:
    """
    Collect an expression into a `Series` by scanning it using a function.

    Nulls will result in the correponding row being a null, and that row will
    not be passed to the given function.

    To support multiple arguments, you can use ``pl.struct()`` to combine
    multiple columns into a ``Struct``.  The struct columns should be in the
    order you wish to pass to the function.

    Streaming is NOT used, so memory usage may be high.

    ``**extra_args`` allows passing in additional constants to the called
    function, in the order they were passed in; they will be passed in at the
    end of the arguments.

    See ``collect_scan()`` for other details.
    """
    numba_function = _compile_function(function)
    np_dtype = _polars_dtype_to_numpy(return_dtype)

    def handle_data(series: pl.Series) -> T:
        if series.dtype == pl.Struct:
            df = series.struct.unnest()
            column_names = df.columns
        else:
            df = pl.DataFrame({"fold": series})
            column_names = ["fold"]
        is_null = reduce(or_, (df[s].is_null() for s in df.columns))
        result = np.empty((len(df),), dtype=np_dtype)
        df = df.fill_null(strategy="zero")
        scanner = _get_scanner(len(column_names))
        scanner(
            numba_function,
            initial_accumulator,
            tuple(extra_args),
            result,
            is_null.to_numpy(),
            *(df[n].to_numpy() for n in column_names),
        )
        return pl.Series(series.name, result, dtype=return_dtype).set(is_null, None)

    return expr.map_batches(
        handle_data,
        is_elementwise=False,
        returns_scalar=False,
        return_dtype=return_dtype,
    )


@pl.api.register_expr_namespace("plumba")
class _PolarsNumbaExprNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def fold(
        self,
        function: Callable[Concatenate[T, P], T],
        initial_accumulator: T,
        return_dtype: PolarsDataType,
        extra_args: Sequence[Any] = (),
    ) -> pl.Expr:
        return fold(self._expr, function, initial_accumulator, return_dtype, extra_args)

    def scan(
        self,
        function: Callable[Concatenate[T, P], T],
        initial_accumulator: T,
        return_dtype: PolarsDataType,
        extra_args: Sequence[Any] = (),
    ) -> pl.Expr:
        return scan(self._expr, function, initial_accumulator, return_dtype, extra_args)


_PolarsNumbaExprNamespace.fold.__doc__ = fold.__doc__
_PolarsNumbaExprNamespace.scan.__doc__ = scan.__doc__
