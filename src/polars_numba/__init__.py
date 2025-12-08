"""
Higher-level programmable Polars APIs, using Numba.

TODO:
- README
- Examples of fold()
"""

from __future__ import annotations

from types import FunctionType
from typing import Callable, Concatenate, TypeVar, ParamSpec, TYPE_CHECKING
from inspect import signature, getclosurevars
import polars as pl
from numba import jit

if TYPE_CHECKING:
    from polars._typing import Literal

T = TypeVar("T")
P = ParamSpec("P")


@jit(nogil=True)
def _folder1(numba_function, acc, arr1):
    """Loop and fold a 1-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, arr1[i])
    return acc


@jit(nogil=True)
def _folder2(numba_function, acc, arr1, arr2):
    """Loop and fold a 2-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, arr1[i], arr2[i])
    return acc


@jit(nogil=True)
def _folder3(numba_function, acc, arr1, arr2, arr3):
    """Loop and fold a 3-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, arr1[i], arr2[i], arr3[i])
    return acc


@jit(nogil=True)
def _folder4(numba_function, acc, arr1, arr2, arr3, arr4):
    """Loop and fold a 4-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i])
    return acc


@jit(nogil=True)
def _folder5(numba_function, acc, arr1, arr2, arr3, arr4, arr5):
    """Loop and fold a 5-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i])
    return acc


@jit(nogil=True)
def _folder6(numba_function, acc, arr1, arr2, arr3, arr4, arr5, arr6):
    """Loop and fold a 6-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i])
    return acc


@jit(nogil=True)
def _folder7(numba_function, acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7):
    """Loop and fold a 7-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(
            acc,
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
def _folder8(numba_function, acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8):
    """Loop and fold a 8-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(
            acc,
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
def _folder9(numba_function, acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9):
    """Loop and fold a 9-argument function."""
    for i in range(len(arr1)):
        acc = numba_function(
            acc,
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


def collect_fold(
    df: pl.DataFrame | pl.LazyFrame,
    initial_accumulator: Literal,
    function: Callable[Concatenate[Literal, P], T],
    column_names: None | list[str] = None,
) -> T:
    """
    Collect a frame into a literal value by folding it using a function.

    Streaming is used to save memory.

    If column names are not given, the names of the arguments to the passed in
    function will be used (skipping the first one, since that's the
    accumulator).

    For each row, the accumulator will be passed in to the given function along
    with the values for respective columns.  The result will be a new
    accumulator used for the next row.  The final accumulator is the result of
    this function.

    Rows with nulls are filtered out before processing.

    The given function will be compiled with Numba.  If it captures variables,
    those variables must not change over time, since the function will only be
    compiled once.
    """
    assert isinstance(function, FunctionType)
    _ensure_captured_vars_are_unchanged(function)

    if column_names is None:
        column_names = [p for p in signature(function).parameters.keys()][1:]
    lazy_df = df.lazy().select_seq(*column_names).drop_nulls()

    if function in _NUMBA_CACHE:
        numba_function = _NUMBA_CACHE[function]
    else:
        numba_function = jit(nogil=True)(function)
        _NUMBA_CACHE[function] = numba_function

    # As an alternative to doing dispatch here, we could do the dispatch inside
    # the Numba function, which would be less verbose and duplicative. However,
    # that means longer compilation time since the Numba function will be much
    # more complex.
    match len(column_names):
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
                f"You passed in {len(column_names)} columns, but currently "
                "only up to 9 columns are supported; if you need more, file "
                "an issue."
            )

    acc = initial_accumulator
    for batch_df in lazy_df.collect_batches(chunk_size=50_000, lazy=True):
        acc = folder(
            numba_function, acc, *(batch_df[n].to_numpy() for n in column_names)
        )
    return acc
