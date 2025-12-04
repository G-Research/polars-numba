"""
Higher-level programmable Polars APIs, using Numba.

TODO:
- fold()
- Examples of fold()
- Extract column names from given function
- Caching of compilation
   - Complain if function's bound variables change
"""

from __future__ import annotations

from typing import Any, Callable, Concatenate, TypeVar, ParamSpec, TYPE_CHECKING

import polars as pl
from numba import jit

if TYPE_CHECKING:
    from polars._typing import Literal


T = TypeVar("T")
P = ParamSpec("P")


def collect_fold(
    df: pl.DataFrame | pl.LazyFrame,
    initial_accumulator: Literal,
    function: Callable[Concatenate[Literal, P], Any],
    column_names: list[str],
) -> Literal:
    """
    Collect a frame into a literal value by folding it using a function.

    For each row, the accumulator will be passed in to the function along with
    the values for respective columns.  The result will be a new accumulator
    used for the next row.  The final accumulator is the result of this
    function.

    Rows with nulls are filtered out before processing.
    """
    lazy_df = df.lazy().select_seq(*column_names).drop_nulls()
    numba_function = jit(nogil=True)(function)
    num_columns = len(column_names)

    if num_columns == 0:
        raise ValueError("You must pass in at least one column name")

    # As an alternative we could do the dispatch inside the Numba function.
    # However, that means longer compilation time since the function will be
    # much more complex.
    if num_columns == 1:

        @jit(nogil=True)
        def folder(acc, arr1):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i])
            return acc

    elif num_columns == 2:

        @jit(nogil=True)
        def folder(acc, arr1, arr2):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i], arr2[i])
            return acc

    elif num_columns == 3:

        @jit(nogil=True)
        def folder(acc, arr1, arr2, arr3):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i], arr2[i], arr3[i])
            return acc

    elif num_columns == 4:

        @jit(nogil=True)
        def folder(acc, arr1, arr2, arr3, arr4):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i])
            return acc

    elif num_columns == 5:

        @jit(nogil=True)
        def folder(acc, arr1, arr2, arr3, arr4, arr5):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i])
            return acc

    elif num_columns == 6:

        @jit(nogil=True)
        def folder(acc, arr1, arr2, arr3, arr4, arr5, arr6):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i])
            return acc

    elif num_columns == 7:

        @jit(nogil=True)
        def folder(acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i])
            return acc

    elif num_columns == 8:

        @jit(nogil=True)
        def folder(acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i], arr8[i])
            return acc

    elif num_columns == 9:

        @jit(nogil=True)
        def folder(acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9):
            for i in range(len(arr1)):
                acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i], arr8[i], arr9[i])
            return acc

    else:
        raise RuntimeError(
            f"You passed in {num_columns} columns, but currently only up to 9"
            " columns are supported; if you need more, file an issue"
        )

    acc = initial_accumulator
    for batch_df in lazy_df.collect_batches(chunk_size=50_000):
        acc = folder(acc, *(batch_df[n].to_numpy() for n in column_names))
    return acc
