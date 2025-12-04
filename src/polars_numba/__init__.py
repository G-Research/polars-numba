"""
Higher-level programmable Polars APIs, using Numba.

TODO:
- Examples of fold()
- Caching of compilation
   - Complain if function's bound variables change
"""

from __future__ import annotations

from typing import Callable, Concatenate, TypeVar, ParamSpec, TYPE_CHECKING
from inspect import signature
import polars as pl
from numba import jit

if TYPE_CHECKING:
    from polars._typing import Literal

T = TypeVar("T")
P = ParamSpec("P")


def collect_fold(
    df: pl.DataFrame | pl.LazyFrame,
    initial_accumulator: Literal,
    function: Callable[Concatenate[Literal, P], T],
    column_names: None | list[str] = None,
) -> T:
    """
    Collect a frame into a literal value by folding it using a function.

    If column names are not given, the names of the arguments to the passed in
    function will be used (skipping the first one, since that's the
    accumulator).

    For each row, the accumulator will be passed in to the function along with
    the values for respective columns.  The result will be a new accumulator
    used for the next row.  The final accumulator is the result of this
    function.

    Rows with nulls are filtered out before processing.
    """
    if column_names is None:
        column_names = [p for p in signature(function).parameters.keys()][1:]
    lazy_df = df.lazy().select_seq(*column_names).drop_nulls()
    numba_function = jit(nogil=True)(function)

    # As an alternative to doing dispatch here, we could do the dispatch inside
    # the Numba function, which would be less verbose and duplicative. However,
    # that means longer compilation time since the Numba function will be much
    # more complex.
    match len(column_names):
        case 0:
            raise ValueError("You must pass in at least one column name")

        case 1:

            @jit(nogil=True)
            def folder(acc, arr1):
                for i in range(len(arr1)):
                    acc = numba_function(acc, arr1[i])
                return acc

        case 2:

            @jit(nogil=True)
            def folder(acc, arr1, arr2):
                for i in range(len(arr1)):
                    acc = numba_function(acc, arr1[i], arr2[i])
                return acc

        case 3:

            @jit(nogil=True)
            def folder(acc, arr1, arr2, arr3):
                for i in range(len(arr1)):
                    acc = numba_function(acc, arr1[i], arr2[i], arr3[i])
                return acc

        case 4:

            @jit(nogil=True)
            def folder(acc, arr1, arr2, arr3, arr4):
                for i in range(len(arr1)):
                    acc = numba_function(acc, arr1[i], arr2[i], arr3[i], arr4[i])
                return acc

        case 5:

            @jit(nogil=True)
            def folder(acc, arr1, arr2, arr3, arr4, arr5):
                for i in range(len(arr1)):
                    acc = numba_function(
                        acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i]
                    )
                return acc

        case 6:

            @jit(nogil=True)
            def folder(acc, arr1, arr2, arr3, arr4, arr5, arr6):
                for i in range(len(arr1)):
                    acc = numba_function(
                        acc, arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i]
                    )
                return acc

        case 7:

            @jit(nogil=True)
            def folder(acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7):
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

        case 8:

            @jit(nogil=True)
            def folder(acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8):
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

        case 9:

            @jit(nogil=True)
            def folder(acc, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9):
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

        case _:
            raise RuntimeError(
                f"You passed in {len(column_names)} columns, but currently only up to 9"
                " columns are supported; if you need more, file an issue"
            )

    acc = initial_accumulator
    for batch_df in lazy_df.collect_batches(chunk_size=50_000):
        acc = folder(acc, *(batch_df[n].to_numpy() for n in column_names))
    return acc
