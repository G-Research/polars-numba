"""
By default, only a single ArrayBuilder is used.

The exception is if is_elementwise=True is used.  In non-streaming mode, there
will be an ArrayBuilder per chunk.  In streaming mode, there will be smaller
views into the underlying data, regardless of chunking.
"""

import polars as pl
from polars_numba import arrow_jit


df = pl.DataFrame(
    {"values": pl.concat([pl.Series(range(1_000_000)), pl.Series(range(1_000_000))])}
)
assert df.n_chunks() == 2


def eager(f):
    return df.select(f(pl.col("values")))


def lazy(f):
    return df.lazy().select(f(pl.col("values"))).collect()


def streaming(f):
    return df.lazy().select(f(pl.col("values"))).collect(engine="streaming")


def run_all_options(count_items, returns_scalar):
    for is_elementwise in (False, True):
        f = arrow_jit(returns_scalar=returns_scalar, is_elementwise=is_elementwise)(
            count_items
        )
        for operation in (eager, lazy, streaming):
            print(
                f"is_elementwise={is_elementwise}, operation={operation.__name__}, ArrayBuilder called with following chunk sizes:",
                operation(f).get_column("values").to_list(),
            )


def count_items_array(arr, array_builder):
    length = len(arr)
    array_builder.integer(length)


print("returns_scalar = False, i.e. ArrayBuilder is being used")
run_all_options(count_items_array, False)


def count_items_scalar(arr):
    return len(arr)


print()
print("returns_scalar = True, i.e. no ArrayBuilders")
run_all_options(count_items_scalar, True)
