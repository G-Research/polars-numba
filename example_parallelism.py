"""
We can get parllelism in certain edge cases.

How it works:

    - @arrow_jit is a wrapper around Polars' ``Expr.map_batches()``.

    - When ``map_batches()`` has ``is_elementwise=True, returns_scalar=False``,
      the passed in function receives batches rather than the whole series when
      in streaming mode.

    - We can therefore combine a two pass operation to get batched operations,
      which then get run in the thread pool in parallel, especially in
      streaming mode.
"""

from time import time, process_time

import numpy as np
import polars as pl
from polars_numba import arrow_jit


def timeit(prefix, f, count=50):
    start, cpu_start = time(), process_time()
    for _ in range(count):
        f()
    print(
        f"{prefix}:",
        (time() - start) / count,
        "(secs)",
        (process_time() - cpu_start) / count,
        "(CPU secs)",
    )


@arrow_jit(returns_scalar=True, return_dtype=pl.Float64())
def not_parallel_sum(arr):
    result = 0.0
    for value in arr:
        if value is not None:
            # Try a complex expression so we're not bottlenecked on memory
            # bandwidth:
            result += np.log(np.cos(value) + np.sin(value) + 7)
    return result


# is_elementwise means we won't always get the full Series, we might get chunks
# in some cases.
@arrow_jit(returns_scalar=False, is_elementwise=True, return_dtype=pl.Float64())
def sum_chunk(arr, array_builder):
    result = 0
    for value in arr:
        if value is not None:
            result += np.log(np.cos(value) + np.sin(value) + 7)
    array_builder.real(result)


def parallel_sum(column: pl.Expr) -> pl.Expr:
    # First, do sum of chunks, which will result in a Series of patial sums:
    partial_sums = sum_chunk(column)
    # Then do sum of those:
    return partial_sums.sum()


df = pl.DataFrame({"values": range(1_000_000)})

print(df.select(parallel_sum(pl.col("values"))))
print(df.select(not_parallel_sum(pl.col("values"))))
# Wierdly doing just this, and not the above, results in Numba issues?!
print(df.lazy().select(parallel_sum(pl.col("values"))).collect(engine="streaming"))

# Check correctness
assert (
    abs(
        df.select(not_parallel_sum(pl.col("values"))).item()
        - df.lazy()
        .select(parallel_sum(pl.col("values")))
        .collect(engine="streaming")
        .item()
    )
    < 0.00001
)

timeit("Eager, not_parallel:", lambda: df.select(not_parallel_sum(pl.col("values"))))
timeit("Eager, parallel:", lambda: df.select(parallel_sum(pl.col("values"))))
timeit(
    "Lazy, not_parallel:",
    lambda: df.lazy().select(not_parallel_sum(pl.col("values"))).collect(),
)
timeit(
    "Lazy, parallel:",
    lambda: df.lazy().select(parallel_sum(pl.col("values"))).collect(),
)
timeit(
    "Lazy streaming, not_parallel:",
    lambda: df.lazy()
    .select(not_parallel_sum(pl.col("values")))
    .collect(engine="streaming"),
)
timeit(
    "Lazy streaming, parallel:",
    lambda: df.lazy()
    .select(parallel_sum(pl.col("values")))
    .collect(engine="streaming"),
)
