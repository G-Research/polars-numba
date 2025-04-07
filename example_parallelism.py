"""
Lack of multi-threading.

How it works:

    - @arrow_jit is a wrapper around Polars' ``Expr.map_batches()``.

    - When ``map_batches()`` has ``is_elementwise=True, returns_scalar=False``,
      the passed in function receives batches rather than the whole series when
      in streaming mode.

    - We can therefore combine a two pass operation to get chunked operations,
      which then get run in the thread pool in parallel.

One would hope streaming would enable using multiple threads, but I think this
is prevented by a bug (https://github.com/pola-rs/polars/issues/22160).
"""

from time import time, process_time

import polars as pl
from polars_numba import arrow_jit


def timeit(prefix, f, count=100):
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


@arrow_jit(returns_scalar=True)
def not_parallel_sum(arr):
    result = 0
    for value in arr:
        if value is not None:
            result += value
    return result


# is_elementwise means we won't always get the full Series, we might get chunks
# in some cases.
@arrow_jit(returns_scalar=False, is_elementwise=True)
def sum_chunk(arr, array_builder):
    result = 0
    for value in arr:
        if value is not None:
            result += value
    array_builder.integer(result)


def parallel_sum(column: pl.Expr) -> pl.Expr:
    # First, do sum of chunks, which will result in a Series of patial sums:
    partial_sums = sum_chunk(column)
    # Then do sum of those:
    return not_parallel_sum(partial_sums)


df = pl.DataFrame({"values": range(10_000_000)})

print(df.select(parallel_sum(pl.col("values"))))
# Wierdly doing just this, and not the above, results in Numba issues?!
print(df.lazy().select(parallel_sum(pl.col("values"))).collect(engine="streaming"))

# Check correctness
assert (
    df.select(not_parallel_sum(pl.col("values"))).item()
    == df.lazy()
    .select(parallel_sum(pl.col("values")))
    .collect(engine="streaming")
    .item()
)

timeit("Eager, not_parallel:", lambda: df.select(not_parallel_sum(pl.col("values"))))
timeit("Eager, parallel:", lambda: df.select(parallel_sum(pl.col("values"))))
timeit("Eager, builtin:", lambda: df.select(pl.col("values").sum()))
timeit(
    "Lazy, not_parallel:",
    lambda: df.lazy().select(not_parallel_sum(pl.col("values"))).collect(),
)
timeit(
    "Lazy, parallel:",
    lambda: df.lazy().select(parallel_sum(pl.col("values"))).collect(),
)
timeit(
    "Lazy, builtin:",
    lambda: df.lazy().select(pl.col("values").sum()).collect(),
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
timeit(
    "Lazy streaming, builtin:",
    lambda: df.lazy().select(pl.col("values").sum()).collect(engine="streaming"),
)
