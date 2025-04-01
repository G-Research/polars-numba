"""
Exposing Polars data to Numba is zero-copy, involving no extra memory usage.

Creating a new Series involves 2x memory usage at peak because of the ArrayBuilder API.

Note that the memory measurement technique may be underestimating peak memory usage...
"""

import psutil
import gc
import polars as pl
from polars_numba import arrow_jit

MB = 1_000_000


def get_memory():
    gc.collect()
    return psutil.Process().memory_full_info().uss // MB


# Series with two chunks:
df = pl.DataFrame({"v": pl.concat([pl.Series(range(25_000_000)) for _ in range(4)])})
print("DataFrame size in memory (MB):", df.estimated_size() // MB)


@arrow_jit(returns_scalar=True)
def mysum(arr):
    result = 0
    for value in arr:
        if value is not None:
            result += value
    return result


@arrow_jit(returns_scalar=False, is_elementwise=True)
def duplicate(arr, array_builder):
    for value in arr:
        if value is None:
            array_builder.null()
        else:
            array_builder.integer(value)


# Precompile the Numba functions:
small_df = pl.DataFrame({"v": pl.Series([1, 2, 3])})
assert small_df.select(mysum(pl.col("v"))).item() == 6
small_df.select(duplicate(pl.col("v")))

# Baseline memory usage:
mem_start = get_memory()
print("Baseline memory usage (MB):", mem_start)

# Pass Series to Arrow Numba JIT:
mysum(df.select(mysum(pl.col("v"))))

# Memory usage should be basically the same:
print(
    "Extra memory having called Numba with DataFrame -> scalar (MB):",
    (get_memory() - mem_start),
)
mem_start = get_memory()

# Create new Series via Arrow Numba JIT:
result = df.select(duplicate(pl.col("v")))

# Extra allocated memory usage is 2×?! I am confused by why this isn't reflected in the peak though.
print(
    "Extra memory from using Arrow Numba JIT to duplicate Series (MB):",
    (get_memory() - mem_start),
)


mem_start = get_memory()

# Create new Series via Arrow Numba JIT using lazy + streaming:
result3 = df.lazy().select(duplicate(pl.col("v"))).collect(engine="streaming")

# Extra allocated memory usage is 1×:
print(
    "Extra memory from from STREAMING is_elementwise=True using Arrow Numba JIT to duplicate Series (MB):",
    (get_memory() - mem_start),
)
