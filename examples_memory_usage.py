"""
Exposing Polars data to Numba is zero-copy, involving no extra memory usage.

Creating a new Series involves 2x memory usage at peak because of the ArrayBuilder API.
"""

from resource import getrusage, RUSAGE_SELF

import psutil
import polars as pl
from polars_numba import arrow_jit

MB = 1_000_000

def get_memory():
    return psutil.Process().memory_full_info().uss


def get_peak_memory():
    return getrusage(RUSAGE_SELF).ru_maxrss * 1024  # kb on Linux


# Series with two chunks:
df = pl.DataFrame(
    {"v": pl.concat([pl.Series(range(50_000_000)), pl.Series(range(50_000_000))])}
)
print("DataFrame size in memory (MB):", df.estimated_size() // MB)

@arrow_jit(returns_scalar=True)
def mysum(arr):
    result = 0
    for value in arr:
        if value is not None:
            result += value
    return result

@arrow_jit(returns_scalar=False)
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
print("Baseline memory usage (MB):", mem_start // MB)

# Pass Series to Arrow Numba JIT:
mysum(df.select(mysum(pl.col("v"))))

# Memory usage should be basically the same:
print("Extra memory having called Numba with DataFrame -> scalar (MB):", (get_memory() - mem_start) / MB)
mem_start = get_memory()
peak_start = get_peak_memory()

# Create new Series via Arrow Numba JIT:
result = df.select(duplicate(pl.col("v")))

# Extra peak allocated memroy is ...:
print("Extra PEAK memory from using Arrow Numba JIT to duplicate Series (MB):", (get_peak_memory() - peak_start) // MB)

# Extra allocated memory usage is 1×:
print("Extra FINAL memory from using Arrow Numba JIT to duplicate Series (MB):", (get_memory() - mem_start) // MB)
