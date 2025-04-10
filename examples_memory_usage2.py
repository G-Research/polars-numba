"""
Memory usage can be high when reading e.g. using scan_csv(), since the whole
Series will need to be materialized.  We also test another API that fixes this.
"""
import resource
import polars as pl
from polars_numba import arrow_jit


def get_peak_memory():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


PATH = "/tmp/out.csv"
with open(PATH, "w") as f:
    f.write("col\n")
    for i in range(20_000_000):
        f.write(str(i) + "\n")

@arrow_jit(returns_scalar=False, return_dtype=pl.Int64())
def lengths(arr, array_builder):
    result = arr[0]
    for _ in range(len(arr)):
        array_builder.integer(result)

# precompile
pl.DataFrame({"col": pl.Series([1, 2, 3])}).lazy().select(lengths(pl.col("col"))).collect()

df = pl.scan_csv(PATH)
current_memory = get_peak_memory()
result = df.select(lengths(pl.col("col"))).collect(engine="streaming")
#result = df.select(pl.col("col") + 1).collect(engine="streaming")
print("Numba summing:", get_peak_memory() - current_memory, "extra peak MB")
print("Batches:", len(result.get_column("col").unique()))
