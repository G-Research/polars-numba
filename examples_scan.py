"""
Examples of ``collect_scan()``.
"""

import polars as pl

from polars_numba import collect_scan


### Highest so far: Find the highest price seen at this point in time.

df = pl.DataFrame(
    {
        "price": [20, 19, 21, 22, 23, 21, 20, 24, 25],
    }
)


def highest_so_far(highest_so_far, price):
    return max(highest_so_far, price)


series = collect_scan(df, 0, highest_so_far, pl.UInt64)
assert series.to_list() == [20, 20, 21, 22, 23, 23, 23, 24, 25]
