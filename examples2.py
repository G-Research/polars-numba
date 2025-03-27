"""Examples of more extended data types."""

from datetime import datetime

import polars as pl
import numpy as np
from polars_numba import arrow_jit


### Example #1: Datetimes
#
# Datetimes are exposed as np.datetime64, which is probably not ideal, e.g.
# they don't support timezones.
#
# Note: I can't figure out how to create timedelta64 objects within Numba,
# either.
#
# In general time-related types probably need a lot of work.
@arrow_jit(returns_scalar=False)
def add_days(arr, delta, array_builder):
    for value in arr:
        if value is not None:
            array_builder.datetime(value + delta)


df = pl.DataFrame(
    {"dates": [datetime(2025, 3, 17, 18, 42, 36), datetime(2024, 2, 28, 11, 45, 32)]}
)

print("Original:", df)
days = np.timedelta64(3, "D")
print("Added 3 days:", df.select(add_days(pl.col("dates"), days)))


### Example #2: Lists
@arrow_jit(returns_scalar=False)
def multiply_list_values_by_2(arr, array_builder):
    for list_value in arr:
        if list_value is None:
            array_builder.null()
        else:
            array_builder.begin_list()
            for value in list_value:
                if value is None:
                    array_builder.null()
                else:
                    array_builder.integer(value * 2)
            array_builder.end_list()


df = pl.DataFrame({"lists": [[2, 3], [4], [None, 5], None]})

print("Original:", df)
print("Multiply by 2:", df.select(multiply_list_values_by_2(pl.col("lists"))))


### Example #3: Structs
@arrow_jit(returns_scalar=False)
def add_x_and_y(arr, array_builder):
    for struct in arr:
        if struct is None:
            array_builder.null()
            continue
        if struct.x is None or struct.y is None:
            array_builder.null()
        else:
            array_builder.integer(struct.x + struct.y)


df = pl.DataFrame(
    {"structs": [{"x": 10, "y": 3}, {"x": 20, "y": None}, {"x": None, "y": 2}, None]}
)

print("Original:", df)
print("Add x and y:", df.select(add_x_and_y(pl.col("structs"))))
