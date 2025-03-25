import polars as pl
from polars_numba import arrow_jit

# Create an example DataFrame; notice some data can be missing:
df = pl.DataFrame(
    {
        "values": [17, 2, None, 5, 6, 9, 13, None],
        "values2": [3, None, 4, 1, 8, 0, -2, 5],
    }
)

### Example #1: Given a Polars Series, return a scalar result.
#
# By using @arrow_jit(), the code will be compiled via Numba, and run quickly.
#
# We set returns_scalar=True to indicate result is a scalar (i.e. not a
# Series).
#
# The argument `arr` is the Polars Series, accessible the way a list or NumPy
# array would be, via indexing.
@arrow_jit(returns_scalar=True)
def make_scalar(arr):
    result = 0.0
    for i in range(len(arr)):
        value = arr[i]
        if value is not None:
            result += value * 0.7
    return result

# We can use the make_scalar() function instead `df.select()` and similar
# functions; it retuns an expression.
print(df.select(make_scalar(pl.col("values"))))


### Example #2: Given a Polars Series, return a new Series.
#
# Again, we use @arrow_jit(), but set `returns_scalar=False` to indicate we're
# not returning a scalar, but rather a Series. When this is set to False, an
# extra argument is passed to the function automatically, the `array_builder`.
# This is an Awkward Array ArrayBuilder
# (https://awkward-array.org/doc/main/reference/generated/ak.ArrayBuilder.html).
#
# To add values to the result Series, we call methods on the passed in
# ArrayBuilder, in this case `null()` to add a null value, and `integer()` to
# add an integer.
#
# Future versions of this library might have a nicer API for the common case
# where the returned Series is the same length as the input Series.
@arrow_jit(returns_scalar=False)
def make_series(arr, array_builder):
    for i in range(len(arr)):
        value = arr[i]
        if value is None:
            array_builder.null()
        else:
            array_builder.integer(value + 1)

print(df.select(make_series(pl.col("values"))))


### Example #3: A function with additional arguments.
#
# This function takes three arguments:
#
# 1. arr: the input Series we're operating on.
# 2. scalar: some scalar that the user of the function passes in.
# 3. array_builder: passed in automatically by @arrow_jit() because we set
#    returns_scalar=False.
@arrow_jit(returns_scalar=False)
def add_scalar_to_series(arr, scalar, array_builder):
    for i in range(len(arr)):
        value = arr[i]
        if value is None:
            array_builder.null()
        else:
            array_builder.integer(value + scalar)

print(df.select(add_scalar_to_series(pl.col("values"), 100)))


### Example #4: An operation that takes two different Series and returns a
### Series as the result.
@arrow_jit(return_dtype=None, returns_scalar=False)
def add_two_series(arr, arr2, array_builder):
    for i in range(len(arr)):
        value = arr[i]
        value2 = arr2[i]
        if value is None or value2 is None:
            array_builder.null()
        else:
            array_builder.integer(10 * value + value2)

print(df.select(add_two_series(pl.col("values"), pl.col("values2") + 1)))
