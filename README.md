# polars-numba: Write Numba functions for Polars

While Polars does have support for normal Numba functions, these are not ideal: Polars native data type is Arrow, but Numba by default works on NumPy arrays.
As a result, using Numba:

1. Requires converting Arrow columns to NumPy arrays, and back, which can increase memory usage.
2. Does not support missing data correctly, since that concept doesn't exist in NumPy.
   This is especially a problem for operations that aren't just one-value-at-a-time, for a example a `max()` or windowed function can give the wrong results.
3. Supports fewer data types, compared to Arrow's more complex support.

This package aims to fix that, by integrating Numba support for Arrow with Polars, building on the Numba support from [the Awkward Array project](https://awkward-array.org).

## Memory usage

Awkward Array's memory representation is compatible with Arrow.
When doing read-only operations, then, it requires zero additional memory (zero-copy) because it can just use the Arrow data structure directly, with no copying.

When _creating_ a new Series, however, peak memory usage may be twice as high as the resulting Series, perhaps because of the temporary `ArrayBuilder` object required by the current Awkward Array API.
This ought to be fixable with a new API.

## Current status: proof-of-concept

Currently this is just enough code for users to try it out and provide feedback.
Given sufficient positive feedback we will turn this into a real package on PyPI, do supporting work in other packages if necessary (e.g. improvements to Awkward Array's Numba code), etc.

How to install:

```
$ pip install git+https://github.com/g-research/polars-numba.git
```

Or add package `git+https://github.com/g-research/polars-numba.git` as a dependency to your `requirements.txt`/`pyproject.toml`/etc..

## Using `polars-numba`

Again, this is just a proof-of-concept, but you can see usage examples:

* [`examples.py`](examples.py) shows the basic API.
* [`examples2.py`](examples2.py) has examples with datetimes, lists, and structs.

To see API documentation on how to create complex results, see the [API documentation for ArrayBuilder](https://awkward-array.org/doc/main/reference/generated/ak.ArrayBuilder.html).

### Chunking vs whole series

In most cases, the function called with `@arrow_jit` will be called with the _full_ Series.
The only exception is when you explicitly opted-in to receiving partial batches of the `Series`, by using `@arrow_jit(is_elementwise=True, returns_scalar=False)`; both options are necessary.
(The `is_elementwise` parameter is passed to [`map_batches()`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_batches.html).)

See [`example_chunking.py`](example_chunking.py) and look at its output to get a sense of what happens.
