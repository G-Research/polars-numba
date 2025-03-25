# polars-numba: Write Numba functions for Polars

While Polars does have support for normal Numba functions, these are not ideal: Polars native data type is Arrow, but Numba by default works on NumPy Arrows.
As a result, using Numba:

1. Requires converting Arrow columns to NumPy arrows, and back.
2. Does not support missing data correctly, since that concept doesn't exist in NumPy.
   This is especially a problem for operations that aren't just one-value-at-a-time, for a example a `max()` or windowed function can give the wrong results.
3. Supports fewer data types, compared to Arrow's more complex support.

This package aims to fix that, by integrating Numba support for Arrow with Polars, building on the Numba support from [the Awkward Array project](https://awkward-array.org).
Awkward Array's memory representation is compatible with Arrow.

## Current status: proof-of-concept

Currently this is just enough code for users to try it out and provide feedback.
Given sufficient positive feedback we will turn this into a real package on PyPI, do supporting work in other packages if necessary (e.g. improvements to Awkward Array's Numba code), etc.

How to install:

```
$ pip install git+https://github.com/g-research/polars-numba.git
```

Or add package `git+https://github.com/g-research/polars-numba.git` as a dependency to your `requirements.txt`/`pyproject.toml`/etc..

## Using `polars-numba`

Again, this is just a proof-of-concept, but you can see usage examples in [`examples.py`](`examples.py`).
