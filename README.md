# polars-numba: Easily extending Polars

Polars has many built-in functions.
If none of them meet your needs, you can:

* Write an extension in Rust, which is not great for experimentation or one-off functions.
* Write a user-defined function in Python or maybe Numba, which is a pretty low-level API, without much concern for memory usage.

This package is a (tested, so hopefully usable) prototype of a third alternative, providing higher-level APIs that:

* Can be easily extended by writing simple Python functions (that then get compiled to Numba).
* Are fast.
* Use little memory.

How to install:

```
$ pip install git+https://github.com/g-research/polars-numba.git
```

Or add package `git+https://github.com/g-research/polars-numba.git` as a dependency to your `requirements.txt`/`pyproject.toml`/etc..

## Folding with `collect_fold()`

The first API provided by `polars_numba` is folding.

Given a `DataFrame` or `LazyFrame`, you can use `polars_numba.collect_fold()` to run a fold on the data.
A fold is a function that takes an accumulator and the values of specific columns.
It returns a new accumulator, which is passed on to the next call.
The final result is the result of the function.

Data is processed in batches, using the streaming engine, so memory usage should be constrained.
The passed in fold function is compiled with Numba, so runtime should be fast (though the first call for any combination of functions and types will need to be compiled, which adds some fixed overhead).

You can see examples in [`examples_fold.py`](examples_fold.py).

## Scanning with `collect_scan()`

The second API provided by `polars_numba` is scanning (in the functional programming usage).

Given a `DataFrame` or `LazyFrame`, you can use `polars_numba.collect_scan()` to run a scan on the data.
A scan is a function that takes an accumulator and the values of specific columns.
It returns a new accumulator, which is used as the value for a row in the result and also passed on to the next call.
The final result is a `Series`, the result of all the scan calls.

Data is processed in batches, using the streaming engine, so memory usage should be constrained, though the final `Series` is fully in memory.
As with the fold, the passed in scan function is compiled with Numba.

You can see examples in [`examples_scan.py`](examples_scan.py).
