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

## Folding

A fold is a function that takes an accumulator and the values of specific columns.
It returns a new accumulator, which is passed on to the next call.
The final result is the result of the function.

### `Expr`-based folding

While not supporting streaming, so potentially with high memory usage, `Expr.plumba.fold()` can work with arbitrary `Expr`.
This provides lots of flexibility, e.g. it's usable with `group_by()`.
Just make sure to import the package so the custom namespace gets installed.

You can see examples in [`examples_fold.py`](examples_fold.py).

### Streaming folding with `collect_fold()`

Given a `DataFrame` or `LazyFrame`, you can use `polars_numba.collect_fold()` to run a fold on the data.
Data is processed in batches, using the streaming engine, so memory usage should be constrained.

You can see examples in [`examples_collect_fold.py`](examples_collect_fold.py).

## Scanning

The second set of APIs provided by `polars_numba` is scanning (in the functional programming usage).
A scan is a function that takes an accumulator and the values of specific columns.
It returns a new accumulator, which is used as the value for a row in the result and also passed on to the next call.
The final result is a `Series`, the result of all the scan calls.

### `Expr`-based scanning (TODO)

`Expr.numba.scan()` can work with any `Expr` to do a scan.

You can see examples in [`examples_scan.py`](examples_scan.py).

### Streaming scanning with `collect_scan()`

Given a `DataFrame` or `LazyFrame`, you can use `polars_numba.collect_scan()` to run a scan on the data.
Data is processed in batches, using the streaming engine, so memory usage should be constrained, though the final `Series` is fully in memory.

You can see examples in [`examples_collect_scan.py`](examples_collect_scan.py).

## General features

The functions passed in to folds or scans are compiled with Numba, so runtime should be fast.
The caveats:

* The first call for any combination of functions and types will need to be compiled, which adds some fixed overhead; still worth it for large amounts of data.
* Numba only implements a subset of Python, with some limitations, but it is still very flexible.
