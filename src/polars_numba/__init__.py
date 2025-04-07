"""
Proof-of-concept Polars integration.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

import polars as pl
from polars._typing import PolarsDataType
import awkward as ak
import awkward.numba  # register with Numba
from numba import jit

_ReturnType = TypeVar("_ReturnType")


def arrow_jit(
    returns_scalar: bool,
    return_dtype: PolarsDataType | None = None,
    is_elementwise: bool = False,
) -> Callable[[Callable[[pl.Series], _ReturnType]], Callable[[pl.Expr], pl.Expr]]:
    """
    Decorator factory that wrap a function with Numba.

    The final wrapped function takes a ``pl.Expr`` and calls ``map_batches()``
    on it with the original function accelerated by Numba.Expr()

    This would be slightly nicer if it were built-in to Polars, but probably
    not there yet.
    """

    def convert_args(
        series: pl.Series, params_info: list[tuple[bool, Any]]
    ) -> list[Any]:
        fields = series.struct.fields[:]
        result = []
        for is_scalar, maybe_value in params_info:
            if not is_scalar:
                # It's the next sub-Series in the struct:
                maybe_value = ak.from_arrow(
                    series.struct.field(fields.pop(0)).to_arrow()
                )
            result.append(maybe_value)
        return result

    def decorator(
        func: Callable[[pl.Series], _ReturnType]
    ) -> Callable[[pl.Expr], pl.Expr]:
        # Convert to Numba function:
        jit_func = jit(nogil=True)(func)

        if returns_scalar:

            def run_func(
                series: pl.Series, params_info: list[tuple[bool, Any]]
            ) -> _ReturnType:
                args = convert_args(series, params_info)
                return jit_func(*args)

        else:

            def run_func(
                series: pl.Series, params_info: list[tuple[bool, Any]]
            ) -> _ReturnType:
                args = convert_args(series, params_info)
                builder = ak.ArrayBuilder()
                args.append(builder)
                jit_func(*args)
                result = pl.from_arrow(
                    ak.to_arrow(builder.snapshot(), extensionarray=False)
                )
                # TODO do something with return_dtype?
                return result

        @wraps(func)
        def wrapped(expr: pl.Expr, *args: Any) -> pl.Expr:
            # If *args includes more Expr, we need to combine it with initial
            # Expr as a Struct, since you can only use a single Expr with
            # map_batches().
            params_info = [(False, None)]
            all_expr = [expr]
            for arg in args:
                if isinstance(arg, pl.Expr):
                    params_info.append((False, None))
                    all_expr.append(arg)
                else:
                    params_info.append((True, arg))
            expr = pl.struct(all_expr)
            return expr.map_batches(
                lambda series: run_func(series, params_info),
                return_dtype=return_dtype,
                returns_scalar=returns_scalar,
                is_elementwise=is_elementwise,
            )

        return wrapped

    return decorator
