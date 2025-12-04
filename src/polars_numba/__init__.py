"""
Higher-level programmable Polars APIs, using Numba.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

import polars as pl
from polars._typing import PolarsDataType
from numba import jit

