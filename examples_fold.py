"""
Examples of using ``collect_fold()``.
"""

from datetime import date

import polars as pl

from polars_numba import collect_fold

### Longest streak of days below freezing

# We have the daily min and max temperature, in Celsius:

df = pl.DataFrame(
    {
        "date": pl.date_range(date(2025, 1, 1), date(2025, 1, 9), eager=True),
        "min_temp": [-5, -6, -4, 0, -3, -1, -5, 0, -2],
        "max_temp": [2, 0, -2, 3, -2, -1, -4, 1, -1],
    }
)


# We want to know the longest numbers of days in a row that the temperature was
# zero or below:
def freezing_streak(acc, max_temp):
    (prev_max_streak, cur_days) = acc
    if max_temp <= 0:
        cur_days += 1
    else:
        cur_days = 0
    prev_max_streak = max(prev_max_streak, cur_days)
    return (prev_max_streak, cur_days)


streak, _ = collect_fold(df, (0, 0), freezing_streak, ["max_temp"])
assert streak == 3

# Since the argument name to the function is the same as the column name, we
# don't actually have to specificy column names:
streak, _ = collect_fold(df, (0, 0), freezing_streak)
assert streak == 3
