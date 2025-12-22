"""
Examples of using ``collect_fold()``.
"""

from datetime import date

import polars as pl

from polars_numba import collect_fold

#############################################
### Longest streak of days below freezing ###

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


streak, _ = collect_fold(df, freezing_streak, (0, 0), column_names=["max_temp"])
assert streak == 3

# Instead of passing column names, can also just limit the DataFrame to those
# columns that the function expects:
streak, _ = collect_fold(df.select_seq(["max_temp"]), freezing_streak, (0, 0))
assert streak == 3


#########################################
### Calculating a credit card balance ###


def credit_card_balance(
    starting_balance: float, max_allowed_balance: float, attempted_purchases: pl.Series
) -> float:
    """
    Given a starting balance, a maximum allowed balance, and a series of
    attempted purchase amounts, return the final balance.

    Any purchase that takes the balance above the maximum allowed balance will
    be rejected.
    """

    def maybe_sum(current_balance, max_allowed_balance, attempted_purchase):
        new_balance = current_balance + attempted_purchase
        if new_balance <= max_allowed_balance:
            current_balance = new_balance
        return current_balance

    # For performance reasons changing a function's bound variable is not
    # allowed. So, we pass in the parameters using extra_args. These get passed
    # in right after the accumulator.
    df = pl.DataFrame({"attempted_purchase": attempted_purchases})
    return collect_fold(
        df, maybe_sum, starting_balance, extra_args=[max_allowed_balance]
    )


attempted_purchases = pl.Series([900, 70, -400, 60])
final_balance = credit_card_balance(50, 1000, attempted_purchases)
# We expect the 70 purchase to be rejected because it will take the balance
# above 1000:
assert final_balance == 50 + 900 - 400 + 60
