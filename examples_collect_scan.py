"""
Examples of ``collect_scan()``.
"""

import polars as pl

from polars_numba import collect_scan

#########################################################################
### Highest so far: Find the highest price seen at this point in time ###

df = pl.DataFrame(
    {
        "price": [20, 19, 21, 22, 23, 21, 20, 24, 25],
    }
)


def highest_so_far(highest_so_far, price):
    return max(highest_so_far, price)


series = collect_scan(df, 0, highest_so_far, pl.UInt64)
#         Original values: [20, 19, 21, 22, 23, 21, 20, 24, 25]
assert series.to_list() == [20, 20, 21, 22, 23, 23, 23, 24, 25]


#################################################
### Calculating a running credit card balance ###


def credit_card_balance(
    starting_balance: float, max_allowed_balance: float, attempted_purchases: pl.Series
) -> pl.Series:
    """
    Given a starting balance, a maximum allowed balance, and a series of
    attempted purchase amounts, return the resulting balances.

    Any purchase that takes the balance above the maximum allowed balance will
    be rejected.
    """

    def maybe_sum(current_balance, attempted_purchase, max_allowed_balance):
        new_balance = current_balance + attempted_purchase
        if new_balance <= max_allowed_balance:
            current_balance = new_balance
        return current_balance

    # For performance reasons changing a function's bound variable is not
    # allowed. So, we pass in the parameter by adding it as a column, and use a
    # LazyFrame for that so it doesn't have to be fully in memory.
    df = (
        pl.DataFrame({"attempted_purchase": attempted_purchases})
        .lazy()
        .with_columns(max_allowed_balance=max_allowed_balance)
    )
    return collect_scan(df, starting_balance, maybe_sum, pl.Float64)


attempted_purchases = pl.Series([900, 70, -400, 60])
balances = credit_card_balance(50, 1000, attempted_purchases)
# We expect the 70 purchase to be rejected because it will take the balance
# above the max credit limit of 1000.
#  Initially 50, purchases = [900, 70, -400,  60]
assert balances.to_list() == [950, 950, 550, 610]
