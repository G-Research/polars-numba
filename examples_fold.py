"""
Examples of using ``Expr.plumba.fold()``.
"""

from datetime import date

import polars as pl
import polars_numba  # noqa: F401


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
def freezing_streak(expr: pl.Expr) -> pl.Expr:
    def impl(acc, max_temp):
        (prev_max_streak, cur_days) = acc
        if max_temp <= 0:
            cur_days += 1
        else:
            cur_days = 0
        prev_max_streak = max(prev_max_streak, cur_days)
        return (prev_max_streak, cur_days)

    return expr.plumba.fold(impl, (0, 0), pl.Array(pl.Int64, 2)).arr.first()


streak = df.select(pl.col("max_temp").pipe(freezing_streak)).item()
assert streak == 3


#########################################
### Calculating a credit card balance ###


def credit_card_balance(
    attempted_purchases: pl.Expr, max_allowed_balance: float
) -> pl.Expr:
    """
    Given a series of attempted purchase amounts, a starting balance, and a
    maximum allowed balance, return the final balance.

    Any purchase that takes the balance above the maximum allowed balance will
    be rejected.
    """

    def maybe_sum(current_balance, max_allowed_balance, attempted_purchase):
        new_balance = current_balance + attempted_purchase
        if new_balance <= max_allowed_balance:
            current_balance = new_balance
        return current_balance

    return attempted_purchases.plumba.fold(
        maybe_sum, 0.0, pl.Float64, extra_args=[max_allowed_balance]
    ).alias("balance")


df = pl.DataFrame({"attempted_purchases": [50, 900, 70, -400, 60]})
final_balance = df.select(
    pl.col("attempted_purchases").pipe(credit_card_balance, max_allowed_balance=1000)
).item()
# We expect the 70 purchase to be rejected because it will take the balance
# above 1000:
assert final_balance == 50 + 900 - 400 + 60


###################################################
### Calculating a credit card balance, per user ###

df = pl.DataFrame(
    {
        "user": ["bob", "alice", "alice", "alice", "alice", "alice", "bob"],
        "attempted_purchase": [17.0, 50.0, 900.0, 70.0, -400.0, 60.0, 0.5],
    }
)
final_balance = (
    df.group_by("user")
    .agg(
        pl.col("attempted_purchase").pipe(credit_card_balance, max_allowed_balance=1000)
    )
    .sort("user")
)
assert final_balance.to_dict(as_series=False) == {
    "user": ["alice", "bob"],
    "balance": [610.0, 17.5],
}


###########################################################
## Multiple inputs and multiple outputs                  ##
## Inputs: Purchasing limit is both amount and units     ##
## Outputs: Balance and number of purchased units        ##

def purchase_order_balance(
    attempted_purchase_prices_and_amounts: pl.Expr,
    max_allowed_balance: float,
    max_allowed_units: int,
) -> pl.Expr:
    """
    Any purchase that takes the balance above the maximum allowed balance, or
    cumulative number of units above the maximum, will be totally rejected.
    """

    def maybe_sum(
        acc,
        max_allowed_balance,
        max_allowed_units,
        attempted_purchase_price,
        attempted_purchase_units,
    ):
        (current_balance, current_bought) = acc
        new_balance = current_balance + (
            attempted_purchase_price * attempted_purchase_units
        )
        new_units = current_bought + attempted_purchase_units
        if new_balance <= max_allowed_balance and new_units <= max_allowed_units:
            current_balance = new_balance
            current_bought = new_units
        return (current_balance, current_bought)

    return attempted_purchase_prices_and_amounts.plumba.fold(
        maybe_sum,
        (0.0, 0.0),
        return_dtype=pl.Array(pl.Float64, 2),
        extra_args=[max_allowed_balance, max_allowed_units],
    )


df = pl.DataFrame(
    {
        "attempted_purchase_prices": [5.0, 400.0, 70.0, 4.0, 60.0],
        "attempted_purchase_units": [20.0, 2.0, 2.0, 10.0, 1.0],
    }
)
result = df.select(
    pl.struct("attempted_purchase_prices", "attempted_purchase_units").pipe(
        purchase_order_balance, max_allowed_balance=1000, max_allowed_units=25
    )
).item()
final_balance, final_purchased_units = result
assert final_balance == 960
assert final_purchased_units == 23

