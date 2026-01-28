"""
Examples of using ``Expr.plumba.fold()``.
"""

import polars as pl
import polars_numba  # noqa: F401


#########################################################################
### Highest so far: Find the highest price seen at this point in time ###

df = pl.DataFrame(
    {
        "price": [20, 19, 21, 22, 23, 21, 20, 24, 25],
    }
)


def highest_so_far(highest_so_far, price):
    return max(highest_so_far, price)


result = df.select(pl.col("price").plumba.scan(highest_so_far, 0, pl.UInt64))
#                  Original values: [20, 19, 21, 22, 23, 21, 20, 24, 25]
assert result["price"].to_list() == [20, 20, 21, 22, 23, 23, 23, 24, 25]


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

    return attempted_purchases.plumba.scan(
        maybe_sum, 0.0, pl.Float64, extra_args=[max_allowed_balance]
    ).alias("balance")


df = pl.DataFrame({"attempted_purchases": [50, 900, 70, -400, 60]})
result = df.select(
    pl.col("attempted_purchases").pipe(credit_card_balance, max_allowed_balance=1000)
)["balance"]
# We expect the 70 purchase to be rejected because it will take the balance
# above 1000, so we get 950 twice:
assert result.to_list() == [50, 950, 950, 550, 610], result


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
    "balance": [[50, 950, 950, 550, 610], [17.0, 17.5]],
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

    return attempted_purchase_prices_and_amounts.plumba.scan(
        maybe_sum,
        (0.0, 0.0),
        pl.Array(pl.Float64, 2),
        extra_args=[max_allowed_balance, max_allowed_units],
    ).alias("balance_and_units")


df = pl.DataFrame(
    {
        "attempted_purchase_prices": [5, 400, 70, 4, 60],
        "attempted_purchase_units": [20, 2, 2, 10, 1],
    }
)
result = (
    df.select(
        pl.struct("attempted_purchase_prices", "attempted_purchase_units").pipe(
            purchase_order_balance, max_allowed_balance=1000, max_allowed_units=25
        )
    )["balance_and_units"]
    .arr.to_struct(["balance", "bought_units"])
    .struct.unnest()
)
assert result["balance"].to_list() == [100, 900, 900, 900, 960]
assert result["bought_units"].to_list() == [20, 22, 22, 22, 23]
