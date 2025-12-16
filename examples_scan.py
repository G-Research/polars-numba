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
