import numpy as np
import pandas as pd


def calculate_buy_and_hold_portfolio(
    prices,
    weights,
    initial_investment=100000
):
    if prices.empty:
        raise ValueError(
            "Price data cannot be empty."
        )

    if not isinstance(weights, dict):
        raise TypeError(
            "Weights must be provided as a dictionary."
        )

    if set(weights.keys()) != set(prices.columns):
        raise ValueError(
            "Weight tickers must match price columns."
        )

    if any(
        weight < 0
        for weight in weights.values()
    ):
        raise ValueError(
            "Portfolio weights cannot be negative."
        )

    if not np.isclose(
        sum(weights.values()),
        1.0
    ):
        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    if initial_investment <= 0:
        raise ValueError(
            "Initial investment must be greater than zero."
        )

    if prices.isna().any().any():
        raise ValueError(
            "Price data cannot contain missing values."
        )

    if (prices <= 0).any().any():
        raise ValueError(
            "All prices must be greater than zero."
        )

    weight_series = (
        pd.Series(weights, dtype=float)
        .reindex(prices.columns)
    )

    initial_prices = prices.iloc[0]

    initial_allocations = (
        weight_series * initial_investment
    )

    shares = (
        initial_allocations / initial_prices
    )

    shares.name = "Shares"

    asset_values = prices.mul(
        shares,
        axis="columns"
    )

    portfolio_value = asset_values.sum(axis=1)

    portfolio_value.name = (
        "Buy and Hold Portfolio Value"
    )

    portfolio_returns = (
        portfolio_value
        .pct_change(fill_method=None)
        .dropna()
    )

    portfolio_returns.name = (
        "Buy and Hold Portfolio Return"
    )

    weights_over_time = asset_values.div(
        portfolio_value,
        axis="index"
    )

    return {
        "shares": shares,
        "asset_values": asset_values,
        "portfolio_value": portfolio_value,
        "portfolio_returns": portfolio_returns,
        "weights_over_time": weights_over_time
    }