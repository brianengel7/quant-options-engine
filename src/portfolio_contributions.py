import numpy as np
import pandas as pd

def calculate_return_contributions(
    asset_returns,
    weights,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError(
            "Asset returns cannot be empty."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    weight_series = pd.Series(weights, dtype=float)

    missing_assets = set(asset_returns.columns) - set(
        weight_series.index
    )

    if missing_assets:
        raise ValueError(
            f"Missing weights for: {sorted(missing_assets)}"
        )

    weight_series = weight_series.reindex(
        asset_returns.columns
    )

    if not np.isclose(weight_series.sum(), 1.0):
        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    annualized_asset_returns = (
        asset_returns.mean() * trading_days
    )

    return_contributions = (
        weight_series * annualized_asset_returns
    )

    return_contributions.name = "Return Contribution"

    return return_contributions

def calculate_volatility_contributions(
    asset_returns,
    weights,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError(
            "Asset returns cannot be empty."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    weight_series = pd.Series(weights, dtype=float)

    missing_assets = set(asset_returns.columns) - set(
        weight_series.index
    )

    if missing_assets:
        raise ValueError(
            f"Missing weights for: {sorted(missing_assets)}"
        )

    weight_series = weight_series.reindex(
        asset_returns.columns
    )

    if not np.isclose(weight_series.sum(), 1.0):
        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    annualized_covariance = (
        asset_returns.cov() * trading_days
    )

    portfolio_variance = (
        weight_series.to_numpy()
        @ annualized_covariance.to_numpy()
        @ weight_series.to_numpy()
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    if np.isclose(portfolio_volatility, 0):
        raise ValueError(
            "Volatility contributions are undefined "
            "when portfolio volatility is zero."
        )

    marginal_contributions = (
        annualized_covariance
        @ weight_series
        / portfolio_volatility
    )

    volatility_contributions = (
        weight_series * marginal_contributions
    )

    percentage_contributions = (
        volatility_contributions
        / portfolio_volatility
    )

    results = pd.DataFrame({
        "Weight": weight_series,
        "Volatility Contribution": volatility_contributions,
        "Percentage of Portfolio Risk": percentage_contributions
    })

    return results