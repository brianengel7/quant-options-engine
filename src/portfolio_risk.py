import numpy as np
import pandas as pd


def calculate_correlated_monte_carlo_risk(
    asset_returns,
    weights,
    confidence_level=0.95,
    number_of_simulations=10000,
    horizon_days=1,
    rebalancing_mode="daily",
    random_seed=None
):
    if asset_returns.empty:
        raise ValueError(
            "Asset returns cannot be empty."
        )

    if not isinstance(weights, dict):
        raise TypeError(
            "Weights must be provided as a dictionary."
        )

    if set(weights.keys()) != set(
        asset_returns.columns
    ):
        raise ValueError(
            "Weight tickers must match asset-return columns."
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

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    if number_of_simulations <= 0:
        raise ValueError(
            "Number of simulations must be greater than zero."
        )

    if horizon_days <= 0:
        raise ValueError(
            "Horizon days must be greater than zero."
        )

    valid_modes = {
        "daily",
        "buy_and_hold"
    }

    if rebalancing_mode not in valid_modes:
        raise ValueError(
            "Rebalancing mode must be "
            "'daily' or 'buy_and_hold'."
        )

    weight_series = (
        pd.Series(weights, dtype=float)
        .reindex(asset_returns.columns)
    )

    weight_array = weight_series.to_numpy()

    daily_mean_returns = (
        asset_returns.mean().to_numpy()
    )

    daily_covariance = (
        asset_returns.cov().to_numpy()
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    simulated_daily_asset_returns = (
        random_generator.multivariate_normal(
            mean=daily_mean_returns,
            cov=daily_covariance,
            size=(
                number_of_simulations,
                horizon_days
            ),
            check_valid="raise"
        )
    )

    if rebalancing_mode == "daily":
        simulated_daily_portfolio_returns = (
            simulated_daily_asset_returns
            @ weight_array
        )

        simulated_portfolio_returns = (
            np.prod(
                1 + simulated_daily_portfolio_returns,
                axis=1
            ) - 1
        )

    else:
        simulated_asset_growth = np.prod(
            1 + simulated_daily_asset_returns,
            axis=1
        )

        simulated_portfolio_returns = (
            simulated_asset_growth
            @ weight_array
            - 1
        )

    tail_probability = (
        1 - confidence_level
    )

    return_threshold = np.quantile(
        simulated_portfolio_returns,
        tail_probability
    )

    monte_carlo_var = max(
        -return_threshold,
        0.0
    )

    tail_returns = simulated_portfolio_returns[
        simulated_portfolio_returns
        <= return_threshold
    ]

    if len(tail_returns) == 0:
        raise ValueError(
            "No simulated returns were found "
            "beyond the VaR threshold."
        )

    monte_carlo_expected_shortfall = max(
        -tail_returns.mean(),
        0.0
    )

    return {
        "var": monte_carlo_var,
        "expected_shortfall": (
            monte_carlo_expected_shortfall
        ),
        "return_threshold": return_threshold,
        "simulated_portfolio_returns": (
            simulated_portfolio_returns
        ),
        "simulated_daily_asset_returns": (
            simulated_daily_asset_returns
        )
    }