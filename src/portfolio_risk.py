import numpy as np
import pandas as pd

from scipy.stats import norm

def calculate_historical_var(
    portfolio_returns,
    confidence_level=0.95
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    tail_probability = 1 - confidence_level

    return_quantile = portfolio_returns.quantile(
        tail_probability
    )

    historical_var = max(-return_quantile, 0.0)

    return historical_var

def calculate_dollar_var(
    portfolio_var,
    portfolio_value
):
    if portfolio_var < 0:
        raise ValueError(
            "Portfolio VaR cannot be negative."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be greater than zero."
        )

    dollar_var = portfolio_var * portfolio_value

    return dollar_var

def calculate_historical_expected_shortfall(
    portfolio_returns,
    confidence_level=0.95
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    tail_probability = 1 - confidence_level

    return_quantile = portfolio_returns.quantile(
        tail_probability
    )

    tail_returns = portfolio_returns[
        portfolio_returns <= return_quantile
    ]

    if tail_returns.empty:
        raise ValueError(
            "No returns were found beyond the VaR threshold."
        )

    expected_shortfall = max(
        -tail_returns.mean(),
        0.0
    )

    return expected_shortfall

def calculate_dollar_expected_shortfall(
    expected_shortfall,
    portfolio_value
):
    if expected_shortfall < 0:
        raise ValueError(
            "Expected shortfall cannot be negative."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be greater than zero."
        )

    dollar_expected_shortfall = (
        expected_shortfall * portfolio_value
    )

    return dollar_expected_shortfall

def calculate_parametric_var(
    portfolio_returns,
    confidence_level=0.95
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    if np.isclose(volatility, 0):
        raise ValueError(
            "Parametric VaR is undefined when volatility is zero."
        )

    tail_probability = 1 - confidence_level

    z_score = norm.ppf(tail_probability)

    return_threshold = (
        mean_return + z_score * volatility
    )

    parametric_var = max(
        -return_threshold,
        0.0
    )

    return parametric_var

def calculate_parametric_expected_shortfall(
    portfolio_returns,
    confidence_level=0.95
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    if np.isclose(volatility, 0):
        raise ValueError(
            "Parametric Expected Shortfall is undefined "
            "when volatility is zero."
        )

    tail_probability = 1 - confidence_level

    z_score = norm.ppf(tail_probability)

    expected_shortfall = (
        -mean_return
        + volatility
        * norm.pdf(z_score)
        / tail_probability
    )

    expected_shortfall = max(
        expected_shortfall,
        0.0
    )

    return expected_shortfall

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

