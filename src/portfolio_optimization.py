import numpy as np
import pandas as pd
from scipy.optimize import minimize

def _create_weight_bounds(
    number_of_assets,
    max_weight
):
    if not 0 < max_weight <= 1:
        raise ValueError(
            "Maximum weight must be between 0 and 1."
        )

    if number_of_assets * max_weight < 1:
        raise ValueError(
            "Maximum weight is too low for the "
            "number of portfolio assets."
        )

    return tuple(
        (0.0, max_weight)
        for _ in range(number_of_assets)
    )

def calculate_portfolio_statistics(
    weights,
    expected_returns,
    covariance_matrix,
    risk_free_rate=0.0
):
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(
        expected_returns,
        dtype=float
    )
    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float
    )

    portfolio_return = (
        weights @ expected_returns
    )

    portfolio_variance = (
        weights
        @ covariance_matrix
        @ weights
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    if np.isclose(portfolio_volatility, 0):
        raise ValueError(
            "Portfolio volatility cannot be zero."
        )

    sharpe_ratio = (
        portfolio_return - risk_free_rate
    ) / portfolio_volatility

    return {
        "return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio
    }


def optimize_minimum_volatility(
    asset_returns,
    risk_free_rate=0.0,
    max_weight=1.0,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError(
            "Asset returns cannot be empty."
        )

    if asset_returns.shape[1] < 2:
        raise ValueError(
            "Optimization requires at least two assets."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    if risk_free_rate <= -1:
        raise ValueError(
            "Risk-free rate must be greater than -100%."
        )

    expected_returns = (
        asset_returns.mean() * trading_days
    )

    covariance_matrix = (
        asset_returns.cov() * trading_days
    )

    number_of_assets = len(
        asset_returns.columns
    )

    initial_weights = np.full(
        number_of_assets,
        1 / number_of_assets
    )

    bounds = _create_weight_bounds(
        number_of_assets,
        max_weight
    )

    constraints = {
        "type": "eq",
        "fun": lambda candidate_weights:
            np.sum(candidate_weights) - 1.0
    }

    def portfolio_volatility(candidate_weights):
        portfolio_variance = (
            candidate_weights
            @ covariance_matrix.to_numpy()
            @ candidate_weights
        )

        return np.sqrt(portfolio_variance)

    optimization_result = minimize(
        fun=portfolio_volatility,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not optimization_result.success:
        raise RuntimeError(
            "Minimum-volatility optimization failed: "
            f"{optimization_result.message}"
        )

    optimal_weight_array = np.clip(
        optimization_result.x,
        0.0,
        1.0
    )

    optimal_weight_array = (
        optimal_weight_array
        / optimal_weight_array.sum()
    )

    optimal_weights = pd.Series(
        optimal_weight_array,
        index=asset_returns.columns,
        name="Minimum Volatility Weight"
    )

    statistics = calculate_portfolio_statistics(
        optimal_weight_array,
        expected_returns.to_numpy(),
        covariance_matrix.to_numpy(),
        risk_free_rate=risk_free_rate
    )

    return {
        "weights": optimal_weights,
        "expected_return": statistics["return"],
        "volatility": statistics["volatility"],
        "sharpe_ratio": statistics["sharpe_ratio"]
    }

def optimize_maximum_sharpe(
    asset_returns,
    risk_free_rate=0.0,
    max_weight=1.0,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError(
            "Asset returns cannot be empty."
        )

    if asset_returns.shape[1] < 2:
        raise ValueError(
            "Optimization requires at least two assets."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    if risk_free_rate <= -1:
        raise ValueError(
            "Risk-free rate must be greater than -100%."
        )

    expected_returns = (
        asset_returns.mean() * trading_days
    )

    covariance_matrix = (
        asset_returns.cov() * trading_days
    )

    number_of_assets = len(
        asset_returns.columns
    )

    initial_weights = np.full(
        number_of_assets,
        1 / number_of_assets
    )

    bounds = _create_weight_bounds(
        number_of_assets,
        max_weight
    )

    constraints = {
        "type": "eq",
        "fun": lambda candidate_weights:
            np.sum(candidate_weights) - 1.0
    }

    def negative_sharpe_ratio(candidate_weights):
        portfolio_return = (
            candidate_weights
            @ expected_returns.to_numpy()
        )

        portfolio_variance = (
            candidate_weights
            @ covariance_matrix.to_numpy()
            @ candidate_weights
        )

        portfolio_volatility = np.sqrt(
            portfolio_variance
        )

        if np.isclose(portfolio_volatility, 0):
            return np.inf

        sharpe_ratio = (
            portfolio_return - risk_free_rate
        ) / portfolio_volatility

        return -sharpe_ratio

    optimization_result = minimize(
        fun=negative_sharpe_ratio,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not optimization_result.success:
        raise RuntimeError(
            "Maximum-Sharpe optimization failed: "
            f"{optimization_result.message}"
        )

    optimal_weight_array = np.clip(
        optimization_result.x,
        0.0,
        1.0
    )

    optimal_weight_array = (
        optimal_weight_array
        / optimal_weight_array.sum()
    )

    optimal_weights = pd.Series(
        optimal_weight_array,
        index=asset_returns.columns,
        name="Maximum Sharpe Weight"
    )

    statistics = calculate_portfolio_statistics(
        optimal_weight_array,
        expected_returns.to_numpy(),
        covariance_matrix.to_numpy(),
        risk_free_rate=risk_free_rate
    )

    return {
        "weights": optimal_weights,
        "expected_return": statistics["return"],
        "volatility": statistics["volatility"],
        "sharpe_ratio": statistics["sharpe_ratio"]
    }

def generate_efficient_frontier(
    asset_returns,
    risk_free_rate=0.0,
    max_weight=1.0,
    number_of_points=50,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError(
            "Asset returns cannot be empty."
        )

    if asset_returns.shape[1] < 2:
        raise ValueError(
            "Optimization requires at least two assets."
        )

    if number_of_points < 2:
        raise ValueError(
            "Number of frontier points must be at least two."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    expected_returns = (
        asset_returns.mean() * trading_days
    )

    covariance_matrix = (
        asset_returns.cov() * trading_days
    )

    expected_return_array = (
        expected_returns.to_numpy()
    )

    covariance_array = (
        covariance_matrix.to_numpy()
    )

    number_of_assets = len(
        asset_returns.columns
    )

    bounds = _create_weight_bounds(
        number_of_assets,
        max_weight
    )

    minimum_volatility_portfolio = (
        optimize_minimum_volatility(
            asset_returns,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            trading_days=trading_days
        )
    )

    minimum_frontier_return = (
        minimum_volatility_portfolio[
            "expected_return"
        ]
    )

    initial_weights = (
        minimum_volatility_portfolio["weights"]
        .reindex(asset_returns.columns)
        .to_numpy()
    )

    maximum_return_result = minimize(
        fun=lambda candidate_weights: -(
            candidate_weights @ expected_return_array
        ),
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints={
            "type": "eq",
            "fun": lambda candidate_weights:
                np.sum(candidate_weights) - 1.0
        }
    )

    if not maximum_return_result.success:
        raise RuntimeError(
            "Maximum-return optimization failed: "
            f"{maximum_return_result.message}"
        )

    maximum_frontier_return = (
        -maximum_return_result.fun
    )

    target_returns = np.linspace(
        minimum_frontier_return,
        maximum_frontier_return,
        number_of_points
    )

    frontier_results = []

    def portfolio_volatility(candidate_weights):
        portfolio_variance = (
            candidate_weights
            @ covariance_array
            @ candidate_weights
        )

        return np.sqrt(portfolio_variance)

    for target_return in target_returns:
        constraints = (
            {
                "type": "eq",
                "fun": lambda candidate_weights:
                    np.sum(candidate_weights) - 1.0
            },
            {
                "type": "eq",
                "fun": (
                    lambda candidate_weights,
                    target=target_return:
                    candidate_weights
                    @ expected_return_array
                    - target
                )
            }
        )

        optimization_result = minimize(
            fun=portfolio_volatility,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not optimization_result.success:
            continue

        frontier_weight_array = np.clip(
            optimization_result.x,
            0.0,
            1.0
        )

        frontier_weight_array = (
            frontier_weight_array
            / frontier_weight_array.sum()
        )

        statistics = calculate_portfolio_statistics(
            frontier_weight_array,
            expected_return_array,
            covariance_array,
            risk_free_rate=risk_free_rate
        )

        frontier_row = {
            "Expected Return": statistics["return"],
            "Volatility": statistics["volatility"],
            "Sharpe Ratio": statistics["sharpe_ratio"]
        }

        for ticker, weight in zip(
            asset_returns.columns,
            frontier_weight_array
        ):
            frontier_row[f"{ticker} Weight"] = weight

        frontier_results.append(frontier_row)

        # Use this solution as the starting point
        # for the next nearby target return.
        initial_weights = frontier_weight_array

    if not frontier_results:
        raise RuntimeError(
            "Efficient frontier optimization failed."
        )

    efficient_frontier = pd.DataFrame(
        frontier_results
    )

    efficient_frontier = (
        efficient_frontier
        .sort_values("Volatility")
        .reset_index(drop=True)
    )

    return efficient_frontier