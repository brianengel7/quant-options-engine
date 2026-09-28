import numpy as np
import pandas as pd
import pytest

from src.portfolio_optimization import (
    calculate_portfolio_statistics,
    optimize_minimum_volatility,
    optimize_maximum_sharpe,
    generate_efficient_frontier
)


def _prepare_optimization_inputs(
    asset_returns,
    weights
):
    expected_returns = (
        asset_returns.mean() * 252
    )

    covariance_matrix = (
        asset_returns.cov() * 252
    )

    weight_array = (
        pd.Series(weights, dtype=float)
        .reindex(asset_returns.columns)
        .to_numpy()
    )

    return (
        weight_array,
        expected_returns.to_numpy(),
        covariance_matrix.to_numpy()
    )


def test_portfolio_statistics_match_formulas(
    sample_asset_returns,
    sample_weights
):
    (
        weight_array,
        expected_returns,
        covariance_matrix
    ) = _prepare_optimization_inputs(
        sample_asset_returns,
        sample_weights
    )

    risk_free_rate = 0.04

    actual = calculate_portfolio_statistics(
        weight_array,
        expected_returns,
        covariance_matrix,
        risk_free_rate=risk_free_rate
    )

    expected_return = (
        weight_array @ expected_returns
    )

    expected_volatility = np.sqrt(
        weight_array
        @ covariance_matrix
        @ weight_array
    )

    expected_sharpe = (
        expected_return - risk_free_rate
    ) / expected_volatility

    assert actual["return"] == pytest.approx(
        expected_return
    )

    assert actual["volatility"] == pytest.approx(
        expected_volatility
    )

    assert actual["sharpe_ratio"] == pytest.approx(
        expected_sharpe
    )


def test_minimum_volatility_weights_are_valid(
    sample_asset_returns
):
    result = optimize_minimum_volatility(
        sample_asset_returns,
        risk_free_rate=0.04
    )

    weights = result["weights"]

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert (weights <= 1).all()
    assert list(weights.index) == list(
        sample_asset_returns.columns
    )


def test_minimum_volatility_improves_current_volatility(
    sample_asset_returns,
    sample_weights
):
    (
        weight_array,
        expected_returns,
        covariance_matrix
    ) = _prepare_optimization_inputs(
        sample_asset_returns,
        sample_weights
    )

    current_statistics = (
        calculate_portfolio_statistics(
            weight_array,
            expected_returns,
            covariance_matrix,
            risk_free_rate=0.04
        )
    )

    optimized = optimize_minimum_volatility(
        sample_asset_returns,
        risk_free_rate=0.04
    )

    assert optimized["volatility"] <= (
        current_statistics["volatility"]
        + 1e-7
    )


def test_maximum_sharpe_weights_are_valid(
    sample_asset_returns
):
    result = optimize_maximum_sharpe(
        sample_asset_returns,
        risk_free_rate=0.04
    )

    weights = result["weights"]

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert (weights <= 1).all()


def test_maximum_sharpe_improves_current_sharpe(
    sample_asset_returns,
    sample_weights
):
    (
        weight_array,
        expected_returns,
        covariance_matrix
    ) = _prepare_optimization_inputs(
        sample_asset_returns,
        sample_weights
    )

    current_statistics = (
        calculate_portfolio_statistics(
            weight_array,
            expected_returns,
            covariance_matrix,
            risk_free_rate=0.04
        )
    )

    optimized = optimize_maximum_sharpe(
        sample_asset_returns,
        risk_free_rate=0.04
    )

    assert optimized["sharpe_ratio"] >= (
        current_statistics["sharpe_ratio"]
        - 1e-7
    )


def test_optimizers_respect_maximum_weight(
    sample_asset_returns
):
    max_weight = 0.50

    minimum_volatility = (
        optimize_minimum_volatility(
            sample_asset_returns,
            risk_free_rate=0.04,
            max_weight=max_weight
        )
    )

    maximum_sharpe = optimize_maximum_sharpe(
        sample_asset_returns,
        risk_free_rate=0.04,
        max_weight=max_weight
    )

    assert minimum_volatility[
        "weights"
    ].max() <= max_weight + 1e-7

    assert maximum_sharpe[
        "weights"
    ].max() <= max_weight + 1e-7


def test_efficient_frontier_is_valid(
    sample_asset_returns
):
    number_of_points = 25
    max_weight = 0.60

    frontier = generate_efficient_frontier(
        sample_asset_returns,
        risk_free_rate=0.04,
        max_weight=max_weight,
        number_of_points=number_of_points
    )

    required_columns = {
        "Expected Return",
        "Volatility",
        "Sharpe Ratio",
        "AAPL Weight",
        "JPM Weight",
        "MSFT Weight"
    }

    assert required_columns.issubset(
        frontier.columns
    )

    assert len(frontier) == number_of_points

    weight_columns = [
        "AAPL Weight",
        "JPM Weight",
        "MSFT Weight"
    ]

    frontier_weights = frontier[
        weight_columns
    ]

    np.testing.assert_allclose(
        frontier_weights.sum(axis=1),
        1.0,
        atol=1e-6
    )

    assert (
        frontier_weights >= -1e-7
    ).all().all()

    assert (
        frontier_weights <= max_weight + 1e-7
    ).all().all()

    assert np.isfinite(
        frontier["Expected Return"]
    ).all()

    assert np.isfinite(
        frontier["Volatility"]
    ).all()

    assert np.isfinite(
        frontier["Sharpe Ratio"]
    ).all()

    # The function sorts the frontier by volatility.
    assert (
        frontier["Volatility"]
        .diff()
        .dropna()
        >= -1e-8
    ).all()


def test_optimizer_rejects_infeasible_maximum_weight(
    sample_asset_returns
):
    # Three assets capped at 30% can total only 90%.
    with pytest.raises(
        ValueError,
        match="too low"
    ):
        optimize_minimum_volatility(
            sample_asset_returns,
            max_weight=0.30
        )