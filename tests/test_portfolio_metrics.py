import numpy as np
import pandas as pd
import pytest

from src.portfolio_metrics import (
    calculate_portfolio_returns,
    calculate_cumulative_returns,
    calculate_portfolio_value,
    calculate_total_return,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_drawdowns,
    calculate_max_drawdown,
    calculate_calmar_ratio
)


def test_portfolio_returns_match_weighted_returns(
    sample_asset_returns,
    sample_weights
):
    actual = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    expected = sample_asset_returns.dot(
        pd.Series(sample_weights)
    )

    pd.testing.assert_series_equal(
        actual,
        expected.rename("Portfolio Return")
    )


def test_cumulative_returns_compound_correctly(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    actual = calculate_cumulative_returns(
        portfolio_returns
    )

    expected = (
        (1 + portfolio_returns).cumprod() - 1
    )

    expected.name = "Cumulative Return"

    pd.testing.assert_series_equal(
        actual,
        expected
    )


def test_portfolio_value_matches_compounded_growth(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    initial_investment = 100000

    portfolio_value = calculate_portfolio_value(
        portfolio_returns,
        initial_investment
    )

    expected_final_value = (
        initial_investment
        * (1 + portfolio_returns).prod()
    )

    assert portfolio_value.iloc[-1] == pytest.approx(
        expected_final_value
    )


def test_total_return_matches_final_growth(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    actual = calculate_total_return(
        portfolio_returns
    )

    expected = (
        (1 + portfolio_returns).prod() - 1
    )

    assert actual == pytest.approx(expected)


def test_annualized_return_matches_formula(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    actual = calculate_annualized_return(
        portfolio_returns
    )

    expected = (
        (1 + portfolio_returns).prod()
        ** (252 / len(portfolio_returns))
        - 1
    )

    assert actual == pytest.approx(expected)


def test_annualized_volatility_matches_formula(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    actual = calculate_annualized_volatility(
        portfolio_returns
    )

    expected = (
        portfolio_returns.std()
        * np.sqrt(252)
    )

    assert actual == pytest.approx(expected)


def test_covariance_volatility_matches_return_volatility(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    direct_volatility = (
        calculate_annualized_volatility(
            portfolio_returns
        )
    )

    covariance_volatility = (
        calculate_portfolio_volatility(
            sample_asset_returns,
            sample_weights
        )
    )

    assert covariance_volatility == pytest.approx(
        direct_volatility
    )


def test_initial_loss_is_counted_as_drawdown():
    returns = pd.Series(
        [-0.10, 0.05, 0.10]
    )

    drawdowns = calculate_drawdowns(returns)

    assert drawdowns.iloc[0] == pytest.approx(
        -0.10
    )

    assert calculate_max_drawdown(
        returns
    ) == pytest.approx(-0.10)


def test_risk_adjusted_metrics_are_finite(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    sharpe = calculate_sharpe_ratio(
        portfolio_returns,
        risk_free_rate=0.04
    )

    sortino = calculate_sortino_ratio(
        portfolio_returns,
        minimum_acceptable_return=0.04
    )

    calmar = calculate_calmar_ratio(
        portfolio_returns
    )

    assert np.isfinite(sharpe)
    assert np.isfinite(sortino)
    assert np.isfinite(calmar)


def test_portfolio_rejects_weights_that_do_not_sum_to_one(
    sample_asset_returns
):
    invalid_weights = {
        "AAPL": 0.50,
        "JPM": 0.30,
        "MSFT": 0.30
    }

    with pytest.raises(
        ValueError,
        match="sum to 1"
    ):
        calculate_portfolio_returns(
            sample_asset_returns,
            invalid_weights
        )