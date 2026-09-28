import numpy as np
import pandas as pd
import pytest

from src.portfolio_holdings import (
    calculate_buy_and_hold_portfolio
)


def test_buy_and_hold_initial_allocations(
    sample_prices,
    sample_weights
):
    result = calculate_buy_and_hold_portfolio(
        sample_prices,
        sample_weights,
        initial_investment=100000
    )

    expected_shares = pd.Series(
        {
            "AAPL": 400.0,
            "JPM": 500.0,
            "MSFT": 175.0
        },
        name="Shares"
    )

    pd.testing.assert_series_equal(
        result["shares"],
        expected_shares
    )

    assert result[
        "portfolio_value"
    ].iloc[0] == pytest.approx(100000)

    assert result[
        "portfolio_value"
    ].iloc[-1] == pytest.approx(113500)


def test_buy_and_hold_returns_match_value_changes(
    sample_prices,
    sample_weights
):
    result = calculate_buy_and_hold_portfolio(
        sample_prices,
        sample_weights,
        initial_investment=100000
    )

    expected_returns = (
        result["portfolio_value"]
        .pct_change(fill_method=None)
        .dropna()
    )

    expected_returns.name = (
        "Buy and Hold Portfolio Return"
    )

    pd.testing.assert_series_equal(
        result["portfolio_returns"],
        expected_returns
    )


def test_buy_and_hold_weights_drift(
    sample_prices,
    sample_weights
):
    result = calculate_buy_and_hold_portfolio(
        sample_prices,
        sample_weights
    )

    initial_weights = (
        pd.Series(sample_weights)
        .reindex(sample_prices.columns)
    )

    actual_initial_weights = (
        result["weights_over_time"].iloc[0]
    )

    final_weights = (
        result["weights_over_time"].iloc[-1]
    )

    np.testing.assert_allclose(
        actual_initial_weights,
        initial_weights
    )

    assert not np.allclose(
        final_weights,
        initial_weights
    )

    assert final_weights.sum() == pytest.approx(
        1.0
    )


def test_buy_and_hold_rejects_invalid_weights(
    sample_prices
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
        calculate_buy_and_hold_portfolio(
            sample_prices,
            invalid_weights
        )