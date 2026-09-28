import numpy as np
import pandas as pd
import pytest

from scipy.stats import norm

from src.portfolio_metrics import (
    calculate_portfolio_returns
)

from src.portfolio_risk import (
    calculate_historical_var,
    calculate_historical_expected_shortfall,
    calculate_parametric_var,
    calculate_parametric_expected_shortfall,
    calculate_dollar_var,
    calculate_dollar_expected_shortfall,
    calculate_correlated_monte_carlo_risk
)


def test_historical_var_matches_return_quantile(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    confidence_level = 0.95

    actual = calculate_historical_var(
        portfolio_returns,
        confidence_level
    )

    expected = max(
        -portfolio_returns.quantile(0.05),
        0.0
    )

    assert actual == pytest.approx(expected)


def test_historical_expected_shortfall_matches_tail_mean(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    return_threshold = (
        portfolio_returns.quantile(0.05)
    )

    tail_returns = portfolio_returns[
        portfolio_returns <= return_threshold
    ]

    expected = max(
        -tail_returns.mean(),
        0.0
    )

    actual = (
        calculate_historical_expected_shortfall(
            portfolio_returns,
            confidence_level=0.95
        )
    )

    assert actual == pytest.approx(expected)


def test_parametric_var_matches_normal_formula(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    tail_probability = 0.05
    z_score = norm.ppf(tail_probability)

    expected = max(
        -(
            portfolio_returns.mean()
            + z_score * portfolio_returns.std()
        ),
        0.0
    )

    actual = calculate_parametric_var(
        portfolio_returns,
        confidence_level=0.95
    )

    assert actual == pytest.approx(expected)


def test_parametric_expected_shortfall_matches_formula(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    tail_probability = 0.05
    z_score = norm.ppf(tail_probability)

    expected = max(
        -mean_return
        + volatility
        * norm.pdf(z_score)
        / tail_probability,
        0.0
    )

    actual = (
        calculate_parametric_expected_shortfall(
            portfolio_returns,
            confidence_level=0.95
        )
    )

    assert actual == pytest.approx(expected)


def test_dollar_risk_conversions():
    portfolio_value = 100000
    percentage_var = 0.02
    percentage_es = 0.03

    assert calculate_dollar_var(
        percentage_var,
        portfolio_value
    ) == pytest.approx(2000)

    assert calculate_dollar_expected_shortfall(
        percentage_es,
        portfolio_value
    ) == pytest.approx(3000)


def test_correlated_monte_carlo_is_reproducible(
    sample_asset_returns,
    sample_weights
):
    first_result = (
        calculate_correlated_monte_carlo_risk(
            sample_asset_returns,
            sample_weights,
            number_of_simulations=5000,
            random_seed=42
        )
    )

    second_result = (
        calculate_correlated_monte_carlo_risk(
            sample_asset_returns,
            sample_weights,
            number_of_simulations=5000,
            random_seed=42
        )
    )

    np.testing.assert_allclose(
        first_result[
            "simulated_portfolio_returns"
        ],
        second_result[
            "simulated_portfolio_returns"
        ]
    )

    assert first_result["var"] == pytest.approx(
        second_result["var"]
    )

    assert first_result[
        "expected_shortfall"
    ] == pytest.approx(
        second_result["expected_shortfall"]
    )


def test_monte_carlo_expected_shortfall_exceeds_var(
    sample_asset_returns,
    sample_weights
):
    result = calculate_correlated_monte_carlo_risk(
        sample_asset_returns,
        sample_weights,
        confidence_level=0.95,
        number_of_simulations=10000,
        random_seed=42
    )

    assert result["expected_shortfall"] >= (
        result["var"]
    )

    assert len(
        result["simulated_portfolio_returns"]
    ) == 10000

    assert result[
        "simulated_daily_asset_returns"
    ].shape == (10000, 1, 3)


def test_one_day_rebalancing_modes_are_identical(
    sample_asset_returns,
    sample_weights
):
    daily_result = (
        calculate_correlated_monte_carlo_risk(
            sample_asset_returns,
            sample_weights,
            number_of_simulations=5000,
            horizon_days=1,
            rebalancing_mode="daily",
            random_seed=42
        )
    )

    buy_and_hold_result = (
        calculate_correlated_monte_carlo_risk(
            sample_asset_returns,
            sample_weights,
            number_of_simulations=5000,
            horizon_days=1,
            rebalancing_mode="buy_and_hold",
            random_seed=42
        )
    )

    np.testing.assert_allclose(
        daily_result[
            "simulated_portfolio_returns"
        ],
        buy_and_hold_result[
            "simulated_portfolio_returns"
        ]
    )


def test_monte_carlo_var_is_close_to_parametric_var(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    parametric_var = calculate_parametric_var(
        portfolio_returns,
        confidence_level=0.95
    )

    monte_carlo_result = (
        calculate_correlated_monte_carlo_risk(
            sample_asset_returns,
            sample_weights,
            confidence_level=0.95,
            number_of_simulations=50000,
            random_seed=42
        )
    )

    assert monte_carlo_result[
        "var"
    ] == pytest.approx(
        parametric_var,
        abs=0.002
    )


def test_monte_carlo_rejects_invalid_weights(
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
        calculate_correlated_monte_carlo_risk(
            sample_asset_returns,
            invalid_weights
        )