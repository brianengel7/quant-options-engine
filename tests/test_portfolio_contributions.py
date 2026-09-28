import numpy as np
import pandas as pd
import pytest

from src.portfolio_contributions import (
    calculate_return_contributions,
    calculate_volatility_contributions
)

from src.portfolio_metrics import (
    calculate_portfolio_returns,
    calculate_portfolio_volatility
)


def test_return_contributions_match_formula(
    sample_asset_returns,
    sample_weights
):
    actual = calculate_return_contributions(
        sample_asset_returns,
        sample_weights
    )

    expected = (
        pd.Series(sample_weights)
        .reindex(sample_asset_returns.columns)
        * sample_asset_returns.mean()
        * 252
    )

    expected.name = "Return Contribution"

    pd.testing.assert_series_equal(
        actual,
        expected
    )


def test_return_contributions_sum_to_expected_return(
    sample_asset_returns,
    sample_weights
):
    contributions = (
        calculate_return_contributions(
            sample_asset_returns,
            sample_weights
        )
    )

    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    expected_portfolio_return = (
        portfolio_returns.mean() * 252
    )

    assert contributions.sum() == pytest.approx(
        expected_portfolio_return
    )


def test_volatility_contributions_reconcile(
    sample_asset_returns,
    sample_weights
):
    contributions = (
        calculate_volatility_contributions(
            sample_asset_returns,
            sample_weights
        )
    )

    portfolio_volatility = (
        calculate_portfolio_volatility(
            sample_asset_returns,
            sample_weights
        )
    )

    assert contributions[
        "Volatility Contribution"
    ].sum() == pytest.approx(
        portfolio_volatility
    )

    assert contributions[
        "Percentage of Portfolio Risk"
    ].sum() == pytest.approx(1.0)

    assert np.isfinite(
        contributions.to_numpy()
    ).all()


def test_contributions_reject_missing_weight(
    sample_asset_returns
):
    incomplete_weights = {
        "AAPL": 0.50,
        "JPM": 0.50
    }

    with pytest.raises(
        ValueError,
        match="Missing weights"
    ):
        calculate_return_contributions(
            sample_asset_returns,
            incomplete_weights
        )