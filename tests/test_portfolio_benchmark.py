import numpy as np
import pandas as pd
import pytest

from src.portfolio_benchmark import (
    align_portfolio_and_benchmark,
    calculate_beta,
    calculate_alpha,
    calculate_tracking_error,
    calculate_information_ratio
)

from src.portfolio_metrics import (
    calculate_portfolio_returns
)


def test_alignment_keeps_only_shared_dates():
    portfolio_returns = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range(
            "2026-01-01",
            periods=3
        )
    )

    benchmark_returns = pd.Series(
        [0.005, 0.010, 0.015],
        index=pd.date_range(
            "2026-01-02",
            periods=3
        )
    )

    aligned = align_portfolio_and_benchmark(
        portfolio_returns,
        benchmark_returns
    )

    assert len(aligned) == 2

    assert list(aligned.columns) == [
        "Portfolio",
        "Benchmark"
    ]


def test_beta_matches_covariance_formula(
    sample_asset_returns,
    sample_weights,
    sample_benchmark_returns
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    expected = (
        portfolio_returns.cov(
            sample_benchmark_returns
        )
        / sample_benchmark_returns.var()
    )

    actual = calculate_beta(
        portfolio_returns,
        sample_benchmark_returns
    )

    assert actual == pytest.approx(expected)


def test_alpha_matches_capm_formula(
    sample_asset_returns,
    sample_weights,
    sample_benchmark_returns
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    risk_free_rate = 0.04

    beta = calculate_beta(
        portfolio_returns,
        sample_benchmark_returns
    )

    daily_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / 252)
        - 1
    )

    expected = (
        (
            portfolio_returns
            - daily_risk_free_rate
            - beta
            * (
                sample_benchmark_returns
                - daily_risk_free_rate
            )
        ).mean()
        * 252
    )

    actual = calculate_alpha(
        portfolio_returns,
        sample_benchmark_returns,
        risk_free_rate=risk_free_rate
    )

    assert actual == pytest.approx(expected)


def test_tracking_error_and_information_ratio(
    sample_asset_returns,
    sample_weights,
    sample_benchmark_returns
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    active_returns = (
        portfolio_returns
        - sample_benchmark_returns
    )

    expected_tracking_error = (
        active_returns.std() * np.sqrt(252)
    )

    expected_information_ratio = (
        active_returns.mean() * 252
        / expected_tracking_error
    )

    actual_tracking_error = (
        calculate_tracking_error(
            portfolio_returns,
            sample_benchmark_returns
        )
    )

    actual_information_ratio = (
        calculate_information_ratio(
            portfolio_returns,
            sample_benchmark_returns
        )
    )

    assert actual_tracking_error == pytest.approx(
        expected_tracking_error
    )

    assert actual_information_ratio == pytest.approx(
        expected_information_ratio
    )


def test_beta_rejects_constant_benchmark(
    sample_asset_returns,
    sample_weights
):
    portfolio_returns = calculate_portfolio_returns(
        sample_asset_returns,
        sample_weights
    )

    constant_benchmark = pd.Series(
        0.01,
        index=portfolio_returns.index
    )

    with pytest.raises(
        ValueError,
        match="variance is zero"
    ):
        calculate_beta(
            portfolio_returns,
            constant_benchmark
        )