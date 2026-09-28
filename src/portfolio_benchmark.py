import numpy as np
import pandas as pd

def align_portfolio_and_benchmark(
    portfolio_returns,
    benchmark_returns
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if benchmark_returns.empty:
        raise ValueError(
            "Benchmark returns cannot be empty."
        )

    aligned_returns = pd.concat(
        [portfolio_returns, benchmark_returns],
        axis=1,
        join="inner"
    ).dropna()

    if aligned_returns.empty:
        raise ValueError(
            "Portfolio and benchmark have no overlapping dates."
        )

    aligned_returns.columns = [
        "Portfolio",
        "Benchmark"
    ]

    return aligned_returns

def calculate_beta(
    portfolio_returns,
    benchmark_returns
):
    aligned_returns = align_portfolio_and_benchmark(
        portfolio_returns,
        benchmark_returns
    )

    benchmark_variance = (
        aligned_returns["Benchmark"].var()
    )

    if np.isclose(benchmark_variance, 0):
        raise ValueError(
            "Beta is undefined when benchmark variance is zero."
        )

    covariance = aligned_returns[
        ["Portfolio", "Benchmark"]
    ].cov().loc["Portfolio", "Benchmark"]

    beta = covariance / benchmark_variance

    return beta

def calculate_alpha(
    portfolio_returns,
    benchmark_returns,
    risk_free_rate=0.0,
    trading_days=252
):
    if risk_free_rate <= -1:
        raise ValueError(
            "Risk-free rate must be greater than -100%."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    aligned_returns = align_portfolio_and_benchmark(
        portfolio_returns,
        benchmark_returns
    )

    beta = calculate_beta(
        aligned_returns["Portfolio"],
        aligned_returns["Benchmark"]
    )

    daily_risk_free_rate = (
        (1 + risk_free_rate)
        ** (1 / trading_days)
    ) - 1

    portfolio_excess_returns = (
        aligned_returns["Portfolio"]
        - daily_risk_free_rate
    )

    benchmark_excess_returns = (
        aligned_returns["Benchmark"]
        - daily_risk_free_rate
    )

    daily_alpha_series = (
        portfolio_excess_returns
        - beta * benchmark_excess_returns
    )

    annualized_alpha = (
        daily_alpha_series.mean()
        * trading_days
    )

    return annualized_alpha

def calculate_tracking_error(
    portfolio_returns,
    benchmark_returns,
    trading_days=252
):
    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    aligned_returns = align_portfolio_and_benchmark(
        portfolio_returns,
        benchmark_returns
    )

    active_returns = (
        aligned_returns["Portfolio"]
        - aligned_returns["Benchmark"]
    )

    tracking_error = (
        active_returns.std()
        * np.sqrt(trading_days)
    )

    if np.isclose(tracking_error, 0):
        raise ValueError(
            "Tracking error is zero because the portfolio "
            "does not deviate from the benchmark."
        )

    return tracking_error

def calculate_information_ratio(
    portfolio_returns,
    benchmark_returns,
    trading_days=252
):
    aligned_returns = align_portfolio_and_benchmark(
        portfolio_returns,
        benchmark_returns
    )

    active_returns = (
        aligned_returns["Portfolio"]
        - aligned_returns["Benchmark"]
    )

    tracking_error = calculate_tracking_error(
        aligned_returns["Portfolio"],
        aligned_returns["Benchmark"],
        trading_days
    )

    annualized_active_return = (
        active_returns.mean() * trading_days
    )

    information_ratio = (
        annualized_active_return / tracking_error
    )

    return information_ratio