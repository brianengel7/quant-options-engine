import numpy as np
import pandas as pd

from src import portfolio_engine


def create_test_market_data():
    random_generator = np.random.default_rng(42)

    dates = pd.bdate_range(
        start="2024-01-02",
        periods=260
    )

    mean_returns = np.array([
        0.0006,
        0.0005,
        0.0004
    ])

    covariance_matrix = np.array([
        [0.00016, 0.00008, 0.00005],
        [0.00008, 0.00014, 0.00004],
        [0.00005, 0.00004, 0.00012]
    ])

    generated_returns = (
        random_generator.multivariate_normal(
            mean=mean_returns,
            cov=covariance_matrix,
            size=len(dates)
        )
    )

    asset_returns = pd.DataFrame(
        generated_returns,
        index=dates,
        columns=["AAPL", "MSFT", "JPM"]
    )

    starting_prices = pd.Series({
        "AAPL": 180.0,
        "MSFT": 380.0,
        "JPM": 160.0
    })

    prices = (
        (1 + asset_returns).cumprod()
        * starting_prices
    )

    benchmark_noise = random_generator.normal(
        loc=0.0,
        scale=0.002,
        size=len(dates)
    )

    benchmark_returns = (
        asset_returns["AAPL"] * 0.35
        + asset_returns["MSFT"] * 0.35
        + asset_returns["JPM"] * 0.30
        + benchmark_noise
    )

    # Asset-return calculation drops the first price date.
    benchmark_returns = benchmark_returns.iloc[1:]
    benchmark_returns.name = "SPY"

    return prices, benchmark_returns


def test_run_portfolio_analysis_returns_complete_report(
    monkeypatch
):
    prices, benchmark_returns = (
        create_test_market_data()
    )

    def fake_get_portfolio_prices(
        tickers,
        start_date,
        end_date=None
    ):
        assert tickers == ["AAPL", "MSFT", "JPM"]
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"

        return prices.copy()

    def fake_get_benchmark_returns(
        benchmark_ticker,
        start_date,
        end_date=None
    ):
        assert benchmark_ticker == "SPY"
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"

        return benchmark_returns.copy()

    monkeypatch.setattr(
        portfolio_engine,
        "get_portfolio_prices",
        fake_get_portfolio_prices
    )

    monkeypatch.setattr(
        portfolio_engine,
        "get_benchmark_returns",
        fake_get_benchmark_returns
    )

    report = portfolio_engine.run_portfolio_analysis(
        tickers=["AAPL", "MSFT", "JPM"],
        weights={
            "AAPL": 0.40,
            "MSFT": 0.35,
            "JPM": 0.25
        },
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_investment=100_000,
        benchmark_ticker="SPY",
        risk_free_rate=0.04,
        confidence_level=0.95,
        maximum_position_weight=0.60,
        number_of_simulations=2_000,
        monte_carlo_horizon_days=1,
        monte_carlo_rebalancing_mode="daily",
        frontier_points=10,
        trading_days=252,
        random_seed=42
    )

    expected_sections = {
        "inputs",
        "previews",
        "performance",
        "risk_adjusted",
        "benchmark",
        "risk",
        "matrices",
        "contributions",
        "optimization",
        "buy_and_hold",
        "strategy_comparison",
        "strategy_value_history",
        "data"
    }

    assert expected_sections.issubset(report.keys())

    assert report["inputs"]["tickers"] == [
        "AAPL",
        "MSFT",
        "JPM"
    ]

    assert (
        report["inputs"]["initial_investment"]
        == 100_000
    )

    assert np.isfinite(
        report["performance"]["current_value"]
    )

    assert report["performance"]["current_value"] > 0

    assert np.isfinite(
        report["performance"]["annualized_return"]
    )

    assert np.isfinite(
        report["performance"]["annualized_volatility"]
    )

    assert np.isfinite(
        report["risk_adjusted"]["sharpe_ratio"]
    )

    assert report["risk_adjusted"]["maximum_drawdown"] <= 0

    assert np.isfinite(
        report["benchmark"]["beta"]
    )

    assert np.isfinite(
        report["benchmark"]["alpha"]
    )

    for risk_method in [
        "historical",
        "parametric",
        "monte_carlo"
    ]:
        risk_results = report["risk"][risk_method]

        assert np.isfinite(risk_results["var"])
        assert risk_results["var"] >= 0

        assert np.isfinite(
            risk_results["expected_shortfall"]
        )

        assert risk_results["expected_shortfall"] >= 0

    simulated_returns = (
        report["risk"]["monte_carlo"][
            "simulated_returns"
        ]
    )

    assert len(simulated_returns) == 2_000

    minimum_volatility_weights = pd.Series(
        report["optimization"][
            "minimum_volatility"
        ]["weights"],
        dtype=float
    )

    maximum_sharpe_weights = pd.Series(
        report["optimization"][
            "maximum_sharpe"
        ]["weights"],
        dtype=float
    )

    assert np.isclose(
        minimum_volatility_weights.sum(),
        1.0
    )

    assert np.isclose(
        maximum_sharpe_weights.sum(),
        1.0
    )

    assert (
        minimum_volatility_weights
        <= 0.60 + 1e-6
    ).all()

    assert (
        maximum_sharpe_weights
        <= 0.60 + 1e-6
    ).all()

    assert not report[
        "optimization"
    ]["efficient_frontier"].empty

    assert {
        "Daily Rebalanced",
        "Buy and Hold"
    }.issubset(
        report["strategy_comparison"].index
    )

    assert not report[
        "strategy_value_history"
    ].empty

    assert report["data"]["prices"].equals(
        prices
    )