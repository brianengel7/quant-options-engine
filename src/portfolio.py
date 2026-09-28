from pathlib import Path

from .portfolio_engine import run_portfolio_analysis
from .portfolio_reporting import print_portfolio_report
from .portfolio_visualizations import (
    plot_efficient_frontier,
    plot_strategy_growth,
    plot_monte_carlo_distribution
)


def main():
    # -----------------------------
    # Portfolio inputs
    # -----------------------------

    tickers = [
        "AAPL",
        "MSFT",
        "JPM"
    ]

    weights = {
        "AAPL": 0.40,
        "MSFT": 0.35,
        "JPM": 0.25
    }

    # -----------------------------
    # Run portfolio engine
    # -----------------------------

    analysis = run_portfolio_analysis(
        tickers=tickers,
        weights=weights,
        start_date="2025-01-01",
        end_date=None,
        initial_investment=100_000,
        benchmark_ticker="SPY",
        risk_free_rate=0.04,
        confidence_level=0.95,
        maximum_position_weight=0.60,
        number_of_simulations=10_000,
        monte_carlo_horizon_days=1,
        monte_carlo_rebalancing_mode="daily",
        frontier_points=50,
        trading_days=252,
        random_seed=42
    )

    # -----------------------------
    # Output paths
    # -----------------------------

    output_directory = (
        Path(__file__).resolve().parent.parent
        / "outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    efficient_frontier_chart_path = (
        output_directory
        / "efficient_frontier.png"
    )

    strategy_growth_chart_path = (
        output_directory
        / "strategy_growth.png"
    )

    monte_carlo_chart_path = (
        output_directory
        / "monte_carlo_distribution.png"
    )

    plot_efficient_frontier(
        analysis["optimization"]["efficient_frontier"],
        analysis["optimization"]["current"],
        analysis["optimization"]["minimum_volatility"],
        analysis["optimization"]["maximum_sharpe"],
        show=False,
        save_path=efficient_frontier_chart_path
    )

    plot_strategy_growth(
        analysis["strategy_value_history"],
        show=False,
        save_path=strategy_growth_chart_path
    )

    monte_carlo_results = (
        analysis["risk"]["monte_carlo"]
    )

    plot_monte_carlo_distribution(
        monte_carlo_results["simulated_returns"],
        monte_carlo_results["var"],
        monte_carlo_results["expected_shortfall"],
        confidence_level=(
            analysis["inputs"]["confidence_level"]
        ),
        number_of_bins=60,
        show=False,
        save_path=monte_carlo_chart_path
    )

    analysis["charts"] = {
        "efficient_frontier": (
            efficient_frontier_chart_path
        ),
        "strategy_growth": (
            strategy_growth_chart_path
        ),
        "monte_carlo_distribution": (
            monte_carlo_chart_path
        )
    }

    print_portfolio_report(analysis)


if __name__ == "__main__":
    main()