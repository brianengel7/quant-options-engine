import pandas as pd
import numpy as np
from pathlib import Path
from portfolio_optimization import (
    calculate_portfolio_statistics,
    optimize_minimum_volatility,
    optimize_maximum_sharpe,
    generate_efficient_frontier
)
from portfolio_visualizations import (
    plot_efficient_frontier,
    plot_strategy_growth
)
from portfolio_holdings import (
    calculate_buy_and_hold_portfolio
)
from portfolio_risk import (
    calculate_historical_var,
    calculate_dollar_var,
    calculate_historical_expected_shortfall,
    calculate_dollar_expected_shortfall,
    calculate_parametric_var,
    calculate_parametric_expected_shortfall,
    calculate_correlated_monte_carlo_risk
)
from portfolio_metrics import (
    calculate_portfolio_returns,
    calculate_cumulative_returns,
    calculate_portfolio_value,
    calculate_total_return,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_covariance_matrix,
    calculate_correlation_matrix,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_drawdowns,
    calculate_max_drawdown,
    calculate_calmar_ratio
)

from portfolio_data import (
    get_portfolio_prices,
    calculate_asset_returns,
    get_benchmark_returns
)

from portfolio_benchmark import (
    align_portfolio_and_benchmark,
    calculate_beta,
    calculate_alpha,
    calculate_tracking_error,
    calculate_information_ratio
)  

from portfolio_contributions import (
    calculate_return_contributions,
    calculate_volatility_contributions
)

if __name__ == "__main__":
    # -----------------------------
    # Portfolio inputs
    # -----------------------------

    tickers = ["AAPL", "MSFT", "JPM"]

    weights = {
        "AAPL": 0.40,
        "MSFT": 0.35,
        "JPM": 0.25
    }

    start_date = "2025-01-01"
    initial_investment = 100000
    risk_free_rate = 0.04
    confidence_level = 0.95
    benchmark_ticker = "SPY"

    # -----------------------------
    # Market data and returns
    # -----------------------------

    prices = get_portfolio_prices(
        tickers=tickers,
        start_date=start_date
    )

    asset_returns = calculate_asset_returns(
        prices
    )

    portfolio_returns = calculate_portfolio_returns(
        asset_returns,
        weights
    )

    # -----------------------------
    # Portfolio performance
    # -----------------------------

    cumulative_returns = calculate_cumulative_returns(
        portfolio_returns
    )

    portfolio_value_history = calculate_portfolio_value(
        portfolio_returns,
        initial_investment=initial_investment
    )

    initial_value = pd.Series(
        [initial_investment],
        index=[prices.index[0]],
        name="Daily Rebalanced"
    )

    rebalanced_value_history = pd.concat(
        [
            initial_value,
            portfolio_value_history.rename(
                "Daily Rebalanced"
            )
        ]
    ).sort_index()

    current_portfolio_value = (
        portfolio_value_history.iloc[-1]
    )

    total_return = calculate_total_return(
        portfolio_returns
    )

    annualized_return = calculate_annualized_return(
        portfolio_returns
    )

    annualized_volatility = calculate_annualized_volatility(
        portfolio_returns
    )

    # -----------------------------
    # Covariance and correlation
    # -----------------------------

    covariance_matrix = calculate_covariance_matrix(
        asset_returns
    )

    correlation_matrix = calculate_correlation_matrix(
        asset_returns
    )

    covariance_volatility = calculate_portfolio_volatility(
        asset_returns,
        weights
    )

    # -----------------------------
    # Risk-adjusted performance
    # -----------------------------

    sharpe_ratio = calculate_sharpe_ratio(
        portfolio_returns,
        risk_free_rate=risk_free_rate
    )

    sortino_ratio = calculate_sortino_ratio(
        portfolio_returns,
        minimum_acceptable_return=risk_free_rate
    )

    drawdowns = calculate_drawdowns(
        portfolio_returns
    )

    maximum_drawdown = calculate_max_drawdown(
        portfolio_returns
    )

    calmar_ratio = calculate_calmar_ratio(
        portfolio_returns
    )

    # -----------------------------
    # Benchmark analysis
    # -----------------------------

    benchmark_returns = get_benchmark_returns(
        benchmark_ticker=benchmark_ticker,
        start_date=start_date
    )

    aligned_returns = align_portfolio_and_benchmark(
        portfolio_returns,
        benchmark_returns
    )

    portfolio_beta = calculate_beta(
        portfolio_returns,
        benchmark_returns
    )

    portfolio_alpha = calculate_alpha(
        portfolio_returns,
        benchmark_returns,
        risk_free_rate=risk_free_rate
    )

    tracking_error = calculate_tracking_error(
        portfolio_returns,
        benchmark_returns
    )

    information_ratio = calculate_information_ratio(
        portfolio_returns,
        benchmark_returns
    )

    # -----------------------------
    # Historical VaR and ES
    # -----------------------------

    historical_var = calculate_historical_var(
        portfolio_returns,
        confidence_level=confidence_level
    )

    historical_dollar_var = calculate_dollar_var(
        historical_var,
        current_portfolio_value
    )

    historical_expected_shortfall = (
        calculate_historical_expected_shortfall(
            portfolio_returns,
            confidence_level=confidence_level
        )
    )

    historical_dollar_expected_shortfall = (
        calculate_dollar_expected_shortfall(
            historical_expected_shortfall,
            current_portfolio_value
        )
    )

    # -----------------------------
    # Parametric VaR and ES
    # -----------------------------

    parametric_var = calculate_parametric_var(
        portfolio_returns,
        confidence_level=confidence_level
    )

    parametric_dollar_var = calculate_dollar_var(
        parametric_var,
        current_portfolio_value
    )

    parametric_expected_shortfall = (
        calculate_parametric_expected_shortfall(
            portfolio_returns,
            confidence_level=confidence_level
        )
    )

    parametric_dollar_expected_shortfall = (
        calculate_dollar_expected_shortfall(
            parametric_expected_shortfall,
            current_portfolio_value
        )
    )

    # -----------------------------
    # Monte Carlo VaR and ES
    # -----------------------------

    monte_carlo_results = (
        calculate_correlated_monte_carlo_risk(
            asset_returns,
            weights,
            confidence_level=confidence_level,
            number_of_simulations=10000,
            horizon_days=1,
            rebalancing_mode="daily",
            random_seed=42
        )
    )

    monte_carlo_var = (
        monte_carlo_results["var"]
    )

    monte_carlo_expected_shortfall = (
        monte_carlo_results["expected_shortfall"]
    )

    simulated_returns = (
        monte_carlo_results[
            "simulated_portfolio_returns"
        ]
    )

    monte_carlo_dollar_var = calculate_dollar_var(
        monte_carlo_var,
        current_portfolio_value
    )

    monte_carlo_dollar_expected_shortfall = (
        calculate_dollar_expected_shortfall(
            monte_carlo_expected_shortfall,
            current_portfolio_value
        )
    )

    # -----------------------------
    # Asset contribution analysis
    # -----------------------------

    return_contributions = calculate_return_contributions(
        asset_returns,
        weights
    )

    volatility_contributions = (
        calculate_volatility_contributions(
            asset_returns,
            weights
        )
    )

    formatted_volatility_contributions = (
        volatility_contributions.copy()
    )

    for column in formatted_volatility_contributions.columns:
        formatted_volatility_contributions[column] = (
            formatted_volatility_contributions[column]
            .apply(lambda value: f"{value:.2%}")
        )

    maximum_position_weight = 0.60

    minimum_volatility_portfolio = (
        optimize_minimum_volatility(
            asset_returns,
            risk_free_rate=risk_free_rate,
            max_weight=maximum_position_weight
        )
    )

    optimization_expected_returns = (
        asset_returns.mean() * 252
    )

    optimization_covariance_matrix = (
        asset_returns.cov() * 252
    )

    current_weight_array = (
        pd.Series(weights, dtype=float)
        .reindex(asset_returns.columns)
        .to_numpy()
    )

    current_portfolio_statistics = (
        calculate_portfolio_statistics(
            current_weight_array,
            optimization_expected_returns.to_numpy(),
            optimization_covariance_matrix.to_numpy(),
            risk_free_rate=risk_free_rate
        )
    )

    current_portfolio_point = {
        "expected_return": (
            current_portfolio_statistics["return"]
        ),
        "volatility": (
            current_portfolio_statistics["volatility"]
        ),
        "sharpe_ratio": (
            current_portfolio_statistics["sharpe_ratio"]
        )
    }

    maximum_sharpe_portfolio = (
        optimize_maximum_sharpe(
            asset_returns,
            risk_free_rate=risk_free_rate,
            max_weight=maximum_position_weight
        )
    )

    efficient_frontier = generate_efficient_frontier(
        asset_returns,
        risk_free_rate=risk_free_rate,
        max_weight=maximum_position_weight,
        number_of_points=50
    )

    buy_and_hold_results = (
        calculate_buy_and_hold_portfolio(
            prices,
            weights,
            initial_investment=initial_investment
        )
    )

    buy_and_hold_returns = (
        buy_and_hold_results["portfolio_returns"]
    )

    buy_and_hold_value_history = (
        buy_and_hold_results["portfolio_value"]
    )

    buy_and_hold_shares = (
        buy_and_hold_results["shares"]
    )

    buy_and_hold_weights = (
        buy_and_hold_results["weights_over_time"]
    )

    buy_and_hold_total_return = (
        calculate_total_return(
            buy_and_hold_returns
        )
    )

    buy_and_hold_annualized_return = (
        calculate_annualized_return(
            buy_and_hold_returns
        )
    )

    buy_and_hold_annualized_volatility = (
        calculate_annualized_volatility(
            buy_and_hold_returns
        )
    )

    buy_and_hold_sharpe_ratio = (
        calculate_sharpe_ratio(
            buy_and_hold_returns,
            risk_free_rate=risk_free_rate
        )
    )

    buy_and_hold_max_drawdown = (
        calculate_max_drawdown(
            buy_and_hold_returns
        )
    )

    strategy_value_history = pd.concat(
        [
            rebalanced_value_history,
            buy_and_hold_value_history.rename(
                "Buy and Hold"
            )
        ],
        axis=1
    ).dropna()

    strategy_comparison = pd.DataFrame(
        {
            "Daily Rebalanced": {
                "Final Value": (
                    rebalanced_value_history.iloc[-1]
                ),
                "Total Return": total_return,
                "Annualized Return": annualized_return,
                "Annualized Volatility": (
                    annualized_volatility
                ),
                "Sharpe Ratio": sharpe_ratio,
                "Maximum Drawdown": maximum_drawdown
            },
            "Buy and Hold": {
                "Final Value": (
                    buy_and_hold_value_history.iloc[-1]
                ),
                "Total Return": (
                    buy_and_hold_total_return
                ),
                "Annualized Return": (
                    buy_and_hold_annualized_return
                ),
                "Annualized Volatility": (
                    buy_and_hold_annualized_volatility
                ),
                "Sharpe Ratio": (
                    buy_and_hold_sharpe_ratio
                ),
                "Maximum Drawdown": (
                    buy_and_hold_max_drawdown
                )
            }
        }
    ).T

    # -----------------------------
    # Output
    # -----------------------------

    print("\nPORTFOLIO INPUTS")
    print("----------------")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Start date: {start_date}")
    print(f"Initial investment: ${initial_investment:,.2f}")

    print("\nWeights:")
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.2%}")

    print("\nPORTFOLIO RETURNS")
    print("-----------------")
    print(portfolio_returns.head())

    print("\nCUMULATIVE RETURNS")
    print("------------------")
    print(cumulative_returns.head())

    print("\nPORTFOLIO VALUE HISTORY")
    print("-----------------------")
    print(portfolio_value_history.head())

    print("\nPERFORMANCE SUMMARY")
    print("-------------------")
    print(
        f"Current portfolio value: "
        f"${current_portfolio_value:,.2f}"
    )
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized return: {annualized_return:.2%}")
    print(
        f"Annualized volatility: "
        f"{annualized_volatility:.2%}"
    )
    print(
        f"Covariance-based volatility: "
        f"{covariance_volatility:.2%}"
    )

    print("\nRISK-ADJUSTED PERFORMANCE")
    print("-------------------------")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Sortino ratio: {sortino_ratio:.2f}")
    print(f"Maximum drawdown: {maximum_drawdown:.2%}")
    print(f"Calmar ratio: {calmar_ratio:.2f}")

    print("\nBENCHMARK ANALYSIS")
    print("------------------")
    print(f"Benchmark: {benchmark_ticker}")
    print(f"Portfolio beta: {portfolio_beta:.2f}")
    print(f"Annualized alpha: {portfolio_alpha:.2%}")
    print(f"Tracking error: {tracking_error:.2%}")
    print(
        f"Information ratio: "
        f"{information_ratio:.2f}"
    )

    print("\nHISTORICAL RISK")
    print("---------------")
    print(
        f"Historical VaR ({confidence_level:.0%}): "
        f"{historical_var:.2%}"
    )
    print(
        f"Historical dollar VaR: "
        f"${historical_dollar_var:,.2f}"
    )
    print(
        f"Historical Expected Shortfall "
        f"({confidence_level:.0%}): "
        f"{historical_expected_shortfall:.2%}"
    )
    print(
        f"Historical dollar Expected Shortfall: "
        f"${historical_dollar_expected_shortfall:,.2f}"
    )

    print("\nPARAMETRIC RISK")
    print("----------------")
    print(
        f"Parametric VaR ({confidence_level:.0%}): "
        f"{parametric_var:.2%}"
    )
    print(
        f"Parametric dollar VaR: "
        f"${parametric_dollar_var:,.2f}"
    )
    print(
        f"Parametric Expected Shortfall "
        f"({confidence_level:.0%}): "
        f"{parametric_expected_shortfall:.2%}"
    )
    print(
        f"Parametric dollar Expected Shortfall: "
        f"${parametric_dollar_expected_shortfall:,.2f}"
    )

    print("\nCORRELATED MONTE CARLO RISK")
    print("----------------")
    print(
        f"Monte Carlo VaR ({confidence_level:.0%}): "
        f"{monte_carlo_var:.2%}"
    )
    print(
        f"Monte Carlo dollar VaR: "
        f"${monte_carlo_dollar_var:,.2f}"
    )
    print(
        f"Monte Carlo Expected Shortfall "
        f"({confidence_level:.0%}): "
        f"{monte_carlo_expected_shortfall:.2%}"
    )
    print(
        f"Monte Carlo dollar Expected Shortfall: "
        f"${monte_carlo_dollar_expected_shortfall:,.2f}"
    )

    print("\nANNUALIZED COVARIANCE MATRIX")
    print("----------------------------")
    print(covariance_matrix)

    print("\nCORRELATION MATRIX")
    print("------------------")
    print(correlation_matrix)

    print("\nALIGNED PORTFOLIO AND BENCHMARK RETURNS")
    print("---------------------------------------")
    print(aligned_returns.head())

    print("\nANNUALIZED RETURN CONTRIBUTIONS")
    print("--------------------------------")
    print(
        return_contributions.apply(
            lambda value: f"{value:.2%}"
        )
    )
    print(
        f"Total arithmetic return contribution: "
        f"{return_contributions.sum():.2%}"
    )

    print("\nASSET-LEVEL RISK CONTRIBUTIONS")
    print("------------------------------")
    print(formatted_volatility_contributions)

    print("\nMOST RECENT DRAWDOWNS")
    print("---------------------")
    print(drawdowns.tail())

    print("\nPORTFOLIO OPTIMIZATION")
    print("----------------------")

    print("\nCurrent portfolio weights:")
    for ticker, weight in zip(
        asset_returns.columns,
        current_weight_array
    ):
        print(f"{ticker}: {weight:.2%}")

    print(
        f"Expected return: "
        f"{current_portfolio_statistics['return']:.2%}"
    )
    print(
        f"Volatility: "
        f"{current_portfolio_statistics['volatility']:.2%}"
    )
    print(
        f"Sharpe ratio: "
        f"{current_portfolio_statistics['sharpe_ratio']:.2f}"
    )

    print("\nMinimum-volatility weights:")
    for ticker, weight in (
        minimum_volatility_portfolio["weights"].items()
    ):
        print(f"{ticker}: {weight:.2%}")

    print(
        f"Expected return: "
        f"{minimum_volatility_portfolio['expected_return']:.2%}"
    )
    print(
        f"Volatility: "
        f"{minimum_volatility_portfolio['volatility']:.2%}"
    )
    print(
        f"Sharpe ratio: "
        f"{minimum_volatility_portfolio['sharpe_ratio']:.2f}"
    )

    print("\nMaximum-Sharpe weights:")
    for ticker, weight in (
        maximum_sharpe_portfolio["weights"].items()
    ):
        print(f"{ticker}: {weight:.2%}")

    print(
        f"Expected return: "
        f"{maximum_sharpe_portfolio['expected_return']:.2%}"
    )
    print(
        f"Volatility: "
        f"{maximum_sharpe_portfolio['volatility']:.2%}"
    )
    print(
        f"Sharpe ratio: "
        f"{maximum_sharpe_portfolio['sharpe_ratio']:.2f}"
    )

    print("\nEfficient frontier:")
    print(
        efficient_frontier[
            [
                "Expected Return",
                "Volatility",
                "Sharpe Ratio"
            ]
        ].head(10).to_string(
            index=False,
            formatters={
                "Expected Return": lambda value: (
                    f"{value:.2%}"
                ),
                "Volatility": lambda value: (
                    f"{value:.2%}"
                ),
                "Sharpe Ratio": lambda value: (
                    f"{value:.2f}"
                )
            }
        )
    )

    chart_path = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "efficient_frontier.png"
    )

    plot_efficient_frontier(
        efficient_frontier,
        current_portfolio_point,
        minimum_volatility_portfolio,
        maximum_sharpe_portfolio,
        show=False,
        save_path=chart_path
    )

    print(
        f"\nEfficient frontier chart saved to: "
        f"{chart_path}"
    )

    print(
        f"Maximum position weight: "
        f"{maximum_position_weight:.0%}"
    )

    print("\nBUY-AND-HOLD PORTFOLIO")
    print("----------------------")

    print("\nInitial shares:")
    for ticker, shares in buy_and_hold_shares.items():
        print(f"{ticker}: {shares:,.4f}")

    print(
        f"\nFinal portfolio value: "
        f"${buy_and_hold_value_history.iloc[-1]:,.2f}"
    )

    print(
        f"Total return: "
        f"{buy_and_hold_total_return:.2%}"
    )

    print(
        f"Annualized return: "
        f"{buy_and_hold_annualized_return:.2%}"
    )

    print(
        f"Annualized volatility: "
        f"{buy_and_hold_annualized_volatility:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{buy_and_hold_sharpe_ratio:.2f}"
    )

    print("\nEnding weights:")
    for ticker, weight in (
        buy_and_hold_weights.iloc[-1].items()
    ):
        print(f"{ticker}: {weight:.2%}")

    print("\nSTRATEGY COMPARISON")
    print("-------------------")

    print(
        strategy_comparison.to_string(
            formatters={
                "Final Value": lambda value: (
                    f"${value:,.2f}"
                ),
                "Total Return": lambda value: (
                    f"{value:.2%}"
                ),
                "Annualized Return": lambda value: (
                    f"{value:.2%}"
                ),
                "Annualized Volatility": lambda value: (
                    f"{value:.2%}"
                ),
                "Sharpe Ratio": lambda value: (
                    f"{value:.2f}"
                ),
                "Maximum Drawdown": lambda value: (
                    f"{value:.2%}"
                )
            }
        )
    )

    strategy_chart_path = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "strategy_growth.png"
    )

    plot_strategy_growth(
        strategy_value_history,
        show=False,
        save_path=strategy_chart_path
    )

    print(
        f"\nStrategy growth chart saved to: "
        f"{strategy_chart_path}"
    )

