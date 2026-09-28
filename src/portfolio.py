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
    plot_strategy_growth,
    plot_monte_carlo_distribution
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

from portfolio_reporting import (
    print_portfolio_report
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

    monte_carlo_chart_path = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "monte_carlo_distribution.png"
    )

    plot_monte_carlo_distribution(
        simulated_returns,
        monte_carlo_var,
        monte_carlo_expected_shortfall,
        confidence_level=confidence_level,
        number_of_bins=60,
        show=False,
        save_path=monte_carlo_chart_path
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

    current_optimization_weights = pd.Series(
        current_weight_array,
        index=asset_returns.columns
    )

    portfolio_report = {
        "inputs": {
            "tickers": tickers,
            "weights": weights,
            "start_date": start_date,
            "initial_investment": initial_investment,
            "risk_free_rate": risk_free_rate,
            "confidence_level": confidence_level,
            "maximum_position_weight": (
                maximum_position_weight
            )
        },
        "previews": {
            "portfolio_returns": portfolio_returns,
            "cumulative_returns": cumulative_returns,
            "portfolio_value": (
                rebalanced_value_history
            )
        },
        "performance": {
            "current_value": current_portfolio_value,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": (
                annualized_volatility
            ),
            "covariance_volatility": (
                covariance_volatility
            )
        },
        "risk_adjusted": {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "maximum_drawdown": maximum_drawdown,
            "calmar_ratio": calmar_ratio,
            "drawdowns": drawdowns
        },
        "benchmark": {
            "ticker": benchmark_ticker,
            "beta": portfolio_beta,
            "alpha": portfolio_alpha,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "aligned_returns": aligned_returns
        },
        "risk": {
            "historical": {
                "var": historical_var,
                "dollar_var": historical_dollar_var,
                "expected_shortfall": (
                    historical_expected_shortfall
                ),
                "dollar_expected_shortfall": (
                    historical_dollar_expected_shortfall
                )
            },
            "parametric": {
                "var": parametric_var,
                "dollar_var": parametric_dollar_var,
                "expected_shortfall": (
                    parametric_expected_shortfall
                ),
                "dollar_expected_shortfall": (
                    parametric_dollar_expected_shortfall
                )
            },
            "monte_carlo": {
                "var": monte_carlo_var,
                "dollar_var": monte_carlo_dollar_var,
                "expected_shortfall": (
                    monte_carlo_expected_shortfall
                ),
                "dollar_expected_shortfall": (
                    monte_carlo_dollar_expected_shortfall
                )
            }
        },
        "matrices": {
            "covariance": covariance_matrix,
            "correlation": correlation_matrix
        },
        "contributions": {
            "return": return_contributions,
            "volatility": volatility_contributions
        },
        "optimization": {
            "current": {
                "weights": current_optimization_weights,
                "expected_return": (
                    current_portfolio_statistics["return"]
                ),
                "volatility": (
                    current_portfolio_statistics[
                        "volatility"
                    ]
                ),
                "sharpe_ratio": (
                    current_portfolio_statistics[
                        "sharpe_ratio"
                    ]
                )
            },
            "minimum_volatility": (
                minimum_volatility_portfolio
            ),
            "maximum_sharpe": (
                maximum_sharpe_portfolio
            ),
            "efficient_frontier": efficient_frontier
        },
        "buy_and_hold": {
            "shares": buy_and_hold_shares,
            "final_value": (
                buy_and_hold_value_history.iloc[-1]
            ),
            "total_return": (
                buy_and_hold_total_return
            ),
            "annualized_return": (
                buy_and_hold_annualized_return
            ),
            "annualized_volatility": (
                buy_and_hold_annualized_volatility
            ),
            "sharpe_ratio": (
                buy_and_hold_sharpe_ratio
            ),
            "maximum_drawdown": (
                buy_and_hold_max_drawdown
            ),
            "ending_weights": (
                buy_and_hold_weights.iloc[-1]
            )
        },
        "strategy_comparison": strategy_comparison,
        "charts": {
            "efficient_frontier": chart_path,
            "strategy_growth": strategy_chart_path,
            "monte_carlo_distribution": monte_carlo_chart_path
        }
    }

    print_portfolio_report(
        portfolio_report
    )

