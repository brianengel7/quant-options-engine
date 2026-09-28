import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
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

def calculate_historical_var(
    portfolio_returns,
    confidence_level=0.95
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    tail_probability = 1 - confidence_level

    return_quantile = portfolio_returns.quantile(
        tail_probability
    )

    historical_var = max(-return_quantile, 0.0)

    return historical_var

def calculate_dollar_var(
    portfolio_var,
    portfolio_value
):
    if portfolio_var < 0:
        raise ValueError(
            "Portfolio VaR cannot be negative."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be greater than zero."
        )

    dollar_var = portfolio_var * portfolio_value

    return dollar_var

def calculate_historical_expected_shortfall(
    portfolio_returns,
    confidence_level=0.95
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    tail_probability = 1 - confidence_level

    return_quantile = portfolio_returns.quantile(
        tail_probability
    )

    tail_returns = portfolio_returns[
        portfolio_returns <= return_quantile
    ]

    if tail_returns.empty:
        raise ValueError(
            "No returns were found beyond the VaR threshold."
        )

    expected_shortfall = max(
        -tail_returns.mean(),
        0.0
    )

    return expected_shortfall

def calculate_dollar_expected_shortfall(
    expected_shortfall,
    portfolio_value
):
    if expected_shortfall < 0:
        raise ValueError(
            "Expected shortfall cannot be negative."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be greater than zero."
        )

    dollar_expected_shortfall = (
        expected_shortfall * portfolio_value
    )

    return dollar_expected_shortfall

def calculate_parametric_var(
    portfolio_returns,
    confidence_level=0.95
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    if np.isclose(volatility, 0):
        raise ValueError(
            "Parametric VaR is undefined when volatility is zero."
        )

    tail_probability = 1 - confidence_level

    z_score = norm.ppf(tail_probability)

    return_threshold = (
        mean_return + z_score * volatility
    )

    parametric_var = max(
        -return_threshold,
        0.0
    )

    return parametric_var

def calculate_parametric_expected_shortfall(
    portfolio_returns,
    confidence_level=0.95
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    if np.isclose(volatility, 0):
        raise ValueError(
            "Parametric Expected Shortfall is undefined "
            "when volatility is zero."
        )

    tail_probability = 1 - confidence_level

    z_score = norm.ppf(tail_probability)

    expected_shortfall = (
        -mean_return
        + volatility
        * norm.pdf(z_score)
        / tail_probability
    )

    expected_shortfall = max(
        expected_shortfall,
        0.0
    )

    return expected_shortfall

def calculate_monte_carlo_risk(
    portfolio_returns,
    confidence_level=0.95,
    number_of_simulations=10000,
    horizon_days=1,
    random_seed=None
):
    if portfolio_returns.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    if number_of_simulations <= 0:
        raise ValueError(
            "Number of simulations must be greater than zero."
        )

    if horizon_days <= 0:
        raise ValueError(
            "Horizon days must be greater than zero."
        )

    daily_mean_return = portfolio_returns.mean()
    daily_volatility = portfolio_returns.std()

    if np.isclose(daily_volatility, 0):
        raise ValueError(
            "Monte Carlo risk is undefined when volatility is zero."
        )

    horizon_mean_return = (
        daily_mean_return * horizon_days
    )

    horizon_volatility = (
        daily_volatility * np.sqrt(horizon_days)
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    simulated_returns = random_generator.normal(
        loc=horizon_mean_return,
        scale=horizon_volatility,
        size=number_of_simulations
    )

    tail_probability = 1 - confidence_level

    return_threshold = np.quantile(
        simulated_returns,
        tail_probability
    )

    monte_carlo_var = max(
        -return_threshold,
        0.0
    )

    tail_returns = simulated_returns[
        simulated_returns <= return_threshold
    ]

    if len(tail_returns) == 0:
        raise ValueError(
            "No simulated returns were found beyond "
            "the VaR threshold."
        )

    monte_carlo_expected_shortfall = max(
        -tail_returns.mean(),
        0.0
    )

    return (
        monte_carlo_var,
        monte_carlo_expected_shortfall,
        simulated_returns
    )

def calculate_return_contributions(
    asset_returns,
    weights,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError(
            "Asset returns cannot be empty."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    weight_series = pd.Series(weights, dtype=float)

    missing_assets = set(asset_returns.columns) - set(
        weight_series.index
    )

    if missing_assets:
        raise ValueError(
            f"Missing weights for: {sorted(missing_assets)}"
        )

    weight_series = weight_series.reindex(
        asset_returns.columns
    )

    if not np.isclose(weight_series.sum(), 1.0):
        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    annualized_asset_returns = (
        asset_returns.mean() * trading_days
    )

    return_contributions = (
        weight_series * annualized_asset_returns
    )

    return_contributions.name = "Return Contribution"

    return return_contributions

def calculate_volatility_contributions(
    asset_returns,
    weights,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError(
            "Asset returns cannot be empty."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    weight_series = pd.Series(weights, dtype=float)

    missing_assets = set(asset_returns.columns) - set(
        weight_series.index
    )

    if missing_assets:
        raise ValueError(
            f"Missing weights for: {sorted(missing_assets)}"
        )

    weight_series = weight_series.reindex(
        asset_returns.columns
    )

    if not np.isclose(weight_series.sum(), 1.0):
        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    annualized_covariance = (
        asset_returns.cov() * trading_days
    )

    portfolio_variance = (
        weight_series.to_numpy()
        @ annualized_covariance.to_numpy()
        @ weight_series.to_numpy()
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    if np.isclose(portfolio_volatility, 0):
        raise ValueError(
            "Volatility contributions are undefined "
            "when portfolio volatility is zero."
        )

    marginal_contributions = (
        annualized_covariance
        @ weight_series
        / portfolio_volatility
    )

    volatility_contributions = (
        weight_series * marginal_contributions
    )

    percentage_contributions = (
        volatility_contributions
        / portfolio_volatility
    )

    results = pd.DataFrame({
        "Weight": weight_series,
        "Volatility Contribution": volatility_contributions,
        "Percentage of Portfolio Risk": percentage_contributions
    })

    return results
    

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

