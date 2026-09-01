import yfinance as yf
import pandas as pd
import numpy as np


def get_portfolio_prices(tickers, start_date, end_date=None):

    if not isinstance(tickers, list):
        raise TypeError("Tickers must be provided as a list.")

    if len(tickers) < 2:
        raise ValueError("A portfolio must contain at least two assets.")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    close_prices = data["Close"]

    close_prices = close_prices.dropna(how="any")

    if close_prices.empty:
        raise ValueError("No complete price data was returned.")

    return close_prices

def calculate_asset_returns(prices):
    if prices.empty:
        raise ValueError("Price data cannot be empty.")

    asset_returns = prices.pct_change(fill_method=None).dropna()

    return asset_returns

def calculate_portfolio_returns(asset_returns, weights):
    if asset_returns.empty:
        raise ValueError('Asset returns cannot be empty.')
    if not isinstance(weights, dict):
        raise TypeError("Weights must be provided as a dictionary.")
    if set(weights.keys()) != set(asset_returns.columns):
        raise ValueError(
            "Weight tickers must match the asset return columns."
        )
    if any(weight < 0 for weight in weights.values()):
        raise ValueError(
            "Portfolio weights cannot be negative."
        )
    if not abs(sum(weights.values()) - 1.0) < 0.0001:
        raise ValueError(
            "Portfolio weights must sum to 1.0."
        )
    weights_series = pd.Series(weights)
    portfolio_returns = asset_returns.dot(weights_series)
    portfolio_returns.name = "Portfolio Return"


    return portfolio_returns

def calculate_cumulative_returns(portfolio_returns):
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    cumulative_returns.name = "Cumulative Return"

    return cumulative_returns

def calculate_portfolio_value(
    portfolio_returns,
    initial_investment=10000
):
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    if initial_investment <= 0:
        raise ValueError(
            "Initial investment must be greater than zero."
        )

    portfolio_value = (
        initial_investment
        * (1 + portfolio_returns).cumprod()
    )

    portfolio_value.name = "Portfolio Value"

    return portfolio_value

def calculate_total_return(portfolio_returns):
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    total_return = (1 + portfolio_returns).prod() - 1

    return total_return

def calculate_annualized_return(
    portfolio_returns,
    trading_days=252
):
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    number_of_days = len(portfolio_returns)

    total_growth = (1 + portfolio_returns).prod()

    annualized_return = (
        total_growth ** (trading_days / number_of_days)
    ) - 1

    return annualized_return

def calculate_annualized_volatility(
    portfolio_returns,
    trading_days=252
):
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    daily_volatility = portfolio_returns.std()

    annualized_volatility = (
        daily_volatility * np.sqrt(trading_days)
    )

    return annualized_volatility

def calculate_covariance_matrix(
    asset_returns,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError("Asset returns cannot be empty.")

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    daily_covariance = asset_returns.cov()

    annualized_covariance = (
        daily_covariance * trading_days
    )

    return annualized_covariance

def calculate_correlation_matrix(asset_returns):
    if asset_returns.empty:
        raise ValueError("Asset returns cannot be empty.")

    correlation_matrix = asset_returns.corr()

    return correlation_matrix

def calculate_portfolio_volatility(
    asset_returns,
    weights,
    trading_days=252
):
    if asset_returns.empty:
        raise ValueError("Asset returns cannot be empty.")

    if set(weights.keys()) != set(asset_returns.columns):
        raise ValueError(
            "Weight tickers must match the asset return columns."
        )

    if not abs(sum(weights.values()) - 1.0) < 0.0001:
        raise ValueError(
            "Portfolio weights must sum to 1.0."
        )

    covariance_matrix = calculate_covariance_matrix(
        asset_returns,
        trading_days
    )

    weights_series = pd.Series(weights)

    portfolio_variance = (
        weights_series
        @ covariance_matrix
        @ weights_series
    )

    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_volatility

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0, trading_days=252):
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    if risk_free_rate <= -1:
        raise ValueError(
            "Risk-free rate must be greater than -100%."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    daily_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / trading_days)
    ) - 1

    daily_excess_returns = (
        portfolio_returns - daily_risk_free_rate
    )

    daily_volatility = portfolio_returns.std()

    if daily_volatility == 0:
        raise ValueError(
            "Sharpe ratio is undefined when volatility is zero."
        )

    sharpe_ratio = (
        daily_excess_returns.mean()
        / daily_volatility
        * np.sqrt(trading_days)
    )

    return sharpe_ratio

def calculate_sortino_ratio(
    portfolio_returns,
    minimum_acceptable_return=0.0,
    trading_days=252
):
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    if minimum_acceptable_return <= -1:
        raise ValueError(
            "Minimum acceptable return must be greater than -100%."
        )

    if trading_days <= 0:
        raise ValueError(
            "Trading days must be greater than zero."
        )

    daily_target_return = (
        (1 + minimum_acceptable_return)
        ** (1 / trading_days)
    ) - 1

    excess_returns = (
        portfolio_returns - daily_target_return
    )

    downside_returns = excess_returns.clip(upper=0)

    downside_deviation = np.sqrt(
        (downside_returns ** 2).mean()
    )

    if np.isclose(downside_deviation, 0):
        raise ValueError(
            "Sortino ratio is undefined when downside deviation is zero."
        )

    sortino_ratio = (
        excess_returns.mean()
        / downside_deviation
        * np.sqrt(trading_days)
    )

    return sortino_ratio

def calculate_drawdowns(portfolio_returns):
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    growth_index = (1 + portfolio_returns).cumprod()

    running_peak = growth_index.cummax()

    drawdowns = growth_index / running_peak - 1
    drawdowns.name = "Drawdown"

    return drawdowns

def calculate_max_drawdown(portfolio_returns):
    drawdowns = calculate_drawdowns(portfolio_returns)

    maximum_drawdown = drawdowns.min()

    return maximum_drawdown

def calculate_calmar_ratio(portfolio_returns, trading_days=252):
    annualized_return = calculate_annualized_return(
        portfolio_returns,
        trading_days
    )

    maximum_drawdown = calculate_max_drawdown(
        portfolio_returns
    )

    if np.isclose(maximum_drawdown, 0):
        raise ValueError(
            "Calmar ratio is undefined when maximum drawdown is zero."
        )

    calmar_ratio = (
        annualized_return / abs(maximum_drawdown)
    )

    return calmar_ratio

def get_benchmark_returns(
    benchmark_ticker,
    start_date,
    end_date=None
):
    if not isinstance(benchmark_ticker, str):
        raise TypeError(
            "Benchmark ticker must be a string."
        )

    if not benchmark_ticker.strip():
        raise ValueError(
            "Benchmark ticker cannot be empty."
        )

    data = yf.download(
        tickers=benchmark_ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            "No benchmark data was returned."
        )

    close_prices = data["Close"].squeeze()

    benchmark_returns = (
        close_prices
        .pct_change(fill_method=None)
        .dropna()
    )

    benchmark_returns.name = benchmark_ticker

    return benchmark_returns

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
    

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM"]

    weights = {
        "AAPL": 0.40,
        "MSFT": 0.35,
        "JPM": 0.25
    }

    prices = get_portfolio_prices(
        tickers=tickers,
        start_date="2025-01-01"
    )

    asset_returns = calculate_asset_returns(prices)

    portfolio_returns = calculate_portfolio_returns(
        asset_returns,
        weights
    )

    cumulative_returns = calculate_cumulative_returns(
        portfolio_returns
    )

    portfolio_value = calculate_portfolio_value(
        portfolio_returns,
        initial_investment=10000
    )

    total_return = calculate_total_return(
        portfolio_returns
    )

    annualized_return = calculate_annualized_return(
        portfolio_returns
    )

    print("Portfolio returns:")
    print(portfolio_returns.head())

    print("\nCumulative returns:")
    print(cumulative_returns.head())

    print("\nPortfolio value:")
    print(portfolio_value.head())

    print(f"\nFinal portfolio value: ${portfolio_value.iloc[-1]:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized return: {annualized_return:.2%}")

    annualized_volatility = calculate_annualized_volatility(
    portfolio_returns
    )

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

    risk_free_rate = 0.04

    sharpe_ratio = calculate_sharpe_ratio(
        portfolio_returns,
        risk_free_rate=risk_free_rate
    )

    sortino_ratio = calculate_sortino_ratio(
        portfolio_returns,
        minimum_acceptable_return=0.04
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

    benchmark_ticker = "SPY"

    benchmark_returns = get_benchmark_returns(
        benchmark_ticker=benchmark_ticker,
        start_date="2025-01-01"
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

    confidence_level = 0.95
    portfolio_value = 100000

    historical_var = calculate_historical_var(
        portfolio_returns,
        confidence_level=confidence_level
    )

    historical_dollar_var = calculate_dollar_var(
        historical_var,
        portfolio_value
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
            portfolio_value
        )
    )

    print(
        f"Annualized volatility: {annualized_volatility:.2%}"
    )

    print(
        f"Covariance-based volatility: "
        f"{covariance_volatility:.2%}"
    )

    print(f"Annualized return: {annualized_return:.2%}")
    print(f"Annualized volatility: {annualized_volatility:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Sortino ratio: {sortino_ratio:.2f}")
    print(f"Maximum drawdown: {maximum_drawdown:.2%}")
    print(f"Calmar ratio: {calmar_ratio:.2f}")
    print(f"Benchmark: {benchmark_ticker}")
    print(f"Portfolio beta: {portfolio_beta:.2f}")
    print(f"Annualized alpha: {portfolio_alpha:.2%}")
    print(f"Tracking error: {tracking_error:.2%}")
    print(f"Information ratio: {information_ratio:.2f}")
    print(f"Historical VaR ({confidence_level:.0%}): {historical_var:.2%}")
    print(f"Historical dollar VaR: ${historical_dollar_var:,.2f}")
    print(f"Historical Expected Shortfall ({confidence_level:.0%}): {historical_expected_shortfall:.2%}")
    print(f"Historical dollar Expected Shortfall: ${historical_dollar_expected_shortfall:,.2f}")

    print("\nAligned portfolio and benchmark returns:")
    print(aligned_returns.head())



