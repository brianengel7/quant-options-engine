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



