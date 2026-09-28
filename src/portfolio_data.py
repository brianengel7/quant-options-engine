import yfinance as yf

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