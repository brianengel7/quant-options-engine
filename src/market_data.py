import yfinance as yf
import numpy as np
from datetime import datetime

def get_historical_prices(ticker, period = "1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period = period)
    return data["Close"]

def get_option_expirations(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    return expirations

def get_risk_free_rate():
    treasury = yf.Ticker("^IRX")
    data = treasury.history(period = "5d")
    latest_yield = data["Close"].iloc[-1] / 100
    return latest_yield

def get_option_chain(ticker, expiration):
    stock = yf.Ticker(ticker)
    option_chain = stock.option_chain(expiration)
    return option_chain.calls, option_chain.puts

def calculate_log_returns(prices):
    price_ratios = prices / prices.shift(1)
    log_returns = np.log(price_ratios).dropna()
    return log_returns

def calculate_historical_volatility(log_returns):
    daily_volatility = log_returns.std()
    historical_volatility = daily_volatility * np.sqrt(252)

    return historical_volatility

def time_to_expiration(expiration):
    expiration_date = datetime.strptime(expiration, "%Y-%m-%d")
    today = datetime.today()
    T = (expiration_date - today).days / 365
    return T

def calculate_closest_option(options, S):
    strike_distances = (options["strike"] - S).abs()
    closest_index = strike_distances.idxmin()
    closest_option = options.loc[closest_index]
    return closest_option


if __name__ == "__main__":
    prices = get_historical_prices("AAPL")
    expirations = get_option_expirations("AAPL")
    expiration = expirations[0]
    log_returns = calculate_log_returns(prices)
    historical_volatility = calculate_historical_volatility(log_returns)
    T = time_to_expiration(expiration)
    calls, puts = get_option_chain("AAPL", expiration)
    closest_call = calculate_closest_option(calls, prices.iloc[-1])
    closest_put = calculate_closest_option(puts, prices.iloc[-1])

    print(f"Closest Call: {closest_call}")
    print(f"Closest Put: {closest_put}")

    print(f"Annualized historical volatility: {historical_volatility:.2%}")
    print(T)
