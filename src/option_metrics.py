from black_scholes import calculate_call_price, calculate_put_price
from greeks import calculate_call_delta, calculate_put_delta, calculate_gamma, calculate_vega, calculate_theta_call, calculate_theta_put, calculate_rho_call, calculate_rho_put
from market_data import get_historical_prices, get_option_expirations, get_option_chain, get_risk_free_rate, calculate_log_returns, calculate_historical_volatility, time_to_expiration, calculate_closest_option
from implied_volatility import calculate_call_iv, calculate_put_iv, calculate_iv_smile
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from binomial_tree import calculate_binomial_price

def calculate_option_metrics(S, K, T, r, sigma):

    call_price = calculate_call_price(S, K, T, r, sigma)
    put_price = calculate_put_price(S, K, T, r, sigma)
    call_delta = calculate_call_delta(S, K, T, r, sigma)
    put_delta = calculate_put_delta(S, K, T, r, sigma)
    gamma = calculate_gamma(S, K, T, r, sigma)
    vega = calculate_vega(S, K, T, r, sigma)
    call_theta = calculate_theta_call(S, K, T, r, sigma)
    put_theta = calculate_theta_put(S, K, T, r, sigma)
    call_rho = calculate_rho_call(S, K, T, r, sigma)
    put_rho = calculate_rho_put(S, K, T, r, sigma)

    metrics = {
        "Call Price" : call_price,
        "Put Price" : put_price,
        "Call Delta" : call_delta,
        "Put Delta" : put_delta,
        "Gamma" : gamma,
        "Vega" : vega,
        "Call Theta" : call_theta,
        "Put Theta" : put_theta,
        "Call Rho" : call_rho,
        "Put Rho" : put_rho,
    }

    return metrics

if __name__ == "__main__":

    ticker = "AAPL"
    prices = get_historical_prices(ticker)
    expirations = get_option_expirations(ticker)
    for expiration in expirations:
        T = time_to_expiration(expiration)

        if T > 0:
            break
    if T <= 0:
        raise ValueError("No expiration with positive time remaining was found.")
    log_returns = calculate_log_returns(prices)
    sigma = calculate_historical_volatility(log_returns)
    S = prices.iloc[-1]
    r = get_risk_free_rate()

    calls, puts = get_option_chain(ticker, expiration)
    closest_call = calculate_closest_option(calls, S)
    closest_put = calculate_closest_option(puts, S)
    call_K = closest_call["strike"]
    put_K = closest_put["strike"]
    call_bid = closest_call['bid']
    call_ask = closest_call['ask']
    put_bid = closest_put['bid']
    put_ask = closest_put['ask']
    call_market_price = (call_bid + call_ask) / 2
    put_market_price = (put_bid + put_ask) / 2
    if call_K != put_K:
        raise ValueError("The closest call and put strikes do not match.")
    K =  call_K
    results = calculate_option_metrics(S, K, T, r, sigma)

    call_model_price = results['Call Price']
    put_model_price = results['Put Price']

    steps = 500

    european_binomial_call = calculate_binomial_price(
        S=S,
        K=call_K,
        T=T,
        r=r,
        sigma=sigma,
        steps=steps,
        type="call",
        style="european",
    )

    american_binomial_call = calculate_binomial_price(
        S=S,
        K=call_K,
        T=T,
        r=r,
        sigma=sigma,
        steps=steps,
        type="call",
        style="american",
    )

    european_binomial_put = calculate_binomial_price(
        S=S,
        K=put_K,
        T=T,
        r=r,
        sigma=sigma,
        steps=steps,
        type="put",
        style="european",
    )

    american_binomial_put = calculate_binomial_price(
        S=S,
        K=put_K,
        T=T,
        r=r,
        sigma=sigma,
        steps=steps,
        type="put",
        style="american",
    )

    call_binomial_difference = (
        european_binomial_call - call_model_price
    )

    put_binomial_difference = (
        european_binomial_put - put_model_price
    )

    call_early_exercise_premium = (
        american_binomial_call - european_binomial_call
    )

    put_early_exercise_premium = (
        american_binomial_put - european_binomial_put
    )


    call_price_difference = call_model_price - call_market_price
    put_price_difference = put_model_price - put_market_price
    call_iv = calculate_call_iv(call_market_price, S, call_K, T, r)
    put_iv = calculate_put_iv(put_market_price, S, put_K, T, r)

    call_smile_df = calculate_iv_smile(
        calls,
        "call",
        S,
        T,
        r
    )

    put_smile_df = calculate_iv_smile(
        puts,
        "put",
        S,
        T,
        r
    )


    print(f"Current stock price: ${S:.2f}")
    print(f"Strike price: ${K:.2f}")
    print(f"Historical Volatility: {sigma:.2%}")
    print(f"Expiration date: {expiration}")
    print(f"Time to expiration: {T:.6f} years")
    print(f"Risk-Free Rate: {r:.2%}")
    print(f'Call Implied Volatility: {call_iv:.2%}')
    print(f'Put Implied Volatility: {put_iv:.2%}')
    print()
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")

    print()
    print(f"Binomial Tree Results ({steps} steps)")
    print()

    print("Call:")
    print(f"  Market midpoint: ${call_market_price:.4f}")
    print(f"  Black-Scholes price: ${call_model_price:.4f}")
    print(f"  European binomial price: ${european_binomial_call:.4f}")
    print(f"  American binomial price: ${american_binomial_call:.4f}")
    print(f"  Binomial vs. Black-Scholes: ${call_binomial_difference:.4f}")
    print(f"  Early-exercise premium: ${call_early_exercise_premium:.4f}")

    print()

    print("Put:")
    print(f"  Market midpoint: ${put_market_price:.4f}")
    print(f"  Black-Scholes price: ${put_model_price:.4f}")
    print(f"  European binomial price: ${european_binomial_put:.4f}")
    print(f"  American binomial price: ${american_binomial_put:.4f}")
    print(f"  Binomial vs. Black-Scholes: ${put_binomial_difference:.4f}")
    print(f"  Early-exercise premium: ${put_early_exercise_premium:.4f}")
    
    fig, ax = plt.subplots(figsize = (10,6))
    ax.plot(
        call_smile_df["strike"],
        call_smile_df["implied_volatility"],
        marker="o",
        label="Calls"
    )
    ax.plot(
        put_smile_df["strike"],
        put_smile_df["implied_volatility"],
        marker="o",
        label="Puts"
    )
    ax.axvline(
        S,
        color="black",
        linestyle="--",
        label=f"Stock price: ${S:.2f}"
    )
    ax.set_xlabel("Strike Price ($)")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"{ticker} Implied Volatility by Strike")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    plt.show()

