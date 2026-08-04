from black_scholes import calculate_call_price, calculate_put_price
from greeks import calculate_call_delta, calculate_put_delta, calculate_gamma, calculate_vega, calculate_theta_call, calculate_theta_put, calculate_rho_call, calculate_rho_put
from market_data import get_historical_prices, get_option_expirations, get_option_chain, get_risk_free_rate, calculate_log_returns, calculate_historical_volatility, time_to_expiration, calculate_closest_option
from implied_volatility import calculate_call_iv

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
    T = time_to_expiration(expiration)
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

    call_price_difference = call_model_price - call_market_price
    put_price_difference = put_model_price - put_market_price
    call_iv = calculate_call_iv(call_market_price, S, K, T, r)

    print(f"Current stock price: ${S:.2f}")
    print(f"Strike price: ${K:.2f}")
    print(f"Historical Volatility: {sigma:.2%}")
    print(f"Expiration date: {expiration}")
    print(f"Time to expiration: {T:.6f} years")
    print(f"Risk-Free Rate: {r:.2%}")
    print(f'Call Implied Volatility: {call_iv:.2%}')
    print()
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")