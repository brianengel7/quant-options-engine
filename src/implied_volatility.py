from black_scholes import calculate_call_price, calculate_put_price

def calculate_call_iv(market_price, S, K, T, r, tolerance = 0.0001, max_iterations = 100):
    low_volatility = 0.0001
    high_volatility = 5
    for iteration in range(max_iterations):
        mid_volatility = (low_volatility + high_volatility) / 2
        model_price = calculate_call_price(S, K, T, r, mid_volatility)
        price_difference = abs(model_price - market_price)
        if price_difference < tolerance:
            return mid_volatility
        elif model_price < market_price:
            low_volatility = mid_volatility
        else:
            high_volatility = mid_volatility
    raise RuntimeError(
    "Call implied volatility did not converge within the maximum iterations."
    )