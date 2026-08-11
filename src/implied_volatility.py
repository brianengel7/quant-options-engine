from black_scholes import calculate_call_price, calculate_put_price
import numpy as np
import pandas as pd

def calculate_iv(pricing_function, market_price, S, K, T, r, tolerance = 0.0001, max_iterations = 100):
    low_volatility = 0.0001
    high_volatility = 5
    for iteration in range(max_iterations):
        mid_volatility = (low_volatility + high_volatility) / 2
        model_price = pricing_function(S, K, T, r, mid_volatility)
        price_difference = abs(model_price - market_price)
        if price_difference < tolerance:
            return mid_volatility
        elif model_price < market_price:
            low_volatility = mid_volatility
        else:
            high_volatility = mid_volatility
    raise RuntimeError(
        "Implied volatility did not converge within the maximum iterations."
    )

def calculate_iv_smile(option_chain, option_type, S, T, r, strike_range = 0.2):
    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")
    if option_type == "call":
        iv_function = calculate_call_iv
    else:
        iv_function = calculate_put_iv
    valid_options = option_chain[
        (option_chain["bid"] > 0) &
        (option_chain["ask"] > 0) &
        (option_chain["ask"] >= option_chain["bid"])
    ]

    nearby_options = valid_options[
        (valid_options["strike"] >= S * (1 - strike_range)) &
        (valid_options["strike"] <= S * (1 + strike_range))
    ]
    smile_data = []
    for index, row in nearby_options.iterrows():
        strike = row['strike']
        market_midpoint = (row["bid"] + row["ask"]) / 2
        try:
            iv = iv_function(market_midpoint, S, strike, T, r)
            smile_data.append({'strike': strike, 'implied_volatility': iv})
        except (ValueError, RuntimeError):
            continue
    if not smile_data:
        return pd.DataFrame(
            columns=["strike", "implied_volatility"]
        )
    smile_df = pd.DataFrame(smile_data)
    return smile_df.sort_values("strike").reset_index(drop=True)

def calculate_call_iv(market_price, S, K, T, r, tolerance = 0.0001, max_iterations = 100):
    if market_price <= 0:
        raise ValueError("Market price must be greater than zero.")

    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike must be greater than zero.")

    if T <= 0:
        raise ValueError("Time to expiration must be greater than zero.")

    low_volatility = 0.0001
    high_volatility = 5
    call_lower_bound = max(0, S - K * np.exp(-r * T))
    if market_price < call_lower_bound:
        raise ValueError('Call market price is below lower bound.')
    elif market_price >= S:
        raise ValueError("Call market price must be below stock price.")
    return calculate_iv(
        calculate_call_price,
        market_price,
        S,
        K,
        T,
        r,
        tolerance,
        max_iterations
    )

def calculate_put_iv(market_price, S, K, T, r, tolerance = 0.0001, max_iterations = 100):

    if market_price <= 0:
        raise ValueError("Market price must be greater than zero.")

    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike must be greater than zero.")

    if T <= 0:
        raise ValueError("Time to expiration must be greater than zero.")

    low_volatility = 0.0001
    high_volatility = 5
    put_lower_bound = max(0, K * np.exp(-r * T) - S)
    put_upper_bound = K * np.exp(-r * T)
    if market_price < put_lower_bound:
        raise ValueError('Put market price is below lower bound.')
    if market_price >= put_upper_bound:
        raise ValueError('Put market price must be below upper bound')
    return calculate_iv(
            calculate_put_price,
            market_price,
            S,
            K,
            T,
            r,
            tolerance,
            max_iterations
        )
