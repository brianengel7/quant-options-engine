import numpy as np

def calculate_binomial_price(S, K, T, r, sigma, steps, type = 'call', style = 'american'):

    if S <= 0:
        raise ValueError("Stock price must be greater than zero.")

    if K <= 0:
        raise ValueError("Strike price must be greater than zero.")

    if T <= 0:
        raise ValueError("Time to expiration must be greater than zero.")

    if sigma <= 0:
        raise ValueError("Volatility must be greater than zero.")

    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("Steps must be a positive integer.")

    if type not in ("call", "put"):
        raise ValueError(
            "Option type must be either 'call' or 'put'."
        )

    if style not in ("american", "european"):
        raise ValueError(
            "Exercise style must be either 'american' or 'european'."
        )

    dt = T / steps
    up_factor = np.exp(sigma * np.sqrt(dt))
    down_factor = 1 / up_factor
    risk_neutral_probability = (np.exp(r * dt) - down_factor) / (up_factor - down_factor)
    discount_factor = np.exp(-r * dt)

    if not 0 < risk_neutral_probability < 1:
        raise ValueError(
        "Risk-neutral probability must be between zero and one."
    )

    stock_prices = np.zeros(steps + 1)

    for down_moves in range(steps + 1):
        up_moves = steps - down_moves
        stock_prices[down_moves] = (S * up_factor**up_moves * down_factor**down_moves)

    if type == 'call':
        option_values = np.maximum(stock_prices - K, 0)
    elif type == 'put':
        option_values = np.maximum(K - stock_prices, 0)

    for step in range(steps - 1, -1, -1):
        for down_moves in range(step + 1):
            continuation_value = discount_factor * (risk_neutral_probability 
                            * option_values[down_moves] + 
                            (1 - risk_neutral_probability)
                            * option_values[down_moves + 1]
                            )
            option_values[down_moves] = continuation_value
            stock_price = (S * up_factor**(step - down_moves) * down_factor**(down_moves))
            if type == "call":
                exercise_value = max(stock_price - K, 0)
            else:
                exercise_value = max(K - stock_price, 0)

            if style == 'american':
                option_values[down_moves] = max(continuation_value, exercise_value)
            elif style == 'european':
                option_values[down_moves] = continuation_value

    return option_values[0]