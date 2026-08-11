import numpy as np
from black_scholes import calculate_call_price, calculate_put_price

def simulate_terminal_prices(
        S,
        T,
        r,
        sigma,
        simulations = 100_000,
        seed = None,
):

    if not isinstance(S, (int, float)) or isinstance(S, bool):
        raise TypeError("Stock price must be a number.")

    if not isinstance(T, (int, float)) or isinstance(T, bool):
        raise TypeError("Time to expiration must be a number.")

    if not isinstance(r, (int, float)) or isinstance(r, bool):
        raise TypeError("Risk-free rate must be a number.")

    if not isinstance(sigma, (int, float)) or isinstance(sigma, bool):
        raise TypeError("Volatility must be a number.")

    if S <= 0:
        raise ValueError("Stock price must be greater than zero.")

    if T <= 0:
        raise ValueError("Time to expiration must be greater than zero.")

    if sigma <= 0:
        raise ValueError("Volatility must be greater than zero.")

    if (
        not isinstance(simulations, int)
        or isinstance(simulations, bool)
        or simulations <= 0
    ):
        raise ValueError("Simulations must be a positive integer.")

    if seed is not None and (
        not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("Seed must be an integer or None.")

    rng = np.random.default_rng(seed)
    random_shocks = rng.standard_normal(simulations)
    terminal_prices = S * np.exp((r - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * random_shocks
    )
    return terminal_prices

def calculate_monte_carlo_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type,
    simulations=100_000,
    seed=None,
):
    if not isinstance(K, (int, float)) or isinstance(K, bool):
        raise TypeError("Strike price must be a number.")
    if K <= 0:
        raise ValueError("Strike price must be greater than zero.")
    if option_type not in ("call", "put"):
        raise ValueError("Option type must be either 'call' or 'put'.")
    terminal_prices = simulate_terminal_prices(
        S=S,
        T=T,
        r=r,
        sigma=sigma,
        simulations=simulations,
        seed=seed,
    )
    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0)
    discounted_payoffs = np.exp(-r * T) * payoffs
    option_price = np.mean(discounted_payoffs)
    standard_deviation = np.std(discounted_payoffs, ddof=1)
    standard_error = standard_deviation / np.sqrt(simulations)
    confidence_margin = 1.96 * standard_error

    confidence_interval = (
        option_price - confidence_margin,
        option_price + confidence_margin,
    )

    return {
    "price": option_price,
    "standard_error": standard_error,
    "confidence_interval": confidence_interval,
    }


if __name__ == "__main__":

    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    simulations = 100_000
    seed = 42

    monte_carlo_call = calculate_monte_carlo_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        simulations=simulations,
        seed=seed,
    )

    monte_carlo_put = calculate_monte_carlo_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="put",
        simulations=simulations,
        seed=seed,
    )
    black_scholes_call = calculate_call_price(
        S,
        K,
        T,
        r,
        sigma,
    )

    black_scholes_put = calculate_put_price(
        S,
        K,
        T,
        r,
        sigma,
    )
    call_difference = (
        monte_carlo_call["price"] - black_scholes_call
    )

    put_difference = (
        monte_carlo_put["price"] - black_scholes_put
    )

    print("Call option:")
    print(f"  Monte Carlo: ${monte_carlo_call['price']:.4f}")
    print(f"  Standard error: ${monte_carlo_call['standard_error']:.4f}")
    print(
        "  95% confidence interval: "
        f"(${monte_carlo_call['confidence_interval'][0]:.4f}, "
        f"${monte_carlo_call['confidence_interval'][1]:.4f})"
    )
    print(f"  Black-Scholes: ${black_scholes_call:.4f}")
    print(f"  Difference: ${call_difference:.4f}")

    print()

    print("Put option:")
    print(f"  Monte Carlo: ${monte_carlo_put['price']:.4f}")
    print(f"  Standard error: ${monte_carlo_put['standard_error']:.4f}")
    print(
        "  95% confidence interval: "
        f"(${monte_carlo_put['confidence_interval'][0]:.4f}, "
        f"${monte_carlo_put['confidence_interval'][1]:.4f})"
    )
    print(f"  Black-Scholes: ${black_scholes_put:.4f}")
    print(f"  Difference: ${put_difference:.4f}")
