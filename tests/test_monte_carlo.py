import pytest
from src.black_scholes import calculate_call_price, calculate_put_price
from src.monte_carlo import calculate_monte_carlo_price

def test_monte_carlo_call_matches_black_scholes():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    simulations = 100_000
    seed = 42

    monte_carlo_result = calculate_monte_carlo_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        simulations=simulations,
        seed=seed,
    )

    black_scholes_price = calculate_call_price(S, K, T, r, sigma)

    assert monte_carlo_result["price"] == pytest.approx(
        black_scholes_price,
        abs=0.10,
    )

    lower_bound, upper_bound = monte_carlo_result["confidence_interval"]

    assert lower_bound <= black_scholes_price <= upper_bound

def test_monte_carlo_put_matches_black_scholes():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    simulations = 100_000
    seed = 42

    monte_carlo_result = calculate_monte_carlo_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="put",
        simulations=simulations,
        seed=seed,
    )

    black_scholes_price = calculate_put_price(S, K, T, r, sigma)

    assert monte_carlo_result["price"] == pytest.approx(
        black_scholes_price,
        abs=0.10,
    )

    lower_bound, upper_bound = monte_carlo_result["confidence_interval"]

    assert lower_bound <= black_scholes_price <= upper_bound

def test_monte_carlo_rejects_invalid_option_type():
    with pytest.raises(
        ValueError,
        match="Option type must be either 'call' or 'put'.",
    ):
        calculate_monte_carlo_price(
            S=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            option_type="invalid",
            simulations=100_000,
            seed=42,
        )

def test_monte_carlo_rejects_negative_strike():
    with pytest.raises(
        ValueError,
        match="Strike price must be greater than zero.",
    ):
        calculate_monte_carlo_price(
            S=100,
            K=0,
            T=1,
            r=0.05,
            sigma=0.20,
            option_type="call",
            simulations=100_000,
            seed=42,
        )

def test_monte_carlo_rejects_strike():
    with pytest.raises(
        TypeError,
        match="Strike price must be a number.",
    ):
        calculate_monte_carlo_price(
            S=100,
            K="100",
            T=1,
            r=0.05,
            sigma=0.20,
            option_type="call",
            simulations=100_000,
            seed=42,
        )

def test_monte_carlo_rejects_negative_stock_price():
    with pytest.raises(
        ValueError,
        match="Stock price must be greater than zero.",
    ):
        calculate_monte_carlo_price(
            S=0,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            option_type="call",
            simulations=100_000,
            seed=42,
        )

def test_monte_carlo_rejects_stock_price():
    with pytest.raises(
        TypeError,
        match="Stock price must be a number.",
    ):
        calculate_monte_carlo_price(
            S="100",
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            option_type="call",
            simulations=100_000,
            seed=42,
        )