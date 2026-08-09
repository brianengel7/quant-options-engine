import pytest
from src.black_scholes import calculate_call_price, calculate_put_price
from src.implied_volatility import calculate_call_iv, calculate_put_iv
import numpy as np

def test_call_iv():
    S = 100
    K = 105
    T = 0.5
    r = 0.05
    expected_sigma = 0.3
    market_price = calculate_call_price(S, K, T, r, expected_sigma)
    calculated_iv = calculate_call_iv(market_price, S, K, T, r)
    assert calculated_iv == pytest.approx(expected_sigma, abs = 0.0001)

def test_put_iv():
    S = 100
    K = 105
    T = 0.5
    r = 0.05
    expected_sigma = 0.3
    market_price = calculate_put_price(S, K, T, r, expected_sigma)
    calculated_iv = calculate_put_iv(market_price, S, K, T, r)
    assert calculated_iv == pytest.approx(expected_sigma, abs = 0.0001)

def test_call_iv_neg_price_check():
    with pytest.raises(ValueError):
        calculate_call_iv(-1, 100, 105, 0.5, 0.05)

def test_put_iv_neg_price_check():
    with pytest.raises(ValueError):
        calculate_put_iv(-1, 100, 105, 0.5, 0.05)

def test_call_iv_rejects_upper_bound():
    S = 100
    with pytest.raises(ValueError, match = 'Call market price must be below stock price.'):
        calculate_call_iv(
            market_price = S,
            S = S, 
            K = 105,
            T = 0.5,
            r = 0.05
        )

def test_put_iv_rejects_upper_bound():
    K = 105
    T = 0.5
    r = 0.05
    put_upper_bound = K * np.exp(-r * T)
    with pytest.raises(ValueError, match = 'Put market price must be below upper bound'):
        calculate_put_iv(
            market_price = put_upper_bound, 
            S = 100,
            K = 105,
            T = 0.5,
            r = 0.05
        )

def test_call_iv_rejects_lower_bound():
    S = 120
    K = 100
    T = 0.5
    r = 0.05
    call_lower_bound = max(0, S - K * np.exp(-r * T))
    invalid_market_price = call_lower_bound - 0.01
    with pytest.raises(ValueError, match = 'Call market price is below lower bound.'):
        calculate_call_iv(
            market_price = invalid_market_price,
            S = S,
            K = K,
            T = T,
            r = r
        )

def test_put_iv_rejects_lower_bound():
    S = 80
    K = 100
    T = 0.5
    r = 0.05
    put_lower_bound = max(0, K * np.exp(-r * T) - S)
    invalid_market_price = put_lower_bound - 0.01
    with pytest.raises(ValueError, match = 'Put market price is below lower bound.'):
        calculate_put_iv(
            market_price = invalid_market_price,
            S = S,
            K = K,
            T = T,
            r = r
        )