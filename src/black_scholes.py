import numpy as np
from scipy.stats import norm

def calculate_d1(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return d1

def calculate_d2(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return d2   

def calculate_call_price(S, K, T, r, sigma):

    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price

def calculate_put_price(S, K, T, r, sigma):

    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return put_price

S = 100
K = 105
T = 30 / 365
r = 0.05
sigma = 0.40

call_price = calculate_call_price(
    S,
    K,
    T,
    r,
    sigma
)

put_price = calculate_put_price(
    S,
    K,
    T,
    r,
    sigma
)

if __name__ == "__main__":
    validation_price = call_price - put_price

    print("Call Price:", call_price)
    print("Put Price:", put_price)
    print("Validation Price:", validation_price)
