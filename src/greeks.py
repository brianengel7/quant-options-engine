import numpy as np
from scipy.stats import norm
from black_scholes import calculate_call_price, calculate_put_price, calculate_d1, calculate_d1, calculate_d2

def calculate_call_delta(S, K, T, r, sigma):
    
    d1 = calculate_d1(S, K, T, r, sigma)

    call_delta = norm.cdf(d1)

    return call_delta

def calculate_put_delta(S, K, T, r, sigma):

    d1 = calculate_d1(S, K, T, r, sigma)

    put_delta = norm.cdf(d1) - 1

    return put_delta

def calculate_gamma(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return gamma

def calculate_vega(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)

    vega = S * norm.pdf(d1) * np.sqrt(T)

    return vega

def calculate_theta_call(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)

    return theta_call

def calculate_theta_put(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    return theta_put

def calculate_rho_call(S, K, T, r, sigma):
    d2 = calculate_d2(S, K, T, r, sigma)

    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2)

    return rho_call

def calculate_rho_put(S, K, T, r, sigma):
    d2 = calculate_d2(S, K, T, r, sigma)

    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return rho_put


#validation
if __name__ == "__main__":
    S = 100
    K = 105
    T = 30 / 365
    T2 = 28/365
    r = 0.05
    sigma = 0.20
    h = 0.0001

    call_delta = calculate_call_delta(S, K, T, r, sigma)
    put_delta = calculate_put_delta(S, K, T, r, sigma)

    delta_up = calculate_call_delta(S + h, K, T, r, sigma)
    delta_down = calculate_call_delta(S - h, K, T, r, sigma)

    price_1 = calculate_call_price(S, K, T, r, sigma)
    price_2 = calculate_call_price(S + h, K, T, r, sigma)
    numerical_delta = (price_2 - price_1) / h

    analytical_gamma = calculate_gamma(S, K, T, r, sigma)
    numerical_gamma = (delta_up - delta_down) / (2 * h)

    analytical_vega = calculate_vega(S, K, T, r, sigma)
    numerical_vega = (calculate_call_price(S, K, T, r, sigma + h) - calculate_call_price(S, K, T, r, sigma - h)) / (2 * h)

    analytical_theta_call = calculate_theta_call(S, K, T, r, sigma)
    analytical_theta_put = calculate_theta_put(S, K, T, r, sigma)
    numerical_theta_call = (calculate_call_price(S, K, T2, r, sigma) - calculate_call_price(S, K, T, r, sigma)) / (T - T2)
    numerical_theta_put = (calculate_put_price(S, K, T2, r, sigma) - calculate_put_price(S, K, T, r, sigma)) / (T - T2)
    analytical_theta_callPerDay = analytical_theta_call / 365
    analytical_theta_putPerDay = analytical_theta_put / 365

    analytical_rho_call = calculate_rho_call(S, K, T, r, sigma)
    analytical_rho_put = calculate_rho_put(S, K, T, r, sigma)
    numerical_rho_call = (calculate_call_price(S, K, T, r + h, sigma) - calculate_call_price(S, K, T, r - h, sigma))/ (2 * h)
    numerical_rho_put = (calculate_put_price(S, K, T, r + h, sigma) - calculate_put_price(S, K, T, r - h, sigma))/ (2 * h)

    print("Call Delta:", call_delta)
    print("Put Delta:", put_delta)
    print("Gamma:", analytical_gamma)
    print("Vega:", analytical_vega)
    print("Theta Call:", analytical_theta_call)
    print("Theta Put:", analytical_theta_put)
    print("Theta Call/Day:", analytical_theta_callPerDay)
    print("Theta Put/Day:", analytical_theta_putPerDay)
    print("Rho Call:", analytical_rho_call)
    print("Rho Put:", analytical_rho_put)
    print(call_delta - put_delta)
    print("Delta difference:", call_delta - numerical_delta)
    print ("Gamma difference:", analytical_gamma - numerical_gamma)
    print ("Vega difference:", analytical_vega - numerical_vega)
    print ("Theta Call difference:", analytical_theta_call - numerical_theta_call)
    print ("Theta Put difference:", analytical_theta_put - numerical_theta_put)
    print ("Rho Call difference:", analytical_rho_call - numerical_rho_call)
    print ("Rho Put difference:", analytical_rho_put - numerical_rho_put)