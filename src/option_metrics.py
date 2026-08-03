from black_scholes import calculate_call_price, calculate_put_price
from greeks import calculate_call_delta, calculate_put_delta, calculate_gamma, calculate_vega, calculate_theta_call, calculate_theta_put, calculate_rho_call, calculate_rho_put

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

    S = 100
    K = 105
    T = 30 / 365
    r = 0.05
    sigma = 0.20   

    results = calculate_option_metrics(S, K, T, r, sigma)

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")