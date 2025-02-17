import numpy as np
from scipy.stats import norm  # Needed for the cumulative distribution function (CDF)

def calculate_d1(S, K, T, r, sigma=0.2, q=0):
    return np.log(S/K) + (r + 0.5*sigma**2)*T / sigma * np.sqrt(T)

def black_scholes(S, K, T, r, sigma, q=0):
    d1 = calculate_d1(S, K, T, r, sigma, q)        
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    return call_price

def calculate_vega(S, K, T, r, sigma, q=0):
    d1 = calculate_d1(S, K, T, r, sigma, q)

    return S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)

def implied_volatility(C_market, S, K, r, T, sigma_0=0.2, q=0, tolerance=1e-5, max_iterations=100):
    sigma = sigma_0
    for i in range(max_iterations):
        C_BS = black_scholes(S, K, T, r, sigma, q)
        
        vega = calculate_vega(S, K, T, r, sigma, q)
        
        if vega == 0:
            return sigma
        
        sigma = sigma - (C_BS - C_market) / vega

        if abs(C_BS - C_market) < tolerance:
            return sigma

    return sigma

# Testing the function
if __name__ == "__main__":
    S = 100     # Example stock price
    K = 100     # Example strike price
    T = 1       # Time to expiry (1 year)
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)
    C_market = 10

    print(implied_volatility(C_market, S, K, r, T, sigma))
