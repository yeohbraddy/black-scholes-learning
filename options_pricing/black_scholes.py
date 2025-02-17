import numpy as np
from scipy.stats import norm  # Needed for the cumulative distribution function (CDF)

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Computes the Black-Scholes option price for European call/put options.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate (as a decimal)
        sigma (float): Volatility of the underlying asset (as a decimal)
        option_type (str): "call" or "put"

    Returns:
        float: Option price
    """
    # Step 1: Compute d1
    d1 = np.log(S/K) + (r + 0.5*sigma**2)*T / sigma * np.sqrt(T)

    # Step 2: Compute d2
    d2 = d1 - sigma * np.sqrt(T)

    # Step 3: Compute option price based on type
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

# Testing the function
if __name__ == "__main__":
    S = 100     # Example stock price
    K = 100     # Example strike price
    T = 1       # Time to expiry (1 year)
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)

    print("Call Price:", black_scholes(S, K, T, r, sigma, "call"))
    print("Put Price:", black_scholes(S, K, T, r, sigma, "put"))
