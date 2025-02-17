import pytest
from options_pricing.black_scholes import black_scholes
import sys
import os
import numpy as np

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from options_pricing.black_scholes import black_scholes


def test_black_scholes_call():
    """
    Test Black-Scholes call option pricing with known values.
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    expected_price = 10.4506  # Compute or use a known correct value
    computed_price = black_scholes(S, K, T, r, sigma, "call")
    assert abs(computed_price - expected_price) < 1e-4  # Allow small floating-point errors

def test_black_scholes_put():
    """
    Test Black-Scholes put option pricing with known values.
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    expected_price = 10.4506 - 100 + 100 * np.exp(-0.05 * 1)  # Exact put price from put-call parity
    computed_price = black_scholes(S, K, T, r, sigma, "put")
    assert abs(computed_price - expected_price) < 1e-4

def test_invalid_option_type():
    """
    Test that an invalid option type raises a ValueError.
    """
    with pytest.raises(ValueError):
        black_scholes(100, 100, 1, 0.05, 0.2, "invalid")

