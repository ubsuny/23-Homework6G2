import pytest
import numpy as np
from calculus_trithbha_version import simpson, trapezoid, adaptive_trapezoid  
# Function to calculate exp(-1/x) with handling for x near 0
def exp(x):
    if np.isinf(x):
        return 0
    safe_x = np.clip(x, 1e-10, np.inf)
    return np.where(x != 0, np.exp(-1/safe_x), 0)


# Function to calculate cos(1/x) with handling for x near 0
def cos(x):
    safe_x = np.clip(x, 1e-10, np.inf)
    return np.where(x != 0, np.cos(1/safe_x), 1)

# Cubic function for testing: x^3 + constant
def cubic(x, constant=1/2):
    return x**3 + constant

# Test for the exp function
def test_exp():
    assert exp(0) == 0
    assert exp(1) == np.exp(-1)
    assert np.isclose(exp(np.inf), 0)

# Test for the cos function
def test_cos():
    assert cos(0) == 1
    assert np.isclose(cos(1), np.cos(1))
    assert np.isclose(cos(np.inf), np.cos(0))

# Test for the cubic function
def test_cubic():
    assert cubic(1, 0.5) == 1.5
    assert cubic(0) == 0.5
    assert cubic(-1, -0.5) == -1.5

# Test for the simpson function
def test_simpson():
    result, _ = simpson(np.sin, 0, np.pi, 100, 1e-6)
    assert np.isclose(result, 2, atol=1e-6)

# Test for the trapezoid function
def test_trapezoid():
    result, _ = trapezoid(np.sin, 0, np.pi, 100, 1e-6)
    assert np.isclose(result, 2, atol=1e-6)

# Test for the adaptive_trapezoid function
def test_adaptive_trapezoid():
    result, _ = adaptive_trapezoid(np.sin, 0, np.pi, 1e-6)
    assert np.isclose(result, 2, atol=1e-6)

