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

# Test for the 'exp' function with a different function as input
def test_exp_with_function():
    # Define a test function (lambda function) that squares its input
    test_function = lambda x: x**2

    # Test the 'exp' function by applying the test_function to 0.
    # Since test_function(0) is 0, exp(-1/0) should ideally be 0.
    # However, due to handling in 'exp', it should return np.exp(-1/small_number) for small inputs.
    assert np.isclose(exp(test_function(0)), np.exp(-1/test_function(1e-15)))

    # Test the 'exp' function with the test_function applied to 1.
    # Here, test_function(1) is 1, so exp(-1/1) should return np.exp(-1), 
    # which is the expected behavior for input 1 to the 'exp' function.
    assert np.isclose(exp(test_function(1)), np.exp(-1))

# Test for the 'cubic' function with a different function as input
def test_cubic_with_function():
    # Define a linear test function (lambda function) that represents a simple linear relationship
    test_function = lambda x: 2*x + 3

    # Test the 'cubic' function with the test_function applied to 0.
    # The expected result is the cube of the test function's output at 0 plus a constant (0.5).
    # Since test_function(0) is 3, cubic(3) should equal 3^3 + 0.5.
    assert cubic(test_function(0)) == test_function(0)**3 + 0.5

    # Test the 'cubic' function with the test_function applied to 1.
    # The expected result is the cube of the test function's output at 1 plus a constant (0.5).
    # Since test_function(1) is 5, cubic(5) should equal 5^3 + 0.5.
    assert cubic(test_function(1)) == test_function(1)**3 + 0.5

