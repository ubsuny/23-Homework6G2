"""
This module contains unit tests for the calculus_trithbha_version functions, including
tests for exponential, cosine, and cubic calculations, as well as the Simpson and trapezoid
integration methods.
"""

import numpy as np
from calculus_trithbha_version import simpson, trapezoid, adaptive_trapezoid  

def exp(x):
    """
    Calculate the exponential function exp(-1/x), with special handling for x approaching zero.
    
    Parameters:
    - x: The value (or array of values) to evaluate exp(-1/x) at.
    
    Returns:
    - The value of exp(-1/x) or 0 if x is infinite.
    """
    if np.isinf(x):
        return 0
    safe_x = np.clip(x, 1e-10, np.inf)
    return np.where(x != 0, np.exp(-1/safe_x), 0)

def cos(x):
    """
    Calculate the cosine function cos(1/x), with special handling for x approaching zero.
    
    Parameters:
    - x: The value (or array of values) to evaluate cos(1/x) at.
    
    Returns:
    - The value of cos(1/x) or 1 if x is zero.
    """
    safe_x = np.clip(x, 1e-10, np.inf)
    return np.where(x != 0, np.cos(1/safe_x), 1)

def cubic(x, constant=1/2):
    """
    Evaluate a cubic function x^3 with an added constant.
    
    Parameters:
    - x: The value to evaluate the cubic function at.
    - constant: The constant to add to the result (default 0.5).
    
    Returns:
    - The value of x^3 plus the constant.
    """
    return x**3 + constant

def test_exp():
    """
    Unit tests for the exp function.
    """
    assert exp(0) == 0
    assert exp(1) == np.exp(-1)
    assert np.isclose(exp(np.inf), 0)

def test_cos():
    """
    Unit tests for the cos function.
    """
    assert cos(0) == 1
    assert np.isclose(cos(1), np.cos(1))
    assert np.isclose(cos(np.inf), np.cos(0))

def test_cubic():
    """
    Unit tests for the cubic function.
    """
    assert cubic(1, 0.5) == 1.5
    assert cubic(0) == 0.5
    assert cubic(-1, -0.5) == -1.5

def test_simpson():
    """
    Unit test for the Simpson integration method.
    """
    result, _ = simpson(np.sin, 0, np.pi, 100, 1e-6)
    assert np.isclose(result, 2, atol=1e-6)

def test_trapezoid():
    """
    Unit test for the trapezoid integration method.
    """
    result, _ = trapezoid(np.sin, 0, np.pi, 100, 1e-6)
    assert np.isclose(result, 2, atol=1e-6)

def test_adaptive_trapezoid():
    """
    Unit test for the adaptive trapezoid integration method.
    """
    result, _ = adaptive_trapezoid(np.sin, 0, np.pi, 1e-6)
    assert np.isclose(result, 2, atol=1e-6)

def test_exp_with_function():
    """
    Unit test for the exp function using a square function as input.
    """
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

def test_cubic_with_function():
    """
    Unit test for the cubic function using a linear function as input.
    """
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
