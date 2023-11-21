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

# Edge case tests for the exp function
def test_exp_edge_cases():
    # Test with a very small value (close to zero) to check handling of underflow
    assert np.isclose(exp(1e-15), np.exp(-1e15))  # exp should approach zero for small values

    # Test with negative infinity to ensure the function correctly handles it
    # Expected behavior: exp(-inf) should return 0
    assert exp(-np.inf) == 0

    # Test with NaN (Not a Number) to ensure the function correctly handles it
    # Expected behavior: exp(NaN) should return NaN
    assert np.isnan(exp(np.nan))

# Edge case tests for the cos function
def test_cos_edge_cases():
    # Test with a very large value to check the function's behavior
    # Since cos(1/x) approaches cos(0) as x approaches infinity, we expect it to be close to 1
    assert np.isclose(cos(1e15), np.cos(1e-15))

    # Test with negative infinity to check handling of extreme values
    # Expected behavior: cos(-inf) should be close to cos(0)
    assert np.isclose(cos(-np.inf), np.cos(0))

    # Test with NaN to check the function's behavior with undefined input
    # Expected behavior: cos(NaN) should return NaN
    assert np.isnan(cos(np.nan))

# Edge case tests for the cubic function
def test_cubic_edge_cases():
    large_value = 1e15  # A very large value for testing

    # Test with a large positive value
    # Expected behavior: cubic function should correctly compute the cubic value plus the constant
    assert cubic(large_value) == large_value**3 + 0.5

    # Test with a large negative value
    # Expected behavior: cubic function should correctly compute the cubic value plus the constant
    assert cubic(-large_value) == (-large_value)**3 + 0.5

    # Test with NaN to check how the function handles undefined input
    # Expected behavior: cubic(NaN) should return NaN
    assert np.isnan(cubic(np.nan))

# Edge case tests for numerical integration functions (simpson, trapezoid, and adaptive_trapezoid)
def test_numerical_integration_edge_cases():
    # Test integrating a constant function over an interval
    # Expected result: integral should be the constant multiplied by the interval length
    result, _ = simpson(lambda x: 4, 0, 10, 100, 1e-6)
    assert np.isclose(result, 40)  # Integral of constant 4 over interval [0, 10]

    # Test integrating over a zero-length interval
    # Expected result: integral should be zero regardless of the function
    result, _ = trapezoid(np.sin, 1, 1, 100, 1e-6)
    assert np.isclose(result, 0)

# Edge case tests specifically for the adaptive_trapezoid function
def test_adaptive_trapezoid_edge_cases():
    # Test integrating a constant function over an interval
    constant_function = lambda x: 3
    result = adaptive_trapezoid(constant_function, 0, 10, 1e-6)[0]
    assert np.isclose(result, 30)  # Integral of constant 3 over interval [0, 10]

    # Test integrating over a zero-length interval
    zero_interval_result = adaptive_trapezoid(np.sin, 1, 1, 1e-6)[0]
    assert np.isclose(zero_interval_result, 0)

    # Test integrating a function with a known analytical integral
    exp_function = lambda x: np.exp(x)
    exp_integral_result = adaptive_trapezoid(exp_function, 0, 1, 1e-6)[0]
    assert np.isclose(exp_integral_result, np.e - 1, atol=1e-6)  # Should be close to e - 1

    # Test integrating a function with a singularity
    # Note: This test's success depends on the implementation details of adaptive_trapezoid
    singularity_function = lambda x: 1 / np.sqrt(x)
    singularity_result = adaptive_trapezoid(singularity_function, 0, 1, 1e-6)[0]
    # Compare with known result or check if the algorithm handles it without error
