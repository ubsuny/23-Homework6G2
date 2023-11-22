import pytest
import numpy as np
from calculus_trithbha_version import bisection_method, newton_raphson_method, calculate_accuracy

# Function definitions and their derivatives
def tan_func(x):
    return np.tan(x)

def tanh_func(x):
    return np.tanh(x)

def d_tan_func(x):
    return 1 / np.cos(x)**2

def d_tanh_func(x):
    return 1 - np.tanh(x)**2

def test_bisection_method_tan():
    """
    Test the bisection method on the tan function.
    Checks if the method correctly finds the root of tan(x) in the interval [π/4, 3π/4],
    where the known root is π/2.
    """
    root, _ = bisection_method_mod(tan_func, np.pi/4, 3*np.pi/4)
    assert np.isclose(root, np.pi/2, atol=1e-6)

def test_newton_raphson_method_tan():
    """
    Test the Newton-Raphson method on the tan function.
    Checks if the method correctly finds the root of tan(x) using an initial guess
    in the interval [π/4, 3π/4], where the known root is π/2.
    """
    root, _ = newton_raphson_method_mod(tan_func, d_tan_func, (np.pi/4 + 3*np.pi/4) / 2)
    assert np.isclose(root, np.pi/2, atol=1e-6)

def test_bisection_method_tanh():
    """
    Test the bisection method on the tanh function.
    Checks if the method correctly finds the root of tanh(x) in the interval [-2, 2],
    where the known root is 0.
    """
    root, _ = bisection_method_mod(tanh_func, -2, 2)
    assert np.isclose(root, 0, atol=1e-6)

def test_newton_raphson_method_tanh():
    """
    Test the Newton-Raphson method on the tanh function.
    Checks if the method correctly finds the root of tanh(x) using an initial guess of 0.2,
    where the known root is 0.
    """
    root, _ = newton_raphson_method_mod(tanh_func, d_tanh_func, 0.2)
    assert np.isclose(root, 0, atol=1e-6)

def test_calculate_accuracy():
    """
    Test the calculate_accuracy function.
    Checks if the function correctly calculates the number of accurate digits
    when the known root and the obtained root are the same (π/2 in this case).
    The expected accuracy should be infinity as the roots match exactly.
    """
    known_root = np.pi/2
    obtained_root = np.pi/2
    accuracy = calculate_accuracy(known_root, obtained_root, 1e-6)
    assert accuracy == float('inf')
