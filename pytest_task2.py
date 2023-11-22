"""
This module contains definitions and tests for various mathematical functions and methods.
It includes tests for bisection and Newton-Raphson root-finding methods, as well as
a function to calculate the accuracy of the roots found.
"""

import numpy as np
from task3_calculus import bisection_method, newton_raphson_method, calculate_accuracy

def tan_func(x):
    """
    Compute the tangent of x.
    
    Parameters:
    - x: The value to compute the tangent for.
    
    Returns:
    - The tangent of x.
    """
    return np.tan(x)

def tanh_func(x):
    """
    Compute the hyperbolic tangent of x.
    
    Parameters:
    - x: The value to compute the hyperbolic tangent for.
    
    Returns:
    - The hyperbolic tangent of x.
    """
    return np.tanh(x)

def d_tan_func(x):
    """
    Compute the derivative of the tangent function.
    
    Parameters:
    - x: The value to compute the derivative for.
    
    Returns:
    - The derivative of the tangent of x.
    """
    return 1 / np.cos(x)**2

def d_tanh_func(x):
    """
    Compute the derivative of the hyperbolic tangent function.
    
    Parameters:
    - x: The value to compute the derivative for.
    
    Returns:
    - The derivative of the hyperbolic tangent of x.
    """
    return 1 - np.tanh(x)**2
    
def sin_func(x):
    """
    Compute the sine of x.
    
    Parameters:
    - x: The value to compute the sine for.
    
    Returns:
    - The sine of x.
    """
    return np.sin(x)

def cos_func(x):
    """
    Compute the cosine of x.
    
    Parameters:
    - x: The value to compute the cosine for.
    
    Returns:
    - The cosine of x.
    """
    return np.cos(x)

def d_sin_func(x):
    """
    Compute the derivative of the sine function.
    
    Parameters:
    - x: The value to compute the derivative for.
    
    Returns:
    - The derivative of the sine of x.
    """
    return np.cos(x)

def d_cos_func(x):
    """
    Compute the derivative of the cosine function.
    
    Parameters:
    - x: The value to compute the derivative for.
    
    Returns:
    - The derivative of the cosine of x.
    """
    return -np.sin(x)

def test_bisection_method_tan():
    """
    Test the bisection method on the tan function.
    Checks if the method correctly finds the root of tan(x) in the interval [π/4, 3π/4],
    where the known root is π/2.
    """
    root, _ = bisection_method(tan_func, np.pi/4, 3*np.pi/4)
    assert np.isclose(root, np.pi/2, atol=1e-6)

def test_newton_raphson_method_tan():
    """
    Test the Newton-Raphson method on the tan function.
    Checks if the method correctly finds the root of tan(x) using an initial guess
    in the interval [π/4, 3π/4], where the known root is π/2.
    """
    root, _ = newton_raphson_method(tan_func, d_tan_func, (np.pi/4 + 3*np.pi/4) / 2)
    assert np.isclose(root, np.pi/2, atol=1e-6)

def test_bisection_method_tanh():
    """
    Test the bisection method on the tanh function.
    Checks if the method correctly finds the root of tanh(x) in the interval [-2, 2],
    where the known root is 0.
    """
    root, _ = bisection_method(tanh_func, -2, 2)
    assert np.isclose(root, 0, atol=1e-6)

def test_newton_raphson_method_tanh():
    """
    Test the Newton-Raphson method on the tanh function.
    Checks if the method correctly finds the root of tanh(x) using an initial guess of 0.2,
    where the known root is 0.
    """
    root, _ = newton_raphson_method(tanh_func, d_tanh_func, 0.2)
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

def test_bisection_method_no_root():
    """
    Test the bisection method on a function with no root in the given interval.
    The function tested is cos(x) in the interval [0, π/2], which does not contain a root.
    The method should return None.
    """
    root, _ = bisection_method(cos_func, 0, np.pi/2)
    assert root is None

def test_newton_raphson_method_multiple_roots():
    """
    Test the Newton-Raphson method on a function with multiple roots.
    The function tested is sin(x) with an initial guess close to π, ensuring it finds the root at π.
    """
    root, _ = newton_raphson_method(sin_func, d_sin_func, 3)
    assert np.isclose(root, np.pi, atol=1e-6)

def test_edge_case_bisection_method():
    """
    Test the bisection method on an edge case.
    The function tested is tan(x) in an interval close to its vertical asymptote (π/2).
    """
    root, _ = bisection_method(tan_func, np.pi/2 - 0.1, np.pi/2 + 0.1)
    assert root is not None
