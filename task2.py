import numpy as np
import matplotlib.pyplot as plt
from rootfinding import *
from scipy.optimize import newton
from math import tan, tanh, cos

# Bisection and Newton-Raphson methods.

import numpy as np

def tan_function(x):
    """Function y(x) = tan(x)."""
    return np.tan(x)

def tanh_function(x):
    """Function y(x) = tanh(x)."""
    return np.tanh(x)

def d_tan_function(x):
    """Derivative of y(x) = tan(x)."""
    return 1 / np.cos(x)**2

def d_tanh_function(x):
    """Derivative of y(x) = tanh(x)."""
    return 1 - np.tanh(x)**2

def bisection_method(f, a, b, tol=1e-5, max_iter=1000):
    """Bisection root-finding method."""
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None, None
    a_n, b_n = a, b
    for n in range(1, max_iter + 1):
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)
        if abs(f_m_n) < tol:
            return m_n, n  # root and number of iterations
        elif f(a_n) * f_m_n < 0:
            a_n, b_n = a_n, m_n
        elif f(b_n) * f_m_n < 0:
            a_n, b_n = m_n, b_n
        else:
            print("Bisection method fails.")
            return None, None
    print("Bisection method did not converge.")
    return None, None

def newton_raphson_method(f, df, x0, tol=1e-5, max_iter=1000):
    """Newton-Raphson root-finding method."""
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < tol:
            return xn, n  # root and number of iterations
        dfxn = df(xn)
        if dfxn == 0:
            print("Zero derivative. No solution found.")
            return None, None
        xn = xn - fxn / dfxn
    print("Newton-Raphson method did not converge.")
    return None, None

def test_algorithms():
    """Test and compare the Bisection and Newton-Raphson methods."""
    functions = [(tan_function, d_tan_function), (tanh_function, d_tanh_function)]
    initial_guesses = [4.5, 0]  # initial guesses for Newton-Raphson
    intervals = [(4, 5), (-1, 1)]  # intervals for Bisection

    for i, (f, df) in enumerate(functions):
        print(f"\nTesting on function {i+1}:")
        
        # Bisection Method
        root_bisection, steps_bisection = bisection_method(f, *intervals[i])
        print(f"Bisection Method: Root = {root_bisection}, Steps = {steps_bisection}")

        # Newton-Raphson Method
        root_newton, steps_newton = newton_raphson_method(f, df, initial_guesses[i])
        print(f"Newton-Raphson Method: Root = {root_newton}, Steps = {steps_newton}")

# Run the test
test_algorithms()


"""
    The derivative of the function tan(x).

    Parameters:
    - x: float
        The variable.

    Returns:
    float: The derivative value at x.
    """
    return 1 / cos(x)**2

def f_tanh(x):
    return tanh(x)

def df_tanh(x):
    return 1 - tanh(x)**2

def bisection_method(f, a, b, tol=1e-10):
"""
    Find the root of a function using the Bisection method.

    Parameters:
    - f: function
        The function for which roots are to be found.
    - a: float
        The lower bound of the interval.
    - b: float
        The upper bound of the interval.
    - tol: float, optional
        Tolerance for stopping criterion.

    Returns:
    tuple: (root, steps)
        The root of the function and the number of steps taken.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Function must have different signs at the endpoints.")
    
    steps = 0
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if f(midpoint) == 0:
            return midpoint, steps
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        steps += 1
    
    return (a + b) / 2, steps

# Initial guesses and intervals
x0_tan = 1.0  # initial guess for tan(x), avoiding pi/2
a_tan, b_tan = 0, np.pi / 2 - 1e-4  # interval for bisection for tan(x), avoiding pi/2

x0_tanh = 1.0  # initial guess for tanh(x)
a_tanh, b_tanh = -1, 1  # interval for bisection for tanh(x)

# Applying Newton-Raphson and Bisection methods
root_newton_tan, steps_newton_tan = newton(f_tan, x0_tan, df_tan, tol=1e-10, full_output=True)
root_bisection_tan, steps_bisection_tan = bisection_method(f_tan, a_tan, b_tan)

root_newton_tanh, steps_newton_tanh = newton(f_tanh, x0_tanh, df_tanh, tol=1e-10, full_output=True)
root_bisection_tanh, steps_bisection_tanh = bisection_method(f_tanh, a_tanh, b_tanh)

# Displaying the results
print(f"Root of tan(x) using Newton-Raphson: {root_newton_tan}, Steps: {steps_newton_tan.iterations}")
print(f"Root of tan(x) using Bisection: {root_bisection_tan}, Steps: {steps_bisection_tan}")

print(f"Root of tanh(x) using Newton-Raphson: {root_newton_tanh}, Steps: {steps_newton_tanh.iterations}")
print(f"Root of tanh(x) using Bisection: {root_bisection_tanh}, Steps: {steps_bisection_tanh}")

