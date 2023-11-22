import numpy as np
import matplotlib.pyplot as plt
from rootfinding import *
from scipy.optimize import newton
from math import tan, tanh, cos

def calc_tan(x):
    x=np.tan(y)


def f_tan(x):
    return tan(x)

def df_tan(x):
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

