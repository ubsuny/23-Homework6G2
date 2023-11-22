import sys
import numpy as np

def simpson(f, a, b, n, accuracy):
    """
    Simpson's Rule for numerical integration with adaptive iteration.
    Computes the integral of the function 'f' from 'a' to 'b' with initial
    number of subintervals 'n' and increases 'n' until the desired 'accuracy' is achieved.

    Args:
    f (function): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): Initial number of subintervals.
    accuracy (float): Desired accuracy for the integral calculation.

    Returns:
    float, int: The calculated integral and the number of iterations taken.
    """
    def compute_simpson_integral(n):
        """ Helper function to compute Simpson's integral for 'n' subintervals. """
        h = (b - a) / n
        i = np.arange(0, n)
        s = f(a) + f(b) + 4 * np.sum(f(a + i[1::2] * h)) + 2 * np.sum(f(a + i[2:-1:2] * h))
        return s * h / 3

    old_integral = compute_simpson_integral(n)
    iterations = 1
    while True:
        n *= 2
        new_integral = compute_simpson_integral(n)
        if np.abs(new_integral - old_integral) < accuracy:
            return new_integral, iterations
        old_integral = new_integral
        iterations += 1

def trapezoid(f, a, b, n, accuracy):
    """
    Trapezoidal Rule for numerical integration with adaptive iteration.
    Computes the integral of the function 'f' from 'a' to 'b' with initial
    number of subintervals 'n' and increases 'n' until the desired 'accuracy' is achieved.

    Args:
    f (function): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): Initial number of subintervals.
    accuracy (float): Desired accuracy for the integral calculation.

    Returns:
    float, int: The calculated integral and the number of iterations taken.
    """
    def compute_trapezoid_integral(n):
        """ Helper function to compute the trapezoidal integral for 'n' subintervals. """
        h = (b - a) / n
        s = f(a) + f(b) + 2 * np.sum(f(a + np.arange(1, n) * h))
        return s * h / 2

    old_integral = compute_trapezoid_integral(n)
    iterations = 1
    while True:
        n *= 2
        new_integral = compute_trapezoid_integral(n)
        if np.abs(new_integral - old_integral) < accuracy:
            return new_integral, iterations
        old_integral = new_integral
        iterations += 1

def adaptive_trapezoid(f, a, b, accuracy):
    """
    Adaptive Trapezoidal Rule for numerical integration.
    Continuously halves the step size 'h' and doubles the number of subintervals
    until the desired 'accuracy' is achieved.

    Args:
    f (function): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    accuracy (float): Desired accuracy for the integral calculation.

    Returns:
    float, int: The calculated integral and the number of iterations taken.
    """
    old_s = np.inf
    h = b - a
    n = 1
    s = (f(a) + f(b)) * 0.5
    iterations = 0

    while abs(h * (old_s - s * 0.5)) > accuracy:
        iterations += 1
        old_s = s
        for i in np.arange(n):
            s += f(a + (i + 0.5) * h)
        n *= 2
        h *= 0.5

    return h * s, iterations  # Return the integral and the number of iterations


# root finding
def calculate_accuracy(known_root, obtained_root, tolerance):
    """
    Calculate the number of correct digits in the obtained root compared to the known root.
    If the obtained root is within the specified tolerance of the known root,
    the function calculates the number of digits that match.
    """
    if np.isclose(obtained_root, known_root, atol=tolerance):
        difference = abs(obtained_root - known_root)
        if difference == 0:
            return float('inf')  # Perfect match, return infinity
        correct_digits = -np.log10(difference)
        return correct_digits
    else:
        return 0

# Bisection and Newton-Raphson methods with iteration tracking
def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method to find the root of a function.
    This method returns the root and the number of iterations taken.
    It is a bracketing method that repeatedly bisects an interval and selects a subinterval
    in which a root must lie, assuming the function changes sign over the interval.
    """
    if func(a) * func(b) >= 0:
        return None, 0  # No root if function does not change sign

    a_n, b_n = a, b
    for n in range(1, max_iter + 1):
        m_n = (a_n + b_n) / 2  # Midpoint
        f_m_n = func(m_n)
        if abs(f_m_n) < tol:  # Check if the midpoint is close enough to root
            return m_n, n
        elif func(a_n) * f_m_n < 0:  # If sign changes over [a_n, m_n]
            b_n = m_n
        elif func(b_n) * f_m_n < 0:  # If sign changes over [m_n, b_n]
            a_n = m_n

    return (a_n + b_n) / 2, max_iter  # Return the best estimate after max_iter

def newton_raphson_method(func, d_func, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method to find the root of a function.
    This method returns the root and the number of iterations taken.
    It uses the formula x1 = x0 - f(x0)/f'(x0) to iteratively find the root,
    starting from an initial guess x0.
    """
    x_n = x0
    for n in range(max_iter):
        f_x_n = func(x_n)
        if abs(f_x_n) < tol:  # Check if current estimate is close enough to root
            return x_n, n
        df_x_n = d_func(x_n)
        if df_x_n == 0:
            return None, n  # Return None if derivative is zero (to avoid division by zero)
        x_n = x_n - f_x_n / df_x_n

    return x_n, max_iter  # Return the best estimate after max_iter
