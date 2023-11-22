
"""
This module provides functions for numerical integration using the Simpson's rule
and also calculates the cumulative integral of a given function. It includes error
handling for subinterval counts and division by zero in function evaluations.
"""

import numpy as np

def simpson(f, a, b, n):
    """Approximates the definite integral of f from a to b by
    the composite Simpson's rule, using n subintervals.
    From http://en.wikipedia.org/wiki/Simpson's_rule

import numpy as np
import matplotlib.pyplot as plt

def simpson(f, a, b, n):
     """
    Approximate the definite integral of a function using Simpson's rule.
    
    Parameters:
    f : callable
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n : int
        The number of subintervals to use (must be even).
        
    Returns:
    float
        The approximate integral of the function.

    """

    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even.")

    h = (b - a) / n

    i = np.arange(0, n)

    s = f(a) + f(b)
    s += 4 * np.sum(f(a + i[1::2] * h))
    s += 2 * np.sum(f(a + i[2:-1:2] * h))

    i = np.arange(0,n)

    s = f(a) + f(b)
    s += 4 * np.sum( f( a + i[1::2] * h ) )
    s += 2 * np.sum( f( a + i[2:-1:2] * h ) )




    return s * h / 3



def cumulative_integral_simpson(func, a, b, dx):
    """Function that calculates the antiderivative of func"""

    x_values = np.arange(a, b, dx)
    num_of_steps = np.floor((b - a) / dx)

    if num_of_steps < 2:
        raise ValueError("(b - a) / dx must be at least 2")

    antiderivative = np.empty(len(x_values) + 1)
    antiderivative[0] = simpson(func, a, a + dx, 4)

    j = 1
    while j <= num_of_steps:
        antiderivative[j] = antiderivative[j - 1] + simpson(func, a + ((j - 1) * dx), a + (j * dx), 4)
        j = j + 1

    return antiderivative[:-1], x_values

def f(x):
    """Function to calculate exponential of -1 divided by x, handling zero by using a small positive value."""
    # Replace zero values in x with a small positive value to avoid division by zero
    x_safe = np.where(x == 0, np.finfo(float).eps, x)
    return np.exp(-1 / x_safe)

def g(x):
    """Function to calculate cosine of 1 divided by x, handling zero by using a small positive value."""
    # Replace zero values in x with a small positive value to avoid division by zero
    x_safe = np.where(x == 0, np.finfo(float).eps, x)
    return np.cos(1 / x_safe)

def h(x):
    """Function to calculate a cubic polynomial x^3 + 1/2."""
    return (x * x * x) + (1 / 2)

def cumulative_integral_simpson(f,a,b,dx):
    """Function that calculates the antidervative of f"""

    x_values = np.arange(a,b,dx)

    num_of_steps = np.floor((b-a)/dx)

    if num_of_steps < 2:
        raise ValueError("(b-a)/dx  must be at least 2")

    antiderivative = np.empty(len(x_values)+1)

    antiderivative[0] = simpson(f, a, a+dx, 4)

    j = 1

    while j <= num_of_steps :

      antiderivative[j] = antiderivative[j-1] + simpson(f, a+((j-1)*dx), a+(j*dx), 4)

      j = j + 1

    return antiderivative[:-1], x_values


def trapezoid(f, a, b, n):
    """
    Approximate the definite integral of a function using the trapezoidal rule.
    
    Parameters:
    f : callable
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n : int
        The number of subintervals to use.
        
    Returns:
    float
        The approximate integral of the function.
    """
    h = (b - a) / n
    s = f(a) + f(b)
    i = np.arange(0,n)
    s += 2 * np.sum( f(a + i[1:] * h) )
    return s * h / 2


def cumulative_integral_trapezoid(f,a,b,dx):
    """Function that calculates the antidervative function of f"""

    x_values = np.arange(a,b,dx)

    num_of_steps = np.floor((b-a)/dx)

    antiderivative = np.empty(len(x_values)+1)

    antiderivative[0] = trapezoid(f, a, a+dx, 4)

    j = 1

    while j <= num_of_steps :

      antiderivative[j] = antiderivative[j-1] + trapezoid(f, a+((j-1)*dx), a+(j*dx), 4)

      j = j + 1

    return antiderivative[:-1], x_values


def adaptive_trapezoid(f, a, b, acc, output=False):
    """
    Compute the definite integral of a function using the adaptive trapezoidal method
    to a desired accuracy.
    
    Parameters:
    f : callable
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    acc : float
        The desired accuracy of the result.
    output : bool, optional
        If True, print intermediate values (default is False).
        
    Returns:
    float
        The approximate integral of the function.
    """
    old_s = np.inf
    h = b - a
    n = 1
    s = (f(a) + f(b)) * 0.5
    if output == True :
        print ("N = " + str(n+1) + ",  Integral = " + str( h*s ))
    while abs(h * (old_s - s*0.5)) > acc :
        old_s = s
        for i in np.arange(n) :
            s += f(a + (i + 0.5) * h)
        n *= 2.
        h *= 0.5
        if output == True :
            print ("N = " + str(n) + ",  Integral = " + str( h*s ))
    return h * s


def cumulative_integral_adaptive_trap(f,a,b,dx):
    """Function that calculates the antidervative function of f"""

    x_values = np.arange(a,b,dx)

    num_of_steps = np.floor((b-a)/dx)

    antiderivative = np.empty(len(x_values)+1)

    antiderivative[0] = adaptive_trapezoid(f, a, a+dx, 10)

    j = 1

    while j <= num_of_steps :

      antiderivative[j] = antiderivative[j-1] + adaptive_trapezoid(f, a+((j-1)*dx), a+(j*dx), 10)

      j = j + 1

    return antiderivative[:-1], x_values


def f(x):
    # Replace zero values in x with a small positive value to avoid division by zero
    x_safe = np.where(x == 0, np.finfo(float).eps, x)

    return np.exp(-1 / x_safe)

def g(x):
    # Replace zero values in x with a small positive value to avoid division by zero
    x_safe = np.where(x == 0, np.finfo(float).eps, x)

    return np.cos(1 / x_safe)

def h(x):
  return (x*x*x)+(1/2)

