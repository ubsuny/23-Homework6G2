import numpy as np
import matplotlib.pyplot as plt

def simpson(f, a, b, n):
    """Approximates the definite integral of f from a to b by
    the composite Simpson's rule, using n subintervals.
    From http://en.wikipedia.org/wiki/Simpson's_rule
    """

    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even.")

    h = (b - a) / n
    i = np.arange(0,n)

    s = f(a) + f(b)
    s += 4 * np.sum( f( a + i[1::2] * h ) )
    s += 2 * np.sum( f( a + i[2:-1:2] * h ) )



    return s * h / 3


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
    """Approximates the definite integral of f from a to b by
    the composite trapezoidal rule, using n subintervals.
    From http://en.wikipedia.org/wiki/Trapezoidal_rule
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
    Uses the adaptive trapezoidal method to compute the definite integral
    of f from a to b to desired accuracy acc.
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
