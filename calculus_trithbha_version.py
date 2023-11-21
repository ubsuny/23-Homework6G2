import sys
import numpy as np

# Modified Simpson's Rule with iteration tracking
def simpson(f, a, b, n, accuracy):
    def compute_simpson_integral(n):
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

# Modified Trapezoidal Rule with iteration tracking
def trapezoid(f, a, b, n, accuracy):
    def compute_trapezoid_integral(n):
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

# Existing adaptive_trapezoid function from your code
# Adaptive Trapezoidal Rule with iteration tracking
def adaptive_trapezoid(f, a, b, accuracy):
    old_s = np.inf
    h = b - a
    n = 1
    s = (f(a) + f(b)) * 0.5
    iterations = 0  # Initialize iteration counter

    while abs(h * (old_s - s * 0.5)) > accuracy:
        iterations += 1  # Increment the iteration counter
        old_s = s
        for i in np.arange(n):
            s += f(a + (i + 0.5) * h)
        n *= 2
        h *= 0.5

    return h * s, iterations  # Return the integral and the number of iterations


# root finding
def root_print_header(algorithm, accuracy):
    """Prints the header for root-finding output, detailing the steps taken by the algorithm.
    
    Parameters:
    algorithm (str): The algorithm name.
    accuracy (float): The desired accuracy of the root.
    """
    # Write the header information to stdout, including algorithm name and requested accuracy
    sys.stdout.write("\n ROOT FINDING using " + algorithm +
                     "\n Requested accuracy = " +repr(accuracy) +
                     "\n Step     Guess For Root          Step Size      " +
                     "     Function Value" +
                     "\n ----  --------------------  --------------------" +
                     "  --------------------" + "\n")

def root_print_step(step, x, dx, f_of_x):
    """Prints the details of a single step in the root-finding process.
    
    Parameters:
    step (int): The current step number.
    x (float): The current estimate of the root.
    dx (float): The change in x for this step.
    f_of_x (float): The function value at the current estimate.
    """
    # Format and write the current step's details to stdout
    sys.stdout.write(repr(step).rjust(5))
    for val in [x, dx, f_of_x]:
        sys.stdout.write("  " + repr(val).ljust(20))
    sys.stdout.write("\n")

def root_max_steps(algorithm, max_steps):
    """Raises an exception when the maximum number of steps is exceeded in the root-finding algorithm.
    
    Parameters:
    algorithm (str): The algorithm name.
    max_steps (int): The maximum number of steps allowed.
    """
    # Raise an exception with a message indicating the maximum number of steps has been exceeded
    raise Exception(" " + algorithm + ": maximum number of steps " +
                    repr(max_steps) + " exceeded\n")

def root_simple(f, x, dx, accuracy=1.0e-6, max_steps=1000, root_debug=False):
    """Simple root-finding algorithm that incrementally searches for a root using step halving.
    
    Parameters:
    f (callable): The function for which to find the root.
    x (float): Initial guess for the root.
    dx (float): Initial step size for searching the root.
    accuracy (float, optional): Desired accuracy for the root. Default is 1.0e-6.
    max_steps (int, optional): Maximum number of steps in the search. Default is 1000.
    root_debug (bool, optional): If True, print debug information. Default is False.
    
    Returns:
    tuple: The estimated root and an array of iteration details if root_debug is True.
    """
    # Initialize the function value at the starting guess and set the step counter to zero
    f0 = f(x)
    fx = f0
    step = 0
    iterations = []  # to store iteration details if needed for debugging
    # If debugging is enabled, print the header and the first step
    if root_debug:        
        root_print_header("Simple Search with Step Halving", accuracy)
        root_print_step(step, x, dx, f0)
        iterations.append([x,f0])
    # Loop until the step size is smaller than the accuracy or the function value is zero
    while abs(dx) > abs(accuracy) and f0 != 0.0:
        x += dx
        fx = f(x)
        # If the sign of the function changes, we've stepped over the root
        if f0 * fx < 0.0:   
            x -= dx         # step back
            dx /= 2.0       # use smaller step
        step += 1
        # If the maximum number of steps is exceeded, raise an exception
        if step > max_steps:
            root_max_steps("root_simple", max_steps)
        # If debugging is enabled, print the details of the current step
        if root_debug:
            root_print_step(step, x, dx, fx)
            iterations.append([x,fx])
    return x,np.array(iterations)

def root_bisection(f, x1, x2, accuracy=1.0e-6, max_steps=1000, root_debug=False):
    """
    Finds a root of the function f using the bisection method within the interval [x1, x2].
    
    Parameters:
    f (callable): The function for which to find the root.
    x1 (float): The lower bound of the interval.
    x2 (float): The upper bound of the interval.
    accuracy (float): The desired accuracy for the root. Default is 1.0e-6.
    max_steps (int): The maximum number of steps to be taken. Default is 1000.
    root_debug (bool): If True, print debug information. Default is False.
    
    Returns:
    tuple: The estimated root and an array of iteration details if root_debug is True.
    """
    # Ensure the interval is valid for the bisection method
    f1 = f(x1)
    f2 = f(x2)
    if f1 * f2 > 0.0:
        raise Exception("The function must have different signs at x1 and x2 for bisection to work.")

    # Initial midpoint and its function value
    x_mid = (x1 + x2) / 2.0
    f_mid = f(x_mid)

    # Initial interval size
    dx = x2 - x1
    step = 0
    iterations = []  # Store iteration details if debugging is enabled

    # Print debug header if enabled
    if root_debug:
        root_print_header("Bisection Search", accuracy)
        root_print_step(step, x_mid, dx, f_mid)
        iterations.append([x_mid, f_mid])

    # Bisection method loop
    while abs(dx) > accuracy:
        # Check if the function value at the midpoint is zero (root found)
        if f_mid == 0.0:
            dx = 0.0
        else:
            # Determine the subinterval [x1, x_mid] or [x_mid, x2] containing the root
            if f1 * f_mid > 0:
                x1 = x_mid
                f1 = f_mid
            else:
                x2 = x_mid
                f2 = f_mid

            # Update the midpoint and its function value
            x_mid = (x1 + x2) / 2.0
            f_mid = f(x_mid)

            # Update the interval size
            dx = x2 - x1

        # Increment the step count and check for maximum steps
        step += 1
        if step > max_steps:
            raise Exception("Too many steps in root_bisection")

        # Print debug information if enabled
        if root_debug:
            root_print_step(step, x_mid, dx, f_mid)
            iterations.append([x_mid, f_mid])

    # Return the estimated root and iteration details if debugging is enabled
    return x_mid, np.array(iterations)

def root_secant(f, x0, x1, accuracy=1.0e-6, max_steps=20, root_debug=False):
    """
    Finds a root of the function f using the secant method, given two initial guesses.

    Parameters:
    f (callable): The function for which to find the root.
    x0 (float): The first initial guess for the root.
    x1 (float): The second initial guess for the root.
    accuracy (float): The desired accuracy for the root. Default is 1.0e-6.
    max_steps (int): The maximum number of steps to be taken. Default is 20.
    root_debug (bool): If True, print debug information. Default is False.

    Returns:
    tuple: The estimated root and an array of iteration details if root_debug is True.
    """
    # List to store iteration details if debugging is enabled
    iterations = []
    f0 = f(x0)
    # Initial difference between the two guesses
    dx = x1 - x0
    step = 0

    # Print debug header if enabled
    if root_debug:
        root_print_header("Secant Search", accuracy)
        root_print_step(step, x0, dx, f0)
        iterations.append([x0, f0])

    # If the function at the first guess is zero, we've found a root
    if f0 == 0:
        return x0

    # Main loop of the secant method
    while abs(dx) > abs(accuracy):
        f1 = f(x1)
        # If the function at the second guess is zero, we've found a root
        if f1 == 0:
            return x1
        # Prevent division by zero if f1 is the same as f0
        if f1 == f0:
            raise Exception("Secant horizontal f(x0) = f(x1) algorithm fails")

        # Update the difference using the secant formula
        dx *= - f1 / (f1 - f0)
        x0, f0 = x1, f1  # Update previous guess to the current one
        x1 += dx  # Update current guess
        step += 1

        # Check for exceeding the maximum number of steps
        if step > max_steps:
            root_max_steps("root_secant", max_steps)

        # Print debug information if enabled
        if root_debug:
            root_print_step(step, x1, dx, f1)
            iterations.append([x1, f1])

    # Return the estimated root and iteration details if debugging is enabled
    return x1, np.array(iterations)

def root_tangent(f, fp, x0, accuracy=1.0e-6, max_steps=20, root_debug=False):
    """
    Finds a root of the function f using the Newton-Raphson (tangent) method, given an initial guess.

    Parameters:
    f (callable): The function for which to find the root.
    fp (callable): The derivative of the function f.
    x0 (float): The initial guess for the root.
    accuracy (float): The desired accuracy for the root. Default is 1.0e-6.
    max_steps (int): The maximum number of steps to be taken. Default is 20.
    root_debug (bool): If True, print debug information. Default is False.

    Returns:
    float: The estimated root.
    """
    # List to store iteration details if debugging is enabled
    iterations = []
    f0 = f(x0)
    fp0 = fp(x0)

    # Check for the derivative being zero at the initial guess, which would fail the method
    if fp0 == 0.0:
        raise Exception("Root tangent df/dx = 0 algorithm fails")

    # Initial change is based on the function and its derivative
    dx = - f0 / fp0
    step = 0

    # Print debug header if enabled
    if root_debug:
        root_print_header("Tangent Search", accuracy)
        root_print_step(step, x0, dx, f0)
        iterations.append([x0, f0])

    # If the function at the initial guess is zero, we've found a root
    if f0 == 0.0:
        return x0

    # Main loop of the Newton-Raphson method
    while True:
        # Update the derivative at the new guess
        fp0 = fp(x0)
        if fp0 == 0.0:
            raise Exception("Root tangent df/dx = 0 algorithm fails")

        # Apply the Newton-Raphson update step
        dx = - f0 / fp0
        x0 += dx
        f0 = f(x0)

        # Check for convergence within the desired accuracy
        if abs(dx) <= accuracy or f0 == 0.0:
            return x0

        # Increment step and check for exceeding maximum steps
        step += 1
        if step > max_steps:
            root_max_steps("root_tangent", max_steps)

        # Print debug information if enabled
        if root_debug:
            root_print_step(step, x0, dx, f0)
            iterations.append([x0,f0])
    return x0
