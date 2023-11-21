import pytest
import numpy as np
from calculus_trithbha_version import *

# Test for the exp function
def test_exp():
    assert exp(0) == 0  # Check if exp function returns 0 for input 0
    assert exp(1) == np.exp(-1)  # Check if exp function returns e^-1 for input 1
    assert np.isclose(exp(np.inf), 0)  # Check if exp function returns close to 0 for input infinity

# Test for the cos function
def test_cos():
    assert cos(0) == 1  # Check if cos function returns 1 for input 0
    assert np.isclose(cos(1), np.cos(1))  # Check if cos function returns cos(1) for input 1
    assert np.isclose(cos(np.inf), np.cos(0))  # Check if cos function returns cos(0) for input infinity

# Test for the cubic function
def test_cubic():
    assert cubic(1, 0.5) == 1.5  # Check if cubic function returns correct value for input 1 with constant 0.5
    assert cubic(0) == 0.5  # Check if cubic function returns constant value (0.5 by default) for input 0
    assert cubic(-1, -0.5) == -0.5  # Check if cubic function returns correct value for input -1 with constant -0.5

# Test for the simpson function
def test_simpson():
    result, _ = simpson(np.sin, 0, np.pi, 100, 1e-6)  # Apply simpson rule to integrate sin from 0 to pi
    assert np.isclose(result, 2, atol=1e-6)  # Check if result is close to 2 with a tolerance of 1e-6

# Test for the trapezoid function
def test_trapezoid():
    result, _ = trapezoid(np.sin, 0, np.pi, 100, 1e-6)  # Apply trapezoid rule to integrate sin from 0 to pi
    assert np.isclose(result, 2, atol=1e-6)  # Check if result is close to 2 with a tolerance of 1e-6

# Test for the adaptive_trapezoid function
def test_adaptive_trapezoid():
    result, _ = adaptive_trapezoid(np.sin, 0, np.pi, 1e-6)  # Apply adaptive trapezoid rule to integrate sin from 0 to pi
    assert np.isclose(result, 2, atol=1e-6)  # Check if result is close to 2 with a tolerance of 1e-6

# Test for the correct_digits function
def test_correct_digits():
    assert correct_digits(1.999, 2) > 2  # Check if correct digits are greater than 2 for a close approximation
    assert correct_digits(2, 2) == np.inf  # Check if correct digits are infinite for an exact match
    assert correct_digits(0, 0) == np.inf  # Check if correct digits are infinite for zero error

def test_exp_edge_cases():
    assert exp(-1) == np.exp(-1)  # Check exp for negative input
    assert exp(np.nan) != exp(np.nan)  # Check exp for NaN input (result should also be NaN)

# Additional tests for cos function
def test_cos_edge_cases():
    assert cos(-1) == np.cos(-1)  # Check cos for negative input
    assert cos(np.nan) != cos(np.nan)  # Check cos for NaN input (result should also be NaN)

# Additional tests for cubic function
def test_cubic_edge_cases():
    assert cubic(2) == 8 + 0.5  # Check cubic for positive input
    assert cubic(-2) == -8 + 0.5  # Check cubic for negative input
    assert cubic(np.nan) == np.nan  # Check cubic for NaN input

# Additional tests for simpson function
def test_simpson_edge_cases():
    result, _ = simpson(lambda x: x**2, -1, 1, 100, 1e-6)  # Integrate x^2 from -1 to 1
    assert np.isclose(result, 2/3, atol=1e-6)  # Check if result is close to 2/3

# Additional tests for trapezoid function
def test_trapezoid_edge_cases():
    result, _ = trapezoid(lambda x: x**2, -1, 1, 100, 1e-6)  # Integrate x^2 from -1 to 1
    assert np.isclose(result, 2/3, atol=1e-6)  # Check if result is close to 2/3

# Additional tests for adaptive_trapezoid function
def test_adaptive_trapezoid_edge_cases():
    result, _ = adaptive_trapezoid(lambda x: x**2, -1, 1, 1e-6)  # Integrate x^2 from -1 to 1
    assert np.isclose(result, 2/3, atol=1e-6)  # Check if result is close to 2/3

# Additional tests for correct_digits function
def test_correct_digits_edge_cases():
    assert correct_digits(1.9999, 2) == pytest.approx(4, abs=0.1)  # Check for high accuracy approximation
    assert correct_digits(0, 1) == -np.log10(1)  # Check for zero actual value and nonzero reference

