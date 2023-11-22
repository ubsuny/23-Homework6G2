# Task 3: Comparison of NumPy Integration Functions and Custom Integration Methods

## Introduction
This task involves comparing the performance and accuracy of NumPy integration functions with custom numerical integration methods. We focus on three functions - `exp_function`, `cos_function`, and `cubic_function` - over specified intervals. The comparison is based on the results obtained from both the custom and NumPy implementations of Simpson's and Trapezoidal rules, including an Adaptive Trapezoidal method.

## Methodology
- **Functions Analyzed**: 
  - `exp_function`
  - `cos_function`
  - `cubic_function`
- **Custom Integration Methods**: 
  - Simpson's Rule
  - Trapezoidal Rule
  - Adaptive Trapezoidal Rule 
- **Numpy Integration Methods**: 
  - NumPy Simpson's Integral
  - NumPy Trapezoidal Integral

## Results and Analysis

## Integration Results Interpretation

### Function: exp_function (Interval: [0, 20])
- **Custom Simpson's Integral**: 16.557
- **Custom Trapezoidal Integral**: 16.557
- **Custom Adaptive Trapezoidal Integral**: 16.557
- **NumPy Simpson's Integral**: 16.550
- **NumPy Trapezoidal Integral**: 16.557

For `exp_function`, the results are remarkably consistent across all custom methods, suggesting that the integration of this function over the given interval is well-handled by these techniques. The minor deviation in the NumPy Simpson's result (16.550) could be due to the internal implementation of Simpson's rule in NumPy, possibly involving different handling of endpoint calculations or interval subdivisions.

### Function: cos_function (Interval: [0, 2π])
- **Custom Simpson's Integral**: 4.790
- **Custom Trapezoidal Integral**: 4.793
- **Custom Adaptive Trapezoidal Integral**: 4.792
- **NumPy Simpson's Integral**: 4.782
- **NumPy Trapezoidal Integral**: 4.785

For `cos_function`, the custom methods provide similar results, with a slight spread from 4.790 to 4.793, indicating that this periodic function poses a slightly more complex integration challenge. The larger deviation in the NumPy Simpson's method (4.782) suggests a possible difference in handling the oscillatory nature of the cosine function, which might affect the accuracy of numerical integration using Simpson's rule in NumPy.

### Function: cubic_function (Interval: [-1, 1])
- **Custom Simpson's Integral**: 1.000
- **Custom Trapezoidal Integral**: 1.000
- **Custom Adaptive Trapezoidal Integral**: 1.000
- **NumPy Simpson's Integral**: 0.999
- **NumPy Trapezoidal Integral**: 1.000

For `cubic_function`, a straightforward polynomial, all methods except the NumPy Simpson's yield an exact result (1.000). The slight deviation in the NumPy Simpson's method (0.999) is interesting and could be related to how numerical approximations are applied in the integration process, especially for polynomials.

## Detailed Efficiency Comparison (Iterations)

### Function: exp_function (Interval: [0, 20])
- **Simpson Iterations**: 1
- **Trapezoid Iterations**: 1
- **Adaptive Trapezoid Iterations**: 7

The `exp_function` is well-handled with minimal iterations in both the Simpson and Trapezoidal methods. The adaptive trapezoidal method, however, required more iterations (7), which could suggest that its adaptive algorithm is more sensitive to achieving the accuracy threshold for this particular function.

### Function: cos_function (Interval: [0, 2π])
- **Simpson Iterations**: 2
- **Trapezoid Iterations**: 5
- **Adaptive Trapezoid Iterations**: 12

The `cos_function` presents a more challenging scenario, particularly for the adaptive trapezoidal method which needed up to 12 iterations to reach the desired accuracy. This increased number of iterations reflects the complexity of integrating an oscillatory function and the efficiency of the adaptive algorithm in handling such scenarios.

### Function: cubic_function (Interval: [-1, 1])
- **Simpson Iterations**: 1
- **Trapezoid Iterations**: 1
- **Adaptive Trapezoid Iterations**: 1

For the simple `cubic_function`, all methods converge quickly, indicating their effectiveness and efficiency in dealing with straightforward polynomial integrations.

## Conclusion
The comparison reveals that custom integration methods perform comparably to NumPy's functions in terms of accuracy. The custom methods are particularly effective for the polynomial function. However, for functions with more complex behaviors, like `exp_function` and `cos_function`, slight differences are observed, likely due to the nuances in how numerical approximations are handled. Efficiency-wise, the custom adaptive method, while slightly more iterative, provides a good balance between accuracy and computational cost, particularly for complex integrations.

