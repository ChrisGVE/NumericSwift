# Optimization

Root finding and function minimization.

## Overview

The Optimization module provides algorithms for finding roots of equations and minimizing functions, following scipy.optimize patterns.

## Scalar Root Finding

### Bisection Method

```swift
// Find root of x^2 - 2 = 0 in [0, 2]
let result = bisect({ x in x*x - 2 }, a: 0, b: 2)
print(result.root)  // ≈ 1.414 (sqrt(2))
```

### Newton-Raphson Method

```swift
// With analytic derivative
let result = newton(
    { x in x*x - 2 },
    x0: 1.5,
    fprime: { x in 2*x }
)

// Without derivative (uses finite differences)
let result = newton({ x in x*x - 2 }, x0: 1.5)
```

### Brent's Method

```swift
// Combines bisection and inverse quadratic interpolation
let result = brent({ x in x*x - 2 }, a: 0, b: 2)
print(result.root)
```

### Secant Method

```swift
let result = secant({ x in x*x - 2 }, x0: 0, x1: 2)
```

## Scalar Minimization

### Golden Section Search

```swift
let result = goldenSection({ x in (x-2)*(x-2) }, a: 0, b: 4)
print(result.x)  // ≈ 2.0
```

## Multivariate Optimization

### Nelder-Mead Simplex

```swift
let result = nelderMead(
    { x in (x[0]-1)*(x[0]-1) + (x[1]-2)*(x[1]-2) },
    x0: [0, 0]
)
print(result.x)  // [1, 2]
```

### Multivariate Root Finding

```swift
let result = newtonMulti(
    { x in [x[0] + x[1] - 3, x[0] - x[1] - 1] },
    x0: [0, 0]
)
print(result.x)  // ≈ [2, 1]
```

## Curve Fitting

```swift
// Fit model y = a * exp(b * x)
func model(_ params: [Double], _ x: Double) -> Double {
    return params[0] * exp(params[1] * x)
}

let (popt, pcov, info) = curveFit(
    model,
    xdata: xData,
    ydata: yData,
    p0: [1.0, 0.1]  // Initial guess
)
print(popt)  // Optimal parameters
```

## Topics

### Result Types

- ``MinimizeScalarResult``
- ``RootScalarResult``
- ``MinimizeResult``
- ``RootResult``
- ``LeastSquaresResult``

### Root Finding

- ``bisect(_:a:b:xtol:maxiter:)``
- ``newton(_:fprime:x0:xtol:maxiter:)``
- ``brent(_:a:b:xtol:maxiter:)``
- ``secant(_:x0:x1:xtol:maxiter:)``

### Scalar Minimization

- ``goldenSection(_:a:b:xtol:maxiter:)``

### Multivariate Optimization

- ``nelderMead(_:x0:xtol:ftol:maxiter:)``
- ``newtonMulti(_:x0:tol:maxiter:)``

### Curve Fitting

- ``curveFit(_:xdata:ydata:p0:ftol:xtol:maxiter:)``
- ``leastSquares(_:x0:ftol:xtol:maxiter:)``
