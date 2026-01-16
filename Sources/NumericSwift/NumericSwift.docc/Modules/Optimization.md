# Optimization

Root finding and function minimization.

## Overview

The Optimization module provides algorithms for finding roots of equations and minimizing functions, following scipy.optimize patterns.

## Scalar Root Finding

### Bisection Method

```swift
// Find root of x^2 - 2 = 0 in [0, 2]
let root = bisect({ x in x*x - 2 }, a: 0, b: 2)
// root ≈ 1.414 (sqrt(2))
```

### Newton-Raphson Method

```swift
// With analytic derivative
let root = newton(
    { x in x*x - 2 },
    x0: 1.5,
    fprime: { x in 2*x }
)

// Without derivative (uses secant method internally)
let root = newton({ x in x*x - 2 }, x0: 1.5)
```

### Brent's Method

```swift
// Combines bisection and inverse quadratic interpolation
let root = brent({ x in x*x - 2 }, a: 0, b: 2)
```

### Secant Method

```swift
let root = secant({ x in x*x - 2 }, x0: 0, x1: 2)
```

## Scalar Minimization

### Golden Section Search

```swift
let minimum = goldenSection({ x in (x-2)*(x-2) }, a: 0, b: 4)
// minimum ≈ 2.0
```

### Brent's Minimization

```swift
let minimum = brent({ x in (x-2)*(x-2) }, a: 0, b: 4)
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
// result ≈ [2, 1]
```

## Curve Fitting

```swift
// Fit model y = a * exp(b * x)
func model(_ x: Double, _ p: [Double]) -> Double {
    return p[0] * exp(p[1] * x)
}

let (popt, pcov, info) = curveFit(
    model,
    xdata: xData,
    ydata: yData,
    p0: [1.0, 0.1]  // Initial guess
)
```

## Topics

### Root Finding

- ``bisect(_:a:b:xtol:maxiter:)``
- ``newton(_:x0:fprime:tol:maxiter:)``
- ``brent(_:a:b:xtol:maxiter:)``
- ``secant(_:x0:x1:tol:maxiter:)``

### Scalar Minimization

- ``goldenSection(_:a:b:tol:maxiter:)``

### Multivariate Optimization

- ``nelderMead(_:x0:tol:maxiter:)``
- ``newtonMulti(_:x0:tol:maxiter:)``
- ``minimize(_:x0:method:tol:maxiter:)``

### Curve Fitting

- ``curveFit(_:xdata:ydata:p0:)``
- ``leastSquares(_:x0:)``
