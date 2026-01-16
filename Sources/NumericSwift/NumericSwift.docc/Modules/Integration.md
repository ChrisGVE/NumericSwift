# Numerical Integration

Numerical integration and ODE solvers.

## Overview

The Integration module provides adaptive quadrature for definite integrals and numerical methods for solving ordinary differential equations (ODEs).

## Quadrature

### Single Integrals

```swift
// Adaptive quadrature (Simpson's rule with adaptive subdivision)
let (result, error) = quad({ x in sin(x) }, 0, .pi)
// result ≈ 2.0

// Fixed-point Gauss-Legendre quadrature
let (result, error) = fixedQuad({ x in exp(-x*x) }, -1, 1, n: 5)

// Romberg integration
let (result, error, iterations) = romberg({ x in 1/x }, 1, 2)
```

### Multiple Integrals

```swift
// Double integral
let (result, error) = dblquad(
    { y, x in x * y },
    0, 1,                    // x bounds
    { _ in 0 }, { _ in 1 }   // y bounds (can depend on x)
)

// Triple integral
let (result, error) = tplquad(
    { z, y, x in x * y * z },
    0, 1,                              // x bounds
    { _ in 0 }, { _ in 1 },            // y bounds
    { _, _ in 0 }, { _, _ in 1 }       // z bounds
)
```

## ODE Solvers

### Initial Value Problems

```swift
// Solve dy/dt = f(t, y) with y(t0) = y0

// Using solveIVP (recommended)
let solution = solveIVP(
    f: { t, y in [-y[1], y[0]] },  // Simple harmonic oscillator
    tSpan: (0, 10),
    y0: [1.0, 0.0],
    method: .rk45                   // Adaptive Runge-Kutta
)

// Access solution
for (t, y) in zip(solution.t, solution.y) {
    print("t=\(t): y=\(y)")
}
```

### Using odeint (scipy-compatible)

```swift
let result = odeint(
    func: { t, y in [y[1], -y[0]] },
    y0: [0.0, 1.0],
    t: Array(stride(from: 0, through: 10, by: 0.1))
)
```

## Topics

### Quadrature Functions

- ``quad(_:_:_:epsabs:epsrel:limit:)``
- ``dblquad(_:_:_:_:_:epsabs:epsrel:)``
- ``tplquad(_:_:_:_:_:_:_:epsabs:epsrel:)``
- ``fixedQuad(_:_:_:n:)``
- ``romberg(_:_:_:tol:maxIterations:)``

### ODE Solvers

- ``solveIVP(f:tSpan:y0:method:rtol:atol:maxStep:)``
- ``odeint(func:y0:t:args:)``
- ``ODEMethod``
