# Numerical Integration

Numerical integration and ODE solvers.

## Overview

The Integration module provides adaptive quadrature for definite integrals and numerical methods for solving ordinary differential equations (ODEs).

## Quadrature

### Single Integrals

```swift
// Adaptive quadrature (Gauss-Kronrod with adaptive subdivision)
let result = quad({ x in sin(x) }, 0, .pi)
print(result.value)  // ≈ 2.0
print(result.error)  // Estimated error

// Fixed-point Gauss-Legendre quadrature
let value = fixedQuad({ x in exp(-x*x) }, -1, 1, n: 5)

// Romberg integration
let result = romberg({ x in 1/x }, 1, 2)
print(result.value, result.error, result.iterations)

// Simple integration rules
let y = [1.0, 4.0, 9.0, 16.0, 25.0]
let simpsonResult = simps(y, dx: 1.0)
let trapezoidResult = trapz(y, dx: 1.0)
```

### Multiple Integrals

```swift
// Double integral with function limits
let result = dblquad(
    { y, x in x * y },
    xa: 0, xb: 1,                  // x bounds
    ya: { _ in 0 }, yb: { _ in 1 } // y bounds (can depend on x)
)

// Double integral with constant limits
let result2 = dblquad(
    { y, x in x * y },
    xa: 0, xb: 1,
    ya: 0, yb: 1
)

// Triple integral
let result3 = tplquad(
    { z, y, x in x * y * z },
    xa: 0, xb: 1,
    ya: { _ in 0 }, yb: { _ in 1 },
    za: { _, _ in 0 }, zb: { _, _ in 1 }
)
```

## ODE Solvers

### Initial Value Problems

```swift
// Solve dy/dt = f(y, t) with y(t0) = y0

// Using solveIVP (recommended)
let solution = solveIVP(
    { y, t in [-y[1], y[0]] },  // Simple harmonic oscillator
    tSpan: (0, 10),
    y0: [1.0, 0.0],
    method: .rk45               // Adaptive Runge-Kutta
)

// Access solution
for (t, y) in zip(solution.t, solution.y) {
    print("t=\(t): y=\(y)")
}
```

### Using odeint (scipy-compatible)

```swift
let result = odeint(
    { y, t in [y[1], -y[0]] },
    y0: [0.0, 1.0],
    t: Array(stride(from: 0, through: 10, by: 0.1))
)
```

### ODE Methods

- `.rk45` - Dormand-Prince 4(5) (default, adaptive)
- `.rk23` - Bogacki-Shampine 2(3) (adaptive)
- `.rk4` - Classic Runge-Kutta 4 (fixed step)

## Topics

### Quadrature Functions

- ``quad(_:_:_:epsabs:epsrel:limit:)``
- ``dblquad(_:xa:xb:ya:yb:epsabs:epsrel:)-(_,_,_,(Double)->Double,_,_,_)``
- ``dblquad(_:xa:xb:ya:yb:epsabs:epsrel:)-(_,_,_,Double,_,_,_)``
- ``tplquad(_:xa:xb:ya:yb:za:zb:epsabs:epsrel:)-(_,_,_,(Double)->Double,_,_,_,_,_)``
- ``tplquad(_:xa:xb:ya:yb:za:zb:epsabs:epsrel:)-(_,_,_,Double,_,_,_,_,_)``
- ``fixedQuad(_:_:_:n:)``
- ``romberg(_:_:_:tol:divmax:)``
- ``simps(_:dx:)``
- ``simps(_:x:)``
- ``trapz(_:dx:)``
- ``trapz(_:x:)``
- ``QuadResult``

### ODE Solvers

- ``solveIVP(_:tSpan:y0:method:tEval:maxStep:rtol:atol:firstStep:)``
- ``odeint(_:y0:t:rtol:atol:)``
- ``ODEMethod``
- ``ODEResult``
