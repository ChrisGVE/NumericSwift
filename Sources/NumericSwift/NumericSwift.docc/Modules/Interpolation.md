# Interpolation

Interpolation methods for data fitting.

## Overview

The Interpolation module provides various interpolation methods including cubic splines, PCHIP (Piecewise Cubic Hermite Interpolating Polynomial), and Akima interpolation.

## Cubic Spline Interpolation

```swift
let x = [0.0, 1.0, 2.0, 3.0, 4.0]
let y = [0.0, 1.0, 4.0, 9.0, 16.0]

// Compute spline coefficients
let coeffs = computeSplineCoeffs(x: x, y: y, bc: .notAKnot)

// Evaluate at new points
let yNew = evalCubicSpline(x: x, coeffs: coeffs, xNew: 2.5)

// Evaluate derivative
let deriv = evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 2.5)

// Integrate over interval
let integral = integrateCubicSpline(x: x, coeffs: coeffs, a: 0, b: 4)
```

### Boundary Conditions

- `.natural` - Second derivative is zero at boundaries
- `.clamped(d0, dn)` - Specify first derivative at boundaries
- `.notAKnot` - Not-a-knot condition (default)

## PCHIP Interpolation

Monotonicity-preserving interpolation that avoids overshoots:

```swift
let x = [0.0, 1.0, 2.0, 3.0]
let y = [0.0, 1.0, 1.5, 1.6]

// Compute derivatives
let derivs = computePchipDerivatives(x: x, y: y)

// Evaluate
let yNew = evalPchip(x: x, y: y, d: derivs, xNew: 1.5)
```

## Akima Interpolation

Smooth interpolation that handles outliers well:

```swift
let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
let y = [0.0, 1.0, 2.0, 2.5, 2.8, 3.0]

// Compute Akima coefficients
let coeffs = computeAkimaCoeffs(x: x, y: y)

// Evaluate
let yNew = evalAkima(x: x, coeffs: coeffs, xNew: 2.5)
```

## Generic 1D Interpolation

```swift
// Using interp1d with various methods
let yNew = interp1d(x: x, y: y, xNew: 2.5, kind: .linear)
let yNew = interp1d(x: x, y: y, xNew: 2.5, kind: .cubic)
let yNew = interp1d(x: x, y: y, xNew: 2.5, kind: .nearest)
```

## Topics

### Cubic Spline

- ``computeSplineCoeffs(x:y:bc:)``
- ``evalCubicSpline(x:coeffs:xNew:)``
- ``evalCubicSplineDerivative(x:coeffs:xNew:)``
- ``integrateCubicSpline(x:coeffs:a:b:)``
- ``SplineBoundaryCondition``

### PCHIP

- ``computePchipDerivatives(x:y:)``
- ``evalPchip(x:y:d:xNew:)``

### Akima

- ``computeAkimaCoeffs(x:y:)``
- ``evalAkima(x:coeffs:xNew:)``

### Generic

- ``interp1d(x:y:xNew:kind:)``
- ``InterpolationKind``
