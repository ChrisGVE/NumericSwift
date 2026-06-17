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

- `.natural` - Second derivative is zero at boundaries (f″ = 0)
- `.clamped` - Zero first derivative at both ends (f′ = 0); shorthand for `.clamped(dStart: 0, dEnd: 0)`
- `.clamped(dStart:dEnd:)` - User-supplied first derivatives at each end; matches SciPy `CubicSpline(bc_type=((1, d0), (1, d1)))`
- `.notAKnot` - Not-a-knot condition (default; requires ≥ 4 points)

### RawRepresentable (string serialisation)

`SplineBoundaryCondition` conforms to `RawRepresentable` with `RawValue == String`.
These raw strings are frozen as part of the published 0.2.x API contract:

| Case                     | `rawValue`     |
|--------------------------|----------------|
| `.natural`               | `"natural"`    |
| `.clamped(dStart:dEnd:)` | `"clamped"`    |
| `.notAKnot`              | `"not-a-knot"` |

```swift
// Serialise
let raw = SplineBoundaryCondition.notAKnot.rawValue  // "not-a-knot"

// Deserialise — "clamped" always restores the zero-slope default
let bc = SplineBoundaryCondition(rawValue: "clamped")  // .clamped(dStart: 0, dEnd: 0)
```

Because the raw value identifies the *kind* of boundary condition (not the
derivative values), `init?(rawValue: "clamped")` always returns
`.clamped(dStart: 0, dEnd: 0)` — the historic zero-slope default.  Callers
that need to persist a specific derivative pair must serialise those values
separately alongside the raw string.

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

## Lagrange and Barycentric Interpolation

```swift
// Lagrange interpolation
let yNew = evalLagrange(x: x, y: y, xNew: 2.5)

// Barycentric interpolation (more stable)
let weights = computeBarycentricWeights(x: x)
let yNew = evalBarycentric(x: x, y: y, w: weights, xNew: 2.5)
```

## N-Dimensional Grid Interpolation

`InterpolationND.interpn` evaluates an N-dimensional interpolant on a regular
(but not necessarily uniform) grid. The API mirrors `scipy.interpolate.interpn`.

```swift
// 2-D example: interpolate on a 3×4 grid
let x = [0.0, 1.0, 2.0]          // x-axis (3 nodes)
let y = [0.0, 1.0, 2.0, 3.0]     // y-axis (4 nodes)
let values = (0..<12).map { i in Double(i) }  // row-major, shape [3, 4]

// Query at a single point
let result = try InterpolationND.interpn(
    points: [x, y],
    values: values,
    xi: [[0.5, 1.5]],             // query point
    method: .linear,
    boundsHandling: .fillValue(.nan)
)
print(result[0])  // interpolated value
```

### Out-of-Bounds Policy

```swift
// Throw on out-of-bounds (default)
let strict = try InterpolationND.interpn(
    points: [x, y], values: values, xi: [[5.0, 0.0]],
    boundsHandling: .error)       // throws InterpError.outOfBounds

// Fill with NaN (SciPy default)
let filled = try InterpolationND.interpn(
    points: [x, y], values: values, xi: [[5.0, 0.0]],
    boundsHandling: .fillValue(.nan))
```

### Migration from top-level `interpn`

The free function `interpn(...)` at module level is deprecated. Replace calls
with the namespaced form:

```swift
// Before (deprecated)
let v = try interpn(points: axes, values: data, xi: queries)

// After
let v = try InterpolationND.interpn(points: axes, values: data, xi: queries)
```

Similarly, the top-level type aliases `InterpolationNDMethod` and
`InterpolationNDBoundsHandling` are deprecated in favour of
``InterpolationND/Method`` and ``InterpolationND/BoundsHandling``.

## Topics

### Cubic Spline

- ``computeSplineCoeffs(x:y:bc:)``
- ``evalCubicSpline(x:coeffs:xNew:extrapolate:)``
- ``evalCubicSplineDerivative(x:coeffs:xNew:order:)``
- ``integrateCubicSpline(x:coeffs:a:b:)``
- ``CubicCoeffs``
- ``SplineBoundaryCondition``

### PCHIP

- ``computePchipDerivatives(x:y:)``
- ``evalPchip(x:y:d:xNew:)``

### Akima

- ``computeAkimaCoeffs(x:y:)``
- ``evalAkima(x:coeffs:xNew:)``

### Generic

- ``interp1d(x:y:xNew:kind:fillValue:boundsError:coeffs:)->Double``
- ``interp1d(x:y:xNew:kind:fillValue:boundsError:coeffs:)->[Double]``
- ``InterpolationKind``

### Lagrange and Barycentric

- ``evalLagrange(x:y:xNew:)``
- ``computeBarycentricWeights(x:)``
- ``evalBarycentric(x:y:w:xNew:)``

### Utilities

- ``findInterval(_:_:)``
- ``solveTridiagonal(diag:offDiag:rhs:)``

### N-Dimensional Grid Interpolation

- ``InterpolationND``
- ``InterpolationND/interpn(points:values:xi:method:boundsHandling:)``
- ``InterpolationND/Method``
- ``InterpolationND/BoundsHandling``
- ``InterpolationND/InterpError``
