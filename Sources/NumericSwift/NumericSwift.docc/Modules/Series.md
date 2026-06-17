# Series

Polynomials, Taylor series, and power series.

## Overview

The `Series` module provides polynomial operations, Taylor series evaluation,
and series summation utilities.

All functions live under the `Series` namespace. The old top-level free
functions (e.g. `polyval`, `polyder`, `seriesSum`) are still available as
`@available(*, deprecated)` shims so existing code continues to compile with a
deprecation warning. New code should use the namespaced forms.

## Migration from Top-Level Functions

```swift
// Before (deprecated)
polyval([1, 2, 3], at: 2.0)
seriesSum(from: 1, tolerance: 1e-12) { n in 1.0 / Double(n * n) }

// After
Series.polyval([1, 2, 3], at: 2.0)
Series.seriesSum(from: 1, tolerance: 1e-12) { n in 1.0 / Double(n * n) }
```

## Polynomial Operations

### Evaluation

```swift
// Evaluate polynomial using Horner's method
// coeffs = [a0, a1, a2, ...] for a0 + a1·x + a2·x² + ...
let y = Series.polyval([1, 2, 3], at: 2.0)  // 1 + 2·2 + 3·4 = 17

// Evaluate centered at a point
let y = Series.polyval(coeffs, at: x, center: a)  // Evaluate at (x − a)
```

### Arithmetic

```swift
let p = [1, 2, 3]  // 1 + 2x + 3x²
let q = [1, 1]     // 1 + x

// Addition
let sum = Series.polyadd(p, q)

// Multiplication
let product = Series.polymul(p, q)
```

### Calculus

```swift
let p = [1, 2, 3]  // 1 + 2x + 3x²

// Derivative
let dp = Series.polyder(p)    // [2, 6] = 2 + 6x

// Integral (constant term = 0)
let ip = Series.polyint(p)    // [0, 1, 1, 1] = x + x² + x³
```

## Taylor Series

```swift
// Get Taylor coefficients for common functions
let sinCoeffs = Series.taylorCoefficients(for: "sin", terms: 10)
let expCoeffs = Series.taylorCoefficients(for: "exp", terms: 20)

// Evaluate Taylor series
let y = Series.taylorEval("exp", at: 0.5, terms: 20)  // e^0.5
let y = Series.taylorEval("sin", at: 0.1, terms: 10)  // sin(0.1)
```

### Supported Functions

- `"sin"`, `"cos"`, `"tan"`
- `"exp"`, `"log"`
- `"sinh"`, `"cosh"`, `"tanh"`
- `"arcsin"`, `"arctan"`

## Series Summation

```swift
// Sum a series until convergence
let result = Series.seriesSum(
    from: 1,
    tolerance: 1e-12
) { n in 1.0 / Double(n * n) }

print(result.sum)        // ≈ π²/6 (Basel sum)
print(result.converged)  // true
print(result.terms)      // Number of terms used

// Product of series
let productResult = Series.seriesProduct(
    from: 1,
    tolerance: 1e-12
) { n in 1.0 - 1.0 / Double(n * n) }

// Partial sums for analysis
let sums = Series.partialSums(from: 1, count: 100) { n in 1.0 / Double(n) }
```

## Chebyshev Points

```swift
// Generate Chebyshev points for stable polynomial interpolation
let points = Series.chebyshevPoints(center: 0, scale: 1, count: 10)
```

## Divided Differences

```swift
// Newton's divided differences for interpolation
let diffs = Series.dividedDifferences(xs: x, ys: y)
```

## Topics

### Namespace

- ``Series``

### Polynomial Evaluation

- ``Series/polyval(_:at:)``
- ``Series/polyval(_:at:center:)``

### Polynomial Arithmetic

- ``Series/polyadd(_:_:)``
- ``Series/polymul(_:_:)``

### Polynomial Calculus

- ``Series/polyder(_:)``
- ``Series/polyint(_:)``

### Taylor Series

- ``Series/taylorCoefficients(for:terms:)``
- ``Series/taylorEval(_:at:terms:)``

### Series Summation

- ``Series/seriesSum(from:to:tolerance:maxIterations:term:)``
- ``Series/seriesProduct(from:to:tolerance:maxIterations:term:)``
- ``Series/partialSums(from:count:term:)``

### Interpolation Support

- ``Series/chebyshevPoints(center:scale:count:)``
- ``Series/dividedDifferences(xs:ys:)``
