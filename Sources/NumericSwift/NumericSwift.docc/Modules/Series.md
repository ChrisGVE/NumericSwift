# Series

Polynomials, Taylor series, and power series.

## Overview

The Series module provides polynomial operations, Taylor series evaluation, and series summation utilities.

## Polynomial Operations

### Evaluation

```swift
// Evaluate polynomial using Horner's method
// coeffs = [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
let y = polyval([1, 2, 3], at: 2.0)  // 1 + 2*2 + 3*4 = 17

// Evaluate centered at a point
let y = polyval(coeffs, at: x, center: a)  // Evaluate at (x-a)
```

### Arithmetic

```swift
let p = [1, 2, 3]  // 1 + 2x + 3x^2
let q = [1, 1]     // 1 + x

// Addition
let sum = polyadd(p, q)

// Multiplication
let product = polymul(p, q)
```

### Calculus

```swift
let p = [1, 2, 3]  // 1 + 2x + 3x^2

// Derivative
let dp = polyder(p)    // [2, 6] = 2 + 6x

// Integral (constant term = 0)
let ip = polyint(p)    // [0, 1, 1, 1] = x + x^2 + x^3
```

## Taylor Series

```swift
// Get Taylor coefficients for common functions
let sinCoeffs = taylorCoefficients(for: "sin", terms: 10)
let expCoeffs = taylorCoefficients(for: "exp", terms: 20)

// Evaluate Taylor series
let y = taylorEval("exp", at: 0.5, terms: 20)  // e^0.5
let y = taylorEval("sin", at: 0.1, terms: 10)  // sin(0.1)
```

### Supported Functions

- `"sin"`, `"cos"`, `"tan"`
- `"exp"`, `"log"`
- `"sinh"`, `"cosh"`, `"tanh"`
- `"arcsin"`, `"arctan"`

## Series Summation

```swift
// Sum a series until convergence
let result = seriesSum(
    from: 1,
    tolerance: 1e-12
) { n in 1.0 / Double(n * n) }

print(result.sum)        // ≈ pi^2/6 (Basel sum)
print(result.converged)  // true
print(result.terms)      // Number of terms used

// Product of series
let productResult = seriesProduct(
    from: 1,
    tolerance: 1e-12
) { n in 1.0 - 1.0 / Double(n * n) }

// Partial sums for analysis
let sums = partialSums(from: 1, count: 100) { n in 1.0 / Double(n) }
```

## Chebyshev Points

```swift
// Generate Chebyshev points for stable polynomial interpolation
let points = chebyshevPoints(center: 0, scale: 1, count: 10)
```

## Divided Differences

```swift
// Newton's divided differences for interpolation
let diffs = dividedDifferences(xs: x, ys: y)
```

## Topics

### Polynomial Evaluation

- ``polyval(_:at:)``
- ``polyval(_:at:center:)``

### Polynomial Arithmetic

- ``polyadd(_:_:)``
- ``polymul(_:_:)``

### Polynomial Calculus

- ``polyder(_:)``
- ``polyint(_:)``

### Taylor Series

- ``taylorCoefficients(for:terms:)``
- ``taylorEval(_:at:terms:)``

### Series Summation

- ``seriesSum(from:to:tolerance:maxIterations:term:)``
- ``seriesProduct(from:to:tolerance:maxIterations:term:)``
- ``partialSums(from:count:term:)``

### Interpolation Support

- ``chebyshevPoints(center:scale:count:)``
- ``dividedDifferences(xs:ys:)``
