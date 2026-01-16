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
let (sum, converged, iterations) = seriesSum(
    from: 1,
    tolerance: 1e-12
) { n in 1.0 / Double(n * n) }
// sum ≈ pi^2/6 (Basel sum)
```

## Mathematical Constants via Series

```swift
let baselSum = baselSum           // pi^2/6
let gamma = eulerMascheroni       // Euler-Mascheroni constant
let catalan = catalanConstant     // Catalan's constant
let apery = aperyConstant         // zeta(3)
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

- ``seriesSum(from:tolerance:term:)``

### Constants

- ``baselSum``
- ``eulerMascheroni``
- ``catalanConstant``
- ``aperyConstant``
