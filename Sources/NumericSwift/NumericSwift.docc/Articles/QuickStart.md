# Quick Start

Get started with NumericSwift's core features.

## Overview

This guide introduces the most commonly used features of NumericSwift through practical examples.

## Complex Numbers

Create and manipulate complex numbers with full arithmetic support:

```swift
import NumericSwift

let z1 = Complex(re: 3, im: 4)
let z2 = Complex.polar(r: 2, theta: .pi/4)

print(z1.abs)        // 5.0 (magnitude)
print(z1.arg)        // 0.927... (phase angle)
print(z1 * z2)       // Complex multiplication
print(z1.exp)        // e^z
print(z1.sqrt)       // Principal square root
```

## Statistical Functions

Compute descriptive statistics and work with probability distributions:

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

print(mean(data))           // 5.5
print(median(data))         // 5.5
print(stddev(data))         // 3.028...
print(percentile(data, 75)) // 7.75

// Probability distributions
let norm = NormalDistribution(loc: 0, scale: 1)
print(norm.pdf(0))          // 0.3989... (probability density)
print(norm.cdf(1.96))       // 0.975 (cumulative distribution)
print(norm.ppf(0.975))      // 1.96 (inverse CDF)
print(norm.rvs(5))          // 5 random samples
```

## Numerical Integration

Integrate functions and solve differential equations:

```swift
// Adaptive quadrature
let result = quad({ x in sin(x) }, 0, .pi)
print(result.value)  // 2.0

// Solve ODE: dy/dt = -y, y(0) = 1
let solution = solveIVP(
    { y, t in [-y[0]] },
    tSpan: (0, 5),
    y0: [1.0]
)
```

## Linear Algebra

Work with matrices and solve linear systems:

```swift
let A = LinAlg.Matrix([[4, 3], [6, 3]])
let b = LinAlg.Matrix([[1], [2]])

// Solve Ax = b
if let x = LinAlg.solve(A, b) {
    print(x)
}

// Matrix decompositions
let (L, U, P) = LinAlg.lu(A)
let (Q, R) = LinAlg.qr(A)
let (values, _, _) = LinAlg.eig(A)
```

## Optimization

Find roots and minimize functions:

```swift
// Find root of x^2 - 2 = 0
let root = bisect({ x in x*x - 2 }, a: 0, b: 2)
print(root.root)  // 1.414... (sqrt(2))

// Minimize (x-2)^2
let minimum = goldenSection({ x in (x-2)*(x-2) }, a: 0, b: 4)
print(minimum.x)  // 2.0
```

## Special Functions

Access mathematical special functions:

```swift
// Error functions
print(erf(1.0))      // 0.8427...
print(erfinv(0.5))   // 0.4769...

// Bessel functions
print(j0(1.0))       // 0.7652...
print(y0(1.0))       // 0.0883...

// Gamma functions
print(digamma(2.0))  // 0.4228...
```

## Next Steps

Explore the individual module documentation for detailed API references:

- <doc:Complex> for complex number operations
- <doc:LinAlg> for linear algebra
- <doc:Distributions> for probability distributions
- <doc:SpecialFunctions> for mathematical special functions
