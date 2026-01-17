# NumericSwift

A comprehensive scientific computing library for Swift, inspired by SciPy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/MIT)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/NumericSwift?style=flat&logo=github)](https://github.com/ChrisGVE/NumericSwift/releases)
[![CI](https://github.com/ChrisGVE/NumericSwift/actions/workflows/ci.yml/badge.svg)](https://github.com/ChrisGVE/NumericSwift/actions/workflows/ci.yml)
[![SPM Compatible](https://img.shields.io/badge/SPM-Compatible-brightgreen.svg?style=flat&logo=swift&logoColor=white)](https://swift.org/package-manager/)
[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9+-F05138.svg?style=flat&logo=swift&logoColor=white)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-iOS%2015+%20|%20macOS%2012+-007AFF.svg?style=flat&logo=apple&logoColor=white)](https://developer.apple.com)
[![Documentation](https://img.shields.io/badge/Documentation-DocC-blue.svg?style=flat&logo=readthedocs&logoColor=white)](https://github.com/ChrisGVE/NumericSwift)

## Overview

NumericSwift brings the power of scientific computing to Swift with an API inspired by SciPy and NumPy. It provides probability distributions, numerical integration, optimization algorithms, interpolation methods, special functions, linear algebra, clustering, and more.

**Design Philosophy:**
- **Complex numbers as first-class citizens** - Use `Double` or `Complex` interchangeably where mathematically appropriate
- **SciPy-inspired API** - Familiar function names and parameter orders for Python developers
- **Pure Swift with vDSP optimizations** - Leverages Apple's Accelerate framework for performance
- **Production quality** - Robust algorithms with proper edge case handling

## Installation

Add NumericSwift to your Swift package:

```swift
dependencies: [
    .package(url: "https://github.com/ChrisGVE/NumericSwift.git", from: "0.1.0")
]
```

Then add it to your target dependencies:

```swift
.target(
    name: "YourTarget",
    dependencies: ["NumericSwift"]
)
```

## Quick Start

### Complex Numbers

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

### Statistical Functions

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

### Numerical Integration

```swift
// Adaptive quadrature
let (result, error) = quad({ x in sin(x) }, 0, .pi)
print(result)  // 2.0

// Solve ODE: dy/dt = -y, y(0) = 1
let solution = solveIVP(
    f: { t, y in [-y[0]] },
    tSpan: (0, 5),
    y0: [1.0]
)
```

### Linear Algebra

```swift
let A = Matrix([[4, 3], [6, 3]])
let b = Matrix([[1], [2]])

// Solve Ax = b
let x = solve(A, b)

// Matrix decompositions
let (L, U, P) = A.lu()
let (Q, R) = A.qr()
let eigenvalues = A.eigenvalues()
```

## Module Overview

| Module | Description |
|--------|-------------|
| **Complex** | Complex number arithmetic with full operator support |
| **Constants** | Mathematical and physical constants (CODATA 2018) |
| **Distributions** | Probability distributions (Normal, T, Chi-squared, F, Gamma, Beta, etc.) |
| **Integration** | Numerical integration (quad, romberg) and ODE solvers (RK4, RK45) |
| **Interpolation** | Cubic spline, PCHIP, Akima interpolation |
| **Optimization** | Root finding (bisect, newton, brent) and minimization (Nelder-Mead) |
| **NumberTheory** | Primes, factorization, GCD, Euler's totient, and more |
| **Series** | Polynomials, Taylor series, power series |
| **SpecialFunctions** | Error functions, Bessel, gamma, beta, elliptic integrals |
| **Statistics** | Descriptive statistics, correlation, statistical tests |
| **LinAlg** | Matrix operations, decompositions (LU, QR, SVD, Cholesky), solvers |
| **Cluster** | K-means, DBSCAN, hierarchical clustering |
| **Spatial** | KDTree, Voronoi diagrams, Delaunay triangulation, distance metrics |
| **Geometry** | 2D/3D geometry with SIMD vectors, coordinate transforms |
| **Utilities** | vDSP-optimized array operations |
| **MathExpr** | Mathematical expression parsing and evaluation |
| **Regression** | Linear/nonlinear regression, ARIMA time series (statsmodels-inspired) |

## Ecosystem

NumericSwift is part of a suite of Swift scientific computing libraries:

- **ArraySwift** (optional) - N-dimensional array support with NumPy-style API
- **PlotSwift** (planned) - Data visualization and plotting

### ArraySwift Integration

When compiled with ArraySwift, NumericSwift provides seamless interoperability between `LinAlg.Matrix` and `NDArray`:

```bash
NUMERICSWIFT_INCLUDE_ARRAYSWIFT=1 swift build
```

This enables:

- **Matrix ↔ NDArray conversion**: `matrix.toNDArray()`, `LinAlg.Matrix(ndarray: arr)`
- **Linear algebra on NDArray**: `arr.solve(b)`, `arr.inv()`, `arr.det()`, `arr.lu()`, `arr.qr()`, `arr.svd()`, `arr.eig()`
- **Statistics functions for NDArray**: `sum(arr)`, `mean(arr)`, `stddev(arr)`, `percentile(arr, p)`
- **Math functions for NDArray**: `sinArray(arr)`, `expArray(arr)`, `sqrtArray(arr)`, etc.

```swift
import NumericSwift
import ArraySwift

// Convert between types
let matrix = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
let arr = matrix.toNDArray()  // NDArray with shape [2, 2]

// Linear algebra directly on NDArray
let A = NDArray([[3.0, 1.0], [1.0, 2.0]])
let b = NDArray([9.0, 8.0])
let x = A.solve(b)  // Solve Ax = b

// Decompositions
let (Q, R) = A.qr()!
let (U, s, Vt) = A.svd()!
```

## Documentation

Full API documentation is available via DocC. Build locally with:

```bash
swift package generate-documentation --target NumericSwift
```

## Requirements

- Swift 5.9+
- iOS 15+ / macOS 12+
- Accelerate framework (included with Apple platforms)

## License

NumericSwift is released under the MIT License. See [LICENSE](LICENSE) for details
