# NumericSwift

A comprehensive scientific computing library for Swift, inspired by SciPy.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg?style=flat)](LICENSE)
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
    .package(url: "https://github.com/ChrisGVE/NumericSwift.git", from: "0.3.0")
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
import NumericSwift

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

print(Stats.mean(data))           // 5.5
print(Stats.median(data))         // 5.5
print(Stats.stddev(data))         // 3.028...
print(Stats.percentile(data, 75)) // 7.75

// Probability distributions
let norm = NormalDistribution(loc: 0, scale: 1)
print(norm.pdf(0))     // 0.3989... (probability density)
print(norm.cdf(1.96))  // 0.975 (cumulative distribution)
print(norm.ppf(0.975)) // 1.96 (inverse CDF)
print(norm.rvs(5))     // 5 random samples
```

### Numerical Integration

```swift
// Adaptive quadrature
let result = quad({ x in sin(x) }, 0, .pi)
print(result.value)  // 2.0

// Non-stiff ODE: dy/dt = -y, y(0) = 1
let solution = solveIVP(
    { y, t in [-y[0]] },
    tSpan: (0, 5),
    y0: [1.0]
)

// Stiff ODE (implicit BDF method)
let stiff = solveIVP(
    { y, t in [1000 * (1 - y[0] * y[0]) * y[1] - y[0], y[1]] },
    tSpan: (0, 3000),
    y0: [2.0, 0.0],
    method: .bdf
)
```

### Linear Algebra

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
let (eigenvalues, _, _) = LinAlg.eig(A)
```

## Module Overview

| Module / Namespace | Description |
|--------------------|-------------|
| **Complex** | Complex number arithmetic with full operator support |
| **Constants** | Mathematical and physical constants (CODATA 2018) |
| **Distributions** | Probability distributions (Normal, T, Chi-squared, F, Gamma, Beta, Poisson, Binomial, etc.) |
| **Integration** | Numerical integration (`quad`, `romberg`) and ODE solvers (RK4, RK45, RK23, BDF for stiff systems) |
| **IntegrationStiff** _(internal)_ | BDF-1 stiff solver driver; reached via `solveIVP(method: .bdf)` |
| **Interpolation** | Cubic spline (natural, clamped, not-a-knot), PCHIP, Akima, barycentric |
| **InterpolationND** | N-dimensional grid interpolation (`interpn`); multilinear and nearest methods |
| **Optimization** | Root finding (`bisect`, `newton`, `brent`) and minimization (Nelder-Mead, BFGS, Levenberg-Marquardt) |
| **NumberTheory** | Primes, factorization, GCD, Euler's totient, and more — via `NumberTheory.*` |
| **Series** | Polynomials, Taylor series, power series — via `Series.*` |
| **SpecialFunctions** | Error functions, Bessel, gamma, beta, elliptic integrals |
| **Stats** | Descriptive statistics, correlation, statistical tests — via `Stats.*` |
| **LinAlg** | Matrix operations, decompositions (LU, QR, SVD, Cholesky), solvers |
| **LinAlg+Arithmetic** | Element-wise and scalar matrix arithmetic |
| **LinAlg+Complex** | Complex matrix arithmetic and decompositions |
| **LinAlg+Decompositions** | LU, QR, SVD, Cholesky, eigendecomposition |
| **LinAlg+MatrixFunctions** | `expm` (Higham 2005), `logm`, `sqrtm`, `logmComplex`, `sqrtmComplex`, `funm` |
| **LinAlg+Solvers** | `solve`, `lstsq`, `solveTriangular`, `choSolve`, `luSolve` |
| **Sparse** | COO and CSR sparse matrices; `spsolve`, `spmv`, `spmm` — via `Sparse.*` |
| **Cluster** | K-means, DBSCAN, hierarchical clustering — via `Cluster.*` |
| **Spatial** | KDTree, Voronoi, Delaunay, convex hull, distance metrics — via `Spatial.*` |
| **Geometry** | 2D/3D geometry with SIMD vectors; coordinate transforms — via `Geometry.*` |
| **ArrayOps** | vDSP-optimized array operations — via `ArrayOps.*` |
| **MathExpr** | Mathematical expression parsing and evaluation (pure-Swift by default; LaTeX via MathLex opt-in) |
| **Regression** | Linear/nonlinear regression, ARIMA time series (statsmodels-inspired) |
| **NumericValue** | Unified numeric type tower (0.3.0) — discriminated union over scalar/complex/matrix/complexMatrix |
| **NumericDispatch** | Unified dispatch surface (0.3.0) — routes (op, kind) pairs over all NumericValue combinations |
| **UnifiedEvaluator** | Single-pass AST evaluator backing `MathExpr.evaluateUnified` |

## Ecosystem

NumericSwift is part of a suite of Swift scientific computing libraries:

- **ArraySwift** (optional) — N-dimensional array support with NumPy-style API;
  compile with `NUMERICSWIFT_INCLUDE_ARRAYSWIFT=1`
- **PlotSwift** (optional) — data visualization and plotting;
  compile with `NUMERICSWIFT_INCLUDE_PLOTSWIFT=1`
- **MathLex** (optional Rust backend) — LaTeX parsing for `MathExpr`;
  compile with `NUMERICSWIFT_INCLUDE_MATHLEX=1`

### MathLex Integration (opt-in Rust backend)

By default, `MathExpr.parse` uses a pure-Swift tokenizer + shunting-yard
parser. This handles the full numeric expression language, including matrix
operations via the `values:` dictionary.

For LaTeX input and richer bracket-literal syntax, build with the MathLex
Rust crate:

```bash
NUMERICSWIFT_INCLUDE_MATHLEX=1 swift build
```

With MathLex enabled:

- `MathExpr.parseLatex(_:)` accepts LaTeX math strings and returns the same
  `MathLexExpression` AST used by all evaluators.
- Bracket-literal expressions like `[1, 2, 3]` or `[[1, 0], [0, 1]]` are
  tokenized directly.

Without MathLex (`NUMERICSWIFT_INCLUDE_MATHLEX=0`, the default):

- `MathExpr.parseLatex(_:)` always throws
  `MathExprError.parseError("LaTeX parsing requires the MathLex backend …")`.
- Bracket-literal expressions throw `MathExprError.parseError`. Supply matrix
  values via the `values:` dictionary instead.

MathLex is a separate optional dependency and is never pulled in by default.
Pure-Swift consumers (e.g. iOS/tvOS targets without a Rust toolchain in CI)
work with no additional setup.

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

## Unified Numeric Pipeline (0.3.0)

NumericSwift 0.3.0 introduces a unified numeric evaluation pipeline that handles scalars, complex
numbers, real matrices, and complex matrices in a single recursive pass.

### NumericValue — the type tower

`NumericValue` is a discriminated union (`enum`) over the four numeric kinds:

```swift
import NumericSwift

// Supply a matrix via the variable binding dictionary
let A = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])

let ast = try MathExpr.parse("det(A) + trace(A)")
let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(A)])
// result == .scalar(det(A) + trace(A)) == .scalar(-2.0 + 5.0) == .scalar(3.0)
```

Pattern-match on the result to extract the payload:

```swift
switch result {
case .scalar(let x):        print("scalar:", x)
case .complex(let z):       print("complex:", z)
case .matrix(let m):        print("matrix:", m.rows, "x", m.cols)
case .complexMatrix(let cm): print("complexMatrix:", cm.rows, "x", cm.cols)
}
```

### Operator semantics

- `*` between two matrices is **matrix multiplication** (matmul), not element-wise.
  Element-wise multiplication uses the `hadamard` named function.
- `dot(u, v)` on two column vectors returns a `.scalar` (1×1 → scalar coercion, §4.3a).
- `dot(u, v)` on complex column vectors is **bilinear** (Σ uᵢ·vᵢ, no conjugation).

### Fallback-parser limitation

The default build has no bracket tokenizer. Expressions like `[1, 2, 3]` or `[[1, 2], [3, 4]]`
throw `MathExprError.parseError`. Supply matrix values via the `values:` dictionary instead.
Bracket-literal parsing is available with the MathLex Rust backend
(`NUMERICSWIFT_INCLUDE_MATHLEX=1`).

### Complex-context promotion

`evaluateComplex` evaluates in **complex mode**: `sqrt(-1)`, `log(-1)`, `ln(-1)`, and the `^`
operator with a negative base and a non-integer exponent are promoted to their complex
principal value (numpy/SciPy convention — `sqrt(-1) = +i`, `log(-1) = +iπ`) instead of `NaN`.
The real `evaluate` keeps the IEEE-754 NaN contract (`eval("sqrt(-4)")` is `NaN`). The
promotion set is narrow: `pow(x, y)` as a *function*, `log10`/`log2`, and inverse-trig still
return `NaN` for negative reals. (Resolved [GitHub issue #1](https://github.com/ChrisGVE/NumericSwift/issues/1).)

## Performance

NumericSwift uses a unified numeric evaluation pipeline (`MathExpr.evaluateUnified`) that routes
mathematical expressions over the `NumericValue` type tower — scalar, complex, real matrix, and
complex matrix — through a single recursive evaluator backed by the `NumericDispatch` surface.
The pipeline is designed to keep overhead minimal relative to calling `LinAlg` primitives directly.

### Ratio gates

Performance is validated by a set of self-relative ratio gates (in `Sources/NumericSwiftBench/`).
Each gate measures two legs on the same machine in the same process and reports the ratio;
no machine-specific baseline file is committed. All gates must satisfy their threshold:

| Gate | Numerator (unified path) | Denominator (direct) | Threshold |
|------|--------------------------|----------------------|-----------|
| Gate 1 | `MathExpr.evaluateUnified` scalar corpus | `MathExpr.evaluate` (legacy) | ≤ 1.15 |
| Gate 2 | `evaluateUnified("A * B")` with real matrices | `LinAlg.dot(A, B)` (BLAS) | ≤ 1.10 |
| Gate 3 | `evaluateUnified("A * B")` with complex matrices | 4× `LinAlg.dot` real-block | ≤ 1.10 |
| Gate 4 | `evaluateUnified("exp(M)")` with matrix | `LinAlg.expm(M)` (Padé) | ≤ 1.10 |

**Interpretation:** a ratio of 1.10 means the unified evaluator path adds at most 10% overhead
on top of the direct `LinAlg` call. The extra cost is evaluator dispatch (variable lookup,
`NumericValue` wrapping, function-registry lookup); the underlying BLAS/Padé computation is
the same in both legs.

### Running the bench

```bash
# Build and run (release mode gives authoritative numbers)
swift build -c release --product NumericSwiftBench
.build/release/NumericSwiftBench

# Tuning environment variables (optional)
BENCH_WARMUP=10 BENCH_SAMPLES=51 .build/release/NumericSwiftBench

# Gate framework self-tests (negative-case FAIL + PENDING proofs)
BENCH_SELF_TEST=1 .build/debug/NumericSwiftBench
```

Typical release-mode results on Apple Silicon (arm64):

```
Gate                                 Denom ns  Numer ns    Ratio  Threshold  State
gate1: unified-vs-legacy (≤1.15)      6.4µs     6.3µs   0.984   ≤ 1.15  PASS
gate2: expr-matmul vs direct-dot     12.0µs    12.3µs   1.026   ≤ 1.10  PASS
gate3: complex-matmul vs real-block  53.4µs    54.1µs   1.013   ≤ 1.10  PASS
gate4: expm-via-expr vs expm-direct   3.0µs     3.2µs   1.077   ≤ 1.10  PASS
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

NumericSwift is released under the Apache License 2.0. See [LICENSE](LICENSE) for details
