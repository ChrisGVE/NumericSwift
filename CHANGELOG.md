# Changelog

All notable changes to NumericSwift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-01-17

### Added

- Swift Package Index documentation support via `.spi.yml` manifest

### Fixed

- DocC documentation symbol links corrected to match actual function signatures
- Removed self-referential links that caused DocC cycle warnings
- Fixed overloaded function disambiguations (dblquad, tplquad, addConstant)
- Corrected tplquad parameter documentation format

## [0.1.0] - 2026-01-16

### Added

- **Complex Numbers**: Full arithmetic, transcendental functions, and polar form support
- **Constants**: Mathematical constants and physical constants (CODATA 2018)
- **Distributions**: Normal, T, Chi-squared, F, Gamma, Beta, Exponential, Uniform distributions
- **Integration**: Adaptive quadrature (`quad`), double/triple integrals, Romberg integration, ODE solvers (RK4, RK45)
- **Interpolation**: Cubic spline, PCHIP, Akima, Lagrange, barycentric interpolation
- **Optimization**: Bisection, Newton-Raphson, Brent's method, Nelder-Mead, curve fitting
- **Number Theory**: Primality testing, factorization, GCD/LCM, Euler's totient, Mobius function
- **Series**: Polynomial operations, Taylor series evaluation, power series summation
- **Special Functions**: Error functions (erf, erfc, erfinv), Bessel functions, gamma/beta functions, elliptic integrals, Lambert W, zeta function
- **Statistics**: Descriptive statistics, statistical tests (t-test, correlation), z-scores, skew, kurtosis
- **Linear Algebra**: Matrix type with LAPACK-backed decompositions (LU, QR, SVD, Cholesky), solvers
- **Cluster**: K-means, DBSCAN, hierarchical clustering with multiple linkage methods
- **Spatial**: KDTree for nearest neighbor queries, distance metrics, Voronoi, Delaunay
- **Geometry**: SIMD-backed vector/matrix types, coordinate transforms, ellipse fitting
- **Utilities**: vDSP-optimized array operations (exp, log, sin, cos, etc.)
- **MathExpr**: Mathematical expression parser and evaluator
- **Regression**: Linear/polynomial regression, nonlinear curve fitting, ARIMA time series

### Fixed

- `erfinv` now achieves ~14 digit precision across entire domain using Winitzki + Halley refinement
- `TDistribution.pdf` uses lgamma for numerical stability with large df values
- `TDistribution.ppf` improved with lgamma computation and tighter tolerance

### Known Limitations

- T-distribution ppf achieves ~5 digits precision at extreme tails (|p| > 0.9999); central and near-tail regions achieve full precision

### Dependencies

- Requires Swift 5.9+, iOS 15+ / macOS 12+
- Uses Apple Accelerate framework for LAPACK/BLAS/vDSP operations
- Optional ArraySwift integration via `NUMERICSWIFT_INCLUDE_ARRAYSWIFT=1` environment variable
