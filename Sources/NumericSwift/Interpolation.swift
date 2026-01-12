//
//  Interpolation.swift
//  NumericSwift
//
//  Interpolation algorithms following scipy.interpolate patterns.
//  Includes cubic splines, PCHIP, Akima, Lagrange, and barycentric interpolation.
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Cubic Spline Coefficients

/// Coefficients for a cubic polynomial segment: a + b*dx + c*dx² + d*dx³
public struct CubicCoeffs: Equatable {
    public let a: Double
    public let b: Double
    public let c: Double
    public let d: Double

    public init(a: Double, b: Double, c: Double, d: Double) {
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    }

    /// Evaluate the cubic polynomial at dx from the segment start.
    public func evaluate(at dx: Double) -> Double {
        a + b * dx + c * dx * dx + d * dx * dx * dx
    }

    /// Evaluate the derivative at dx.
    public func derivative(at dx: Double, order: Int = 1) -> Double {
        switch order {
        case 1:
            return b + 2.0 * c * dx + 3.0 * d * dx * dx
        case 2:
            return 2.0 * c + 6.0 * d * dx
        case 3:
            return 6.0 * d
        default:
            return 0
        }
    }

    /// Integrate the polynomial from 0 to dx.
    public func integrate(to dx: Double) -> Double {
        a * dx + b * dx * dx / 2.0 + c * dx * dx * dx / 3.0 + d * dx * dx * dx * dx / 4.0
    }
}

// MARK: - Interpolation Kind

/// Types of 1D interpolation.
public enum InterpolationKind: String {
    case linear
    case nearest
    case cubic
    case previous
    case next
}

/// Boundary condition types for cubic splines.
public enum SplineBoundaryCondition: String {
    case natural
    case clamped
    case notAKnot = "not-a-knot"
}

// MARK: - Binary Search

/// Find the interval containing x (returns index i such that xs[i] <= x < xs[i+1]).
public func findInterval(_ xs: [Double], _ x: Double) -> Int {
    var lo = 0
    var hi = xs.count - 1
    while hi - lo > 1 {
        let mid = (lo + hi) / 2
        if xs[mid] > x {
            hi = mid
        } else {
            lo = mid
        }
    }
    return lo
}

// MARK: - Tridiagonal Solver

/// Solve tridiagonal system using Thomas algorithm.
///
/// Solves Ax = b where A is tridiagonal with:
/// - `diag`: main diagonal
/// - `offDiag`: sub/super diagonal (symmetric, indexed 0..n-2)
///
/// - Parameters:
///   - diag: Main diagonal coefficients
///   - offDiag: Off-diagonal coefficients
///   - rhs: Right-hand side vector
/// - Returns: Solution vector
public func solveTridiagonal(diag: [Double], offDiag: [Double], rhs: [Double]) -> [Double] {
    let n = diag.count
    guard n > 0 else { return [] }

    var cPrime = [Double](repeating: 0, count: n)
    var dPrime = [Double](repeating: 0, count: n)

    // Forward elimination
    let off0 = offDiag.indices.contains(0) ? offDiag[0] : 0
    cPrime[0] = off0 / diag[0]
    dPrime[0] = rhs[0] / diag[0]

    for i in 1..<n {
        let offPrev = i - 1 < offDiag.count ? offDiag[i - 1] : 0
        let m = diag[i] - offPrev * cPrime[i - 1]
        let offCurr = i < offDiag.count ? offDiag[i] : 0
        cPrime[i] = offCurr / m
        dPrime[i] = (rhs[i] - offPrev * dPrime[i - 1]) / m
    }

    // Back substitution
    var result = [Double](repeating: 0, count: n)
    result[n - 1] = dPrime[n - 1]
    for i in stride(from: n - 2, through: 0, by: -1) {
        result[i] = dPrime[i] - cPrime[i] * result[i + 1]
    }

    return result
}

// MARK: - Cubic Spline

/// Compute cubic spline coefficients.
///
/// - Parameters:
///   - x: x-coordinates (must be sorted, strictly increasing)
///   - y: y-coordinates (function values)
///   - bc: Boundary condition type (natural, clamped, not-a-knot)
/// - Returns: Array of cubic coefficients for each segment
public func computeSplineCoeffs(x: [Double], y: [Double], bc: SplineBoundaryCondition = .notAKnot) -> [CubicCoeffs] {
    let n = x.count
    guard n >= 2 else { return [] }

    // Compute intervals h[i] = x[i+1] - x[i]
    var h = [Double](repeating: 0, count: n - 1)
    for i in 0..<(n - 1) {
        h[i] = x[i + 1] - x[i]
    }

    // Set up tridiagonal system for second derivatives
    var diag = [Double](repeating: 0, count: n)
    var offDiag = [Double](repeating: 0, count: n - 1)
    var rhs = [Double](repeating: 0, count: n)

    switch bc {
    case .natural:
        // Natural boundary: c[0] = 0, c[n-1] = 0
        diag[0] = 1
        rhs[0] = 0

        for i in 1..<(n - 1) {
            diag[i] = 2.0 * (h[i - 1] + h[i])
            offDiag[i - 1] = h[i - 1]
            rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        }

        diag[n - 1] = 1
        rhs[n - 1] = 0
        if n > 2 {
            offDiag[n - 2] = 0
        }

    case .clamped:
        // Clamped: f'(x0) = 0, f'(xn) = 0
        let fp0 = 0.0
        let fpn = 0.0

        diag[0] = 2.0 * h[0]
        rhs[0] = 3.0 * ((y[1] - y[0]) / h[0] - fp0)
        offDiag[0] = h[0]

        for i in 1..<(n - 1) {
            diag[i] = 2.0 * (h[i - 1] + h[i])
            offDiag[i] = h[i]
            rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        }

        diag[n - 1] = 2.0 * h[n - 2]
        rhs[n - 1] = 3.0 * (fpn - (y[n - 1] - y[n - 2]) / h[n - 2])

    case .notAKnot:
        // Not-a-knot requires n >= 4, fall back to natural otherwise
        if n >= 4 {
            // Build interior equations
            for i in 1..<(n - 1) {
                diag[i] = 2.0 * (h[i - 1] + h[i])
                offDiag[i - 1] = h[i - 1]
                if i < n - 1 {
                    offDiag[i] = h[i]
                }
                rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
            }

            // Not-a-knot: third derivatives match at x[1] and x[n-2]
            let alpha0 = h[0] / h[1]
            diag[1] = h[0] * (1.0 + alpha0) + 2.0 * (h[0] + h[1])
            offDiag[1] = h[1] - h[0] * alpha0

            let alphan = h[n - 2] / h[n - 3]
            diag[n - 2] = h[n - 2] * (1.0 + alphan) + 2.0 * (h[n - 3] + h[n - 2])
            if n > 3 {
                offDiag[n - 4] = h[n - 3] - h[n - 2] * alphan
            }

            // Build reduced system (rows 1 to n-2)
            let reducedN = n - 2
            var reducedDiag = [Double](repeating: 0, count: reducedN)
            var reducedOff = [Double](repeating: 0, count: reducedN - 1)
            var reducedRhs = [Double](repeating: 0, count: reducedN)

            for i in 0..<reducedN {
                reducedDiag[i] = diag[i + 1]
                reducedRhs[i] = rhs[i + 1]
                if i < reducedN - 1 {
                    reducedOff[i] = offDiag[i + 1]
                }
            }

            // Solve reduced system
            let cInner = solveTridiagonal(diag: reducedDiag, offDiag: reducedOff, rhs: reducedRhs)

            // Back-substitute to get c[0] and c[n-1]
            var c = [Double](repeating: 0, count: n)
            for i in 0..<reducedN {
                c[i + 1] = cInner[i]
            }
            c[0] = (1.0 + alpha0) * c[1] - alpha0 * c[2]
            c[n - 1] = (1.0 + alphan) * c[n - 2] - alphan * c[n - 3]

            // Compute a, b, d coefficients
            var coeffs = [CubicCoeffs]()
            for i in 0..<(n - 1) {
                let a = y[i]
                let b = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
                let d = (c[i + 1] - c[i]) / (3.0 * h[i])
                coeffs.append(CubicCoeffs(a: a, b: b, c: c[i], d: d))
            }
            return coeffs
        } else {
            // Fall back to natural for n < 4
            return computeSplineCoeffs(x: x, y: y, bc: .natural)
        }
    }

    // Solve for c values (for natural and clamped)
    let c = solveTridiagonal(diag: diag, offDiag: offDiag, rhs: rhs)

    // Compute a, b, d coefficients
    var coeffs = [CubicCoeffs]()
    for i in 0..<(n - 1) {
        let a = y[i]
        let b = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
        let d = (c[i + 1] - c[i]) / (3.0 * h[i])
        coeffs.append(CubicCoeffs(a: a, b: b, c: c[i], d: d))
    }

    return coeffs
}

/// Evaluate cubic spline at a point.
///
/// - Parameters:
///   - x: x-coordinates of data points
///   - coeffs: Spline coefficients
///   - xNew: Point to evaluate at
///   - extrapolate: Whether to extrapolate outside domain
/// - Returns: Interpolated value
public func evalCubicSpline(x: [Double], coeffs: [CubicCoeffs], xNew: Double, extrapolate: Bool = true) -> Double {
    let n = x.count

    // Check bounds
    if xNew < x[0] {
        if !extrapolate { return Double.nan }
        // Linear extrapolation using first segment
        let dx = xNew - x[0]
        return coeffs[0].a + coeffs[0].b * dx
    }

    if xNew > x[n - 1] {
        if !extrapolate { return Double.nan }
        // Linear extrapolation using last segment
        let idx = coeffs.count - 1
        let h = x[n - 1] - x[n - 2]
        let c = coeffs[idx]
        let slope = c.b + 2.0 * c.c * h + 3.0 * c.d * h * h
        let yEnd = c.evaluate(at: h)
        return yEnd + slope * (xNew - x[n - 1])
    }

    let i = findInterval(x, xNew)
    let idx = min(i, coeffs.count - 1)
    let dx = xNew - x[idx]
    return coeffs[idx].evaluate(at: dx)
}

/// Evaluate cubic spline derivative at a point.
public func evalCubicSplineDerivative(x: [Double], coeffs: [CubicCoeffs], xNew: Double, order: Int = 1) -> Double {
    let n = x.count

    // Clamp to domain
    let xEval = max(x[0], min(xNew, x[n - 1]))
    var i = findInterval(x, xEval)
    if i >= n - 1 { i = n - 2 }
    if i >= coeffs.count { i = coeffs.count - 1 }

    let dx = xEval - x[i]
    return coeffs[i].derivative(at: dx, order: order)
}

/// Integrate cubic spline over interval [a, b].
public func integrateCubicSpline(x: [Double], coeffs: [CubicCoeffs], a: Double, b: Double) -> Double {
    let n = x.count

    // Clamp to domain
    let x0 = max(x[0], min(a, x[n - 1]))
    let x1 = max(x[0], min(b, x[n - 1]))

    if x0 > x1 {
        return -integrateCubicSpline(x: x, coeffs: coeffs, a: b, b: a)
    }

    let i0 = findInterval(x, x0)
    let i1 = findInterval(x, x1)

    var total = 0.0

    if i0 == i1 {
        let idx = min(i0, coeffs.count - 1)
        let c = coeffs[idx]
        let dx0 = x0 - x[idx]
        let dx1 = x1 - x[idx]
        total = c.integrate(to: dx1) - c.integrate(to: dx0)
    } else {
        // First partial segment
        let idx0 = min(i0, coeffs.count - 1)
        let c0 = coeffs[idx0]
        let dx0 = x0 - x[idx0]
        let dx1 = x[idx0 + 1] - x[idx0]
        total = c0.integrate(to: dx1) - c0.integrate(to: dx0)

        // Full segments
        for i in (i0 + 1)..<i1 {
            let idx = min(i, coeffs.count - 1)
            let c = coeffs[idx]
            let h = x[i + 1] - x[i]
            total += c.integrate(to: h)
        }

        // Last partial segment
        if i1 < n - 1 && i1 < coeffs.count {
            let c1 = coeffs[i1]
            let dx = x1 - x[i1]
            total += c1.integrate(to: dx)
        }
    }

    return total
}

// MARK: - PCHIP Interpolation

/// Compute PCHIP edge derivative.
private func pchipEdgeDerivative(h1: Double, h2: Double, d1: Double, d2: Double) -> Double {
    let deriv = ((2.0 * h1 + h2) * d1 - h1 * d2) / (h1 + h2)
    if deriv * d1 < 0 {
        return 0
    } else if d1 * d2 < 0 && abs(deriv) > 3.0 * abs(d1) {
        return 3.0 * d1
    }
    return deriv
}

/// Compute PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) derivatives.
///
/// PCHIP preserves monotonicity and produces visually pleasing curves.
///
/// - Parameters:
///   - x: x-coordinates
///   - y: y-coordinates
/// - Returns: Derivative values at each point
public func computePchipDerivatives(x: [Double], y: [Double]) -> [Double] {
    let n = x.count
    guard n >= 2 else { return [] }

    // Compute slopes and intervals
    var h = [Double](repeating: 0, count: n - 1)
    var delta = [Double](repeating: 0, count: n - 1)
    for i in 0..<(n - 1) {
        h[i] = x[i + 1] - x[i]
        delta[i] = (y[i + 1] - y[i]) / h[i]
    }

    var d = [Double](repeating: 0, count: n)

    if n == 2 {
        d[0] = delta[0]
        d[1] = delta[0]
    } else {
        // Endpoints
        d[0] = pchipEdgeDerivative(h1: h[0], h2: h[1], d1: delta[0], d2: delta[1])
        d[n - 1] = pchipEdgeDerivative(h1: h[n - 2], h2: h[n - 3], d1: delta[n - 2], d2: delta[n - 3])

        // Interior points
        for i in 1..<(n - 1) {
            if delta[i - 1] * delta[i] > 0 {
                let w1 = 2.0 * h[i] + h[i - 1]
                let w2 = h[i] + 2.0 * h[i - 1]
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
            } else {
                d[i] = 0
            }
        }
    }

    return d
}

/// Evaluate PCHIP interpolation at a point.
public func evalPchip(x: [Double], y: [Double], d: [Double], xNew: Double) -> Double {
    let n = x.count

    // Extrapolation
    if xNew <= x[0] {
        return y[0] + d[0] * (xNew - x[0])
    }
    if xNew >= x[n - 1] {
        return y[n - 1] + d[n - 1] * (xNew - x[n - 1])
    }

    let i = findInterval(x, xNew)
    let h = x[i + 1] - x[i]
    let t = (xNew - x[i]) / h

    // Hermite basis functions
    let t2 = t * t
    let t3 = t2 * t
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    let h10 = t3 - 2.0 * t2 + t
    let h01 = -2.0 * t3 + 3.0 * t2
    let h11 = t3 - t2

    return h00 * y[i] + h10 * h * d[i] + h01 * y[i + 1] + h11 * h * d[i + 1]
}

// MARK: - Akima Interpolation

/// Compute Akima interpolation coefficients.
///
/// Akima interpolation uses a weighted average of slopes to produce
/// smooth curves without overshooting.
///
/// - Parameters:
///   - x: x-coordinates
///   - y: y-coordinates
/// - Returns: Cubic coefficients for each segment
public func computeAkimaCoeffs(x: [Double], y: [Double]) -> [CubicCoeffs] {
    let n = x.count
    guard n >= 2 else { return [] }

    // Compute slopes between points
    var m = [Double](repeating: 0, count: n + 3)
    for i in 0..<(n - 1) {
        m[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    }

    // Extend slopes at boundaries
    m[1] = 2.0 * m[2] - m[3]
    m[0] = 2.0 * m[1] - m[2]
    m[n + 1] = 2.0 * m[n] - m[n - 1]
    m[n + 2] = 2.0 * m[n + 1] - m[n]

    // Compute Akima derivatives
    var d = [Double](repeating: 0, count: n)
    for i in 0..<n {
        let w1 = abs(m[i + 2] - m[i + 1])
        let w2 = abs(m[i] - m[i + 1])

        if w1 + w2 == 0 {
            d[i] = (m[i + 1] + m[i + 2]) / 2.0
        } else {
            d[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2)
        }
    }

    // Compute coefficients
    var coeffs = [CubicCoeffs]()
    for i in 0..<(n - 1) {
        let h = x[i + 1] - x[i]
        let slope = m[i + 2]
        let a = y[i]
        let b = d[i]
        let c = (3.0 * slope - 2.0 * d[i] - d[i + 1]) / h
        let dd = (d[i] + d[i + 1] - 2.0 * slope) / (h * h)
        coeffs.append(CubicCoeffs(a: a, b: b, c: c, d: dd))
    }

    return coeffs
}

/// Evaluate Akima interpolation at a point.
public func evalAkima(x: [Double], coeffs: [CubicCoeffs], xNew: Double) -> Double {
    let n = x.count

    // Extrapolation
    if xNew <= x[0] {
        let dx = xNew - x[0]
        return coeffs[0].a + coeffs[0].b * dx
    }
    if xNew >= x[n - 1] {
        let dx = xNew - x[n - 2]
        return coeffs[n - 2].evaluate(at: dx)
    }

    let i = findInterval(x, xNew)
    let idx = min(i, coeffs.count - 1)
    let dx = xNew - x[idx]
    return coeffs[idx].evaluate(at: dx)
}

// MARK: - Lagrange Interpolation

/// Evaluate Lagrange interpolation at a point.
///
/// Warning: Lagrange interpolation can be numerically unstable for many points.
/// Consider barycentric interpolation instead.
///
/// - Parameters:
///   - x: x-coordinates
///   - y: y-coordinates
///   - xNew: Point to evaluate at
/// - Returns: Interpolated value
public func evalLagrange(x: [Double], y: [Double], xNew: Double) -> Double {
    let n = x.count
    var result = 0.0

    for i in 0..<n {
        var basis = 1.0
        for j in 0..<n {
            if i != j {
                basis *= (xNew - x[j]) / (x[i] - x[j])
            }
        }
        result += y[i] * basis
    }

    return result
}

// MARK: - Barycentric Interpolation

/// Compute barycentric weights for polynomial interpolation.
///
/// Barycentric interpolation is numerically stable and efficient.
///
/// - Parameter x: x-coordinates
/// - Returns: Barycentric weights
public func computeBarycentricWeights(x: [Double]) -> [Double] {
    let n = x.count
    var w = [Double](repeating: 1, count: n)

    for i in 0..<n {
        for j in 0..<n {
            if i != j {
                w[i] /= (x[i] - x[j])
            }
        }
    }

    return w
}

/// Evaluate barycentric interpolation at a point.
///
/// - Parameters:
///   - x: x-coordinates
///   - y: y-coordinates
///   - w: Barycentric weights
///   - xNew: Point to evaluate at
/// - Returns: Interpolated value
public func evalBarycentric(x: [Double], y: [Double], w: [Double], xNew: Double) -> Double {
    let n = x.count

    // Check for exact match
    for i in 0..<n {
        if xNew == x[i] {
            return y[i]
        }
    }

    var num = 0.0
    var den = 0.0
    for i in 0..<n {
        let term = w[i] / (xNew - x[i])
        num += term * y[i]
        den += term
    }

    return num / den
}

// MARK: - 1D Interpolation

/// Perform 1D interpolation at a single point.
///
/// - Parameters:
///   - x: x-coordinates of data
///   - y: y-coordinates of data
///   - xNew: Point to interpolate at
///   - kind: Interpolation method
///   - fillValue: Value to return for out-of-bounds (if boundsError is false)
///   - boundsError: Whether to return NaN for out-of-bounds
///   - coeffs: Precomputed spline coefficients (for cubic)
/// - Returns: Interpolated value
public func interp1d(
    x: [Double],
    y: [Double],
    xNew: Double,
    kind: InterpolationKind = .linear,
    fillValue: Double = .nan,
    boundsError: Bool = false,
    coeffs: [CubicCoeffs]? = nil
) -> Double {
    let n = x.count

    // Check bounds
    if xNew < x[0] || xNew > x[n - 1] {
        if boundsError {
            return Double.nan
        }
        return fillValue
    }

    // Handle exact boundary values
    if xNew == x[0] { return y[0] }
    if xNew == x[n - 1] { return y[n - 1] }

    let i = findInterval(x, xNew)

    switch kind {
    case .nearest:
        let d1 = abs(xNew - x[i])
        let d2 = abs(xNew - x[i + 1])
        return d1 <= d2 ? y[i] : y[i + 1]

    case .previous:
        return y[i]

    case .next:
        return y[i + 1]

    case .cubic:
        guard let coeffs = coeffs, i < coeffs.count else {
            return Double.nan
        }
        let dx = xNew - x[i]
        return coeffs[i].evaluate(at: dx)

    case .linear:
        let t = (xNew - x[i]) / (x[i + 1] - x[i])
        return y[i] + t * (y[i + 1] - y[i])
    }
}
