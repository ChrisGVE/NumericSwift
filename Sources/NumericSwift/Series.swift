//
//  Series.swift
//  NumericSwift
//
//  Series evaluation, Taylor polynomials, and polynomial utilities.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Series Namespace

/// Polynomial arithmetic, Taylor series, and series summation utilities.
///
/// ## Overview
///
/// ```swift
/// let y = Series.polyval([1, 2, 3], at: 2.0)   // 1 + 2x + 3x² at x=2
/// let s = Series.seriesSum(from: 1) { 1.0 / Double($0 * $0) }
/// ```
public enum Series {

    // MARK: - Polynomial Evaluation

    /// Evaluate polynomial using Horner's method.
    ///
    /// Evaluates c₀ + c₁x + c₂x² + ... + cₙxⁿ efficiently.
    ///
    /// - Parameters:
    ///   - coefficients: Array of coefficients [c₀, c₁, c₂, ...]
    ///   - x: Point to evaluate at
    /// - Returns: Polynomial value at x
    public static func polyval(_ coefficients: [Double], at x: Double) -> Double {
        guard !coefficients.isEmpty else { return 0 }
        var result = coefficients[coefficients.count - 1]
        for i in stride(from: coefficients.count - 2, through: 0, by: -1) {
            result = result * x + coefficients[i]
        }
        return result
    }

    /// Evaluate polynomial centered at a point using Horner's method.
    ///
    /// Evaluates c₀ + c₁(x-a) + c₂(x-a)² + ... where a is the center.
    ///
    /// - Parameters:
    ///   - coefficients: Array of coefficients [c₀, c₁, c₂, ...]
    ///   - x: Point to evaluate at
    ///   - center: Center point for the polynomial
    /// - Returns: Polynomial value at x
    public static func polyval(_ coefficients: [Double], at x: Double, center: Double) -> Double {
        guard !coefficients.isEmpty else { return 0 }
        let dx = x - center
        var result = coefficients[coefficients.count - 1]
        for i in stride(from: coefficients.count - 2, through: 0, by: -1) {
            result = result * dx + coefficients[i]
        }
        return result
    }

    // MARK: - Polynomial Arithmetic

    /// Add two polynomials.
    ///
    /// - Parameters:
    ///   - p: First polynomial coefficients
    ///   - q: Second polynomial coefficients
    /// - Returns: Sum polynomial coefficients
    public static func polyadd(_ p: [Double], _ q: [Double]) -> [Double] {
        let maxLen = max(p.count, q.count)
        var result = [Double](repeating: 0, count: maxLen)
        for i in 0..<p.count { result[i] += p[i] }
        for i in 0..<q.count { result[i] += q[i] }
        while result.count > 1 && result.last == 0 { result.removeLast() }
        return result
    }

    /// Multiply two polynomials.
    ///
    /// - Parameters:
    ///   - p: First polynomial coefficients
    ///   - q: Second polynomial coefficients
    /// - Returns: Product polynomial coefficients
    public static func polymul(_ p: [Double], _ q: [Double]) -> [Double] {
        guard !p.isEmpty && !q.isEmpty else { return [0] }
        var result = [Double](repeating: 0, count: p.count + q.count - 1)
        for i in 0..<p.count {
            for j in 0..<q.count {
                result[i + j] += p[i] * q[j]
            }
        }
        return result
    }

    /// Differentiate a polynomial.
    ///
    /// - Parameter coefficients: Polynomial coefficients [c₀, c₁, c₂, ...]
    /// - Returns: Derivative polynomial coefficients
    public static func polyder(_ coefficients: [Double]) -> [Double] {
        guard coefficients.count > 1 else { return [0] }
        var result = [Double]()
        result.reserveCapacity(coefficients.count - 1)
        for i in 1..<coefficients.count {
            result.append(coefficients[i] * Double(i))
        }
        return result
    }

    /// Integrate a polynomial (with constant 0).
    ///
    /// - Parameter coefficients: Polynomial coefficients
    /// - Returns: Integral polynomial coefficients
    public static func polyint(_ coefficients: [Double]) -> [Double] {
        guard !coefficients.isEmpty else { return [0] }
        var result = [Double](repeating: 0, count: coefficients.count + 1)
        for i in 0..<coefficients.count {
            result[i + 1] = coefficients[i] / Double(i + 1)
        }
        return result
    }

    // MARK: - Taylor Series

    /// Taylor series coefficient generator type.
    public typealias TaylorCoefficient = (Int) -> Double

    /// Get Taylor series coefficients for common functions at x=0.
    ///
    /// Supported functions: sin, cos, exp, log1p, sinh, cosh, tan, atan,
    /// geometric, geometricAlt, sqrt1p, inv1p.
    ///
    /// - Parameters:
    ///   - function: Name of the function
    ///   - terms: Number of terms to generate
    /// - Returns: Array of Taylor coefficients, or nil if function unknown
    public static func taylorCoefficients(for function: String, terms: Int) -> [Double]? {
        guard let generator = knownTaylorSeries[function] else { return nil }
        return (0..<terms).map { generator($0) }
    }

    /// Dictionary of known Taylor series coefficient generators.
    public static let knownTaylorSeries: [String: TaylorCoefficient] = [
        "sin": { n in
            if n % 2 == 0 { return 0 }
            let sign = ((n - 1) / 2) % 2 == 0 ? 1.0 : -1.0
            return sign / NumberTheory.factorial(n)
        },
        "cos": { n in
            if n % 2 != 0 { return 0 }
            let sign = (n / 2) % 2 == 0 ? 1.0 : -1.0
            return sign / NumberTheory.factorial(n)
        },
        "exp": { n in 1.0 / NumberTheory.factorial(n) },
        "log1p": { n in
            if n == 0 { return 0 }
            let sign = n % 2 == 1 ? 1.0 : -1.0
            return sign / Double(n)
        },
        "sinh": { n in
            if n % 2 == 0 { return 0 }
            return 1.0 / NumberTheory.factorial(n)
        },
        "cosh": { n in
            if n % 2 != 0 { return 0 }
            return 1.0 / NumberTheory.factorial(n)
        },
        "tan": { n in
            let coeffs: [Int: Double] = [
                0: 0, 1: 1, 2: 0, 3: 1.0/3.0, 4: 0,
                5: 2.0/15.0, 6: 0, 7: 17.0/315.0, 8: 0,
                9: 62.0/2835.0, 10: 0, 11: 1382.0/155925.0
            ]
            return coeffs[n] ?? 0
        },
        "atan": { n in
            if n % 2 == 0 { return 0 }
            let sign = ((n - 1) / 2) % 2 == 0 ? 1.0 : -1.0
            return sign / Double(n)
        },
        "geometric": { _ in 1.0 },
        "geometricAlt": { n in n % 2 == 0 ? 1.0 : -1.0 },
        "sqrt1p": { n in
            if n == 0 { return 1.0 }
            var coeff = 0.5
            for k in 1..<n { coeff *= (0.5 - Double(k)) / Double(k + 1) }
            return coeff
        },
        "inv1p": { n in n % 2 == 0 ? 1.0 : -1.0 }
    ]

    /// Evaluate a known Taylor series at a point.
    ///
    /// - Parameters:
    ///   - function: Name of the function (sin, cos, exp, etc.)
    ///   - x: Point to evaluate at
    ///   - terms: Number of terms to use (default 20)
    /// - Returns: Approximation using Taylor series
    public static func taylorEval(_ function: String, at x: Double, terms: Int = 20) -> Double? {
        guard let coeffs = taylorCoefficients(for: function, terms: terms) else { return nil }
        return polyval(coeffs, at: x)
    }

    // MARK: - Taylor Series (diagnosed)

    /// Maximum number of Taylor coefficients each generator can supply *exactly*.
    ///
    /// Most generators in ``knownTaylorSeries`` are closed-form and supply an
    /// exact coefficient for every index, so they have no finite limit. The
    /// `"tan"` generator is the documented exception: its coefficients are a
    /// hard-coded table covering only indices `0...11` (12 terms). Requesting
    /// more terms silently returns `0` for every index beyond `11` — including
    /// the genuinely non-zero `x¹³` coefficient (`21844/6081075`) and beyond — so
    /// the resulting evaluation is materially wrong near the radius of
    /// convergence `x = ±π/2`. See CLAUDE.md "Code Review Findings → Series.swift".
    ///
    /// `12` is an *implementation* limit (the size of the hard-coded coefficient
    /// table), not a mathematical boundary — `tan`'s Taylor series exists to all
    /// orders. Extending the table raises the limit; the diagnostic exists so a
    /// caller is never silently handed the zero-padded surplus in the meantime.
    ///
    /// A generator absent from this table is unbounded (closed-form).
    public static let taylorSupportedTermLimit: [String: Int] = [
        "tan": 12
    ]

    /// ``taylorCoefficients(for:terms:)`` with a recoverable limitation diagnostic.
    ///
    /// Identical to ``taylorCoefficients(for:terms:)`` except the result is wrapped
    /// in a ``Diagnosed`` so the caller can detect an out-of-envelope request. For
    /// generators with a finite support limit (see ``taylorSupportedTermLimit`` —
    /// currently only `"tan"`), requesting more than the supported number of terms
    /// attaches an ``NumericDiagnostic/outsideEnvelope(method:reason:)`` diagnostic,
    /// because the surplus coefficients are silently returned as `0` rather than
    /// their true (possibly non-zero) value. The value itself is unchanged — it is
    /// the same best-effort array the bare overload returns.
    ///
    /// - Parameters:
    ///   - function: Name of the function.
    ///   - terms: Number of terms to generate.
    /// - Returns: A ``Diagnosed`` array of coefficients, or `nil` if the function
    ///   is unknown.
    public static func taylorCoefficientsDiagnosed(for function: String, terms: Int) -> Diagnosed<[Double]>? {
        guard let coeffs = taylorCoefficients(for: function, terms: terms) else { return nil }
        var diagnostics: [NumericDiagnostic] = []
        if let limit = taylorSupportedTermLimit[function], terms > limit {
            diagnostics.append(.outsideEnvelope(
                method: "Series.taylor[\(function)]",
                reason: "requested \(terms) terms but the \(function) generator only supplies "
                    + "\(limit) exact coefficients (indices 0...\(limit - 1)); surplus terms are "
                    + "silently zero, so the series is unreliable beyond its support"))
        }
        return Diagnosed(coeffs, diagnostics: diagnostics)
    }

    /// ``taylorEval(_:at:terms:)`` with a recoverable limitation diagnostic.
    ///
    /// Identical to ``taylorEval(_:at:terms:)`` except the result is wrapped in a
    /// ``Diagnosed``. When `function` has a finite support limit (see
    /// ``taylorSupportedTermLimit`` — currently only `"tan"`) and `terms` exceeds
    /// it, the result carries an ``NumericDiagnostic/outsideEnvelope(method:reason:)``
    /// diagnostic: the truncated series silently drops genuinely non-zero
    /// higher-order coefficients and is therefore unreliable, especially near the
    /// radius of convergence. The numeric value matches the bare overload exactly.
    ///
    /// - Parameters:
    ///   - function: Name of the function (sin, cos, exp, etc.).
    ///   - x: Point to evaluate at.
    ///   - terms: Number of terms to use (default 20).
    /// - Returns: A ``Diagnosed`` value, or `nil` if the function is unknown.
    public static func taylorEvalDiagnosed(_ function: String, at x: Double, terms: Int = 20) -> Diagnosed<Double>? {
        guard let diagnosedCoeffs = taylorCoefficientsDiagnosed(for: function, terms: terms) else { return nil }
        return diagnosedCoeffs.map { polyval($0, at: x) }
    }

    // MARK: - Numerical Differentiation Helpers

    /// Chebyshev-like point distribution for numerical approximation.
    ///
    /// Uses cos(linspace(0, π, n)) for numerical stability.
    ///
    /// - Parameters:
    ///   - center: Center point
    ///   - scale: Scaling factor for point spread
    ///   - count: Number of points
    /// - Returns: Array of evaluation points
    public static func chebyshevPoints(center: Double, scale: Double, count: Int) -> [Double] {
        guard count > 1 else { return [center] }
        var points = [Double]()
        points.reserveCapacity(count)
        for i in 0..<count {
            let t = Double(i) / Double(count - 1) * Double.pi
            points.append(scale * Darwin.cos(t) + center)
        }
        return points
    }

    /// Divided differences for polynomial interpolation.
    ///
    /// - Parameters:
    ///   - xs: x-coordinates
    ///   - ys: y-coordinates (function values)
    /// - Returns: Divided difference coefficients
    public static func dividedDifferences(xs: [Double], ys: [Double]) -> [Double] {
        let n = min(xs.count, ys.count)
        guard n > 0 else { return [] }

        var table = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n { table[i][0] = ys[i] }

        for j in 1..<n {
            for i in 0..<(n - j) {
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (xs[i + j] - xs[i])
            }
        }

        return (0..<n).map { table[0][$0] }
    }

    // MARK: - Series Summation

    /// Sum a series using a term generator function.
    ///
    /// - Parameters:
    ///   - from: Starting index
    ///   - to: Ending index (inclusive), or nil for convergence mode
    ///   - tolerance: Convergence tolerance (used when to is nil)
    ///   - maxIterations: Maximum iterations for convergence mode
    ///   - term: Function that generates term for index n
    /// - Returns: Tuple of (sum, converged, iterations)
    public static func seriesSum(
        from: Int,
        to: Int? = nil,
        tolerance: Double = 1e-12,
        maxIterations: Int = 10000,
        term: (Int) -> Double
    ) -> (sum: Double, converged: Bool, iterations: Int) {
        var sum = 0.0
        var prevSum = Double.infinity
        var n = from
        var iterations = 0

        if let toIndex = to {
            while n <= toIndex {
                sum += term(n)
                n += 1
                iterations += 1
            }
            return (sum, true, iterations)
        } else {
            while iterations < maxIterations {
                let t = term(n)
                sum += t
                iterations += 1
                if abs(t) < tolerance && abs(sum - prevSum) < tolerance {
                    return (sum, true, iterations)
                }
                prevSum = sum
                n += 1
            }
            return (sum, false, iterations)
        }
    }

    /// Compute product of a series using a term generator function.
    ///
    /// - Parameters:
    ///   - from: Starting index
    ///   - to: Ending index (inclusive), or nil for convergence mode
    ///   - tolerance: Convergence tolerance (used when to is nil)
    ///   - maxIterations: Maximum iterations for convergence mode
    ///   - term: Function that generates factor for index n
    /// - Returns: Tuple of (product, converged, iterations)
    public static func seriesProduct(
        from: Int,
        to: Int? = nil,
        tolerance: Double = 1e-12,
        maxIterations: Int = 10000,
        term: (Int) -> Double
    ) -> (product: Double, converged: Bool, iterations: Int) {
        var product = 1.0
        var prevProduct = 0.0
        var n = from
        var iterations = 0

        if let toIndex = to {
            while n <= toIndex {
                product *= term(n)
                n += 1
                iterations += 1
            }
            return (product, true, iterations)
        } else {
            while iterations < maxIterations {
                product *= term(n)
                iterations += 1
                if abs(product - prevProduct) < tolerance * abs(product) {
                    return (product, true, iterations)
                }
                prevProduct = product
                n += 1
            }
            return (product, false, iterations)
        }
    }

    /// Generate sequence of partial sums.
    ///
    /// - Parameters:
    ///   - from: Starting index
    ///   - count: Number of partial sums to generate
    ///   - term: Function that generates term for index n
    /// - Returns: Array of partial sums
    public static func partialSums(from: Int, count: Int, term: (Int) -> Double) -> [Double] {
        var sums = [Double]()
        sums.reserveCapacity(count)
        var sum = 0.0
        for i in 0..<count {
            sum += term(from + i)
            sums.append(sum)
        }
        return sums
    }
}

// MARK: - Well-Known Series Constants (top-level — genuinely module-level values)

/// Basel problem: sum of 1/n² from n=1 to infinity = π²/6
public let baselSum: Double = Double.pi * Double.pi / 6.0

/// Sum of 1/n⁴ from n=1 to infinity = π⁴/90
public let zeta4: Double = Darwin.pow(Double.pi, 4) / 90.0

/// Euler-Mascheroni constant γ ≈ 0.5772
public let eulerMascheroni: Double = 0.5772156649015328606065120900824024310421593359

/// Catalan's constant G ≈ 0.9159
public let catalanConstant: Double = 0.9159655941772190150546035149323841107741493742816721342664981196217630197762547694794

/// Apéry's constant ζ(3) ≈ 1.2020
public let aperyConstant: Double = 1.2020569031595942853997381615114499907649862923404988817922715553418382057863130901864

// MARK: - Deprecated shims (backward compatibility)

/// - Note: Deprecated. Use ``Series/polyval(_:at:)`` instead.
@available(*, deprecated, message: "Use Series.polyval(_:at:) instead")
public func polyval(_ coefficients: [Double], at x: Double) -> Double {
    Series.polyval(coefficients, at: x)
}

/// - Note: Deprecated. Use ``Series/polyval(_:at:center:)`` instead.
@available(*, deprecated, message: "Use Series.polyval(_:at:center:) instead")
public func polyval(_ coefficients: [Double], at x: Double, center: Double) -> Double {
    Series.polyval(coefficients, at: x, center: center)
}

/// - Note: Deprecated. Use ``Series/polyadd(_:_:)`` instead.
@available(*, deprecated, message: "Use Series.polyadd(_:_:) instead")
public func polyadd(_ p: [Double], _ q: [Double]) -> [Double] {
    Series.polyadd(p, q)
}

/// - Note: Deprecated. Use ``Series/polymul(_:_:)`` instead.
@available(*, deprecated, message: "Use Series.polymul(_:_:) instead")
public func polymul(_ p: [Double], _ q: [Double]) -> [Double] {
    Series.polymul(p, q)
}

/// - Note: Deprecated. Use ``Series/polyder(_:)`` instead.
@available(*, deprecated, message: "Use Series.polyder(_:) instead")
public func polyder(_ coefficients: [Double]) -> [Double] {
    Series.polyder(coefficients)
}

/// - Note: Deprecated. Use ``Series/polyint(_:)`` instead.
@available(*, deprecated, message: "Use Series.polyint(_:) instead")
public func polyint(_ coefficients: [Double]) -> [Double] {
    Series.polyint(coefficients)
}

/// - Note: Deprecated. Use ``Series/taylorCoefficients(for:terms:)`` instead.
@available(*, deprecated, message: "Use Series.taylorCoefficients(for:terms:) instead")
public func taylorCoefficients(for function: String, terms: Int) -> [Double]? {
    Series.taylorCoefficients(for: function, terms: terms)
}

/// - Note: Deprecated. Use ``Series/knownTaylorSeries`` instead.
@available(*, deprecated, message: "Use Series.knownTaylorSeries instead")
public let knownTaylorSeries: [String: (Int) -> Double] = Series.knownTaylorSeries

/// - Note: Deprecated. Use ``Series/taylorEval(_:at:terms:)`` instead.
@available(*, deprecated, message: "Use Series.taylorEval(_:at:terms:) instead")
public func taylorEval(_ function: String, at x: Double, terms: Int = 20) -> Double? {
    Series.taylorEval(function, at: x, terms: terms)
}

/// - Note: Deprecated. Use ``Series/chebyshevPoints(center:scale:count:)`` instead.
@available(*, deprecated, message: "Use Series.chebyshevPoints(center:scale:count:) instead")
public func chebyshevPoints(center: Double, scale: Double, count: Int) -> [Double] {
    Series.chebyshevPoints(center: center, scale: scale, count: count)
}

/// - Note: Deprecated. Use ``Series/dividedDifferences(xs:ys:)`` instead.
@available(*, deprecated, message: "Use Series.dividedDifferences(xs:ys:) instead")
public func dividedDifferences(xs: [Double], ys: [Double]) -> [Double] {
    Series.dividedDifferences(xs: xs, ys: ys)
}

/// - Note: Deprecated. Use ``Series/seriesSum(from:to:tolerance:maxIterations:term:)`` instead.
@available(*, deprecated, message: "Use Series.seriesSum(from:to:tolerance:maxIterations:term:) instead")
public func seriesSum(
    from: Int,
    to: Int? = nil,
    tolerance: Double = 1e-12,
    maxIterations: Int = 10000,
    term: (Int) -> Double
) -> (sum: Double, converged: Bool, iterations: Int) {
    Series.seriesSum(from: from, to: to, tolerance: tolerance, maxIterations: maxIterations, term: term)
}

/// - Note: Deprecated. Use ``Series/seriesProduct(from:to:tolerance:maxIterations:term:)`` instead.
@available(*, deprecated, message: "Use Series.seriesProduct(from:to:tolerance:maxIterations:term:) instead")
public func seriesProduct(
    from: Int,
    to: Int? = nil,
    tolerance: Double = 1e-12,
    maxIterations: Int = 10000,
    term: (Int) -> Double
) -> (product: Double, converged: Bool, iterations: Int) {
    Series.seriesProduct(from: from, to: to, tolerance: tolerance, maxIterations: maxIterations, term: term)
}

/// - Note: Deprecated. Use ``Series/partialSums(from:count:term:)`` instead.
@available(*, deprecated, message: "Use Series.partialSums(from:count:term:) instead")
public func partialSums(from: Int, count: Int, term: (Int) -> Double) -> [Double] {
    Series.partialSums(from: from, count: count, term: term)
}
