//
//  Series.swift
//  NumericSwift
//
//  Series evaluation, Taylor polynomials, and polynomial utilities.
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Polynomial Evaluation

/// Evaluate polynomial using Horner's method.
///
/// Evaluates c₀ + c₁x + c₂x² + ... + cₙxⁿ efficiently.
///
/// - Parameters:
///   - coefficients: Array of coefficients [c₀, c₁, c₂, ...]
///   - x: Point to evaluate at
/// - Returns: Polynomial value at x
public func polyval(_ coefficients: [Double], at x: Double) -> Double {
    guard !coefficients.isEmpty else { return 0 }

    // Horner's method: evaluate from highest to lowest degree
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
public func polyval(_ coefficients: [Double], at x: Double, center: Double) -> Double {
    guard !coefficients.isEmpty else { return 0 }

    let dx = x - center
    var result = coefficients[coefficients.count - 1]
    for i in stride(from: coefficients.count - 2, through: 0, by: -1) {
        result = result * dx + coefficients[i]
    }
    return result
}

/// Add two polynomials.
///
/// - Parameters:
///   - p: First polynomial coefficients
///   - q: Second polynomial coefficients
/// - Returns: Sum polynomial coefficients
public func polyadd(_ p: [Double], _ q: [Double]) -> [Double] {
    let maxLen = max(p.count, q.count)
    var result = [Double](repeating: 0, count: maxLen)

    for i in 0..<p.count {
        result[i] += p[i]
    }
    for i in 0..<q.count {
        result[i] += q[i]
    }

    // Remove trailing zeros
    while result.count > 1 && result.last == 0 {
        result.removeLast()
    }

    return result
}

/// Multiply two polynomials.
///
/// - Parameters:
///   - p: First polynomial coefficients
///   - q: Second polynomial coefficients
/// - Returns: Product polynomial coefficients
public func polymul(_ p: [Double], _ q: [Double]) -> [Double] {
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
public func polyder(_ coefficients: [Double]) -> [Double] {
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
public func polyint(_ coefficients: [Double]) -> [Double] {
    guard !coefficients.isEmpty else { return [0] }

    var result = [Double](repeating: 0, count: coefficients.count + 1)

    for i in 0..<coefficients.count {
        result[i + 1] = coefficients[i] / Double(i + 1)
    }

    return result
}

// MARK: - Taylor Series Coefficients

/// Taylor series coefficient generator type.
public typealias TaylorCoefficient = (Int) -> Double

/// Get Taylor series coefficients for common functions at x=0.
///
/// Supported functions:
/// - sin, cos, exp, log1p, sinh, cosh, tan, atan
/// - geometric (1/(1-x)), geometricAlt (1/(1+x))
/// - sqrt1p (sqrt(1+x)), inv1p (1/(1+x))
///
/// - Parameters:
///   - function: Name of the function
///   - terms: Number of terms to generate
/// - Returns: Array of Taylor coefficients, or nil if function unknown
public func taylorCoefficients(for function: String, terms: Int) -> [Double]? {
    guard let generator = knownTaylorSeries[function] else { return nil }
    return (0..<terms).map { generator($0) }
}

/// Dictionary of known Taylor series coefficient generators.
public let knownTaylorSeries: [String: TaylorCoefficient] = [
    // sin(x) = x - x³/3! + x⁵/5! - ...
    "sin": { n in
        if n % 2 == 0 { return 0 }
        let sign = ((n - 1) / 2) % 2 == 0 ? 1.0 : -1.0
        return sign / factorial(n)
    },

    // cos(x) = 1 - x²/2! + x⁴/4! - ...
    "cos": { n in
        if n % 2 != 0 { return 0 }
        let sign = (n / 2) % 2 == 0 ? 1.0 : -1.0
        return sign / factorial(n)
    },

    // exp(x) = 1 + x + x²/2! + x³/3! + ...
    "exp": { n in
        return 1.0 / factorial(n)
    },

    // log(1+x) = x - x²/2 + x³/3 - ...
    "log1p": { n in
        if n == 0 { return 0 }
        let sign = n % 2 == 1 ? 1.0 : -1.0
        return sign / Double(n)
    },

    // sinh(x) = x + x³/3! + x⁵/5! + ...
    "sinh": { n in
        if n % 2 == 0 { return 0 }
        return 1.0 / factorial(n)
    },

    // cosh(x) = 1 + x²/2! + x⁴/4! + ...
    "cosh": { n in
        if n % 2 != 0 { return 0 }
        return 1.0 / factorial(n)
    },

    // tan(x) - first few terms using Bernoulli numbers
    "tan": { n in
        let coeffs: [Int: Double] = [
            0: 0, 1: 1, 2: 0, 3: 1.0/3.0, 4: 0,
            5: 2.0/15.0, 6: 0, 7: 17.0/315.0, 8: 0,
            9: 62.0/2835.0, 10: 0, 11: 1382.0/155925.0
        ]
        return coeffs[n] ?? 0
    },

    // atan(x) = x - x³/3 + x⁵/5 - ...
    "atan": { n in
        if n % 2 == 0 { return 0 }
        let sign = ((n - 1) / 2) % 2 == 0 ? 1.0 : -1.0
        return sign / Double(n)
    },

    // 1/(1-x) = 1 + x + x² + x³ + ... (geometric series)
    "geometric": { _ in 1.0 },

    // 1/(1+x) = 1 - x + x² - x³ + ... (alternating geometric)
    "geometricAlt": { n in n % 2 == 0 ? 1.0 : -1.0 },

    // sqrt(1+x) = 1 + x/2 - x²/8 + ... (binomial series (1+x)^(1/2))
    "sqrt1p": { n in
        if n == 0 { return 1.0 }
        var coeff = 0.5
        for k in 1..<n {
            coeff *= (0.5 - Double(k)) / Double(k + 1)
        }
        return coeff
    },

    // 1/(1+x) = 1 - x + x² - ... (same as geometricAlt)
    "inv1p": { n in n % 2 == 0 ? 1.0 : -1.0 }
]

/// Evaluate a known Taylor series at a point.
///
/// - Parameters:
///   - function: Name of the function (sin, cos, exp, etc.)
///   - x: Point to evaluate at
///   - terms: Number of terms to use (default 20)
/// - Returns: Approximation using Taylor series
public func taylorEval(_ function: String, at x: Double, terms: Int = 20) -> Double? {
    guard let coeffs = taylorCoefficients(for: function, terms: terms) else { return nil }
    return polyval(coeffs, at: x)
}

// MARK: - Numerical Differentiation

/// Chebyshev-like point distribution for numerical approximation.
///
/// Uses cos(linspace(0, π, n)) for numerical stability.
///
/// - Parameters:
///   - center: Center point
///   - scale: Scaling factor for point spread
///   - count: Number of points
/// - Returns: Array of evaluation points
public func chebyshevPoints(center: Double, scale: Double, count: Int) -> [Double] {
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
public func dividedDifferences(xs: [Double], ys: [Double]) -> [Double] {
    let n = min(xs.count, ys.count)
    guard n > 0 else { return [] }

    var table = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)

    // Initialize with y values
    for i in 0..<n {
        table[i][0] = ys[i]
    }

    // Compute divided differences
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
public func seriesSum(
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
        // Finite sum
        while n <= toIndex {
            sum += term(n)
            n += 1
            iterations += 1
        }
        return (sum, true, iterations)
    } else {
        // Convergence mode
        while iterations < maxIterations {
            let t = term(n)
            sum += t
            iterations += 1

            // Check convergence
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
public func seriesProduct(
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
        // Finite product
        while n <= toIndex {
            product *= term(n)
            n += 1
            iterations += 1
        }
        return (product, true, iterations)
    } else {
        // Convergence mode
        while iterations < maxIterations {
            product *= term(n)
            iterations += 1

            // Check convergence (product stabilizes)
            if abs(product - prevProduct) < tolerance * abs(product) {
                return (product, true, iterations)
            }

            prevProduct = product
            n += 1
        }
        return (product, false, iterations)
    }
}

// MARK: - Partial Sums

/// Generate sequence of partial sums.
///
/// - Parameters:
///   - from: Starting index
///   - count: Number of partial sums to generate
///   - term: Function that generates term for index n
/// - Returns: Array of partial sums
public func partialSums(from: Int, count: Int, term: (Int) -> Double) -> [Double] {
    var sums = [Double]()
    sums.reserveCapacity(count)

    var sum = 0.0
    for i in 0..<count {
        sum += term(from + i)
        sums.append(sum)
    }

    return sums
}

// MARK: - Well-Known Series Values

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
