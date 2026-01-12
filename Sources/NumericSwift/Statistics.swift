//
//  Statistics.swift
//  NumericSwift
//
//  Basic statistical functions following numpy/scipy patterns.
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Basic Statistics

/// Sum of array elements
public func sum(_ values: [Double]) -> Double {
    values.reduce(0.0, +)
}

/// Arithmetic mean (average)
public func mean(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return .nan }
    return sum(values) / Double(values.count)
}

/// Median (middle value)
public func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return .nan }

    let sorted = values.sorted()
    let count = sorted.count

    if count % 2 == 0 {
        let mid = count / 2
        return (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        return sorted[count / 2]
    }
}

/// Variance with optional delta degrees of freedom
/// - Parameters:
///   - values: Array of values
///   - ddof: Delta degrees of freedom (0 for population variance, 1 for sample variance)
/// - Returns: Variance
public func variance(_ values: [Double], ddof: Int = 0) -> Double {
    guard !values.isEmpty else { return .nan }

    let divisor = values.count - ddof
    guard divisor > 0 else { return .nan }

    let m = mean(values)
    let squaredDiffs = values.map { ($0 - m) * ($0 - m) }
    return squaredDiffs.reduce(0.0, +) / Double(divisor)
}

/// Standard deviation with optional delta degrees of freedom
/// - Parameters:
///   - values: Array of values
///   - ddof: Delta degrees of freedom (0 for population stddev, 1 for sample stddev)
/// - Returns: Standard deviation
public func stddev(_ values: [Double], ddof: Int = 0) -> Double {
    Darwin.sqrt(variance(values, ddof: ddof))
}

/// Percentile using linear interpolation
/// - Parameters:
///   - values: Array of values
///   - p: Percentile (0-100)
/// - Returns: The p-th percentile value
public func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return .nan }
    guard p >= 0 && p <= 100 else { return .nan }

    let sorted = values.sorted()

    let rank = p / 100.0 * Double(sorted.count - 1)
    let lowerIndex = Int(Darwin.floor(rank))
    let upperIndex = Int(Darwin.ceil(rank))

    if lowerIndex == upperIndex || upperIndex >= sorted.count {
        return sorted[min(lowerIndex, sorted.count - 1)]
    }

    let weight = rank - Double(lowerIndex)
    return sorted[lowerIndex] * (1.0 - weight) + sorted[upperIndex] * weight
}

/// Geometric mean: nth root of product of n values
/// gmean([a, b, c]) = (a * b * c)^(1/n)
/// Requires all positive values
public func gmean(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return .nan }

    // Check for non-positive values
    for v in values {
        if v <= 0 { return .nan }
    }

    // Use log-sum-exp for numerical stability
    let logSum = values.reduce(0.0) { $0 + Darwin.log($1) }
    let logMean = logSum / Double(values.count)
    return Darwin.exp(logMean)
}

/// Harmonic mean: n / sum(1/x_i)
/// hmean([a, b, c]) = 3 / (1/a + 1/b + 1/c)
/// Requires all positive values
public func hmean(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return .nan }

    // Check for zero or negative values
    for v in values {
        if v <= 0 { return .nan }
    }

    let reciprocalSum = values.reduce(0.0) { $0 + 1.0 / $1 }
    return Double(values.count) / reciprocalSum
}

/// Mode: most frequently occurring value
/// Returns the smallest mode if there are ties (scipy behavior)
public func mode(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return .nan }

    var counts: [Double: Int] = [:]
    for v in values {
        counts[v, default: 0] += 1
    }

    let maxCount = counts.values.max() ?? 0
    let modes = counts.filter { $0.value == maxCount }.keys.sorted()

    return modes.first ?? values[0]
}

// MARK: - Minimum/Maximum

/// Minimum value in array
public func amin(_ values: [Double]) -> Double {
    values.min() ?? .nan
}

/// Maximum value in array
public func amax(_ values: [Double]) -> Double {
    values.max() ?? .nan
}

/// Range (max - min)
public func ptp(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return .nan }
    return (values.max() ?? 0) - (values.min() ?? 0)
}

// MARK: - Cumulative Functions

/// Cumulative sum
public func cumsum(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }

    var result = [Double]()
    result.reserveCapacity(values.count)
    var running = 0.0

    for v in values {
        running += v
        result.append(running)
    }

    return result
}

/// Cumulative product
public func cumprod(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }

    var result = [Double]()
    result.reserveCapacity(values.count)
    var running = 1.0

    for v in values {
        running *= v
        result.append(running)
    }

    return result
}

// MARK: - Differences

/// First-order discrete difference
public func diff(_ values: [Double]) -> [Double] {
    guard values.count > 1 else { return [] }

    var result = [Double]()
    result.reserveCapacity(values.count - 1)

    for i in 1..<values.count {
        result.append(values[i] - values[i - 1])
    }

    return result
}

// MARK: - Combinatorics

/// Factorial: n!
public func factorial(_ n: Int) -> Double {
    guard n >= 0 else { return .nan }

    if n <= 1 { return 1.0 }

    if n <= 20 {
        var result: Double = 1.0
        for i in 2...n {
            result *= Double(i)
        }
        return result
    } else {
        return Darwin.exp(Darwin.lgamma(Double(n) + 1))
    }
}

/// Permutations: P(n, k) = n! / (n-k)!
public func perm(_ n: Int, _ k: Int) -> Double {
    guard n >= 0 && k >= 0 else { return .nan }

    if k > n { return 0 }
    if k == 0 { return 1 }

    if n <= 20 {
        var result: Double = 1.0
        for i in (n - k + 1)...n {
            result *= Double(i)
        }
        return result
    }

    return Darwin.exp(Darwin.lgamma(Double(n) + 1) - Darwin.lgamma(Double(n - k) + 1))
}

/// Combinations: C(n, k) = n! / (k! * (n-k)!)
public func comb(_ n: Int, _ k: Int) -> Double {
    guard n >= 0 && k >= 0 else { return .nan }

    if k > n { return 0 }
    if k == 0 || k == n { return 1 }

    // Use symmetry for efficiency
    let kUse = min(k, n - k)

    if n <= 20 {
        var result: Double = 1.0
        for i in 0..<kUse {
            result = result * Double(n - i) / Double(i + 1)
        }
        return Darwin.round(result)
    }

    let result = Darwin.exp(
        Darwin.lgamma(Double(n) + 1) -
        Darwin.lgamma(Double(kUse) + 1) -
        Darwin.lgamma(Double(n - kUse) + 1)
    )
    return Darwin.round(result)
}

/// Alias for comb (binomial coefficient)
public func binomial(_ n: Int, _ k: Int) -> Double {
    comb(n, k)
}

// MARK: - Rounding Functions

/// Round to specified number of decimal places
public func round(_ x: Double, decimals: Int = 0) -> Double {
    if decimals == 0 {
        return Darwin.round(x)
    }
    let multiplier = Darwin.pow(10.0, Double(decimals))
    return Darwin.round(x * multiplier) / multiplier
}

/// Truncate toward zero
public func trunc(_ x: Double) -> Double {
    Darwin.trunc(x)
}

/// Sign function: -1, 0, or 1
public func sign(_ x: Double) -> Double {
    if x > 0 { return 1 }
    if x < 0 { return -1 }
    return 0
}

// MARK: - Coordinate Conversions

/// Convert polar coordinates to Cartesian
/// - Parameters:
///   - r: Radius
///   - theta: Angle in radians
/// - Returns: Tuple (x, y)
public func polarToCart(r: Double, theta: Double) -> (x: Double, y: Double) {
    (r * Darwin.cos(theta), r * Darwin.sin(theta))
}

/// Convert Cartesian coordinates to polar
/// - Parameters:
///   - x: X coordinate
///   - y: Y coordinate
/// - Returns: Tuple (r, theta) where theta is in radians
public func cartToPolar(x: Double, y: Double) -> (r: Double, theta: Double) {
    (Darwin.sqrt(x * x + y * y), Darwin.atan2(y, x))
}

/// Convert spherical coordinates to Cartesian
/// - Parameters:
///   - r: Radius
///   - theta: Polar angle (from z-axis) in radians
///   - phi: Azimuthal angle (from x-axis in xy-plane) in radians
/// - Returns: Tuple (x, y, z)
public func sphericalToCart(r: Double, theta: Double, phi: Double) -> (x: Double, y: Double, z: Double) {
    let sinTheta = Darwin.sin(theta)
    let x = r * sinTheta * Darwin.cos(phi)
    let y = r * sinTheta * Darwin.sin(phi)
    let z = r * Darwin.cos(theta)
    return (x, y, z)
}

/// Convert Cartesian coordinates to spherical
/// - Parameters:
///   - x: X coordinate
///   - y: Y coordinate
///   - z: Z coordinate
/// - Returns: Tuple (r, theta, phi) where angles are in radians
public func cartToSpherical(x: Double, y: Double, z: Double) -> (r: Double, theta: Double, phi: Double) {
    let r = Darwin.sqrt(x * x + y * y + z * z)
    let theta = r > 0 ? Darwin.acos(z / r) : 0
    let phi = Darwin.atan2(y, x)
    return (r, theta, phi)
}

// MARK: - Degrees/Radians Conversion

/// Convert degrees to radians
public func deg2rad(_ degrees: Double) -> Double {
    degrees * .pi / 180.0
}

/// Convert radians to degrees
public func rad2deg(_ radians: Double) -> Double {
    radians * 180.0 / .pi
}

// MARK: - Clipping

/// Clip values to a range
public func clip(_ value: Double, min: Double, max: Double) -> Double {
    Swift.min(Swift.max(value, min), max)
}

/// Clip array values to a range
public func clip(_ values: [Double], min: Double, max: Double) -> [Double] {
    values.map { clip($0, min: min, max: max) }
}
