//
//  Statistics.swift
//  NumericSwift
//
//  Basic statistical functions following numpy/scipy patterns.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Stats Namespace

/// Descriptive statistics and array reduction functions.
///
/// ## Overview
///
/// ```swift
/// let m = Stats.mean([1, 2, 3, 4, 5])          // 3.0
/// let v = Stats.variance([1, 2, 3], ddof: 1)   // sample variance
/// ```
public enum Stats {

    // MARK: - Basic Statistics

    /// Sum of array elements.
    public static func sum(_ values: [Double]) -> Double {
        values.reduce(0.0, +)
    }

    /// Arithmetic mean (average).
    public static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        return sum(values) / Double(values.count)
    }

    /// Median (middle value).
    public static func median(_ values: [Double]) -> Double {
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

    /// Variance with optional delta degrees of freedom.
    ///
    /// - Parameters:
    ///   - values: Array of values
    ///   - ddof: Delta degrees of freedom (0 for population variance, 1 for sample variance)
    /// - Returns: Variance
    public static func variance(_ values: [Double], ddof: Int = 0) -> Double {
        guard !values.isEmpty else { return .nan }
        let divisor = values.count - ddof
        guard divisor > 0 else { return .nan }
        let m = mean(values)
        let squaredDiffs = values.map { ($0 - m) * ($0 - m) }
        return squaredDiffs.reduce(0.0, +) / Double(divisor)
    }

    /// Standard deviation with optional delta degrees of freedom.
    ///
    /// - Parameters:
    ///   - values: Array of values
    ///   - ddof: Delta degrees of freedom (0 for population stddev, 1 for sample stddev)
    /// - Returns: Standard deviation
    public static func stddev(_ values: [Double], ddof: Int = 0) -> Double {
        Darwin.sqrt(variance(values, ddof: ddof))
    }

    /// Percentile using linear interpolation.
    ///
    /// - Parameters:
    ///   - values: Array of values
    ///   - p: Percentile (0-100)
    /// - Returns: The p-th percentile value
    public static func percentile(_ values: [Double], _ p: Double) -> Double {
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

    /// Geometric mean: nth root of product of n values.
    ///
    /// Requires all positive values.
    public static func gmean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        for v in values { if v <= 0 { return .nan } }
        let logSum = values.reduce(0.0) { $0 + Darwin.log($1) }
        return Darwin.exp(logSum / Double(values.count))
    }

    /// Harmonic mean: n / sum(1/x_i).
    ///
    /// Requires all positive values.
    public static func hmean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        for v in values { if v <= 0 { return .nan } }
        let reciprocalSum = values.reduce(0.0) { $0 + 1.0 / $1 }
        return Double(values.count) / reciprocalSum
    }

    /// Mode: most frequently occurring value.
    ///
    /// Returns the smallest mode if there are ties (scipy behavior).
    public static func mode(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        var counts: [Double: Int] = [:]
        for v in values { counts[v, default: 0] += 1 }
        let maxCount = counts.values.max() ?? 0
        let modes = counts.filter { $0.value == maxCount }.keys.sorted()
        return modes.first ?? values[0]
    }

    // MARK: - Minimum / Maximum

    /// Minimum value in array.
    public static func amin(_ values: [Double]) -> Double {
        values.min() ?? .nan
    }

    /// Maximum value in array.
    public static func amax(_ values: [Double]) -> Double {
        values.max() ?? .nan
    }

    /// Range (max - min).
    public static func ptp(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        return (values.max() ?? 0) - (values.min() ?? 0)
    }

    // MARK: - Cumulative Functions

    /// Cumulative sum.
    public static func cumsum(_ values: [Double]) -> [Double] {
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

    /// Cumulative product.
    public static func cumprod(_ values: [Double]) -> [Double] {
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

    /// First-order discrete difference.
    public static func diff(_ values: [Double]) -> [Double] {
        guard values.count > 1 else { return [] }
        var result = [Double]()
        result.reserveCapacity(values.count - 1)
        for i in 1..<values.count {
            result.append(values[i] - values[i - 1])
        }
        return result
    }

    // MARK: - Rounding Functions

    /// Round to specified number of decimal places.
    public static func round(_ x: Double, decimals: Int = 0) -> Double {
        if decimals == 0 { return Darwin.round(x) }
        let multiplier = Darwin.pow(10.0, Double(decimals))
        return Darwin.round(x * multiplier) / multiplier
    }

    /// Truncate toward zero.
    public static func trunc(_ x: Double) -> Double {
        Darwin.trunc(x)
    }

    /// Sign function: -1, 0, or 1.
    public static func sign(_ x: Double) -> Double {
        if x > 0 { return 1 }
        if x < 0 { return -1 }
        return 0
    }

    // MARK: - Clipping

    /// Clip value to a range.
    public static func clip(_ value: Double, min: Double, max: Double) -> Double {
        Swift.min(Swift.max(value, min), max)
    }

    /// Clip array values to a range.
    public static func clip(_ values: [Double], min: Double, max: Double) -> [Double] {
        values.map { clip($0, min: min, max: max) }
    }

    // MARK: - NaN-Aware Statistics

    /// Arithmetic mean ignoring NaN values.
    ///
    /// Returns `Double.nan` if the array is empty or all values are NaN.
    public static func nanmean(_ data: [Double]) -> Double {
        let clean = data.filter { !$0.isNaN }
        return mean(clean)
    }

    /// Median ignoring NaN values.
    ///
    /// Returns `Double.nan` if the array is empty or all values are NaN.
    public static func nanmedian(_ data: [Double]) -> Double {
        let clean = data.filter { !$0.isNaN }
        return median(clean)
    }

    /// Variance ignoring NaN values.
    ///
    /// - Parameter ddof: Delta degrees of freedom (0 = population, 1 = sample).
    public static func nanvariance(_ data: [Double], ddof: Int = 0) -> Double {
        let clean = data.filter { !$0.isNaN }
        return variance(clean, ddof: ddof)
    }

    /// Standard deviation ignoring NaN values.
    ///
    /// - Parameter ddof: Delta degrees of freedom (0 = population, 1 = sample).
    public static func nanstd(_ data: [Double], ddof: Int = 0) -> Double {
        let clean = data.filter { !$0.isNaN }
        return stddev(clean, ddof: ddof)
    }

    /// Minimum value ignoring NaN values.
    public static func nanmin(_ data: [Double]) -> Double {
        let clean = data.filter { !$0.isNaN }
        return amin(clean)
    }

    /// Maximum value ignoring NaN values.
    public static func nanmax(_ data: [Double]) -> Double {
        let clean = data.filter { !$0.isNaN }
        return amax(clean)
    }

    /// Sum ignoring NaN values.
    ///
    /// Returns `0.0` when all values are NaN or the array is empty, matching numpy behaviour.
    public static func nansum(_ data: [Double]) -> Double {
        let clean = data.filter { !$0.isNaN }
        return sum(clean)
    }

    /// Percentile using linear interpolation, ignoring NaN values.
    ///
    /// - Parameter p: Percentile in the range 0–100.
    public static func nanpercentile(_ data: [Double], _ p: Double) -> Double {
        let clean = data.filter { !$0.isNaN }
        return percentile(clean, p)
    }
}

// MARK: - Deprecated shims (backward compatibility)

/// - Note: Deprecated. Use ``Stats/sum(_:)`` instead.
@available(*, deprecated, message: "Use Stats.sum(_:) instead")
public func sum(_ values: [Double]) -> Double {
    Stats.sum(values)
}

/// - Note: Deprecated. Use ``Stats/mean(_:)`` instead.
@available(*, deprecated, message: "Use Stats.mean(_:) instead")
public func mean(_ values: [Double]) -> Double {
    Stats.mean(values)
}

/// - Note: Deprecated. Use ``Stats/median(_:)`` instead.
@available(*, deprecated, message: "Use Stats.median(_:) instead")
public func median(_ values: [Double]) -> Double {
    Stats.median(values)
}

/// - Note: Deprecated. Use ``Stats/variance(_:ddof:)`` instead.
@available(*, deprecated, message: "Use Stats.variance(_:ddof:) instead")
public func variance(_ values: [Double], ddof: Int = 0) -> Double {
    Stats.variance(values, ddof: ddof)
}

/// - Note: Deprecated. Use ``Stats/stddev(_:ddof:)`` instead.
@available(*, deprecated, message: "Use Stats.stddev(_:ddof:) instead")
public func stddev(_ values: [Double], ddof: Int = 0) -> Double {
    Stats.stddev(values, ddof: ddof)
}

/// - Note: Deprecated. Use ``Stats/percentile(_:_:)`` instead.
@available(*, deprecated, message: "Use Stats.percentile(_:_:) instead")
public func percentile(_ values: [Double], _ p: Double) -> Double {
    Stats.percentile(values, p)
}

/// - Note: Deprecated. Use ``Stats/gmean(_:)`` instead.
@available(*, deprecated, message: "Use Stats.gmean(_:) instead")
public func gmean(_ values: [Double]) -> Double {
    Stats.gmean(values)
}

/// - Note: Deprecated. Use ``Stats/hmean(_:)`` instead.
@available(*, deprecated, message: "Use Stats.hmean(_:) instead")
public func hmean(_ values: [Double]) -> Double {
    Stats.hmean(values)
}

/// - Note: Deprecated. Use ``Stats/mode(_:)`` instead.
@available(*, deprecated, message: "Use Stats.mode(_:) instead")
public func mode(_ values: [Double]) -> Double {
    Stats.mode(values)
}

/// - Note: Deprecated. Use ``Stats/amin(_:)`` instead.
@available(*, deprecated, message: "Use Stats.amin(_:) instead")
public func amin(_ values: [Double]) -> Double {
    Stats.amin(values)
}

/// - Note: Deprecated. Use ``Stats/amax(_:)`` instead.
@available(*, deprecated, message: "Use Stats.amax(_:) instead")
public func amax(_ values: [Double]) -> Double {
    Stats.amax(values)
}

/// - Note: Deprecated. Use ``Stats/ptp(_:)`` instead.
@available(*, deprecated, message: "Use Stats.ptp(_:) instead")
public func ptp(_ values: [Double]) -> Double {
    Stats.ptp(values)
}

/// - Note: Deprecated. Use ``Stats/cumsum(_:)`` instead.
@available(*, deprecated, message: "Use Stats.cumsum(_:) instead")
public func cumsum(_ values: [Double]) -> [Double] {
    Stats.cumsum(values)
}

/// - Note: Deprecated. Use ``Stats/cumprod(_:)`` instead.
@available(*, deprecated, message: "Use Stats.cumprod(_:) instead")
public func cumprod(_ values: [Double]) -> [Double] {
    Stats.cumprod(values)
}

/// - Note: Deprecated. Use ``Stats/diff(_:)`` instead.
@available(*, deprecated, message: "Use Stats.diff(_:) instead")
public func diff(_ values: [Double]) -> [Double] {
    Stats.diff(values)
}

/// - Note: Deprecated. Use ``Stats/round(_:decimals:)`` instead.
@available(*, deprecated, renamed: "Stats.round(_:decimals:)", message: "Use Stats.round(_:decimals:) instead")
public func round(_ x: Double, decimals: Int = 0) -> Double {
    Stats.round(x, decimals: decimals)
}

/// - Note: Deprecated. Use ``Stats/trunc(_:)`` instead.
@available(*, deprecated, message: "Use Stats.trunc(_:) instead")
public func trunc(_ x: Double) -> Double {
    Stats.trunc(x)
}

/// - Note: Deprecated. Use ``Stats/sign(_:)`` instead.
@available(*, deprecated, message: "Use Stats.sign(_:) instead")
public func sign(_ x: Double) -> Double {
    Stats.sign(x)
}

/// - Note: Deprecated. Use ``Stats/clip(_:min:max:)`` instead.
@available(*, deprecated, message: "Use Stats.clip(_:min:max:) instead")
public func clip(_ value: Double, min: Double, max: Double) -> Double {
    Stats.clip(value, min: min, max: max)
}

/// - Note: Deprecated. Use ``Stats/clip(_:min:max:)`` instead.
@available(*, deprecated, message: "Use Stats.clip(_:min:max:) instead")
public func clip(_ values: [Double], min: Double, max: Double) -> [Double] {
    Stats.clip(values, min: min, max: max)
}

/// - Note: Deprecated. Use ``Stats/nanmean(_:)`` instead.
@available(*, deprecated, message: "Use Stats.nanmean(_:) instead")
public func nanmean(_ data: [Double]) -> Double {
    Stats.nanmean(data)
}

/// - Note: Deprecated. Use ``Stats/nanmedian(_:)`` instead.
@available(*, deprecated, message: "Use Stats.nanmedian(_:) instead")
public func nanmedian(_ data: [Double]) -> Double {
    Stats.nanmedian(data)
}

/// - Note: Deprecated. Use ``Stats/nanvariance(_:ddof:)`` instead.
@available(*, deprecated, message: "Use Stats.nanvariance(_:ddof:) instead")
public func nanvariance(_ data: [Double], ddof: Int = 0) -> Double {
    Stats.nanvariance(data, ddof: ddof)
}

/// - Note: Deprecated. Use ``Stats/nanstd(_:ddof:)`` instead.
@available(*, deprecated, message: "Use Stats.nanstd(_:ddof:) instead")
public func nanstd(_ data: [Double], ddof: Int = 0) -> Double {
    Stats.nanstd(data, ddof: ddof)
}

/// - Note: Deprecated. Use ``Stats/nanmin(_:)`` instead.
@available(*, deprecated, message: "Use Stats.nanmin(_:) instead")
public func nanmin(_ data: [Double]) -> Double {
    Stats.nanmin(data)
}

/// - Note: Deprecated. Use ``Stats/nanmax(_:)`` instead.
@available(*, deprecated, message: "Use Stats.nanmax(_:) instead")
public func nanmax(_ data: [Double]) -> Double {
    Stats.nanmax(data)
}

/// - Note: Deprecated. Use ``Stats/nansum(_:)`` instead.
@available(*, deprecated, message: "Use Stats.nansum(_:) instead")
public func nansum(_ data: [Double]) -> Double {
    Stats.nansum(data)
}

/// - Note: Deprecated. Use ``Stats/nanpercentile(_:_:)`` instead.
@available(*, deprecated, message: "Use Stats.nanpercentile(_:_:) instead")
public func nanpercentile(_ data: [Double], _ p: Double) -> Double {
    Stats.nanpercentile(data, p)
}
