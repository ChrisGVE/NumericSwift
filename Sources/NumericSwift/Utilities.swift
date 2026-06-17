//
//  Utilities.swift
//  NumericSwift
//
//  vDSP-optimized array math utilities.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

// MARK: - ArrayOps Namespace

/// vDSP-optimized element-wise array operations.
///
/// Scalar helpers (`signValue`, `clipValue`, `roundValue`, `truncValue`) are
/// also provided here for convenience. They are thin wrappers around Darwin
/// functions intended for use when operating on individual `Double` values in
/// a context where the full array path would be wasteful.
///
/// ## Overview
///
/// ```swift
/// let s = ArrayOps.sinArray([0, .pi/2, .pi])
/// let c = ArrayOps.clipArray([1, 5, 10], lo: 2, hi: 8)
/// ```
public enum ArrayOps {

    // MARK: - Scalar Helpers

    /// Round a value to the nearest integer.
    @inlinable
    public static func roundValue(_ x: Double) -> Double {
        Darwin.round(x)
    }

    /// Truncate a value toward zero.
    @inlinable
    public static func truncValue(_ x: Double) -> Double {
        Darwin.trunc(x)
    }

    /// Sign function returning -1, 0, or 1.
    @inlinable
    public static func signValue(_ x: Double) -> Double {
        if x > 0 { return 1.0 }
        if x < 0 { return -1.0 }
        return 0.0
    }

    /// Clip a value to a range.
    ///
    /// - Parameters:
    ///   - x: The value to clip
    ///   - lo: Lower bound
    ///   - hi: Upper bound
    @inlinable
    public static func clipValue(_ x: Double, lo: Double, hi: Double) -> Double {
        Swift.min(Swift.max(x, lo), hi)
    }

    // MARK: - Array Functions

    /// Round array elements to the nearest integer.
    public static func roundArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        return values.map { Darwin.round($0) }
    }

    /// Truncate array elements toward zero.
    public static func truncArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        return values.map { Darwin.trunc($0) }
    }

    /// Sign function for array elements (-1, 0, or 1).
    public static func signArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        return values.map { signValue($0) }
    }

    /// Clip array values to a range using vDSP_vclipD.
    ///
    /// - Parameters:
    ///   - values: Array of values to clip
    ///   - lo: Lower bound
    ///   - hi: Upper bound
    public static func clipArray(_ values: [Double], lo: Double, hi: Double) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var low = lo
        var high = hi
        vDSP_vclipD(values, 1, &low, &high, &result, 1, vDSP_Length(values.count))
        return result
    }

    /// Floor array elements using vvfloor.
    public static func floorArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvfloor(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Ceiling array elements using vvceil.
    public static func ceilArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvceil(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Absolute value for array elements using vDSP_vabsD.
    public static func absArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        vDSP_vabsD(values, 1, &result, 1, vDSP_Length(values.count))
        return result
    }

    /// Negate array elements using vDSP_vnegD.
    public static func negArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        vDSP_vnegD(values, 1, &result, 1, vDSP_Length(values.count))
        return result
    }

    /// Square root for array elements using vvsqrt.
    public static func sqrtArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvsqrt(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Square for array elements using vDSP_vsqD.
    public static func squareArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        vDSP_vsqD(values, 1, &result, 1, vDSP_Length(values.count))
        return result
    }

    /// Natural logarithm for array elements using vvlog.
    public static func logArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvlog(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Base-10 logarithm for array elements using vvlog10.
    public static func log10Array(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvlog10(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Exponential for array elements using vvexp.
    public static func expArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvexp(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Power function for array elements using vvpow.
    ///
    /// - Parameters:
    ///   - bases: Array of base values
    ///   - exponents: Array of exponent values (same length as bases)
    public static func powArray(_ bases: [Double], _ exponents: [Double]) -> [Double] {
        guard !bases.isEmpty, bases.count == exponents.count else { return [] }
        var result = [Double](repeating: 0, count: bases.count)
        var n = Int32(bases.count)
        bases.withUnsafeBufferPointer { basesPtr in
            exponents.withUnsafeBufferPointer { expsPtr in
                result.withUnsafeMutableBufferPointer { dst in
                    vvpow(dst.baseAddress!, expsPtr.baseAddress!, basesPtr.baseAddress!, &n)
                }
            }
        }
        return result
    }

    /// Sine for array elements using vvsin.
    public static func sinArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvsin(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Cosine for array elements using vvcos.
    public static func cosArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvcos(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Tangent for array elements using vvtan.
    public static func tanArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvtan(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Arcsine for array elements using vvasin.
    public static func asinArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvasin(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Arccosine for array elements using vvacos.
    public static func acosArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvacos(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Arctangent for array elements using vvatan.
    public static func atanArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvatan(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Hyperbolic sine for array elements using vvsinh.
    public static func sinhArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvsinh(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Hyperbolic cosine for array elements using vvcosh.
    public static func coshArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvcosh(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }

    /// Hyperbolic tangent for array elements using vvtanh.
    public static func tanhArray(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        var n = Int32(values.count)
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vvtanh(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        return result
    }
}

// MARK: - Deprecated shims (backward compatibility)

/// - Note: Deprecated. Use ``ArrayOps/roundValue(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.roundValue(_:) instead")
@inlinable
public func roundValue(_ x: Double) -> Double {
    ArrayOps.roundValue(x)
}

/// - Note: Deprecated. Use ``ArrayOps/truncValue(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.truncValue(_:) instead")
@inlinable
public func truncValue(_ x: Double) -> Double {
    ArrayOps.truncValue(x)
}

/// - Note: Deprecated. Use ``ArrayOps/signValue(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.signValue(_:) instead")
@inlinable
public func signValue(_ x: Double) -> Double {
    ArrayOps.signValue(x)
}

/// - Note: Deprecated. Use ``ArrayOps/clipValue(_:lo:hi:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.clipValue(_:lo:hi:) instead")
@inlinable
public func clipValue(_ x: Double, lo: Double, hi: Double) -> Double {
    ArrayOps.clipValue(x, lo: lo, hi: hi)
}

/// - Note: Deprecated. Use ``ArrayOps/roundArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.roundArray(_:) instead")
public func roundArray(_ values: [Double]) -> [Double] {
    ArrayOps.roundArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/truncArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.truncArray(_:) instead")
public func truncArray(_ values: [Double]) -> [Double] {
    ArrayOps.truncArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/signArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.signArray(_:) instead")
public func signArray(_ values: [Double]) -> [Double] {
    ArrayOps.signArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/clipArray(_:lo:hi:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.clipArray(_:lo:hi:) instead")
public func clipArray(_ values: [Double], lo: Double, hi: Double) -> [Double] {
    ArrayOps.clipArray(values, lo: lo, hi: hi)
}

/// - Note: Deprecated. Use ``ArrayOps/floorArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.floorArray(_:) instead")
public func floorArray(_ values: [Double]) -> [Double] {
    ArrayOps.floorArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/ceilArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.ceilArray(_:) instead")
public func ceilArray(_ values: [Double]) -> [Double] {
    ArrayOps.ceilArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/absArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.absArray(_:) instead")
public func absArray(_ values: [Double]) -> [Double] {
    ArrayOps.absArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/negArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.negArray(_:) instead")
public func negArray(_ values: [Double]) -> [Double] {
    ArrayOps.negArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/sqrtArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.sqrtArray(_:) instead")
public func sqrtArray(_ values: [Double]) -> [Double] {
    ArrayOps.sqrtArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/squareArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.squareArray(_:) instead")
public func squareArray(_ values: [Double]) -> [Double] {
    ArrayOps.squareArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/logArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.logArray(_:) instead")
public func logArray(_ values: [Double]) -> [Double] {
    ArrayOps.logArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/log10Array(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.log10Array(_:) instead")
public func log10Array(_ values: [Double]) -> [Double] {
    ArrayOps.log10Array(values)
}

/// - Note: Deprecated. Use ``ArrayOps/expArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.expArray(_:) instead")
public func expArray(_ values: [Double]) -> [Double] {
    ArrayOps.expArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/powArray(_:_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.powArray(_:_:) instead")
public func powArray(_ bases: [Double], _ exponents: [Double]) -> [Double] {
    ArrayOps.powArray(bases, exponents)
}

/// - Note: Deprecated. Use ``ArrayOps/sinArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.sinArray(_:) instead")
public func sinArray(_ values: [Double]) -> [Double] {
    ArrayOps.sinArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/cosArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.cosArray(_:) instead")
public func cosArray(_ values: [Double]) -> [Double] {
    ArrayOps.cosArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/tanArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.tanArray(_:) instead")
public func tanArray(_ values: [Double]) -> [Double] {
    ArrayOps.tanArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/asinArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.asinArray(_:) instead")
public func asinArray(_ values: [Double]) -> [Double] {
    ArrayOps.asinArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/acosArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.acosArray(_:) instead")
public func acosArray(_ values: [Double]) -> [Double] {
    ArrayOps.acosArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/atanArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.atanArray(_:) instead")
public func atanArray(_ values: [Double]) -> [Double] {
    ArrayOps.atanArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/sinhArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.sinhArray(_:) instead")
public func sinhArray(_ values: [Double]) -> [Double] {
    ArrayOps.sinhArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/coshArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.coshArray(_:) instead")
public func coshArray(_ values: [Double]) -> [Double] {
    ArrayOps.coshArray(values)
}

/// - Note: Deprecated. Use ``ArrayOps/tanhArray(_:)`` instead.
@available(*, deprecated, message: "Use ArrayOps.tanhArray(_:) instead")
public func tanhArray(_ values: [Double]) -> [Double] {
    ArrayOps.tanhArray(values)
}
