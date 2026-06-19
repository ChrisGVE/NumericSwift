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

    /// Apply a vForce unary op `(out, in, count)` over an array of any length,
    /// processing in `Int32.max`-bounded chunks. The vForce ABI takes the element
    /// count as `Int32`; a single `Int32(values.count)` would trap for arrays with
    /// more than `Int32.max` elements, so the count is chunked instead.
    private static func vForceUnary(
        _ values: [Double],
        _ op: (UnsafeMutablePointer<Double>, UnsafePointer<Double>, UnsafePointer<Int32>) -> Void
    ) -> [Double] {
        guard !values.isEmpty else { return [] }
        var result = [Double](repeating: 0, count: values.count)
        let total = values.count
        values.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                var offset = 0
                while offset < total {
                    var c = Int32(Swift.min(total - offset, Int(Int32.max)))
                    op(dst.baseAddress! + offset, src.baseAddress! + offset, &c)
                    offset += Int(c)
                }
            }
        }
        return result
    }

    /// Apply a vForce binary op `(out, a, b, count)` (e.g. `vvpow`) with the same
    /// `Int32.max`-bounded chunking as ``vForceUnary(_:_:)``.
    private static func vForceBinary(
        _ a: [Double], _ b: [Double],
        _ op: (UnsafeMutablePointer<Double>, UnsafePointer<Double>, UnsafePointer<Double>, UnsafePointer<Int32>) -> Void
    ) -> [Double] {
        guard !a.isEmpty, a.count == b.count else { return [] }
        var result = [Double](repeating: 0, count: a.count)
        let total = a.count
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                result.withUnsafeMutableBufferPointer { dst in
                    var offset = 0
                    while offset < total {
                        var c = Int32(Swift.min(total - offset, Int(Int32.max)))
                        op(dst.baseAddress! + offset, aPtr.baseAddress! + offset,
                           bPtr.baseAddress! + offset, &c)
                        offset += Int(c)
                    }
                }
            }
        }
        return result
    }

    /// Floor array elements using vvfloor.
    public static func floorArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvfloor($0, $1, $2) }
    }

    /// Ceiling array elements using vvceil.
    public static func ceilArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvceil($0, $1, $2) }
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
        vForceUnary(values) { vvsqrt($0, $1, $2) }
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
        vForceUnary(values) { vvlog($0, $1, $2) }
    }

    /// Base-10 logarithm for array elements using vvlog10.
    public static func log10Array(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvlog10($0, $1, $2) }
    }

    /// Exponential for array elements using vvexp.
    public static func expArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvexp($0, $1, $2) }
    }

    /// Power function for array elements using vvpow.
    ///
    /// - Parameters:
    ///   - bases: Array of base values
    ///   - exponents: Array of exponent values (same length as bases)
    public static func powArray(_ bases: [Double], _ exponents: [Double]) -> [Double] {
        // vvpow computes out = x^y as `vvpow(out, y, x, count)`, so pass exponents
        // as the first (y) operand and bases as the second (x).
        vForceBinary(exponents, bases) { vvpow($0, $1, $2, $3) }
    }

    /// Sine for array elements using vvsin.
    public static func sinArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvsin($0, $1, $2) }
    }

    /// Cosine for array elements using vvcos.
    public static func cosArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvcos($0, $1, $2) }
    }

    /// Tangent for array elements using vvtan.
    public static func tanArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvtan($0, $1, $2) }
    }

    /// Arcsine for array elements using vvasin.
    public static func asinArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvasin($0, $1, $2) }
    }

    /// Arccosine for array elements using vvacos.
    public static func acosArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvacos($0, $1, $2) }
    }

    /// Arctangent for array elements using vvatan.
    public static func atanArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvatan($0, $1, $2) }
    }

    /// Hyperbolic sine for array elements using vvsinh.
    public static func sinhArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvsinh($0, $1, $2) }
    }

    /// Hyperbolic cosine for array elements using vvcosh.
    public static func coshArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvcosh($0, $1, $2) }
    }

    /// Hyperbolic tangent for array elements using vvtanh.
    public static func tanhArray(_ values: [Double]) -> [Double] {
        vForceUnary(values) { vvtanh($0, $1, $2) }
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
