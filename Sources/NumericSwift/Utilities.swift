//
//  Utilities.swift
//  NumericSwift
//
//  Extended math utilities with optional vDSP optimization when ArraySwift is available.
//
//  Licensed under the MIT License.
//

import Foundation
import Accelerate

// MARK: - Scalar Functions (Always Available)

/// Round a value to the nearest integer.
///
/// - Parameter x: The value to round
/// - Returns: The rounded value
/// - Note: Works with Double only. For complex numbers, use round on real and imaginary parts separately.
@inlinable
public func roundValue(_ x: Double) -> Double {
    Darwin.round(x)
}

/// Truncate a value toward zero.
///
/// - Parameter x: The value to truncate
/// - Returns: The truncated value
/// - Note: Works with Double only. For complex numbers, use trunc on real and imaginary parts separately.
@inlinable
public func truncValue(_ x: Double) -> Double {
    Darwin.trunc(x)
}

/// Sign function returning -1, 0, or 1.
///
/// - Parameter x: The value to check
/// - Returns: -1 if x < 0, 0 if x == 0, 1 if x > 0
/// - Note: Works with Double only. For complex numbers, consider signum(z) = z/|z|.
@inlinable
public func signValue(_ x: Double) -> Double {
    if x > 0 { return 1.0 }
    if x < 0 { return -1.0 }
    return 0.0
}

/// Clip a value to a range.
///
/// - Parameters:
///   - x: The value to clip
///   - lo: The lower bound
///   - hi: The upper bound
/// - Returns: The clipped value
/// - Note: Works with Double only.
@inlinable
public func clipValue(_ x: Double, lo: Double, hi: Double) -> Double {
    Swift.min(Swift.max(x, lo), hi)
}

// MARK: - Array Functions

/// Round array elements to the nearest integer.
///
/// - Parameter values: Array of values to round
/// - Returns: Array of rounded values
public func roundArray(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }
    return values.map { Darwin.round($0) }
}

/// Truncate array elements toward zero.
///
/// - Parameter values: Array of values to truncate
/// - Returns: Array of truncated values
public func truncArray(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }
    return values.map { Darwin.trunc($0) }
}

/// Sign function for array elements.
///
/// - Parameter values: Array of values
/// - Returns: Array of sign values (-1, 0, or 1)
public func signArray(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }
    return values.map { signValue($0) }
}

/// Clip array values to a range.
///
/// Uses vDSP_vclipD for optimal performance.
///
/// - Parameters:
///   - values: Array of values to clip
///   - lo: The lower bound
///   - hi: The upper bound
/// - Returns: Array of clipped values
public func clipArray(_ values: [Double], lo: Double, hi: Double) -> [Double] {
    guard !values.isEmpty else { return [] }

    var result = [Double](repeating: 0, count: values.count)
    var low = lo
    var high = hi

    vDSP_vclipD(values, 1, &low, &high, &result, 1, vDSP_Length(values.count))

    return result
}

/// Floor array elements.
///
/// Uses vvfloor for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of floored values
public func floorArray(_ values: [Double]) -> [Double] {
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

/// Ceiling array elements.
///
/// Uses vvceil for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of ceiling values
public func ceilArray(_ values: [Double]) -> [Double] {
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

/// Absolute value for array elements.
///
/// Uses vDSP_vabsD for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of absolute values
public func absArray(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }

    var result = [Double](repeating: 0, count: values.count)
    vDSP_vabsD(values, 1, &result, 1, vDSP_Length(values.count))

    return result
}

/// Negate array elements.
///
/// Uses vDSP_vnegD for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of negated values
public func negArray(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }

    var result = [Double](repeating: 0, count: values.count)
    vDSP_vnegD(values, 1, &result, 1, vDSP_Length(values.count))

    return result
}

/// Square root for array elements.
///
/// Uses vvsqrt for optimal performance.
///
/// - Parameter values: Array of non-negative values
/// - Returns: Array of square roots
public func sqrtArray(_ values: [Double]) -> [Double] {
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

/// Square for array elements.
///
/// Uses vDSP_vsqD for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of squared values
public func squareArray(_ values: [Double]) -> [Double] {
    guard !values.isEmpty else { return [] }

    var result = [Double](repeating: 0, count: values.count)
    vDSP_vsqD(values, 1, &result, 1, vDSP_Length(values.count))

    return result
}

/// Natural logarithm for array elements.
///
/// Uses vvlog for optimal performance.
///
/// - Parameter values: Array of positive values
/// - Returns: Array of natural logarithms
public func logArray(_ values: [Double]) -> [Double] {
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

/// Base-10 logarithm for array elements.
///
/// Uses vvlog10 for optimal performance.
///
/// - Parameter values: Array of positive values
/// - Returns: Array of base-10 logarithms
public func log10Array(_ values: [Double]) -> [Double] {
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

/// Exponential for array elements.
///
/// Uses vvexp for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of exponentials (e^x)
public func expArray(_ values: [Double]) -> [Double] {
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

/// Power function for array elements.
///
/// Uses vvpow for optimal performance.
///
/// - Parameters:
///   - bases: Array of base values
///   - exponents: Array of exponent values (same length as bases)
/// - Returns: Array of bases^exponents
public func powArray(_ bases: [Double], _ exponents: [Double]) -> [Double] {
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

/// Sine for array elements.
///
/// Uses vvsin for optimal performance.
///
/// - Parameter values: Array of angles in radians
/// - Returns: Array of sine values
public func sinArray(_ values: [Double]) -> [Double] {
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

/// Cosine for array elements.
///
/// Uses vvcos for optimal performance.
///
/// - Parameter values: Array of angles in radians
/// - Returns: Array of cosine values
public func cosArray(_ values: [Double]) -> [Double] {
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

/// Tangent for array elements.
///
/// Uses vvtan for optimal performance.
///
/// - Parameter values: Array of angles in radians
/// - Returns: Array of tangent values
public func tanArray(_ values: [Double]) -> [Double] {
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

/// Arcsine for array elements.
///
/// Uses vvasin for optimal performance.
///
/// - Parameter values: Array of values in range [-1, 1]
/// - Returns: Array of arcsine values in radians
public func asinArray(_ values: [Double]) -> [Double] {
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

/// Arccosine for array elements.
///
/// Uses vvacos for optimal performance.
///
/// - Parameter values: Array of values in range [-1, 1]
/// - Returns: Array of arccosine values in radians
public func acosArray(_ values: [Double]) -> [Double] {
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

/// Arctangent for array elements.
///
/// Uses vvatan for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of arctangent values in radians
public func atanArray(_ values: [Double]) -> [Double] {
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

/// Hyperbolic sine for array elements.
///
/// Uses vvsinh for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of hyperbolic sine values
public func sinhArray(_ values: [Double]) -> [Double] {
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

/// Hyperbolic cosine for array elements.
///
/// Uses vvcosh for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of hyperbolic cosine values
public func coshArray(_ values: [Double]) -> [Double] {
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

/// Hyperbolic tangent for array elements.
///
/// Uses vvtanh for optimal performance.
///
/// - Parameter values: Array of values
/// - Returns: Array of hyperbolic tangent values
public func tanhArray(_ values: [Double]) -> [Double] {
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
