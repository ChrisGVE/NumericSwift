//
//  NumericValue+Equality.swift
//  Sources/NumericSwift
//
//  Explicit IEEE-754-aware equality methods for NumericValue.
//
//  ## Semantic choices (recorded to avoid re-litigating)
//
//  ### NaN
//  Both `isExactlyEqual` and `isApproximatelyEqual` treat NaN as non-reflexive:
//  a value containing NaN in any component is never equal to anything,
//  including itself. This matches IEEE 754 § 5.11 and the behaviour of the
//  built-in `==` on `Double`. It also matches SciPy / NumPy `array_equal`
//  (default `equal_nan=False`).
//
//  ### Signed zero
//  `isExactlyEqual` uses Swift's built-in `Double ==`, which follows IEEE 754
//  § 5.10: +0.0 == −0.0 is *true*. This is the "value equality" interpretation.
//  If bitwise identity (distinguishing signs of zero) is ever needed, use
//  `a.sign == b.sign && a == b` or compare `a.bitPattern == b.bitPattern`.
//  The current choice is documented here; the implementation does not paper
//  over it. `isApproximatelyEqual` inherits the same convention naturally
//  because |+0.0 − (−0.0)| == 0.0 ≤ any positive tolerance.
//
//  ### LinAlg.Matrix tolerant ==
//  `LinAlg.Matrix`'s synthesised `Equatable` implementation uses a 1e-10
//  tolerance (see LinAlg.swift line ~307). `isExactlyEqual` MUST NOT call that
//  operator. Instead, it compares the underlying `[Double]` arrays element by
//  element using raw `Double ==`, bypassing the tolerance entirely.
//
//  Licensed under the Apache License, Version 2.0.
//

// MARK: - NumericValue Equality Methods

extension NumericValue {

    // MARK: - isExactlyEqual

    /// Returns `true` when `self` and `other` carry the same kind and are
    /// value-equal under IEEE 754 equality rules.
    ///
    /// **Semantics:**
    /// - Cross-kind comparisons (e.g. `.scalar` vs `.matrix`) always return `false`.
    /// - NaN is *non-reflexive*: a value containing NaN in any component returns
    ///   `false`, even when compared to itself.
    /// - Signed zero: `+0.0` and `−0.0` compare as *equal* (IEEE 754 § 5.10,
    ///   value equality). Use `bitPattern` comparison if sign-of-zero identity
    ///   is required.
    /// - Matrices of differing shape return `false`.
    /// - Matrix elements are compared with exact `Double ==`, bypassing
    ///   `LinAlg.Matrix`'s tolerance-based `==` operator.
    ///
    /// - Parameter other: The value to compare against.
    /// - Returns: `true` iff the values are of the same kind and exactly equal.
    public func isExactlyEqual(to other: NumericValue) -> Bool {
        switch (self, other) {

        case (.scalar(let a), .scalar(let b)):
            // IEEE 754: NaN != NaN; +0.0 == -0.0
            return a == b

        case (.complex(let za), .complex(let zb)):
            // Both real and imaginary parts must be exactly equal.
            // NaN in either part makes the whole comparison false.
            return za.re == zb.re && za.im == zb.im

        case (.matrix(let ma), .matrix(let mb)):
            guard ma.rows == mb.rows && ma.cols == mb.cols else { return false }
            // Compare raw data element by element with exact Double ==.
            // We do NOT call LinAlg.Matrix.== (which uses a 1e-10 tolerance).
            // We also do NOT use [Double].== because Swift's Array.== can
            // short-circuit to true when both sides share the same backing
            // storage, which would incorrectly return true for a matrix whose
            // data contains NaN (NaN must never equal anything, including itself).
            return zip(ma.data, mb.data).allSatisfy { $0 == $1 }

        case (.complexMatrix(let ca), .complexMatrix(let cb)):
            guard ca.rows == cb.rows && ca.cols == cb.cols else { return false }
            // Same bypass and same reason for element-wise loop over Array.==.
            return zip(ca.real, cb.real).allSatisfy { $0 == $1 }
                && zip(ca.imag, cb.imag).allSatisfy { $0 == $1 }

        default:
            // Cross-kind: scalar vs complex, matrix vs complexMatrix, etc.
            return false
        }
    }

    // MARK: - isApproximatelyEqual

    /// Returns `true` when `self` and `other` carry the same kind and every
    /// corresponding element differs by at most `tolerance` in absolute value.
    ///
    /// **Semantics:**
    /// - Cross-kind comparisons always return `false`.
    /// - NaN: a value containing NaN in any component returns `false` because
    ///   `|NaN − x|` is NaN, which is never `≤ tolerance`.
    /// - Signed zero: |+0.0 − (−0.0)| == 0.0, so signed zeros compare equal
    ///   under any non-negative tolerance.
    /// - Matrices of differing shape return `false`.
    /// - This relation is **non-transitive**: for tolerance `t` it is possible
    ///   that `a ≈ b` and `b ≈ c` yet `a ≉ c` (e.g. a=0, b=t·0.9, c=t·1.8).
    ///
    /// - Parameters:
    ///   - other: The value to compare against.
    ///   - tolerance: Maximum allowed absolute element-wise difference
    ///     (default: `1e-10`).
    /// - Returns: `true` iff every component difference is ≤ `tolerance`.
    public func isApproximatelyEqual(to other: NumericValue,
                                     tolerance: Double = 1e-10) -> Bool {
        switch (self, other) {

        case (.scalar(let a), .scalar(let b)):
            return abs(a - b) <= tolerance

        case (.complex(let za), .complex(let zb)):
            return abs(za.re - zb.re) <= tolerance
                && abs(za.im - zb.im) <= tolerance

        case (.matrix(let ma), .matrix(let mb)):
            guard ma.rows == mb.rows && ma.cols == mb.cols else { return false }
            return zip(ma.data, mb.data).allSatisfy { abs($0 - $1) <= tolerance }

        case (.complexMatrix(let ca), .complexMatrix(let cb)):
            guard ca.rows == cb.rows && ca.cols == cb.cols else { return false }
            return zip(ca.real, cb.real).allSatisfy { abs($0 - $1) <= tolerance }
                && zip(ca.imag, cb.imag).allSatisfy { abs($0 - $1) <= tolerance }

        default:
            return false
        }
    }
}
