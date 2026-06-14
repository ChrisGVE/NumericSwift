//
//  NumericValue.swift
//  Sources/NumericSwift
//
//  The NumericValue tower type — a single enum that unifies all numeric kinds
//  produced and consumed by the unified numeric pipeline.
//
//  Architectural context: NumericValue sits at the boundary between the
//  per-domain modules (Complex, LinAlg) and the pipeline layer being built in
//  the unified-pipeline tag. Every pipeline stage accepts and produces
//  NumericValue, so downstream code never has to switch on concrete types.
//
//  Design decisions recorded here to avoid re-litigating them:
//
//  • No Equatable / Hashable synthesis. Equality is intentionally deferred to
//    Task 7, which provides explicit IEEE-754-aware methods. The synthesised
//    default would violate NaN semantics (NaN == NaN is false by IEEE 754 but
//    true for Swift's synthesised ==) and would clash with LinAlg.Matrix's
//    tolerance-based == operator. Leaving those protocols unimplemented forces
//    callers to use the explicit equality methods, which makes the NaN and
//    tolerance behaviour visible at every call site.
//
//  • Sendable conformance is structural (no @unchecked needed): all four
//    payload types were made Sendable in Task 4:
//      Double               — primitive, always Sendable
//      Complex              — two Double fields
//      LinAlg.Matrix        — Int × 2 + [Double]
//      LinAlg.ComplexMatrix — Int × 2 + [Double] × 2
//
//  Licensed under the Apache License, Version 2.0.
//

// MARK: - NumericValue

/// A discriminated union over the four numeric kinds in the pipeline.
///
/// `NumericValue` is the currency type of the unified numeric pipeline: every
/// stage accepts `NumericValue` inputs and returns `NumericValue` outputs.
/// Pattern-matching on the cases gives access to the concrete payload.
///
/// ```swift
/// func printValue(_ v: NumericValue) {
///     switch v {
///     case .scalar(let x):        print("scalar:", x)
///     case .complex(let z):       print("complex:", z)
///     case .matrix(let m):        print("matrix:", m)
///     case .complexMatrix(let cm): print("complexMatrix:", cm)
///     }
/// }
/// ```
///
/// ## Equality
///
/// `NumericValue` deliberately does **not** conform to `Equatable`. Use the
/// explicit IEEE-754-aware comparison methods provided in Task 7. This makes
/// NaN semantics and matrix-tolerance choices explicit at every call site.
///
/// ## Thread safety
///
/// `NumericValue` conforms to `Sendable`. All payload types carry value
/// semantics and were verified `Sendable` in Task 4.
public enum NumericValue: Sendable {

    // MARK: - Cases

    /// A real scalar (Double-precision floating-point).
    case scalar(Double)

    /// A complex scalar.
    case complex(Complex)

    /// A real matrix (may also represent a real vector when `cols == 1`).
    case matrix(LinAlg.Matrix)

    /// A complex matrix (may also represent a complex vector when `cols == 1`).
    case complexMatrix(LinAlg.ComplexMatrix)
}

// MARK: - CustomStringConvertible

extension NumericValue: CustomStringConvertible {

    /// A short, human-readable description of the value — suitable for
    /// debugging and logging. Each case renders distinctly so a reader can
    /// identify the kind and size at a glance without decoding the enum.
    ///
    /// Examples:
    /// ```
    /// NumericValue.scalar(3.14)            → "scalar(3.14)"
    /// NumericValue.complex(Complex(1, 2)) → "complex(1.0+2.0i)"
    /// NumericValue.matrix(2×3 matrix)     → "matrix(2x3)"
    /// NumericValue.complexMatrix(1×4)     → "complexMatrix(1x4)"
    /// ```
    public var description: String {
        switch self {
        case .scalar(let x):
            return "scalar(\(x))"
        case .complex(let z):
            return "complex(\(z))"
        case .matrix(let m):
            return "matrix(\(m.rows)x\(m.cols))"
        case .complexMatrix(let cm):
            return "complexMatrix(\(cm.rows)x\(cm.cols))"
        }
    }
}
