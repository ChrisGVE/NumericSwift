//
//  NumericDiagnostic.swift
//  NumericSwift
//
//  Diagnostics types for the NumericSwift E2E functional workbench.
//
//  These types serve two roles:
//    1. Public API — result structs carry a `diagnostics` field so callers can
//       inspect computed-at-runtime warnings without turning them into thrown errors.
//    2. Workbench gate — `WorkbenchGateTests` asserts zero `outsideEnvelope`
//       diagnostics across the full fixture corpus, so regressions fail CI.
//
//  Design decisions
//  ─────────────────
//  • `NumericDiagnostic` is an enum rather than a class hierarchy so exhaustive
//    `switch` statements compile-check coverage when new cases are added.
//  • `Diagnosed<Value>` is a simple recoverable wrapper — not a thrown error —
//    because many callers want the best-effort result *and* the warning.
//  • Both types conform to `Sendable` so they can cross actor boundaries safely
//    alongside `NumericValue` and `LinAlg.Matrix`, which are also `Sendable`.
//
//  Licensed under the Apache License, Version 2.0.
//

// MARK: - NumericDiagnostic

/// A structured diagnostic attached to a numeric result.
///
/// Diagnostics are **non-fatal**: the associated value is still valid as a
/// best-effort answer, but callers should inspect the diagnostic to decide
/// whether the result meets their accuracy requirements.
///
/// ## Severity model
///
/// | Case | Meaning |
/// |------|---------|
/// | `outsideEnvelope` | The computation was called outside the domain where its declared accuracy holds. The result may be wrong. |
/// | `precisionDegraded` | The algorithm converged but achieved fewer significant digits than full double precision. |
/// | `nonConvergence` | An iterative algorithm did not converge within its iteration budget. The returned value is the last iterate. |
///
/// ## Self-awareness gate
///
/// `WorkbenchGateTests` asserts that no fixture produces an `outsideEnvelope`
/// diagnostic when the input is within the declared limitation envelope. A
/// violation means NumericSwift either mis-classifies safe inputs or has a
/// correctness regression.
///
/// ## Example
///
/// ```swift
/// let result = someIntegration(f, a: a, b: b)
/// if result.diagnostics.contains(where: \.isOutsideEnvelope) {
///     print("Warning: result may be inaccurate")
/// }
/// ```
///
/// ## Source-stability policy
///
/// This enum is **not** `@frozen`: additional cases may be introduced in a future
/// minor release (a new diagnostic kind such as overflow or cancellation). The
/// enum form was chosen precisely so that adding a case is a compile-checked
/// event. Consumers that `switch` over `NumericDiagnostic` should therefore
/// include a `default:` (or `@unknown default:`) branch so a future case
/// addition does not break their build. The three existing cases will not be
/// renamed or removed.
public enum NumericDiagnostic: Sendable, Equatable, CustomStringConvertible {

    // MARK: Cases

    /// The input was outside the domain where the method's declared accuracy holds.
    ///
    /// - Parameters:
    ///   - method: Short identifier for the algorithm or method (e.g. `"tDist.ppf"`).
    ///   - reason: Human-readable description of the violated precondition
    ///             (e.g. `"|p| > 0.9999 — tail precision is ~5 digits"`).
    case outsideEnvelope(method: String, reason: String)

    /// The algorithm converged but achieved fewer significant digits than
    /// full `Double` precision (~15–16 digits).
    ///
    /// - Parameters:
    ///   - method: Short identifier for the algorithm (e.g. `"erfinv"`).
    ///   - approxDigits: Approximate number of correct significant digits achieved.
    case precisionDegraded(method: String, approxDigits: Int)

    /// An iterative algorithm exhausted its iteration budget without converging.
    ///
    /// The associated value (in the containing result struct) is the last iterate
    /// and may be a useful starting point for further refinement.
    ///
    /// - Parameters:
    ///   - method: Short identifier for the algorithm (e.g. `"bisect"`).
    ///   - reason: Human-readable description of the convergence failure
    ///             (e.g. `"exceeded maxiter=100; residual=3.4e-7"`).
    case nonConvergence(method: String, reason: String)

    // MARK: Helpers

    /// Returns `true` when this diagnostic indicates the computation was outside
    /// the method's declared accuracy envelope.
    ///
    /// Use this to implement the workbench self-awareness gate:
    ///
    /// ```swift
    /// let violations = result.diagnostics.filter(\.isOutsideEnvelope)
    /// XCTAssert(violations.isEmpty, "Envelope violation: \(violations)")
    /// ```
    public var isOutsideEnvelope: Bool {
        if case .outsideEnvelope = self { return true }
        return false
    }

    /// Returns `true` when this diagnostic indicates an iterative algorithm
    /// exhausted its iteration budget without converging.
    public var isNonConvergence: Bool {
        if case .nonConvergence = self { return true }
        return false
    }

    /// Returns `true` when this diagnostic indicates the result achieved fewer
    /// significant digits than full `Double` precision.
    public var isPrecisionDegraded: Bool {
        if case .precisionDegraded = self { return true }
        return false
    }

    // MARK: CustomStringConvertible

    /// A human-readable description of the diagnostic.
    ///
    /// Format: `[<severity>] <method>: <detail>`
    public var description: String {
        switch self {
        case let .outsideEnvelope(method, reason):
            return "[outsideEnvelope] \(method): \(reason)"
        case let .precisionDegraded(method, approxDigits):
            return "[precisionDegraded] \(method): ~\(approxDigits) significant digit(s)"
        case let .nonConvergence(method, reason):
            return "[nonConvergence] \(method): \(reason)"
        }
    }
}

// MARK: - Diagnosed<Value>

/// A recoverable wrapper pairing a best-effort computed value with zero or more
/// ``NumericDiagnostic`` entries.
///
/// `Diagnosed<T>` is the wave-1 public surface for **recoverable** diagnostics.
/// Result structs (`QuadResult`, `OLSResult`, etc.) carry their own flat
/// `diagnostics: [NumericDiagnostic]` field for the common single-result case;
/// `Diagnosed<T>` is for APIs that want to express uncertainty at the type level.
///
/// ## Reliability check
///
/// ```swift
/// let d: Diagnosed<Double> = someComputation(x)
/// guard d.isReliable else {
///     // at least one outsideEnvelope diagnostic — handle with care
///     print(d.diagnostics)
/// }
/// use(d.value)
/// ```
///
/// ## Transforming the value
///
/// ```swift
/// let radians: Diagnosed<Double> = angleInDegrees.map { $0 * .pi / 180 }
/// ```
///
/// - Note: `Value` must itself conform to `Sendable` because `Diagnosed` is
///   `Sendable`. This is satisfied by all numeric types in NumericSwift
///   (`Double`, ``Complex``, ``LinAlg/Matrix``, etc.).
public struct Diagnosed<Value: Sendable>: Sendable {

    // MARK: Stored properties

    /// The best-effort computed value.
    ///
    /// Always present, even when `diagnostics` is non-empty.
    public let value: Value

    /// Zero or more diagnostics produced during the computation.
    ///
    /// Empty means the computation completed without any detected issues.
    public let diagnostics: [NumericDiagnostic]

    // MARK: Initialiser

    /// Create a `Diagnosed` wrapper.
    ///
    /// - Parameters:
    ///   - value: The computed result.
    ///   - diagnostics: Diagnostics from the computation. Defaults to empty.
    public init(_ value: Value, diagnostics: [NumericDiagnostic] = []) {
        self.value = value
        self.diagnostics = diagnostics
    }

    // MARK: Reliability

    /// `true` when no diagnostic is an ``NumericDiagnostic/outsideEnvelope(method:reason:)``
    ///
    /// `precisionDegraded` and `nonConvergence` diagnostics do **not** make a
    /// result unreliable — the value is still a valid (if imprecise or
    /// non-converged) answer. Only `outsideEnvelope` indicates that the method
    /// may produce a fundamentally incorrect result.
    public var isReliable: Bool {
        diagnostics.allSatisfy { !$0.isOutsideEnvelope }
    }

    // MARK: Functor

    /// Transform the value, carrying diagnostics through unchanged.
    ///
    /// - Parameter transform: A closure applied to `value`.
    /// - Returns: A new `Diagnosed` wrapping the transformed value with the
    ///            same `diagnostics`.
    public func map<U: Sendable>(_ transform: (Value) -> U) -> Diagnosed<U> {
        Diagnosed<U>(transform(value), diagnostics: diagnostics)
    }
}

// MARK: - Conditional Equatable

/// `Diagnosed` compares equal when both the wrapped value and the diagnostic
/// list are equal. Synthesised when `Value` is `Equatable`, so callers and tests
/// can assert on a `Diagnosed` result directly instead of unwrapping `.value`
/// and `.diagnostics` separately.
extension Diagnosed: Equatable where Value: Equatable {}
