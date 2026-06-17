//
//  Envelope.swift
//  NumericSwiftWorkbench
//
//  Limitation-envelope registry for the NumericSwift E2E functional workbench.
//
//  An "envelope" is the declared accuracy contract for one (strategy, tier) pair:
//  the maximum absolute error a strategy is expected to achieve on cases of that
//  difficulty tier. The workbench checks every result against its envelope and
//  flags two failure modes:
//
//    • Violation   — result is *worse* than declared (a real regression or bug).
//    • Unexpectedly better — result is consistently better, suggesting the
//      envelope is stale and should be tightened.
//
//  ## Wave coverage
//
//  Wave 1 (this file): types + a worked Integration example.
//  Wave 2: full per-domain registry population.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

// MARK: - EnvelopeEntry

/// The declared accuracy contract for one (strategy, tier) combination.
///
/// The `inEnvelope` predicate is the authoritative check: it takes the actual
/// absolute error and returns `true` when the result is within the declared
/// envelope. For most strategies this is simply `error <= maxAbsError`, but
/// the closure form allows richer contracts (e.g. relative error, ULP bounds).
///
/// ### Example
///
/// ```swift
/// let entry = EnvelopeEntry(
///     strategy: "quad",
///     tier: .hard,
///     maxAbsError: 1e-10,
///     description: "Gauss-Kronrod adaptive quadrature on smooth hard cases"
/// )
/// ```
public struct EnvelopeEntry: Sendable {

    /// Strategy identifier (must match keys in ``WorkbenchCase/tol``).
    public let strategy: String

    /// Difficulty tier this entry applies to.
    public let tier: CaseTier

    /// Maximum absolute error declared for this (strategy, tier) pair.
    ///
    /// Used as the default `inEnvelope` predicate: `absError <= maxAbsError`.
    public let maxAbsError: Double

    /// Human-readable rationale for the declared accuracy.
    public let description: String

    /// Returns `true` when `absError` is within the declared envelope.
    ///
    /// Defaults to `absError <= maxAbsError`. Override via ``init(strategy:tier:maxAbsError:description:inEnvelope:)``
    /// for non-standard contracts.
    public let inEnvelope: @Sendable (Double) -> Bool

    // MARK: Initialisers

    /// Create an entry with the default `<= maxAbsError` predicate.
    public init(
        strategy: String,
        tier: CaseTier,
        maxAbsError: Double,
        description: String
    ) {
        self.strategy = strategy
        self.tier = tier
        self.maxAbsError = maxAbsError
        self.description = description
        self.inEnvelope = { $0 <= maxAbsError }
    }

    /// Create an entry with a custom predicate.
    ///
    /// Use when the accuracy contract is not a simple absolute-error bound
    /// (e.g. for ULP-based or relative-error contracts).
    public init(
        strategy: String,
        tier: CaseTier,
        maxAbsError: Double,
        description: String,
        inEnvelope: @escaping @Sendable (Double) -> Bool
    ) {
        self.strategy = strategy
        self.tier = tier
        self.maxAbsError = maxAbsError
        self.description = description
        self.inEnvelope = inEnvelope
    }
}

// MARK: - EnvelopeKey

/// Lookup key for the envelope registry.
struct EnvelopeKey: Hashable {
    let strategy: String
    let tier: CaseTier
}

// MARK: - EnvelopeRegistry

/// A per-domain registry of ``EnvelopeEntry`` values.
///
/// The registry maps (strategy, tier) → entry. A domain's `Suite` populates
/// this registry; the workbench runner uses it to check results.
///
/// ### Usage
///
/// ```swift
/// var registry = EnvelopeRegistry()
/// registry.register(EnvelopeEntry(strategy: "quad", tier: .hard, maxAbsError: 1e-10, ...))
/// if let entry = registry[strategy: "quad", tier: .hard] {
///     let ok = entry.inEnvelope(absError)
/// }
/// ```
public struct EnvelopeRegistry: Sendable {

    private var entries: [EnvelopeKey: EnvelopeEntry] = [:]

    public init() {}

    /// Register an entry. Later registrations for the same (strategy, tier) overwrite earlier ones.
    public mutating func register(_ entry: EnvelopeEntry) {
        entries[EnvelopeKey(strategy: entry.strategy, tier: entry.tier)] = entry
    }

    /// Look up the entry for a (strategy, tier) pair.
    public subscript(strategy strategy: String, tier tier: CaseTier) -> EnvelopeEntry? {
        entries[EnvelopeKey(strategy: strategy, tier: tier)]
    }

    /// All registered entries, in insertion-stable order.
    public var allEntries: [EnvelopeEntry] {
        entries.values.sorted { lhs, rhs in
            lhs.strategy == rhs.strategy
                ? lhs.tier.rawValue < rhs.tier.rawValue
                : lhs.strategy < rhs.strategy
        }
    }
}

// MARK: - Violation

/// A single envelope violation: a strategy result that exceeded its declared
/// accuracy on a particular fixture case.
///
/// The workbench collects violations across all domains and cases. The
/// `WorkbenchGateTests` XCTest asserts `violations.isEmpty`.
public struct Violation: Sendable, CustomStringConvertible {

    /// The fixture case that produced the violation.
    public let caseID: String

    /// The strategy whose result exceeded its envelope.
    public let strategy: String

    /// Difficulty tier of the case.
    public let tier: CaseTier

    /// Absolute error of the strategy on this case.
    public let absError: Double

    /// The declared maximum absolute error for this (strategy, tier).
    public let declaredMaxError: Double

    /// Whether this is a ``SelfAwarenessFailure`` (case was in-envelope
    /// yet got an ``NumericDiagnostic/outsideEnvelope(method:reason:)`` diagnostic).
    public let isSelfAwarenessFailure: Bool

    public var description: String {
        let kind = isSelfAwarenessFailure ? "SELF-AWARENESS" : "ENVELOPE"
        return "[\(kind)] \(caseID)/\(strategy) (\(tier.rawValue)): "
            + "absError=\(absError) > declared=\(declaredMaxError)"
    }
}

// MARK: - SelfAwarenessFailure

/// A self-awareness failure: the library emitted an
/// ``NumericDiagnostic/outsideEnvelope(method:reason:)`` diagnostic on a case
/// that is actually **within** the declared envelope.
///
/// Per WORKBENCH.md §5, the self-awareness gate checks that the library
/// correctly classifies inputs: it must not warn for safe inputs (false positive)
/// and must not fail to warn for unsafe inputs (false negative).
///
/// Wave 2 populates detection logic; wave 1 defines the model only.
public struct SelfAwarenessFailure: Sendable, CustomStringConvertible {

    /// The fixture case where the mismatch occurred.
    public let caseID: String

    /// The strategy that emitted the spurious or missing diagnostic.
    public let strategy: String

    /// `true` = library warned on an in-envelope input (false positive).
    /// `false` = library did not warn on an out-of-envelope input (false negative).
    public let isFalsePositive: Bool

    /// The diagnostics actually emitted.
    public let emittedDiagnostics: [NumericDiagnostic]

    public var description: String {
        let kind = isFalsePositive ? "FALSE-POSITIVE" : "FALSE-NEGATIVE"
        return "[\(kind)] \(caseID)/\(strategy): \(emittedDiagnostics)"
    }
}

// MARK: - Integration envelope example (wave 1 worked example)

/// Pre-built ``EnvelopeRegistry`` for the Integration (quadrature) domain.
///
/// This is the wave-1 worked example. The envelopes reflect:
///  - `quad` (Gauss-Kronrod adaptive): full double precision on smooth integrands.
///  - `romberg` (Richardson extrapolation): slightly looser on non-smooth integrands.
///  - `simps` (Simpson's rule with uniform spacing): exact for polynomials ≤ degree 2;
///    rough approximation for oscillatory or non-uniform inputs.
///  - `trapz` (trapezoidal rule): first-order; exact only for linear functions.
///  - `fixed_quad` (Gauss-Legendre fixed-point): depends on order; hard cases may need order ≥ 5.
///
/// Wave 2 will populate registries for all other domains.
public func makeIntegrationEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()

    for tier: CaseTier in [.trivial, .hard, .edge] {
        let quadTol: Double = tier == .trivial ? 1e-14 : tier == .hard ? 1e-10 : 1e-6
        reg.register(EnvelopeEntry(
            strategy: "quad",
            tier: tier,
            maxAbsError: quadTol,
            description: "Gauss-Kronrod 15-point adaptive quadrature — \(tier.rawValue) cases"
        ))

        let rombergTol: Double = tier == .trivial ? 1e-12 : tier == .hard ? 1e-8 : 1e-4
        reg.register(EnvelopeEntry(
            strategy: "romberg",
            tier: tier,
            maxAbsError: rombergTol,
            description: "Romberg (Richardson extrapolation) — \(tier.rawValue) cases"
        ))

        let simpsTol: Double = tier == .trivial ? 1e-12 : tier == .hard ? 1e-3 : 1e-1
        reg.register(EnvelopeEntry(
            strategy: "simps",
            tier: tier,
            maxAbsError: simpsTol,
            description: "Simpson's rule (uniform spacing) — \(tier.rawValue) cases"
        ))

        let trapzTol: Double = tier == .trivial ? 1e-10 : tier == .hard ? 1e-2 : 1e-1
        reg.register(EnvelopeEntry(
            strategy: "trapz",
            tier: tier,
            maxAbsError: trapzTol,
            description: "Trapezoidal rule — \(tier.rawValue) cases"
        ))

        let fixedQuadTol: Double = tier == .trivial ? 1e-12 : tier == .hard ? 1e-6 : 1e-3
        reg.register(EnvelopeEntry(
            strategy: "fixed_quad",
            tier: tier,
            maxAbsError: fixedQuadTol,
            description: "Gauss-Legendre fixed quadrature — \(tier.rawValue) cases"
        ))
    }

    return reg
}


