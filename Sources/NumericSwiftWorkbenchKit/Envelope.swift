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
//  ## Architecture
//
//  This file defines the envelope infrastructure types:
//    • `EnvelopeEntry` — the accuracy contract for one (strategy, tier) pair,
//      carrying a `maxAbsError` and an `inEnvelope` predicate closure.
//    • `EnvelopeRegistry` — a `[EnvelopeKey: EnvelopeEntry]` dictionary wrapper.
//    • `Violation` — an accuracy miss: in-envelope result exceeded its declared tol.
//    • `SelfAwarenessFailure` — a hard-gate miss: spurious or missing diagnostic.
//
//  Per-domain envelope registrations (the actual (strategy, tier) bounds) live in
//  `Domains/<Domain>.swift`, each exposed through that domain's ``DomainSuite``.
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

/// A single **accuracy** violation: an in-envelope strategy result whose error
/// exceeded its declared tolerance for the case tier.
///
/// Per WORKBENCH.md §5/§7 an accuracy violation is a *reported flag*, not the
/// hard gate — the hard gate is ``SelfAwarenessFailure``. The workbench collects
/// accuracy violations for the report; they do not fail CI on their own.
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

    public init(
        caseID: String,
        strategy: String,
        tier: CaseTier,
        absError: Double,
        declaredMaxError: Double
    ) {
        self.caseID = caseID
        self.strategy = strategy
        self.tier = tier
        self.absError = absError
        self.declaredMaxError = declaredMaxError
    }

    public var description: String {
        "[ACCURACY] \(caseID)/\(strategy) (\(tier.rawValue)): "
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

// Per-domain envelope registries live in `Domains/<Domain>.swift`, each exposed
// through that domain's ``DomainSuite``. See `Domains/Integration.swift` for the
// reference implementation.


