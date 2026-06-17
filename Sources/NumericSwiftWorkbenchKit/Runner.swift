//
//  Runner.swift
//  NumericSwiftWorkbenchKit
//
//  The workbench runner: applies every strategy of every fixture case and
//  produces the accuracy + self-awareness verdict (WORKBENCH.md §5/§7).
//
//  This logic lives in the Kit library (not the executable) so the XCTest gate
//  (`WorkbenchGateTests`) can import and run it — an XCTest target cannot import
//  an executable target.
//
//  ## The two verdicts
//
//  For each (case, strategy) the runner records:
//    • an **accuracy** check — is the in-envelope result within `tol`? A miss is a
//      reported ``Violation`` (a flag, not a CI failure).
//    • a **self-awareness** check — the hard gate. Each case tags every strategy
//      `inEnvelope: true|false`:
//        - in-envelope  → the library MUST NOT emit an `outsideEnvelope`
//          diagnostic (a spurious one is a FALSE-POSITIVE ``SelfAwarenessFailure``);
//        - out-of-envelope → the library MUST emit an `outsideEnvelope`
//          diagnostic (a missing one is a FALSE-NEGATIVE ``SelfAwarenessFailure``).
//
//  The gate asserts zero self-awareness failures (`SummaryReport/hasFailed`).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

// MARK: - DomainSuite

/// A self-contained per-domain workbench suite: its strategy registrations and
/// its envelope registry, identified by domain name.
///
/// One ``DomainSuite`` is declared per domain in `Domains/<Domain>.swift` and
/// aggregated into ``Workbench/allSuites``.
public struct DomainSuite: Sendable {

    /// Domain name — must match the fixture file stem (`<domain>.json`) and the
    /// `<domain>` prefix of every case id.
    public let name: String

    /// Populates a registry with this domain's strategy closures.
    public let registerStrategies: @Sendable (inout StrategyRegistry) -> Void

    /// Builds this domain's per-(strategy, tier) envelope registry.
    public let makeEnvelopeRegistry: @Sendable () -> EnvelopeRegistry

    public init(
        name: String,
        registerStrategies: @escaping @Sendable (inout StrategyRegistry) -> Void,
        makeEnvelopeRegistry: @escaping @Sendable () -> EnvelopeRegistry
    ) {
        self.name = name
        self.registerStrategies = registerStrategies
        self.makeEnvelopeRegistry = makeEnvelopeRegistry
    }

    /// This suite's populated strategy registry.
    public var strategyRegistry: StrategyRegistry {
        var reg = StrategyRegistry()
        registerStrategies(&reg)
        return reg
    }
}

// MARK: - Workbench

/// The workbench runner namespace.
public enum Workbench {

    /// Every registered domain suite. Each domain appends its suite here.
    ///
    /// Listed explicitly (Swift has no compile-time target reflection) so the
    /// set of active domains is auditable in one place.
    public static let allSuites: [DomainSuite] = [
        integrationSuite,
        odeSuite,
        optrootSuite,
        interpSuite,
        linsolveSuite,
        spatialknnSuite,
        circlefitSuite,
        clusterSuite,
        statisticsSuite,
    ]

    /// Look up a suite by domain name.
    public static func suite(named name: String) -> DomainSuite? {
        allSuites.first { $0.name == name }
    }

    // MARK: Per-case

    /// Run one fixture case through all of its declared strategies.
    ///
    /// - Parameters:
    ///   - wbCase: The fixture case.
    ///   - registry: The domain's strategy registry.
    ///   - envelopes: The domain's envelope registry (used to score accuracy when
    ///     the case omits an explicit `tol` for a strategy).
    public static func runCase(
        _ wbCase: WorkbenchCase,
        registry: StrategyRegistry,
        envelopes: EnvelopeRegistry
    ) -> CaseResult {
        var outcomes: [String: StrategyOutcome] = [:]

        for strategyID in wbCase.strategies {
            let result = registry[strategyID]?(wbCase.inputs)
            let value = result?.value
            let diagnostics = result?.diagnostics ?? []
            let emittedOutside = diagnostics.contains(where: \.isOutsideEnvelope)
            let declaredInEnvelope = wbCase.isInEnvelope(strategyID)

            // ── Self-awareness verdict (the hard gate) ────────────────────────
            // Only meaningful when the strategy actually produced a value;
            // a nil result is recorded as an ERROR, not a self-awareness verdict.
            var selfAwareness: [SelfAwarenessFailure] = []
            if value != nil {
                if declaredInEnvelope && emittedOutside {
                    selfAwareness.append(SelfAwarenessFailure(
                        caseID: wbCase.id,
                        strategy: strategyID,
                        isFalsePositive: true,
                        emittedDiagnostics: diagnostics
                    ))
                } else if !declaredInEnvelope && !emittedOutside {
                    selfAwareness.append(SelfAwarenessFailure(
                        caseID: wbCase.id,
                        strategy: strategyID,
                        isFalsePositive: false,
                        emittedDiagnostics: diagnostics
                    ))
                }
            }

            // ── Accuracy verdict (reported flag, in-envelope cases only) ──────
            // The per-case `tol` is the authoritative envelope (WORKBENCH.md §5);
            // the domain envelope registry is only a fallback when the fixture
            // omits a tol for this strategy. Display and check use the SAME bound.
            let absError = value.map { abs($0 - wbCase.oracle.value) }
            let envelopeEntry = envelopes[strategy: strategyID, tier: wbCase.tier]
            let declaredTol = wbCase.tol[strategyID] ?? envelopeEntry?.maxAbsError

            var accuracy: [Violation] = []
            if declaredInEnvelope, let err = absError, let tol = declaredTol {
                // Prefer the case tol; only consult the registry predicate when the
                // case did not declare one for this strategy.
                let withinTol: Bool
                if wbCase.tol[strategyID] != nil {
                    withinTol = err <= tol
                } else if let entry = envelopeEntry {
                    withinTol = entry.inEnvelope(err)
                } else {
                    withinTol = true
                }
                if !withinTol {
                    accuracy.append(Violation(
                        caseID: wbCase.id,
                        strategy: strategyID,
                        tier: wbCase.tier,
                        absError: err,
                        declaredMaxError: tol
                    ))
                }
            }

            outcomes[strategyID] = StrategyOutcome(
                strategy: strategyID,
                tier: wbCase.tier,
                value: value,
                absError: absError,
                declaredTol: declaredTol,
                inEnvelope: declaredInEnvelope,
                diagnostics: diagnostics,
                accuracyViolations: accuracy,
                selfAwarenessFailures: selfAwareness
            )
        }

        return CaseResult(workbenchCase: wbCase, strategyResults: outcomes)
    }

    // MARK: Per-domain

    /// Run every case for a domain suite.
    public static func runDomain(_ suite: DomainSuite, cases: [WorkbenchCase]) -> DomainReport {
        let registry = suite.strategyRegistry
        let envelopes = suite.makeEnvelopeRegistry()
        let caseResults = cases.map { runCase($0, registry: registry, envelopes: envelopes) }
        return DomainReport(domain: suite.name, caseResults: caseResults)
    }

    /// Run a set of pre-loaded fixtures (domain name → cases) through their suites.
    ///
    /// Domains without a registered suite are skipped with a warning row in the
    /// returned report (zero cases), so a missing suite never silently passes.
    public static func run(fixtures: [String: [WorkbenchCase]]) -> SummaryReport {
        var reports: [DomainReport] = []
        for domain in fixtures.keys.sorted() {
            let cases = fixtures[domain] ?? []
            guard let suite = suite(named: domain) else {
                // No suite registered: surface as an empty (unrun) domain report.
                reports.append(DomainReport(domain: domain, caseResults: []))
                continue
            }
            reports.append(runDomain(suite, cases: cases))
        }
        return SummaryReport(domainReports: reports)
    }
}
