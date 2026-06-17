//
//  Report.swift
//  NumericSwiftWorkbenchKit
//
//  Per-domain table + summary rendering for the NumericSwift E2E workbench.
//
//  Output format (WORKBENCH.md §7):
//    • Per-domain table: case | tier | strategy | error | tol | inEnv | diag | status
//    • Per-domain rollup: N cases, K self-awareness failures, M accuracy flags
//    • Final summary: N domains, M cases, K self-awareness failures
//      (exit nonzero on self-awareness failures — the hard gate)
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

// MARK: - StrategyOutcome

/// The outcome of running one strategy on one case.
public struct StrategyOutcome: Sendable {

    public let strategy: String
    public let tier: CaseTier

    /// The computed value, or nil when the strategy returned nil (recorded as ERROR).
    public let value: Double?

    /// Absolute error vs. oracle (nil when `value` is nil).
    public let absError: Double?

    /// The declared tolerance for this (strategy, tier).
    public let declaredTol: Double?

    /// Whether the fixture declared this case in-envelope for this strategy.
    public let inEnvelope: Bool

    /// Diagnostics emitted by the library.
    public let diagnostics: [NumericDiagnostic]

    /// Accuracy violations (reported flags, not the hard gate).
    public let accuracyViolations: [Violation]

    /// Self-awareness failures (the hard gate).
    public let selfAwarenessFailures: [SelfAwarenessFailure]

    public init(
        strategy: String,
        tier: CaseTier,
        value: Double?,
        absError: Double?,
        declaredTol: Double?,
        inEnvelope: Bool,
        diagnostics: [NumericDiagnostic],
        accuracyViolations: [Violation],
        selfAwarenessFailures: [SelfAwarenessFailure]
    ) {
        self.strategy = strategy
        self.tier = tier
        self.value = value
        self.absError = absError
        self.declaredTol = declaredTol
        self.inEnvelope = inEnvelope
        self.diagnostics = diagnostics
        self.accuracyViolations = accuracyViolations
        self.selfAwarenessFailures = selfAwarenessFailures
    }

    /// Short status label for the report table.
    public var statusLabel: String {
        if value == nil { return "ERROR" }
        if !selfAwarenessFailures.isEmpty { return "SELF-AWARE✗" }
        if !accuracyViolations.isEmpty { return "ACC-FLAG" }
        return "PASS"
    }

    /// `true` if the library emitted any `outsideEnvelope` diagnostic.
    public var emittedLimitationDiagnostic: Bool {
        diagnostics.contains(where: \.isOutsideEnvelope)
    }
}

// MARK: - CaseResult

/// The outcome of running all strategies on a single ``WorkbenchCase``.
public struct CaseResult: Sendable {

    /// The fixture case.
    public let workbenchCase: WorkbenchCase

    /// Per-strategy outcomes, keyed by strategy id.
    public let strategyResults: [String: StrategyOutcome]

    public init(workbenchCase: WorkbenchCase, strategyResults: [String: StrategyOutcome]) {
        self.workbenchCase = workbenchCase
        self.strategyResults = strategyResults
    }

    public var accuracyViolations: [Violation] {
        strategyResults.values.flatMap(\.accuracyViolations)
    }

    public var selfAwarenessFailures: [SelfAwarenessFailure] {
        strategyResults.values.flatMap(\.selfAwarenessFailures)
    }
}

// MARK: - DomainReport

/// Aggregated results for all cases in one domain.
public struct DomainReport: Sendable {

    public let domain: String
    public let caseResults: [CaseResult]

    public init(domain: String, caseResults: [CaseResult]) {
        self.domain = domain
        self.caseResults = caseResults
    }

    public var totalCases: Int { caseResults.count }

    public var selfAwarenessFailures: [SelfAwarenessFailure] {
        caseResults.flatMap(\.selfAwarenessFailures)
    }

    public var accuracyViolations: [Violation] {
        caseResults.flatMap(\.accuracyViolations)
    }

    public var selfAwarenessFailureCount: Int { selfAwarenessFailures.count }
    public var accuracyViolationCount: Int { accuracyViolations.count }

    /// Cases that are fully clean — no self-awareness failure and no accuracy flag.
    public var passCount: Int {
        caseResults.filter {
            $0.selfAwarenessFailures.isEmpty && $0.accuracyViolations.isEmpty
        }.count
    }
}

// MARK: - SummaryReport

/// Final summary across all domains.
public struct SummaryReport: Sendable {

    public let domainReports: [DomainReport]

    public init(domainReports: [DomainReport]) {
        self.domainReports = domainReports
    }

    public var totalDomains: Int { domainReports.count }

    public var totalCases: Int {
        domainReports.reduce(0) { $0 + $1.totalCases }
    }

    public var totalSelfAwarenessFailures: Int {
        domainReports.reduce(0) { $0 + $1.selfAwarenessFailureCount }
    }

    public var totalAccuracyViolations: Int {
        domainReports.reduce(0) { $0 + $1.accuracyViolationCount }
    }

    /// The hard gate: `true` when any self-awareness failure exists. Accuracy
    /// flags do NOT fail the gate (WORKBENCH.md §5/§7).
    public var hasFailed: Bool { totalSelfAwarenessFailures > 0 }

    /// All self-awareness failures across all domains (for the gate assertion message).
    public var allSelfAwarenessFailures: [SelfAwarenessFailure] {
        domainReports.flatMap(\.selfAwarenessFailures)
    }
}

// MARK: - ReportRenderer

/// Renders workbench reports to a `String`.
public enum ReportRenderer {

    // MARK: Per-domain table

    public static func renderDomain(_ report: DomainReport) -> String {
        var lines: [String] = []
        lines.append("")
        lines.append("=== Domain: \(report.domain) ===")

        let col = columnWidths()
        lines.append(headerRow(col))
        lines.append(separator(col))

        for caseResult in report.caseResults {
            for (strategy, outcome) in caseResult.strategyResults.sorted(by: { $0.key < $1.key }) {
                lines.append(tableRow(
                    id: caseResult.workbenchCase.id,
                    tier: caseResult.workbenchCase.tier.rawValue,
                    strategy: strategy,
                    error: outcome.absError.map(formatSci) ?? "n/a",
                    tol: outcome.declaredTol.map(formatSci) ?? "n/a",
                    inEnv: outcome.inEnvelope ? "in" : "OUT",
                    diag: outcome.emittedLimitationDiagnostic ? "yes" : "no",
                    status: outcome.statusLabel,
                    col: col
                ))
            }
        }

        lines.append(separator(col))
        lines.append(
            "  \(report.domain): \(report.totalCases) case(s), "
            + "\(report.passCount) PASS, "
            + "\(report.selfAwarenessFailureCount) self-awareness failure(s), "
            + "\(report.accuracyViolationCount) accuracy flag(s)"
        )
        return lines.joined(separator: "\n")
    }

    // MARK: Summary

    public static func renderSummary(_ report: SummaryReport) -> String {
        var lines: [String] = []
        lines.append("")
        lines.append("╔══════════════════════════════════════════════════════╗")
        lines.append("║          NumericSwift Workbench — Summary            ║")
        lines.append("╚══════════════════════════════════════════════════════╝")
        lines.append("")

        for domain in report.domainReports {
            let status = domain.selfAwarenessFailureCount == 0 ? "✓" : "✗"
            lines.append(
                "  \(status) \(domain.domain): \(domain.totalCases) case(s), "
                + "\(domain.selfAwarenessFailureCount) self-awareness failure(s), "
                + "\(domain.accuracyViolationCount) accuracy flag(s)"
            )
        }

        lines.append("")
        lines.append(
            "Total: \(report.totalDomains) domain(s), "
            + "\(report.totalCases) case(s), "
            + "\(report.totalSelfAwarenessFailures) self-awareness failure(s), "
            + "\(report.totalAccuracyViolations) accuracy flag(s)"
        )

        if report.hasFailed {
            lines.append("")
            lines.append("Self-awareness failures (the gated condition):")
            for failure in report.allSelfAwarenessFailures {
                lines.append("  \(failure)")
            }
            lines.append("")
            lines.append("RESULT: FAIL — \(report.totalSelfAwarenessFailures) self-awareness failure(s).")
        } else if report.totalDomains == 0 {
            lines.append("")
            lines.append("RESULT: 0 domains loaded — no fixtures found.")
        } else {
            lines.append("")
            lines.append("RESULT: PASS — library is self-aware across all out-of-envelope cases.")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: Empty report

    public static func renderNoFixtures() -> String {
        """

        NumericSwift Workbench
        ───────────────────────
        No fixture files found under Tests/NumericSwiftTests/Fixtures/workbench/.

        Generate fixtures with the Python oracle scripts under Tools/workbench_oracles/.
        """
    }

    // MARK: Private layout helpers

    private struct ColWidths {
        let id: Int; let tier: Int; let strategy: Int
        let error: Int; let tol: Int; let inEnv: Int; let diag: Int; let status: Int
    }

    private static func columnWidths() -> ColWidths {
        ColWidths(id: 40, tier: 7, strategy: 14, error: 11, tol: 11, inEnv: 4, diag: 4, status: 12)
    }

    private static func headerRow(_ col: ColWidths) -> String {
        pad("case", col.id) + " | "
        + pad("tier", col.tier) + " | "
        + pad("strategy", col.strategy) + " | "
        + pad("error", col.error) + " | "
        + pad("tol", col.tol) + " | "
        + pad("env", col.inEnv) + " | "
        + pad("dg", col.diag) + " | "
        + "status"
    }

    private static func separator(_ col: ColWidths) -> String {
        String(repeating: "-",
               count: col.id + col.tier + col.strategy + col.error
                    + col.tol + col.inEnv + col.diag + col.status + 24)
    }

    private static func tableRow(
        id: String, tier: String, strategy: String,
        error: String, tol: String, inEnv: String, diag: String, status: String,
        col: ColWidths
    ) -> String {
        truncPad(id, col.id) + " | "
        + pad(tier, col.tier) + " | "
        + pad(strategy, col.strategy) + " | "
        + pad(error, col.error) + " | "
        + pad(tol, col.tol) + " | "
        + pad(inEnv, col.inEnv) + " | "
        + pad(diag, col.diag) + " | "
        + status
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        s.padding(toLength: width, withPad: " ", startingAt: 0)
    }

    private static func truncPad(_ s: String, _ width: Int) -> String {
        s.count <= width ? pad(s, width) : String(s.prefix(width - 1)) + "…"
    }

    private static func formatSci(_ v: Double) -> String {
        String(format: "%.3e", v)
    }
}
