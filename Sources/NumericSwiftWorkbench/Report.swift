//
//  Report.swift
//  NumericSwiftWorkbench
//
//  Per-domain table + summary rendering for the NumericSwift E2E workbench.
//
//  Output format (WORKBENCH.md §7):
//    • Per-domain table: case | tier | strategy | error | tol | status
//    • Per-domain rollup: N cases, M accuracy flags, K violations
//    • Final summary: N domains, M cases, K violations (exit nonzero on violations)
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

// MARK: - CaseResult

/// The outcome of running all strategies on a single ``WorkbenchCase``.
public struct CaseResult: Sendable {

    /// The fixture case.
    public let workbenchCase: WorkbenchCase

    /// Per-strategy outcomes, keyed by strategy id.
    public let strategyResults: [String: StrategyOutcome]

    /// All violations found across all strategies for this case.
    public var violations: [Violation] {
        strategyResults.values.flatMap(\.violations)
    }
}

// MARK: - StrategyOutcome

/// The outcome of running one strategy on one case.
public struct StrategyOutcome: Sendable {

    public let strategy: String
    public let tier: CaseTier

    /// The computed value, or nil when the strategy returned nil.
    public let value: Double?

    /// Absolute error vs. oracle (nil when `value` is nil).
    public let absError: Double?

    /// The declared tolerance for this (strategy, tier).
    public let declaredTol: Double?

    /// Whether the result is within the declared tolerance.
    public let withinTol: Bool

    /// Diagnostics emitted.
    public let diagnostics: [NumericDiagnostic]

    /// Any violations detected for this outcome.
    public let violations: [Violation]

    /// Short status label for the report table.
    public var statusLabel: String {
        if value == nil { return "ERROR" }
        if violations.contains(where: { $0.isSelfAwarenessFailure }) { return "SELF-AWARENESS" }
        if !violations.isEmpty { return "FAIL" }
        return "PASS"
    }
}

// MARK: - DomainReport

/// Aggregated results for all cases in one domain.
public struct DomainReport: Sendable {

    public let domain: String
    public let caseResults: [CaseResult]

    public var totalCases: Int { caseResults.count }

    public var violations: [Violation] {
        caseResults.flatMap(\.violations)
    }

    public var violationCount: Int { violations.count }

    public var passCount: Int {
        caseResults.filter { $0.violations.isEmpty }.count
    }
}

// MARK: - SummaryReport

/// Final summary across all domains.
public struct SummaryReport: Sendable {

    public let domainReports: [DomainReport]

    public var totalDomains: Int { domainReports.count }

    public var totalCases: Int {
        domainReports.reduce(0) { $0 + $1.totalCases }
    }

    public var totalViolations: Int {
        domainReports.reduce(0) { $0 + $1.violationCount }
    }

    /// `true` when there are violations — the executable exits nonzero.
    public var hasFailed: Bool { totalViolations > 0 }
}

// MARK: - ReportRenderer

/// Renders workbench reports to a `String`.
///
/// Consumers can substitute their own output destination (stdout, file, String
/// buffer for tests) by capturing the rendered string.
public enum ReportRenderer {

    // MARK: Per-domain table

    /// Render the per-domain case table plus domain rollup line.
    public static func renderDomain(_ report: DomainReport) -> String {
        var lines: [String] = []

        lines.append("")
        lines.append("=== Domain: \(report.domain) ===")

        // Table header
        let col = columnWidths()
        lines.append(headerRow(col))
        lines.append(separator(col))

        for caseResult in report.caseResults {
            for (strategy, outcome) in caseResult.strategyResults.sorted(by: { $0.key < $1.key }) {
                let errorStr = outcome.absError.map { formatSci($0) } ?? "n/a"
                let tolStr = outcome.declaredTol.map { formatSci($0) } ?? "n/a"
                lines.append(tableRow(
                    id: caseResult.workbenchCase.id,
                    tier: caseResult.workbenchCase.tier.rawValue,
                    strategy: strategy,
                    error: errorStr,
                    tol: tolStr,
                    status: outcome.statusLabel,
                    col: col
                ))
            }
        }

        lines.append(separator(col))
        lines.append(
            "  \(report.domain): \(report.totalCases) case(s), "
            + "\(report.passCount) PASS, "
            + "\(report.violationCount) violation(s)"
        )

        return lines.joined(separator: "\n")
    }

    // MARK: Summary

    /// Render the final summary across all domains.
    public static func renderSummary(_ report: SummaryReport) -> String {
        var lines: [String] = []

        lines.append("")
        lines.append("╔══════════════════════════════════════════════════════╗")
        lines.append("║          NumericSwift Workbench — Summary            ║")
        lines.append("╚══════════════════════════════════════════════════════╝")
        lines.append("")

        for domain in report.domainReports {
            let status = domain.violationCount == 0 ? "✓" : "✗"
            lines.append(
                "  \(status) \(domain.domain): \(domain.totalCases) case(s), "
                + "\(domain.violationCount) violation(s)"
            )
        }

        lines.append("")
        lines.append(
            "Total: \(report.totalDomains) domain(s), "
            + "\(report.totalCases) case(s), "
            + "\(report.totalViolations) violation(s)"
        )

        if report.hasFailed {
            lines.append("")
            lines.append("RESULT: FAIL — \(report.totalViolations) violation(s) detected.")
        } else if report.totalDomains == 0 {
            lines.append("")
            lines.append("RESULT: 0 domains loaded — no fixtures found yet (wave 1 skeleton).")
        } else {
            lines.append("")
            lines.append("RESULT: PASS — all cases within declared envelopes.")
        }

        return lines.joined(separator: "\n")
    }

    // MARK: Empty report

    /// Rendered message when no fixture files were found.
    public static func renderNoFixtures() -> String {
        """

        NumericSwift Workbench — Wave 1 Foundation
        ───────────────────────────────────────────
        No fixture files found under Tests/NumericSwiftTests/Fixtures/workbench/.
        This is expected for wave 1 — fixture generation is a wave 2 deliverable.

        To generate fixtures, run the Python oracle scripts under Tools/workbench_oracles/
        with NUMERICSWIFT_REGENERATE_WORKBENCH=1 once they are committed (wave 2).

        The workbench harness is built and ready.
        """
    }

    // MARK: Private layout helpers

    private struct ColWidths {
        let id: Int; let tier: Int; let strategy: Int
        let error: Int; let tol: Int; let status: Int
    }

    private static func columnWidths() -> ColWidths {
        ColWidths(id: 40, tier: 8, strategy: 14, error: 12, tol: 12, status: 14)
    }

    private static func headerRow(_ col: ColWidths) -> String {
        pad("case", col.id) + " | "
        + pad("tier", col.tier) + " | "
        + pad("strategy", col.strategy) + " | "
        + pad("error", col.error) + " | "
        + pad("tol", col.tol) + " | "
        + "status"
    }

    private static func separator(_ col: ColWidths) -> String {
        String(repeating: "-", count: col.id + col.tier + col.strategy + col.error + col.tol + col.status + 20)
    }

    private static func tableRow(
        id: String, tier: String, strategy: String,
        error: String, tol: String, status: String,
        col: ColWidths
    ) -> String {
        truncPad(id, col.id) + " | "
        + pad(tier, col.tier) + " | "
        + pad(strategy, col.strategy) + " | "
        + pad(error, col.error) + " | "
        + pad(tol, col.tol) + " | "
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
