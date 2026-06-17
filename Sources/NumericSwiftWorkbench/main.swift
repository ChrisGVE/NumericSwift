//
//  main.swift
//  NumericSwiftWorkbench
//
//  Entry point for the NumericSwift E2E functional workbench executable.
//
//  Usage:
//    .build/debug/NumericSwiftWorkbench                      # all domains
//    .build/debug/NumericSwiftWorkbench integration          # selected domains
//    .build/debug/NumericSwiftWorkbench integration optimization
//
//  Fixture files are read from:
//    Tests/NumericSwiftTests/Fixtures/workbench/<domain>.json
//
//  Exit code:
//    0 — all cases pass (or no fixtures found yet)
//    1 — one or more envelope violations detected
//
//  Build:
//    swift build --product NumericSwiftWorkbench
//
//  NUMERICSWIFT_REGENERATE_WORKBENCH=1 triggers the Python oracle regeneration
//  path (implemented in wave 2; guarded here so the variable is documented).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

// MARK: - Regeneration guard

if ProcessInfo.processInfo.environment["NUMERICSWIFT_REGENERATE_WORKBENCH"] == "1" {
    fputs(
        """
        NUMERICSWIFT_REGENERATE_WORKBENCH=1 is set.
        Fixture regeneration via Python oracle scripts is a wave-2 deliverable.
        Run the scripts under Tools/workbench_oracles/ directly with Python + SciPy.
        """,
        stderr
    )
    exit(0)
}

// MARK: - Fixture directory resolution

/// Resolve the fixtures directory relative to this source file's location.
///
/// In debug builds the executable lives at .build/debug/NumericSwiftWorkbench;
/// fixtures are at Tests/NumericSwiftTests/Fixtures/workbench/ relative to the
/// package root (two levels up from .build/).
func resolveFixturesDirectory() -> URL? {
    // Try #file-based resolution first (available when run from the package root).
    let sourceURL = URL(fileURLWithPath: #file)
    // #file → Sources/NumericSwiftWorkbench/main.swift
    // Go up: NumericSwiftWorkbench → Sources → package root
    let packageRoot = sourceURL
        .deletingLastPathComponent() // main.swift → NumericSwiftWorkbench
        .deletingLastPathComponent() // NumericSwiftWorkbench → Sources
        .deletingLastPathComponent() // Sources → package root

    let candidate = packageRoot
        .appendingPathComponent("Tests")
        .appendingPathComponent("NumericSwiftTests")
        .appendingPathComponent("Fixtures")
        .appendingPathComponent("workbench")

    if FileManager.default.fileExists(atPath: candidate.path) {
        return candidate
    }

    // Fallback: current working directory / Tests / ... / workbench
    let cwdCandidate = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("Tests")
        .appendingPathComponent("NumericSwiftTests")
        .appendingPathComponent("Fixtures")
        .appendingPathComponent("workbench")

    if FileManager.default.fileExists(atPath: cwdCandidate.path) {
        return cwdCandidate
    }

    return nil
}

// MARK: - Fixture loading

/// Load workbench fixture files for the requested domains.
///
/// - Parameters:
///   - domains: Domain names to load. Empty means load all available.
///   - directory: URL of the fixture directory.
/// - Returns: Dictionary mapping domain name → array of cases.
func loadFixtures(
    domains: [String],
    from directory: URL
) -> [String: [WorkbenchCase]] {
    var result: [String: [WorkbenchCase]] = [:]
    let fm = FileManager.default

    guard let files = try? fm.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: nil,
        options: .skipsHiddenFiles
    ) else { return result }

    let jsonFiles = files.filter { $0.pathExtension == "json" }

    for file in jsonFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
        let domain = file.deletingPathExtension().lastPathComponent
        guard domains.isEmpty || domains.contains(domain) else { continue }

        guard let data = try? Data(contentsOf: file) else {
            fputs("Warning: cannot read fixture file \(file.path)\n", stderr)
            continue
        }

        let decoder = JSONDecoder()
        guard let cases = try? decoder.decode(FixtureFile.self, from: data) else {
            fputs("Warning: cannot decode fixture file \(file.path)\n", stderr)
            continue
        }

        result[domain] = cases
    }

    return result
}

// MARK: - Strategy registry setup

func buildStrategyRegistry() -> StrategyRegistry {
    var registry = StrategyRegistry()
    registerIntegrationStrategies(into: &registry)
    // Wave 2: register remaining domain strategies here.
    return registry
}

// MARK: - Per-case runner

func runCase(
    _ wbCase: WorkbenchCase,
    registry: StrategyRegistry,
    envelopeRegistry: EnvelopeRegistry
) -> CaseResult {
    var strategyOutcomes: [String: StrategyOutcome] = [:]

    for strategyID in wbCase.strategies {
        let fn = registry[strategyID]
        let stratResult = fn?(wbCase.inputs)

        let value = stratResult?.value
        let diagnostics = stratResult?.diagnostics ?? []
        let absError = value.map { abs($0 - wbCase.oracle.value) }
        let declaredTol = wbCase.tol[strategyID]
            ?? envelopeRegistry[strategy: strategyID, tier: wbCase.tier]?.maxAbsError

        let withinTol: Bool
        if let err = absError, let tol = declaredTol {
            withinTol = envelopeRegistry[strategy: strategyID, tier: wbCase.tier]
                .map { $0.inEnvelope(err) }
                ?? (err <= tol)
        } else {
            withinTol = value != nil
        }

        var violations: [Violation] = []
        if let err = absError, let tol = declaredTol, !withinTol {
            violations.append(Violation(
                caseID: wbCase.id,
                strategy: strategyID,
                tier: wbCase.tier,
                absError: err,
                declaredMaxError: tol,
                isSelfAwarenessFailure: false
            ))
        }

        strategyOutcomes[strategyID] = StrategyOutcome(
            strategy: strategyID,
            tier: wbCase.tier,
            value: value,
            absError: absError,
            declaredTol: declaredTol,
            withinTol: withinTol,
            diagnostics: diagnostics,
            violations: violations
        )
    }

    return CaseResult(workbenchCase: wbCase, strategyResults: strategyOutcomes)
}

// MARK: - Main

let requestedDomains = Array(CommandLine.arguments.dropFirst())
let registry = buildStrategyRegistry()

// Locate fixtures directory.
guard let fixturesDir = resolveFixturesDirectory() else {
    print(ReportRenderer.renderNoFixtures())
    exit(0)
}

// Load fixture files.
let allFixtures = loadFixtures(domains: requestedDomains, from: fixturesDir)

guard !allFixtures.isEmpty else {
    print(ReportRenderer.renderNoFixtures())
    exit(0)
}

// Build per-domain envelope registries.
// Wave 1: only Integration is wired; others pass through with empty registries.
func envelopeRegistry(for domain: String) -> EnvelopeRegistry {
    switch domain {
    case "integration": return makeIntegrationEnvelopeRegistry()
    default: return EnvelopeRegistry()
    }
}

// Run all domains.
var domainReports: [DomainReport] = []

for domain in allFixtures.keys.sorted() {
    let cases = allFixtures[domain] ?? []
    let envReg = envelopeRegistry(for: domain)

    let caseResults = cases.map { runCase($0, registry: registry, envelopeRegistry: envReg) }
    let domainReport = DomainReport(domain: domain, caseResults: caseResults)
    domainReports.append(domainReport)

    print(ReportRenderer.renderDomain(domainReport))
}

let summary = SummaryReport(domainReports: domainReports)
print(ReportRenderer.renderSummary(summary))

exit(summary.hasFailed ? 1 : 0)
