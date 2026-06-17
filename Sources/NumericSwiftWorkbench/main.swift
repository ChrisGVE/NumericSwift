//
//  main.swift
//  NumericSwiftWorkbench
//
//  Thin CLI over NumericSwiftWorkbenchKit. The runner, fixture model, strategy
//  and envelope registries, and per-domain suites all live in the Kit library so
//  the XCTest gate can share them.
//
//  Usage:
//    .build/debug/NumericSwiftWorkbench                      # all domains
//    .build/debug/NumericSwiftWorkbench integration          # selected domains
//
//  Exit code: 0 — no self-awareness failures; 1 — one or more (the hard gate).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwiftWorkbenchKit

let requestedDomains = Array(CommandLine.arguments.dropFirst())

guard let fixturesDir = FixtureLoader.fixturesDirectory() else {
    print(ReportRenderer.renderNoFixtures())
    exit(0)
}

let fixtures = FixtureLoader.load(domains: requestedDomains, from: fixturesDir)
guard !fixtures.isEmpty else {
    print(ReportRenderer.renderNoFixtures())
    exit(0)
}

let summary = Workbench.run(fixtures: fixtures)

for domain in summary.domainReports {
    print(ReportRenderer.renderDomain(domain))
}
print(ReportRenderer.renderSummary(summary))

exit(summary.hasFailed ? 1 : 0)
