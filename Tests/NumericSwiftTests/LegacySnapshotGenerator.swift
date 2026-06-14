// LegacySnapshotGenerator.swift
// Tests/NumericSwiftTests/
//
// One-shot test that exercises the three legacy evaluators over the full parity
// corpus and writes the results to the committed fixture file.
//
// IMPORTANT: This test must ONLY run against the legacy evaluators — never
// against unified-pipeline code. Regenerating from a unified path would produce
// a tautologically-true fixture that cannot catch regressions (vacuous-gate bug).
//
// Usage:
//   NUMERICSWIFT_REGENERATE_SNAPSHOT=1 swift test \
//       --filter NumericSwiftTests.LegacySnapshotGeneratorTests/testGenerateSnapshot
//
// The test is unconditionally SKIPPED when the env var is absent, so it never
// runs as part of the normal CI suite.
//
// Fixture output path (resolved at compile time):
//   Tests/NumericSwiftTests/Fixtures/LegacySnapshot.json

import Foundation
import XCTest
@testable import NumericSwift

final class LegacySnapshotGeneratorTests: XCTestCase {

  // MARK: Fixture URL (compile-time path resolution)

  /// Absolute path to the committed fixture file.
  ///
  /// Resolved relative to this source file so it works from any working
  /// directory (Xcode, `swift test`, CI).
  static var fixtureURL: URL {
    // #file resolves to the .swift source path at compile time.
    let sourceDir = URL(fileURLWithPath: #file).deletingLastPathComponent()
    return sourceDir
      .appendingPathComponent("Fixtures")
      .appendingPathComponent("LegacySnapshot.json")
  }

  // MARK: Generator test (env-gated)

  /// Build the corpus, run the legacy evaluators, and write the frozen snapshot.
  ///
  /// Skipped unless `NUMERICSWIFT_REGENERATE_SNAPSHOT=1` is set so this never
  /// runs automatically in CI or normal `swift test` invocations.
  func testGenerateSnapshot() throws {
    guard ProcessInfo.processInfo.environment["NUMERICSWIFT_REGENERATE_SNAPSHOT"] == "1"
    else {
      throw XCTSkip("Set NUMERICSWIFT_REGENERATE_SNAPSHOT=1 to regenerate the fixture")
    }

    let entries = try ParityCorpusBuilder.buildAll()

    let snapshot = LegacySnapshot(
      schemaVersion: "1.0",
      capturedAt: ISO8601DateFormatter().string(from: Date()),
      evaluatorPaths: [
        "scalar": "MathExpr.evaluate(_:variables:) — MathExpr.swift:150",
        "complex": "MathExpr.evaluateComplex(_:variables:complexVariables:) — MathExpr.swift:311",
        "linAlg": "LinAlg direct function/operator calls — LinAlg.swift",
      ],
      entries: entries)

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(snapshot)

    // Ensure the Fixtures directory exists.
    let fixturesDir = Self.fixtureURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(
      at: fixturesDir, withIntermediateDirectories: true)

    try data.write(to: Self.fixtureURL, options: .atomic)

    let byteCount = data.count
    let entryCount = entries.count
    print("Snapshot written: \(Self.fixtureURL.path)")
    print("  \(entryCount) entries, \(byteCount) bytes")

    // Verify the file is readable and round-trips cleanly.
    let readBack = try Data(contentsOf: Self.fixtureURL)
    let decoded = try JSONDecoder().decode(LegacySnapshot.self, from: readBack)
    XCTAssertEqual(decoded.entries.count, entries.count,
      "Round-trip entry count mismatch — snapshot may be corrupt")
    XCTAssertEqual(decoded.schemaVersion, "1.0")
  }
}
