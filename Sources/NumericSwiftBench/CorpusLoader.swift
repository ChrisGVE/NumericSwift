// CorpusLoader.swift
// Sources/NumericSwiftBench/
//
// Reads the frozen LegacySnapshot.json from the committed fixture path so
// bench legs that must exercise the legacy evaluators can do so against the
// canonical corpus without recomputing it from scratch.
//
// The path is resolved relative to this source file at compile time (#file),
// mirroring exactly how LegacySnapshotGeneratorTests and ParityCorpusTests
// locate the fixture. This ensures the bench always reads the committed
// snapshot and never silently falls back to a stale or missing file.

import Foundation

// MARK: - Snapshot entry types (duplicated from ParityCorpus for bench isolation)
//
// The bench executable is a separate target that cannot @testable-import
// NumericSwiftTests. Rather than making the corpus types part of the main
// NumericSwift library (that would bloat the public surface), we re-declare
// a minimal read-only model here that is sufficient for the bench legs.
// These types mirror the Codable layout of LegacySnapshot.json exactly.

/// Evaluator tag stored in the snapshot.
enum BenchEvaluatorTag: String, Decodable {
  case scalar
  case complex
  case linAlg
}

/// Tagged result from the snapshot — only the fields the bench legs need.
enum BenchResult: Decodable {
  case scalar(Double)
  case complex(re: Double, im: Double)
  case matrix(rows: Int, cols: Int, data: [Double])
  case complexMatrix(rows: Int, cols: Int, real: [Double], imag: [Double])
  case nilResult
  case error(category: String)

  private enum CodingKeys: String, CodingKey {
    case type, value, re, im, rows, cols, data, real, imag, category
  }

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    let type_ = try c.decode(String.self, forKey: .type)
    switch type_ {
    case "scalar":
      let bits = try c.decode(UInt64.self, forKey: .value)
      self = .scalar(Double(bitPattern: bits))
    case "complex":
      let rBits = try c.decode(UInt64.self, forKey: .re)
      let iBits = try c.decode(UInt64.self, forKey: .im)
      self = .complex(re: Double(bitPattern: rBits), im: Double(bitPattern: iBits))
    case "matrix":
      let rows = try c.decode(Int.self, forKey: .rows)
      let cols = try c.decode(Int.self, forKey: .cols)
      let bits = try c.decode([UInt64].self, forKey: .data)
      self = .matrix(rows: rows, cols: cols, data: bits.map { Double(bitPattern: $0) })
    case "complexMatrix":
      let rows = try c.decode(Int.self, forKey: .rows)
      let cols = try c.decode(Int.self, forKey: .cols)
      let rBits = try c.decode([UInt64].self, forKey: .real)
      let iBits = try c.decode([UInt64].self, forKey: .imag)
      self = .complexMatrix(
        rows: rows, cols: cols,
        real: rBits.map { Double(bitPattern: $0) },
        imag: iBits.map { Double(bitPattern: $0) })
    case "nilResult":
      self = .nilResult
    case "error":
      let cat = try c.decode(String.self, forKey: .category)
      self = .error(category: cat)
    default:
      throw DecodingError.dataCorruptedError(
        forKey: .type, in: c, debugDescription: "Unknown result type: \(type_)")
    }
  }
}

/// One entry in the frozen snapshot.
struct BenchSnapshotEntry: Decodable {
  let id: String
  let description: String
  let evaluator: BenchEvaluatorTag
  let result: BenchResult
}

/// The top-level snapshot document.
struct BenchSnapshot: Decodable {
  let schemaVersion: String
  let capturedAt: String
  let entries: [BenchSnapshotEntry]
}

// MARK: - Loader

/// Loads the committed LegacySnapshot.json fixture.
enum CorpusLoader {

  /// Absolute URL to the committed fixture file.
  ///
  /// Resolved at compile time relative to this source file, identical to the
  /// approach used in the test target so both always read the same artifact.
  static var fixtureURL: URL {
    // #file = .../Sources/NumericSwiftBench/CorpusLoader.swift
    // Navigate up two directories and into Tests/NumericSwiftTests/Fixtures/
    let sourceDir = URL(fileURLWithPath: #file)
      .deletingLastPathComponent()   // NumericSwiftBench/
      .deletingLastPathComponent()   // Sources/
      .deletingLastPathComponent()   // package root
    return sourceDir
      .appendingPathComponent("Tests")
      .appendingPathComponent("NumericSwiftTests")
      .appendingPathComponent("Fixtures")
      .appendingPathComponent("LegacySnapshot.json")
  }

  /// Load and decode the snapshot, aborting with a clear message on failure.
  ///
  /// The bench harness calls this at startup. A missing or corrupt snapshot is
  /// a configuration error — it means the repo is in an inconsistent state
  /// and the bench should not proceed silently.
  ///
  /// - Returns: The decoded ``BenchSnapshot``.
  static func loadOrExit() -> BenchSnapshot {
    let url = fixtureURL
    guard FileManager.default.fileExists(atPath: url.path) else {
      fputs("""
        FATAL: LegacySnapshot.json not found at expected path:
          \(url.path)

        This file is committed as part of Task #2 (Phase 0 parity baseline).
        Re-run the snapshot generator to recreate it:
          NUMERICSWIFT_REGENERATE_SNAPSHOT=1 swift test \\
              --filter NumericSwiftTests.LegacySnapshotGeneratorTests

        Then commit Tests/NumericSwiftTests/Fixtures/LegacySnapshot.json.

        """, stderr)
      exit(2)
    }

    let data: Data
    do {
      data = try Data(contentsOf: url)
    } catch {
      fputs("FATAL: Cannot read LegacySnapshot.json: \(error)\n", stderr)
      exit(2)
    }

    do {
      return try JSONDecoder().decode(BenchSnapshot.self, from: data)
    } catch {
      fputs("""
        FATAL: Cannot decode LegacySnapshot.json: \(error)

        The snapshot may be corrupt or from an incompatible schema version.
        Regenerate it with the procedure above.

        """, stderr)
      exit(2)
    }
  }
}
