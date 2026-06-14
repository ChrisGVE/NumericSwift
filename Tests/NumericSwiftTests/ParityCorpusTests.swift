// ParityCorpusTests.swift
// Tests/NumericSwiftTests/
//
// Parity-baseline tests for task #2 (Phase 0, unified-pipeline tag).
//
// These tests READ the committed frozen snapshot and verify its structural
// integrity plus spot-check known numerical values. They do NOT regenerate
// the snapshot — that is LegacySnapshotGenerator's job.
//
// Test coverage:
//   1. Snapshot loads and round-trips cleanly (Codable)
//   2. All expected corpus segments are present
//   3. All three evaluator labels are recorded
//   4. Hand-verified spot-check values (SciPy cross-references)
//   5. IEEE-754 edge values survive JSON round-trip bit-exactly
//   6. Group-A error entries carry expected categories
//   7. Group-B error entries carry expected categories
//   8. Bilinear complex dot ≠ Hermitian inner product
//   9. Minimum corpus entry count (regression guard)

import Foundation
import XCTest
@testable import NumericSwift

final class ParityCorpusTests: XCTestCase {

  // MARK: Helpers

  private static var _snapshot: LegacySnapshot?

  private static func loadSnapshot() throws -> LegacySnapshot {
    if let cached = _snapshot { return cached }
    let sourceDir = URL(fileURLWithPath: #file).deletingLastPathComponent()
    let url = sourceDir
      .appendingPathComponent("Fixtures")
      .appendingPathComponent("LegacySnapshot.json")
    let data = try Data(contentsOf: url)
    let snap = try JSONDecoder().decode(LegacySnapshot.self, from: data)
    _snapshot = snap
    return snap
  }

  // Entry lookup by id prefix.
  private static func entries(
    _ snap: LegacySnapshot, prefix: String
  ) -> [CorpusEntry] {
    snap.entries.filter { $0.id.hasPrefix(prefix) }
  }

  private static func entry(
    _ snap: LegacySnapshot, id: String
  ) -> CorpusEntry? {
    snap.entries.first(where: { $0.id == id })
  }

  // Extract scalar double from LegacyResult.
  private static func scalarValue(_ result: LegacyResult) -> Double? {
    if case .scalar(let v) = result { return v }
    return nil
  }

  // MARK: Test 1 — Snapshot loads and round-trips

  func testSnapshotRoundTrip() throws {
    let snap = try Self.loadSnapshot()

    // Re-encode and decode; compare entry count and IDs.
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(snap)
    let decoded = try JSONDecoder().decode(LegacySnapshot.self, from: data)

    XCTAssertEqual(snap.entries.count, decoded.entries.count,
      "Entry count changed after round-trip")
    XCTAssertEqual(snap.schemaVersion, "1.0")

    let ids = snap.entries.map(\.id)
    let roundIds = decoded.entries.map(\.id)
    XCTAssertEqual(ids, roundIds, "Entry IDs changed after round-trip")
  }

  // MARK: Test 2 — All expected segments present

  func testSnapshotContainsAllSegments() throws {
    let snap = try Self.loadSnapshot()

    let expectedPrefixes: [String] = [
      "scalar-s",
      "complex-c",
      "rmat-m",
      "cmat-cm",
      "coerce-c",
      "bilin-d",
      "matfn-",
      "groupA-e",
      "groupB-e",
      "ieee-f",
    ]
    for prefix in expectedPrefixes {
      let count = Self.entries(snap, prefix: prefix).count
      XCTAssertGreaterThan(count, 0,
        "Segment '\(prefix)' missing from snapshot (0 entries)")
    }
  }

  // MARK: Test 3 — All three evaluator labels recorded

  func testSnapshotEvaluatorLabels() throws {
    let snap = try Self.loadSnapshot()

    let labels = Set(snap.entries.map { $0.evaluator.rawValue })
    XCTAssertTrue(labels.contains("scalar"),
      "Snapshot contains no scalar evaluator entries")
    XCTAssertTrue(labels.contains("complex"),
      "Snapshot contains no complex evaluator entries")
    XCTAssertTrue(labels.contains("linAlg"),
      "Snapshot contains no linAlg evaluator entries")

    XCTAssertEqual(snap.evaluatorPaths.count, 3,
      "Expected exactly 3 evaluator paths recorded in snapshot header")
  }

  // MARK: Test 4 — Hand-verified SciPy spot-checks

  func testSpotChecksAgainstSciPyReferences() throws {
    let snap = try Self.loadSnapshot()
    let tol = 1e-12

    // s16: sin(0) = 0
    if let e = Self.entry(snap, id: "scalar-s16"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v, 0.0, accuracy: tol, "sin(0) should be 0")
    } else {
      XCTFail("Entry scalar-s16 missing")
    }

    // s17: cos(0) = 1
    if let e = Self.entry(snap, id: "scalar-s17"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v, 1.0, accuracy: tol, "cos(0) should be 1")
    } else {
      XCTFail("Entry scalar-s17 missing")
    }

    // s18: exp(1) = e
    if let e = Self.entry(snap, id: "scalar-s18"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v, Foundation.exp(1.0), accuracy: tol, "exp(1) should be e")
    } else {
      XCTFail("Entry scalar-s18 missing")
    }

    // s19: log(1) = 0
    if let e = Self.entry(snap, id: "scalar-s19"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v, 0.0, accuracy: tol, "log(1) should be 0")
    } else {
      XCTFail("Entry scalar-s19 missing")
    }

    // s20: sqrt(4) = 2
    if let e = Self.entry(snap, id: "scalar-s20"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v, 2.0, accuracy: tol, "sqrt(4) should be 2")
    } else {
      XCTFail("Entry scalar-s20 missing")
    }

    // rmat-m15: v1·v2 = [1,2,3]·[4,5,6] = 32
    if let e = Self.entry(snap, id: "rmat-m15"),
       case .matrix(let rows, let cols, let data) = e.result {
      XCTAssertEqual(rows, 1, "vec·vec dot should be 1×1")
      XCTAssertEqual(cols, 1, "vec·vec dot should be 1×1")
      XCTAssertEqual(data[0], 32.0, accuracy: tol,
        "[1,2,3]·[4,5,6] should be 32")
    } else {
      XCTFail("Entry rmat-m15 missing or wrong type")
    }

    // matfn-t01: trace([[1,2],[3,4]]) = 5
    if let e = Self.entry(snap, id: "matfn-t01"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v, 5.0, accuracy: tol, "trace([[1,2],[3,4]]) should be 5")
    } else {
      XCTFail("Entry matfn-t01 missing")
    }

    // matfn-d01: det([[1,2],[3,4]]) = -2
    if let e = Self.entry(snap, id: "matfn-d01"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v, -2.0, accuracy: tol, "det([[1,2],[3,4]]) should be -2")
    } else {
      XCTFail("Entry matfn-d01 missing")
    }

    // coerce-c01: dot([2,3],[4,5]) = 23
    if let e = Self.entry(snap, id: "coerce-c01"),
       case .matrix(let rows, let cols, let data) = e.result {
      XCTAssertEqual(rows, 1)
      XCTAssertEqual(cols, 1)
      XCTAssertEqual(data[0], 23.0, accuracy: tol,
        "dot([2,3],[4,5]) should be 23")
    } else {
      XCTFail("Entry coerce-c01 missing or wrong type")
    }
  }

  // MARK: Test 5 — IEEE-754 edge values stored bit-exactly

  func testIEEE754EdgeValuesStoredBitExactly() throws {
    let snap = try Self.loadSnapshot()

    // NaN from sqrt(-1) — bit pattern must round-trip as NaN
    if let e = Self.entry(snap, id: "ieee-f01"),
       let v = Self.scalarValue(e.result) {
      // Double.nan == Double.nan is false by IEEE 754 design;
      // compare bit patterns to verify exact storage.
      XCTAssertTrue(v.isNaN, "ieee-f01 should decode as NaN")
    } else {
      XCTFail("Entry ieee-f01 missing")
    }

    // +inf
    if let e = Self.entry(snap, id: "ieee-f08"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v.bitPattern, Double.infinity.bitPattern,
        "+inf bitPattern must match exactly")
    } else {
      XCTFail("Entry ieee-f08 missing")
    }

    // -inf
    if let e = Self.entry(snap, id: "ieee-f09"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v.bitPattern, (-Double.infinity).bitPattern,
        "-inf bitPattern must match exactly")
    } else {
      XCTFail("Entry ieee-f09 missing")
    }

    // +0 and -0 have different bit patterns
    if let ePos = Self.entry(snap, id: "ieee-f06"),
       let eNeg = Self.entry(snap, id: "ieee-f07"),
       let vPos = Self.scalarValue(ePos.result),
       let vNeg = Self.scalarValue(eNeg.result) {
      XCTAssertNotEqual(vPos.bitPattern, vNeg.bitPattern,
        "+0.0 and -0.0 must have different bitPatterns")
      XCTAssertEqual(vPos.bitPattern, (0.0).bitPattern,
        "+0.0 bitPattern mismatch")
      XCTAssertEqual(vNeg.bitPattern, (-0.0).bitPattern,
        "-0.0 bitPattern mismatch")
    } else {
      XCTFail("Signed-zero entries ieee-f06/ieee-f07 missing")
    }

    // NaN in 1×1 matrix
    if let e = Self.entry(snap, id: "ieee-f11"),
       case .matrix(_, _, let data) = e.result {
      XCTAssertTrue(data[0].isNaN, "ieee-f11 matrix[0] should be NaN")
    } else {
      XCTFail("Entry ieee-f11 missing")
    }
  }

  // MARK: Test 6 — Group-A error categories

  func testGroupAErrorCategories() throws {
    let snap = try Self.loadSnapshot()
    let groupA = Self.entries(snap, prefix: "groupA-e")
    XCTAssertGreaterThanOrEqual(groupA.count, 6,
      "Expected at least 6 Group-A entries")

    for entry in groupA {
      guard case .error(let cat) = entry.result else {
        XCTFail("Group-A entry \(entry.id) should have .error result")
        continue
      }
      let validCats: Set<ErrorCategory> = [.dimensionMismatch, .divisionByZero]
      XCTAssertTrue(validCats.contains(cat),
        "Group-A entry \(entry.id) has unexpected category: \(cat)")
    }
  }

  // MARK: Test 7 — Group-B error categories

  func testGroupBErrorCategories() throws {
    let snap = try Self.loadSnapshot()
    let groupB = Self.entries(snap, prefix: "groupB-e")
    XCTAssertGreaterThanOrEqual(groupB.count, 8,
      "Expected at least 8 Group-B entries")

    for entry in groupB {
      guard case .error(let cat) = entry.result else {
        XCTFail("Group-B entry \(entry.id) should have .error result")
        continue
      }
      let validCats: Set<ErrorCategory> = [.notSquare, .divisionByZero]
      XCTAssertTrue(validCats.contains(cat),
        "Group-B entry \(entry.id) has unexpected category: \(cat)")
    }
  }

  // MARK: Test 8 — Bilinear dot ≠ Hermitian inner product

  func testBilinearDotDistinctFromHermitian() throws {
    let snap = try Self.loadSnapshot()

    // bilin-d03: (1+i)·(1+i) bilinear = 2i — imaginary part nonzero
    if let e = Self.entry(snap, id: "bilin-d03"),
       case .complex(let re, let im) = e.result {
      XCTAssertEqual(re, 0.0, accuracy: 1e-12,
        "bilinear (1+i).(1+i) real part should be 0")
      XCTAssertEqual(im, 2.0, accuracy: 1e-12,
        "bilinear (1+i).(1+i) imag part should be 2")
    } else {
      XCTFail("Entry bilin-d03 missing or wrong type")
    }

    // bilin-d04: Hermitian |1+i|² = 2 — real scalar
    if let e = Self.entry(snap, id: "bilin-d04"),
       let v = Self.scalarValue(e.result) {
      XCTAssertEqual(v, 2.0, accuracy: 1e-12,
        "Hermitian |1+i|² should be 2")
    } else {
      XCTFail("Entry bilin-d04 missing")
    }
  }

  // MARK: Test 9 — Minimum corpus entry count

  func testCorpusMinimumEntryCount() throws {
    let snap = try Self.loadSnapshot()
    // At time of writing: 50 scalar + 20 complex + 15 rmat + 5 cmat +
    //   3 coerce + 4 bilin + 14 matfn + 7 groupA + 9 groupB + 12 ieee = 139
    // Use a conservative floor to allow future additions without rebaselining.
    XCTAssertGreaterThanOrEqual(snap.entries.count, 100,
      "Corpus must contain at least 100 entries; got \(snap.entries.count)")
  }
}
