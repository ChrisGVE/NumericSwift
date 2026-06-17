//
//  WorkbenchGateTests.swift
//  NumericSwiftTests
//
//  Wave-1 gate skeleton for the NumericSwift E2E functional workbench.
//
//  ## Wave-1 scope
//
//  These tests assert the diagnostics core behaves correctly and that the fixture
//  decoder round-trips a hand-authored sample case. They are intentionally
//  non-vacuous per FP3: every assertion exercises real semantics.
//
//  The full fixture-driven gate (load all workbench/<domain>.json files, run every
//  strategy, assert zero envelope violations) is a wave-3 deliverable once fixture
//  generation (wave 2) is complete.
//
//  ## CI behaviour
//
//  Run with:
//    swift test --filter WorkbenchGateTests
//
//  All tests must pass on `swift test` (no environment gates). The slow fixture
//  corpus tests will be added in wave 3 behind NUMERICSWIFT_WORKBENCH_GATE=1.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift
import NumericSwiftWorkbenchKit

// MARK: - WorkbenchGateTests

final class WorkbenchGateTests: XCTestCase {

    // MARK: NumericDiagnostic semantics

    func testNumericDiagnostic_outsideEnvelope_isOutsideEnvelope() {
        let d = NumericDiagnostic.outsideEnvelope(method: "tDist.ppf", reason: "|p| > 0.9999")
        XCTAssertTrue(d.isOutsideEnvelope)
    }

    func testNumericDiagnostic_precisionDegraded_isNotOutsideEnvelope() {
        let d = NumericDiagnostic.precisionDegraded(method: "erfinv", approxDigits: 5)
        XCTAssertFalse(d.isOutsideEnvelope)
    }

    func testNumericDiagnostic_nonConvergence_isNotOutsideEnvelope() {
        let d = NumericDiagnostic.nonConvergence(method: "bisect", reason: "exceeded maxiter=100")
        XCTAssertFalse(d.isOutsideEnvelope)
    }

    func testNumericDiagnostic_description_outsideEnvelope() {
        let d = NumericDiagnostic.outsideEnvelope(method: "quad", reason: "singularity detected")
        XCTAssertTrue(d.description.contains("[outsideEnvelope]"))
        XCTAssertTrue(d.description.contains("quad"))
        XCTAssertTrue(d.description.contains("singularity detected"))
    }

    func testNumericDiagnostic_description_precisionDegraded() {
        let d = NumericDiagnostic.precisionDegraded(method: "erfinv", approxDigits: 5)
        XCTAssertTrue(d.description.contains("[precisionDegraded]"))
        XCTAssertTrue(d.description.contains("5"))
    }

    func testNumericDiagnostic_description_nonConvergence() {
        let d = NumericDiagnostic.nonConvergence(method: "bisect", reason: "exceeded maxiter=100")
        XCTAssertTrue(d.description.contains("[nonConvergence]"))
        XCTAssertTrue(d.description.contains("bisect"))
    }

    func testNumericDiagnostic_equatable_sameCase() {
        let a = NumericDiagnostic.outsideEnvelope(method: "m", reason: "r")
        let b = NumericDiagnostic.outsideEnvelope(method: "m", reason: "r")
        XCTAssertEqual(a, b)
    }

    func testNumericDiagnostic_equatable_differentMethod() {
        let a = NumericDiagnostic.outsideEnvelope(method: "A", reason: "r")
        let b = NumericDiagnostic.outsideEnvelope(method: "B", reason: "r")
        XCTAssertNotEqual(a, b)
    }

    func testNumericDiagnostic_equatable_differentCase() {
        let a = NumericDiagnostic.outsideEnvelope(method: "m", reason: "r")
        let b = NumericDiagnostic.nonConvergence(method: "m", reason: "r")
        XCTAssertNotEqual(a, b)
    }

    // MARK: Diagnosed<Value> semantics

    func testDiagnosed_emptyDiagnostics_isReliable() {
        let d = Diagnosed(3.14)
        XCTAssertEqual(d.value, 3.14, accuracy: 1e-15)
        XCTAssertTrue(d.isReliable)
        XCTAssertTrue(d.diagnostics.isEmpty)
    }

    func testDiagnosed_outsideEnvelopeDiagnostic_isNotReliable() {
        let d = Diagnosed(
            42.0,
            diagnostics: [.outsideEnvelope(method: "m", reason: "r")]
        )
        XCTAssertFalse(d.isReliable)
    }

    func testDiagnosed_precisionDegradedOnly_isReliable() {
        // precisionDegraded does not make a result unreliable
        let d = Diagnosed(
            1.0,
            diagnostics: [.precisionDegraded(method: "m", approxDigits: 5)]
        )
        XCTAssertTrue(d.isReliable)
    }

    func testDiagnosed_nonConvergenceOnly_isReliable() {
        // nonConvergence does not make a result unreliable
        let d = Diagnosed(
            0.5,
            diagnostics: [.nonConvergence(method: "m", reason: "exceeded maxiter")]
        )
        XCTAssertTrue(d.isReliable)
    }

    func testDiagnosed_mixedDiagnostics_isNotReliable() {
        let d = Diagnosed(
            0.0,
            diagnostics: [
                .precisionDegraded(method: "m", approxDigits: 3),
                .outsideEnvelope(method: "n", reason: "bad"),
            ]
        )
        XCTAssertFalse(d.isReliable)
    }

    func testDiagnosed_map_transformsValuePreservingDiagnostics() {
        let diag = NumericDiagnostic.precisionDegraded(method: "x", approxDigits: 8)
        let d = Diagnosed(2.0, diagnostics: [diag])
        let doubled = d.map { $0 * 2 }
        XCTAssertEqual(doubled.value, 4.0, accuracy: 1e-15)
        XCTAssertEqual(doubled.diagnostics, [diag])
    }

    func testDiagnosed_map_preservesReliability() {
        let d = Diagnosed(1.0, diagnostics: [.outsideEnvelope(method: "m", reason: "r")])
        let mapped = d.map { $0 + 1 }
        XCTAssertFalse(mapped.isReliable)
    }

    // MARK: Result-struct diagnostics fields (source-compatibility)

    func testQuadResult_defaultDiagnostics_isEmpty() {
        let r = QuadResult(value: 1.0, error: 1e-12, neval: 15)
        XCTAssertTrue(r.diagnostics.isEmpty)
    }

    func testODEResult_defaultDiagnostics_isEmpty() {
        let r = ODEResult(t: [0.0, 1.0], y: [[0.0], [1.0]], success: true, message: "ok", nfev: 42)
        XCTAssertTrue(r.diagnostics.isEmpty)
    }

    func testOLSResult_defaultDiagnostics_isEmpty() {
        // Use the real ols() function to construct an OLSResult — not a hand-built stub.
        // Simple 3-point regression: y = 2x + 1 with intercept.
        let X: [[Double]] = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        let y: [Double] = [3.0, 5.0, 7.0]
        guard let result = ols(y, X) else {
            XCTFail("ols() returned nil on valid input")
            return
        }
        XCTAssertTrue(result.diagnostics.isEmpty, "Default diagnostics must be empty")
        // Smoke-check the regression output while we're here.
        XCTAssertEqual(result.params[1], 2.0, accuracy: 1e-10)
    }

    func testGLMResult_defaultDiagnostics_isEmpty() {
        let X: [[Double]] = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
        let y: [Double] = [3.0, 5.0, 7.0, 9.0]
        guard let result = glm(y, X) else {
            XCTFail("glm() returned nil on valid input")
            return
        }
        XCTAssertTrue(result.diagnostics.isEmpty)
    }

    func testARIMAResult_defaultDiagnostics_isEmpty() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        guard let result = arima(y, p: 1, d: 0, q: 0) else {
            XCTFail("arima() returned nil on valid input")
            return
        }
        XCTAssertTrue(result.diagnostics.isEmpty)
    }

    // MARK: Fixture decoder round-trip

    /// Verifies that a hand-authored JSON fixture case decodes correctly,
    /// including bit-exact oracle reconstruction from the `bits` hex string.
    func testFixtureDecoder_sampleCase_roundTrip() throws {
        // This is the canonical example from WORKBENCH.md §3.
        // √π = 1.7724538509055159 — bit pattern frozen from numpy.
        let json = """
        [
          {
            "id": "integration.hard.gaussian_bell",
            "tier": "hard",
            "inputs": { "a": -5.0, "b": 5.0, "tag": "gaussian_bell" },
            "oracle": { "value": 1.7724538509055159, "bits": "0x3FFC5BF891B4EF6A" },
            "source": "scipy.integrate.quad 1.17.1",
            "strategies": ["quad", "romberg"],
            "tol": { "quad": 1e-10, "romberg": 1e-8 }
          }
        ]
        """.data(using: .utf8)!

        let cases = try JSONDecoder().decode(FixtureFile.self, from: json)

        XCTAssertEqual(cases.count, 1)
        let c = cases[0]

        XCTAssertEqual(c.id, "integration.hard.gaussian_bell")
        XCTAssertEqual(c.tier, .hard)
        XCTAssertEqual(c.source, "scipy.integrate.quad 1.17.1")
        XCTAssertEqual(c.strategies, ["quad", "romberg"])
        XCTAssertEqual(c.tol["quad"], 1e-10)
        XCTAssertEqual(c.tol["romberg"], 1e-8)
        XCTAssertEqual(c.domain, "integration")

        // Bit-exact oracle: the value must be reconstructed from the UInt64 bitPattern,
        // not from the floating-point "value" key. If bit-exact decode is broken,
        // this test detects it because the bitPattern is the ground truth.
        let expectedBits: UInt64 = 0x3FFC_5BF8_91B4_EF6A
        XCTAssertEqual(c.oracle.bits, expectedBits, "bits field must decode as hex string")
        XCTAssertEqual(
            c.oracle.value,
            Double(bitPattern: expectedBits),
            accuracy: 0,
            "oracle.value must be reconstructed from bits, not from JSON float"
        )

        // Inputs bag round-trip.
        XCTAssertEqual(c.inputs["a"]?.doubleValue, -5.0)
        XCTAssertEqual(c.inputs["b"]?.doubleValue, 5.0)
        XCTAssertEqual(c.inputs["tag"]?.stringValue, "gaussian_bell")
    }

    func testFixtureDecoder_oracleValue_negativeZero() throws {
        // IEEE-754 negative zero has a distinct bit pattern from +0.
        // Verify the decoder preserves it via bits.
        let negZeroBits: UInt64 = 0x8000_0000_0000_0000  // -0.0
        // %016llX correctly formats a 64-bit unsigned value (vs %X which is 32-bit).
        let hexStr = String(format: "0x%016llX", negZeroBits)
        let json = """
        [
          {
            "id": "edge.negative_zero",
            "tier": "edge",
            "inputs": {},
            "oracle": { "value": 0.0, "bits": "\(hexStr)" },
            "source": "manual",
            "strategies": [],
            "tol": {}
          }
        ]
        """.data(using: .utf8)!

        let cases = try JSONDecoder().decode(FixtureFile.self, from: json)
        let oracle = cases[0].oracle
        XCTAssertEqual(oracle.bits, negZeroBits)
        // Bit-exact: -0.0 bitPattern == negZeroBits, +0.0 bitPattern != negZeroBits
        XCTAssertEqual(oracle.value.bitPattern, negZeroBits,
                       "Decoder must preserve -0.0 bit pattern")
    }

    func testFixtureDecoder_caseTier_edge() throws {
        let json = """
        [{"id":"x","tier":"edge","inputs":{},"oracle":{"value":0.0,"bits":"0x0000000000000000"},
          "source":"manual","strategies":[],"tol":{}}]
        """.data(using: .utf8)!
        let cases = try JSONDecoder().decode(FixtureFile.self, from: json)
        XCTAssertEqual(cases[0].tier, .edge)
    }

    func testFixtureDecoder_caseTier_trivial() throws {
        let json = """
        [{"id":"x","tier":"trivial","inputs":{},"oracle":{"value":0.0,"bits":"0x0000000000000000"},
          "source":"manual","strategies":[],"tol":{}}]
        """.data(using: .utf8)!
        let cases = try JSONDecoder().decode(FixtureFile.self, from: json)
        XCTAssertEqual(cases[0].tier, .trivial)
    }

    func testFixtureDecoder_inputValue_arrayOfDoubles() throws {
        let json = """
        [{"id":"x","tier":"hard","inputs":{"xs":[1.0,2.0,3.0]},
          "oracle":{"value":0.0,"bits":"0x0000000000000000"},
          "source":"manual","strategies":[],"tol":{}}]
        """.data(using: .utf8)!
        let cases = try JSONDecoder().decode(FixtureFile.self, from: json)
        let xs = cases[0].inputs["xs"]?.arrayValue
        XCTAssertEqual(xs?.count, 3)
        XCTAssertEqual(xs?[0].doubleValue, 1.0)
    }
}
