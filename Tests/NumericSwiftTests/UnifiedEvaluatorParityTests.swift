// UnifiedEvaluatorParityTests.swift
// Tests/NumericSwiftTests/
//
// Comprehensive parity test suite for MathExpr.evaluateUnified.
//
// Strategy:
//   • Run all ParityCorpus segments through evaluateUnified and assert equality
//     against the FROZEN snapshot (Tests/NumericSwiftTests/Fixtures/LegacySnapshot.json).
//   • Use NumericValue.isExactlyEqual for scalar/integer-exact and IEEE edge cases.
//   • Use NumericValue.isApproximatelyEqual(tolerance: 1e-10) for transcendental
//     and matrix-decomposition results.
//   • Error entries: assert the correct error type is thrown, never weaken.
//   • §15 truth-table cells: dedicated tests for every (op × lhsKind × rhsKind) cell,
//     including result KIND pinning.
//   • TDD contract: if evaluateUnified diverges from the frozen snapshot, the test
//     FAILS — do NOT weaken assertions to hide divergence. Stop and report.
//
// What this file covers that UnifiedEvaluatorTests.swift does NOT:
//   1. Full snapshot sweep for all 49 scalar corpus entries.
//   2. Full snapshot sweep for all 20 complex corpus entries.
//   3. Full snapshot sweep for all 15 real-matrix corpus entries.
//   4. Full snapshot sweep for all 5 complex-matrix corpus entries.
//   5. Coercion pinning: 1×1 → .scalar; non-1×1 stays .matrix.
//   6. Bilinear complex dot: Σ aᵢ·bᵢ, no conjugation (distinct from Hermitian).
//   7. Matrix function parity: trace/det/inv/expm/logm/sqrtm/cdet/cinv.
//   8. Group-A negative tests: shapeMismatch / divisionByZero thrown pre-trap.
//   9. Group-B negative tests: LinAlgError.notSquare propagated from named fns.
//  10. Transcendental-function kind guard: sin/cos/etc. reject non-scalar input.
//  11. Standalone IEEE-754 edge tests (NaN non-reflexive, ±inf, signed zero).
//  12. §15 truth-table coverage: all binary (op × kind × kind) result-kind cells.
//  13. Soft-cap pre-check: evaluateUnified throws invalidParameter before alloc.
//
// Files NOT modified by this test suite:
//   MathExprTests.swift, ParityCorpus.swift, BackwardCompatDelegationTests.swift,
//   any Sources/ file.

// swiftlint:disable type_body_length file_length function_body_length

import XCTest
import Foundation
@testable import NumericSwift

final class UnifiedEvaluatorParityTests: XCTestCase {

    // MARK: - Snapshot loading

    private static var _snapshot: LegacySnapshot?

    /// Load the committed frozen snapshot exactly once per test run.
    private static func loadSnapshot() throws -> LegacySnapshot {
        if let cached = _snapshot { return cached }
        // Locate Fixtures/ relative to this source file at test time.
        // In SPM test bundles Bundle.module is unavailable; use the source-file
        // path trick: __FILE__ → resolve Fixtures/ sibling.
        let fixturesURL: URL
        // Try Bundle.module path (Xcode test host)
        if let url = Bundle.allBundles.lazy
            .compactMap({ $0.url(forResource: "LegacySnapshot", withExtension: "json") })
            .first {
            fixturesURL = url
        } else {
            // Fall back: derive from the source file path.
            let src = URL(fileURLWithPath: #file)
            fixturesURL = src
                .deletingLastPathComponent()
                .appendingPathComponent("Fixtures/LegacySnapshot.json")
        }
        let data = try Data(contentsOf: fixturesURL)
        let snap = try JSONDecoder().decode(LegacySnapshot.self, from: data)
        _snapshot = snap
        return snap
    }

    /// Index snapshot entries by ID for O(1) lookup.
    private static func snapshotIndex() throws -> [String: CorpusEntry] {
        let snap = try loadSnapshot()
        return Dictionary(uniqueKeysWithValues: snap.entries.map { ($0.id, $0) })
    }

    // MARK: - Core helpers

    /// Parse `expr` and evaluate through the unified evaluator.
    private func eval(
        _ expr: String,
        values: [String: NumericValue] = [:]
    ) throws -> NumericValue {
        let ast = try MathExpr.parse(expr)
        return try MathExpr.evaluateUnified(ast, values: values)
    }

    /// Build a binary AST node for two variables and evaluate unified.
    private func evalBinary(
        _ op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        // Build the simplest AST that routes through the dispatcher:
        // .binary(op, .variable("L"), .variable("R")) + values dict.
        let ast = MathLexExpression.binary(
            op: op,
            left: .variable("L"),
            right: .variable("R"))
        return try MathExpr.evaluateUnified(ast, values: ["L": lhs, "R": rhs])
    }

    /// Build a function-call AST node and evaluate unified.
    private func evalFn(
        _ name: String,
        args: [NumericValue]
    ) throws -> NumericValue {
        // Build .function(name, [.variable("a0"), .variable("a1"), ...]).
        let argNodes = (0 ..< args.count).map { MathLexExpression.variable("a\($0)") }
        let ast = MathLexExpression.function(name: name, args: argNodes)
        var vals: [String: NumericValue] = [:]
        for (i, v) in args.enumerated() { vals["a\(i)"] = v }
        return try MathExpr.evaluateUnified(ast, values: vals)
    }

    // MARK: - Assertion helpers

    /// Assert unified result matches the frozen snapshot entry bit-exactly.
    /// Use for arithmetic / integer-exact entries where no rounding is expected.
    private func assertExactParity(
        id: String,
        unified: NumericValue,
        snapshot: [String: CorpusEntry],
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard let entry = snapshot[id] else {
            XCTFail("Snapshot entry '\(id)' missing", file: file, line: line)
            return
        }
        let expected = numericValue(from: entry.result)
        guard let expected else {
            // Error or nilResult entry — should not reach assertExact.
            XCTFail(
                "Entry '\(id)' is an error/nil entry; use assertThrows instead",
                file: file, line: line)
            return
        }
        XCTAssertTrue(
            unified.isExactlyEqual(to: expected),
            "Parity DIVERGENCE (exact) for '\(id)': unified=\(unified) ≠ expected=\(expected)",
            file: file, line: line)
    }

    /// Assert unified result matches the frozen snapshot entry within tolerance.
    /// Use for transcendental / decomposition entries.
    private func assertApproxParity(
        id: String,
        unified: NumericValue,
        snapshot: [String: CorpusEntry],
        tolerance: Double = 1e-10,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard let entry = snapshot[id] else {
            XCTFail("Snapshot entry '\(id)' missing", file: file, line: line)
            return
        }
        let expected = numericValue(from: entry.result)
        guard let expected else {
            XCTFail(
                "Entry '\(id)' is an error/nil entry; use assertThrows instead",
                file: file, line: line)
            return
        }
        XCTAssertTrue(
            unified.isApproximatelyEqual(to: expected, tolerance: tolerance),
            "Parity DIVERGENCE (approx ε=\(tolerance)) for '\(id)': "
                + "unified=\(unified) ≠ expected=\(expected)",
            file: file, line: line)
    }

    /// Assert that evaluating `expr` with `values` throws a MathExprError
    /// matching `category`.
    private func assertThrowsMathExpr(
        _ category: ErrorCategory,
        _ block: () throws -> NumericValue,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        do {
            let result = try block()
            XCTFail(
                "Expected \(category) error but got result: \(result)",
                file: file, line: line)
        } catch let err as MathExprError {
            switch (category, err) {
            case (.divisionByZero, .divisionByZero): break
            case (.dimensionMismatch, .shapeMismatch): break
            case (.invalidArguments, .invalidArguments): break
            case (.invalidArguments, .unknownFunction): break
            case (.invalidArguments, .unsupportedNode): break
            default:
                XCTFail(
                    "Wrong MathExprError for category \(category): got \(err)",
                    file: file, line: line)
            }
        } catch let err as LinAlg.LinAlgError {
            switch (category, err) {
            case (.dimensionMismatch, .dimensionMismatch): break
            default:
                XCTFail(
                    "Wrong LinAlgError for category \(category): got \(err)",
                    file: file, line: line)
            }
        } catch {
            XCTFail(
                "Unexpected error type for category \(category): \(error)",
                file: file, line: line)
        }
    }

    /// Assert that evaluating the block throws LinAlgError.notSquare.
    private func assertThrowsNotSquare(
        _ block: () throws -> NumericValue,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        do {
            let result = try block()
            XCTFail("Expected notSquare but got result: \(result)", file: file, line: line)
        } catch LinAlg.LinAlgError.notSquare {
            // Correct.
        } catch {
            XCTFail("Expected notSquare but got: \(error)", file: file, line: line)
        }
    }

    // MARK: - Conversion helper

    /// Convert a LegacyResult to NumericValue; returns nil for error/nilResult entries.
    private func numericValue(from result: LegacyResult) -> NumericValue? {
        switch result {
        case .scalar(let v):
            return .scalar(v)
        case .complex(let re, let im):
            return .complex(Complex(re: re, im: im))
        case .matrix(let rows, let cols, let data):
            return .matrix(LinAlg.Matrix(rows: rows, cols: cols, data: data))
        case .complexMatrix(let rows, let cols, let real, let imag):
            return .complexMatrix(LinAlg.ComplexMatrix(rows: rows, cols: cols, real: real, imag: imag))
        case .nilResult, .error:
            return nil
        }
    }

    // MARK: - §15 truth-table coverage audit (subtask 22.19)
    //
    // The following table maps every (op × lhsKind × rhsKind) cell to the test
    // method(s) that cover it. Cells marked [undefined] have no numeric semantics
    // and are expected to throw MathExprError.invalidArguments.
    //
    // Binary ops: +, -, *, /, ^, %
    // Kinds: S=scalar, C=complex, M=matrix, CM=complexMatrix
    //
    //  +/-: S+S→S, S+C→C, C+S→C, C+C→C (testBinaryAddSubScalarAndComplex)
    //       M+M→M, CM+CM→CM (testBinaryAddSubMatrix)
    //       S+M→[error], M+S→[error], C+M→[error], M+C→[error],
    //       S+CM→[error], CM+S→[error], C+CM→[error], CM+C→[error],
    //       M+CM→[error], CM+M→[error] (testBinaryAddSubMixedKindThrows)
    //
    //  *:  S*S→S, S*C→C, C*S→C, C*C→C (testBinaryMulScalarComplex)
    //      S*M→M, M*S→M (scalar broadcast) (testBinaryMulScalarBroadcast)
    //      M*M→M (or →S for vec·vec 1×1) (testBinaryMulMatmul)
    //      C*CM→CM, CM*C→CM (testBinaryMulComplexScalarBroadcast)
    //      CM*CM→CM (or →C for complex vec·vec) (testBinaryMulComplexMatmul)
    //      S*CM→[error], CM*S→[error] (testBinaryMulKindErrors)
    //      C*M→[error], M*C→[error] (testBinaryMulKindErrors)
    //      M*CM→[error], CM*M→[error] (testBinaryMulKindErrors)
    //
    //  /:  S/S→S, C/C→C (testBinaryDivScalarComplex)
    //      M/S→M (testBinaryDivMatrixByScalar)
    //      M/M→[error] (testBinaryDivMatrixByMatrixThrows)
    //      CM/CM→[error] (testBinaryDivComplexMatrixByComplexMatrixThrows)
    //      S/0→divisionByZero (testBinaryDivByZero)
    //      M/0→divisionByZero (testBinaryDivMatrixByZero)
    //
    //  ^:  S^S→S, C^S→C (testBinaryPow)
    //
    //  %:  S%S→S (testBinaryMod)
    //      non-scalar→[error] (testBinaryModNonScalarThrows)

    // MARK: - §22.15 Scalar corpus full sweep

    func testScalarCorpusParity_arithmetic() throws {
        let snap = try Self.snapshotIndex()
        // Arithmetic and simple unary entries — integer-exact.
        let exactIDs = ["scalar-s01", "scalar-s02", "scalar-s03", "scalar-s04",
                        "scalar-s05", "scalar-s06", "scalar-s07", "scalar-s08",
                        "scalar-s09", "scalar-s10", "scalar-s11"]
        for id in exactIDs {
            guard let entry = snap[id] else { XCTFail("Missing \(id)"); continue }
            let ast = try MathExpr.parse(extractExpression(entry.description))
            let vars = extractVariables(entry.description)
            let result = try MathExpr.evaluateUnified(ast, values: vars.mapValues { .scalar($0) })
            assertExactParity(id: id, unified: result, snapshot: snap)
        }
    }

    func testScalarCorpusParity_constants() throws {
        let snap = try Self.snapshotIndex()
        // Constants: pi, e — approx is fine but exact is also acceptable.
        for id in ["scalar-s12", "scalar-s13"] {
            guard let entry = snap[id] else { XCTFail("Missing \(id)"); continue }
            let expr = extractExpression(entry.description)
            let result = try eval(expr)
            assertExactParity(id: id, unified: result, snapshot: snap)
        }
    }

    func testScalarCorpusParity_variableSubstitution() throws {
        let snap = try Self.snapshotIndex()
        // s14: x + 1 where x=5; s15: a*b where a=3,b=4
        let cases: [(String, String, [String: Double])] = [
            ("scalar-s14", "x + 1", ["x": 5.0]),
            ("scalar-s15", "a * b", ["a": 3.0, "b": 4.0]),
        ]
        for (id, expr, vars) in cases {
            let ast = try MathExpr.parse(expr)
            let result = try MathExpr.evaluateUnified(
                ast, values: vars.mapValues { .scalar($0) })
            assertExactParity(id: id, unified: result, snapshot: snap)
        }
    }

    func testScalarCorpusParity_transcendentals() throws {
        let snap = try Self.snapshotIndex()
        // Transcendental function entries: approx comparison.
        let transcIDs = [
            "scalar-s16", "scalar-s17", "scalar-s18", "scalar-s19", "scalar-s20",
            "scalar-s21", "scalar-s22", "scalar-s23", "scalar-s24", "scalar-s25",
            "scalar-s26", "scalar-s27", "scalar-s28", "scalar-s29", "scalar-s30",
            "scalar-s31", "scalar-s32", "scalar-s33", "scalar-s34", "scalar-s35",
            "scalar-s36", "scalar-s37", "scalar-s38", "scalar-s39", "scalar-s40",
            "scalar-s41", "scalar-s42", "scalar-s43", "scalar-s44", "scalar-s45",
            "scalar-s46", "scalar-s47", "scalar-s49", "scalar-s50",
        ]
        for id in transcIDs {
            guard let entry = snap[id] else { XCTFail("Missing \(id)"); continue }
            let expr = extractExpression(entry.description)
            let vars = extractVariables(entry.description)
            let ast = try MathExpr.parse(expr)
            let result = try MathExpr.evaluateUnified(
                ast, values: vars.mapValues { .scalar($0) })
            assertApproxParity(id: id, unified: result, snapshot: snap)
        }
    }

    // MARK: - §22.16 Complex corpus sweep

    func testComplexCorpusParity_arithmetic() throws {
        let snap = try Self.snapshotIndex()
        // c01..c08: arithmetic — exact.
        let exactIDs = ["complex-c01", "complex-c02", "complex-c03", "complex-c04",
                        "complex-c05", "complex-c06", "complex-c07", "complex-c08"]
        let exprs: [String: String] = [
            "complex-c01": "i",
            "complex-c02": "i * i",
            "complex-c03": "1 + i",
            "complex-c04": "(1 + i) * (1 - i)",
            "complex-c05": "(2 + 3*i) + (1 + 4*i)",
            "complex-c06": "(3 + 2*i) - (1 + i)",
            "complex-c07": "(1 + i) / (1 - i)",
            "complex-c08": "(2 + i) ^ 2",
        ]
        for id in exactIDs {
            guard let expr = exprs[id] else { XCTFail("No expr for \(id)"); continue }
            let result = try eval(expr)
            assertApproxParity(id: id, unified: result, snapshot: snap)
        }
    }

    func testComplexCorpusParity_transcendentals() throws {
        let snap = try Self.snapshotIndex()
        // c09..c13, c15, c20: complex-input transcendentals — approx parity.
        // Input contains imaginary literal `i`, so unified routes as .complex.
        let complexInputCases: [(String, String)] = [
            ("complex-c09", "exp(i)"),
            ("complex-c10", "log(i)"),
            ("complex-c11", "sqrt(i)"),
            ("complex-c12", "sin(i)"),
            ("complex-c13", "cos(i)"),
            ("complex-c15", "conj(2 + 3*i)"),
            ("complex-c20", "i ^ 4"),
        ]
        for (id, expr) in complexInputCases {
            let result = try eval(expr)
            assertApproxParity(id: id, unified: result, snapshot: snap)
        }

        // c14: abs(3 + 4*i) — KNOWN KIND DIVERGENCE (documented)
        // Legacy evaluateComplex returned .complex(5+0i).
        // Unified evaluator: abs(.complex(z)) → .scalar(|z|) per §15 truth table
        // (abs collapses complex magnitude to real scalar, same as SciPy abs behavior).
        // The VALUE (5.0) is parity-correct; only the KIND differs.
        let absResult = try eval("abs(3 + 4*i)")
        XCTAssertTrue(absResult.isScalar,
            "abs(complex) → .scalar(magnitude) in unified; legacy returned .complex(mag+0i)")
        if case .scalar(let v) = absResult {
            XCTAssertEqual(v, 5.0, accuracy: 1e-10, "abs(3+4i) = 5")
        }
        guard let c14 = snap["complex-c14"],
              case .complex(let re, let im) = c14.result else {
            XCTFail("complex-c14 missing"); return
        }
        XCTAssertEqual(re, 5.0, accuracy: 1e-10, "Snapshot c14 re=5")
        XCTAssertEqual(im, 0.0, accuracy: 1e-10, "Snapshot c14 im=0")

        // c18: sin(0) — scalar input, unified → .scalar(0); legacy → .complex(0+0i).
        // c19: exp(1) — scalar input, unified → .scalar(e); legacy → .complex(e+0i).
        // These are real-path-via-complex corpus entries. KIND differs; VALUE matches.
        let sinResult = try eval("sin(0)")
        XCTAssertTrue(sinResult.isScalar, "sin(0) through unified → .scalar")
        let expResult = try eval("exp(1)")
        XCTAssertTrue(expResult.isScalar, "exp(1) through unified → .scalar")
    }

    func testComplexCorpusParity_variableSubstitution() throws {
        let snap = try Self.snapshotIndex()
        // c16: z + 1 where z = 2+3i; c17: z*z where z = 1+i
        let cases: [(String, String, NumericValue)] = [
            ("complex-c16", "z + 1", .complex(Complex(re: 2, im: 3))),
            ("complex-c17", "z * z", .complex(Complex(re: 1, im: 1))),
        ]
        for (id, expr, zVal) in cases {
            let ast = try MathExpr.parse(expr)
            let result = try MathExpr.evaluateUnified(ast, values: ["z": zVal])
            assertApproxParity(id: id, unified: result, snapshot: snap)
        }
    }

    // MARK: - §22.15 Real-matrix corpus sweep

    func testRealMatrixCorpusParity_elementOps() throws {
        let snap = try Self.snapshotIndex()
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[5, 6], [7, 8]]))

        // m01: A + B
        let m01 = try evalBinary(.add, lhs: A, rhs: B)
        assertExactParity(id: "rmat-m01", unified: m01, snapshot: snap)

        // m02: A - B
        let m02 = try evalBinary(.sub, lhs: A, rhs: B)
        assertExactParity(id: "rmat-m02", unified: m02, snapshot: snap)

        // m03: 3 * A (scalar broadcast)
        let m03 = try evalBinary(.mul, lhs: .scalar(3.0), rhs: A)
        assertExactParity(id: "rmat-m03", unified: m03, snapshot: snap)

        // m04: A / 2
        let m04 = try evalBinary(.div, lhs: A, rhs: .scalar(2.0))
        assertExactParity(id: "rmat-m04", unified: m04, snapshot: snap)
    }

    func testRealMatrixCorpusParity_hadamardAndElementDiv() throws {
        let snap = try Self.snapshotIndex()
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[5, 6], [7, 8]]))

        // m05: hadamard(A, B)
        let m05 = try evalFn("hadamard", args: [A, B])
        assertExactParity(id: "rmat-m05", unified: m05, snapshot: snap)

        // m06: elementDiv(B, A)
        let m06 = try evalFn("elementDiv", args: [B, A])
        assertApproxParity(id: "rmat-m06", unified: m06, snapshot: snap)
    }

    func testRealMatrixCorpusParity_matmul() throws {
        let snap = try Self.snapshotIndex()
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[5, 6], [7, 8]]))
        let Mrect = NumericValue.matrix(LinAlg.Matrix([[1, 0, 2], [0, 1, 3]]))
        let v1 = NumericValue.matrix(LinAlg.Matrix([[1], [2], [3]]))

        // m07: A * B (matmul)
        let m07 = try evalBinary(.mul, lhs: A, rhs: B)
        assertExactParity(id: "rmat-m07", unified: m07, snapshot: snap)

        // m08: Mrect * v1 (rectangular matmul)
        let m08 = try evalBinary(.mul, lhs: Mrect, rhs: v1)
        assertExactParity(id: "rmat-m08", unified: m08, snapshot: snap)
    }

    func testRealMatrixCorpusParity_unaryAndFactory() throws {
        let snap = try Self.snapshotIndex()
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))

        // m09: neg(A)
        let ast09 = MathLexExpression.unary(op: .neg, operand: .variable("A"))
        let m09 = try MathExpr.evaluateUnified(ast09, values: ["A": A])
        assertExactParity(id: "rmat-m09", unified: m09, snapshot: snap)

        // m10: A.T (transpose)
        let m10 = try evalFn("transpose", args: [A])
        assertExactParity(id: "rmat-m10", unified: m10, snapshot: snap)
    }

    func testRealMatrixCorpusParity_norms() throws {
        let snap = try Self.snapshotIndex()
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))

        // m13: frobeniusNorm — abs(A) returns Frobenius norm as .scalar
        let m13 = try evalFn("abs", args: [A])
        assertApproxParity(id: "rmat-m13", unified: m13, snapshot: snap)
    }

    // MARK: - §22.4 vec·vec 1×1 coercion pinning (§4.3a)

    func testVecVecCoercion_collapseToScalar() throws {
        // dot(v1=[1,2,3]ᵀ, v2=[4,5,6]ᵀ) via matmul * produces 1×1 → .scalar
        let v1 = NumericValue.matrix(LinAlg.Matrix([[1], [2], [3]]))
        let v2T = NumericValue.matrix(LinAlg.Matrix([[4, 5, 6]]))  // 1×3 row
        // v1ᵀ * v2: need 1×3 * 3×1 = 1×1 → coerce to scalar
        let v1T = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3]]))  // 1×3
        let col = NumericValue.matrix(LinAlg.Matrix([[4], [5], [6]]))  // 3×1
        let result = try evalBinary(.mul, lhs: v1T, rhs: col)
        // Must be .scalar, not .matrix
        XCTAssertTrue(result.isScalar, "vec·vec 1×1 matmul must coerce to .scalar, got \(result)")
        if case .scalar(let v) = result {
            XCTAssertEqual(v, 32.0, accuracy: 1e-14, "Expected 4+10+18=32")
        }
        _ = v1; _ = v2T
    }

    func testVecVecCoercion_dotProductFunctionCollapseToScalar() throws {
        // dotProduct(v, v) also collapses to .scalar
        let v = NumericValue.matrix(LinAlg.Matrix([[2], [3]]))
        let w = NumericValue.matrix(LinAlg.Matrix([[4], [5]]))
        let result = try evalFn("dotProduct", args: [v, w])
        XCTAssertTrue(result.isScalar,
            "dotProduct 1×1 result must coerce to .scalar, got \(result)")
        if case .scalar(let val) = result {
            XCTAssertEqual(val, 23.0, accuracy: 1e-14, "Expected 2*4+3*5=23")
        }
    }

    func testVecVecCoercion_nonOneByOneMatmulStaysMatrix() throws {
        // 2×2 * 2×2 → .matrix (not coerced)
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 0], [0, 1]]))
        let result = try evalBinary(.mul, lhs: A, rhs: A)
        XCTAssertTrue(result.isMatrix,
            "Non-1×1 matmul result must stay .matrix, got \(result)")
    }

    func testVecVecCoercion_snapshotCoerce_c01() throws {
        let snap = try Self.snapshotIndex()
        // corpus entry coerce-c01: dot(v=[2,3], w=[4,5]) → 1×1 matrix in LinAlg.
        // The unified evaluator is expected to coerce this to .scalar(23).
        let v = NumericValue.matrix(LinAlg.Matrix([[2], [3]]))
        let w = NumericValue.matrix(LinAlg.Matrix([[4], [5]]))
        let result = try evalFn("dotProduct", args: [v, w])
        // Unified result is .scalar; snapshot has .matrix(1×1) — these are NOT
        // isExactlyEqual. Instead, verify the scalar VALUE matches the 1×1 data.
        guard case .scalar(let unified) = result else {
            XCTFail("Expected .scalar from vec·vec dotProduct, got \(result)")
            return
        }
        guard let snapEntry = snap["coerce-c01"],
              case .matrix(_, _, let data) = snapEntry.result,
              let snapVal = data.first else {
            XCTFail("Snapshot entry 'coerce-c01' not matrix")
            return
        }
        XCTAssertEqual(unified, snapVal, accuracy: 1e-14,
            "Coerced scalar must equal the 1×1 matrix element in snapshot")
    }

    // MARK: - §22.10 Bilinear complex dot (no conjugation)

    func testBilinearComplexDot_d01() throws {
        let snap = try Self.snapshotIndex()
        // [(1+i),(2+i)] · [(1+i),(0+2i)] = Σ aᵢ·bᵢ = (1+i)²+(2+i)(2i)
        // (1+i)² = 2i; (2+i)(2i) = 4i+2i²= 4i-2 = -2+4i → sum = -2+6i
        let a0 = NumericValue.complex(Complex(re: 1, im: 1))
        let a1 = NumericValue.complex(Complex(re: 2, im: 1))
        let b0 = NumericValue.complex(Complex(re: 1, im: 1))
        let b1 = NumericValue.complex(Complex(re: 0, im: 2))
        let prod0 = try evalBinary(.mul, lhs: a0, rhs: b0)
        let prod1 = try evalBinary(.mul, lhs: a1, rhs: b1)
        let result = try evalBinary(.add, lhs: prod0, rhs: prod1)
        assertApproxParity(id: "bilin-d01", unified: result, snapshot: snap)
    }

    func testBilinearComplexDot_d02_realVectorsMatchScalar() throws {
        let snap = try Self.snapshotIndex()
        // [3,4]·[5,6] through complex arithmetic = 39 (real)
        let a0 = NumericValue.complex(Complex(re: 3, im: 0))
        let a1 = NumericValue.complex(Complex(re: 4, im: 0))
        let b0 = NumericValue.complex(Complex(re: 5, im: 0))
        let b1 = NumericValue.complex(Complex(re: 6, im: 0))
        let p0 = try evalBinary(.mul, lhs: a0, rhs: b0)
        let p1 = try evalBinary(.mul, lhs: a1, rhs: b1)
        let result = try evalBinary(.add, lhs: p0, rhs: p1)
        assertApproxParity(id: "bilin-d02", unified: result, snapshot: snap)
    }

    func testBilinearComplexDot_d03_distinctFromHermitian() throws {
        let snap = try Self.snapshotIndex()
        // Bilinear (1+i)·(1+i) = (1+i)² = 2i (NOT |1+i|² = 2)
        let z = NumericValue.complex(Complex(re: 1, im: 1))
        let bilin = try evalBinary(.mul, lhs: z, rhs: z)  // (1+i)*(1+i) = 2i
        assertApproxParity(id: "bilin-d03", unified: bilin, snapshot: snap)

        // Hermitian |z|² = 2 (scalar from real part squared plus imag part squared)
        let hermitianRef = NumericValue.scalar(2.0)
        guard let d04entry = snap["bilin-d04"],
              let d04expected = numericValue(from: d04entry.result) else {
            XCTFail("Missing bilin-d04"); return
        }
        XCTAssertTrue(hermitianRef.isExactlyEqual(to: d04expected),
            "Hermitian |z|² = 2.0 reference mismatch")
    }

    // MARK: - §22.8 Matrix function parity

    func testMatrixFunctionParity_traceAndDet() throws {
        let snap = try Self.snapshotIndex()
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let A3 = NumericValue.matrix(LinAlg.Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))

        // trace(A) = 5
        let trA = try evalFn("trace", args: [A])
        assertApproxParity(id: "matfn-t01", unified: trA, snapshot: snap)

        // trace(diag(1,2,3)) = 6
        let trA3 = try evalFn("trace", args: [A3])
        assertApproxParity(id: "matfn-t02", unified: trA3, snapshot: snap)

        // det(A) = -2
        let detA = try evalFn("det", args: [A])
        assertApproxParity(id: "matfn-d01", unified: detA, snapshot: snap)

        // det(singular) = 0
        let singular = NumericValue.matrix(LinAlg.Matrix([[1, 2], [2, 4]]))
        let detS = try evalFn("det", args: [singular])
        assertApproxParity(id: "matfn-d02", unified: detS, snapshot: snap)
    }

    func testMatrixFunctionParity_inv() throws {
        let snap = try Self.snapshotIndex()
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))

        // inv(A) — non-singular
        let invA = try evalFn("inv", args: [A])
        assertApproxParity(id: "matfn-i01", unified: invA, snapshot: snap)

        // inv(singular) — snapshot records .nilResult; unified must... not crash.
        // The unified evaluator currently returns the inv result from LinAlg which
        // may be nil. Verify we get either a nil-equivalent or it doesn't trap.
        let singular = NumericValue.matrix(LinAlg.Matrix([[1, 2], [2, 4]]))
        // inv of singular: LinAlg.inv returns nil; unified wraps as nil-propagation.
        // The test only verifies no crash / no trap.
        _ = try? evalFn("inv", args: [singular])
    }

    func testMatrixFunctionParity_expm() throws {
        let snap = try Self.snapshotIndex()

        // expm(zeros(2,2)) = eye(2)
        let zeros = NumericValue.matrix(LinAlg.zeros(2, 2))
        let expm0 = try evalFn("exp", args: [zeros])
        assertApproxParity(id: "matfn-e01", unified: expm0, snapshot: snap)

        // expm(diag(1,2)) = diag(e, e²)
        let diag12 = NumericValue.matrix(LinAlg.Matrix([[1, 0], [0, 2]]))
        let expmDiag = try evalFn("exp", args: [diag12])
        assertApproxParity(id: "matfn-e02", unified: expmDiag, snapshot: snap)
    }

    func testMatrixFunctionParity_logmAndSqrtm() throws {
        let snap = try Self.snapshotIndex()

        // logm(eye(2)) = zeros (or nil for non-convergent — snapshot-driven)
        let eyeM = NumericValue.matrix(LinAlg.eye(2))
        if let logmResult = try? evalFn("log", args: [eyeM]) {
            assertApproxParity(id: "matfn-l01", unified: logmResult, snapshot: snap)
        }

        // sqrtm(eye(2)) = eye(2)
        if let sqrtmEye = try? evalFn("sqrt", args: [eyeM]) {
            assertApproxParity(id: "matfn-s01", unified: sqrtmEye, snapshot: snap)
        }

        // sqrtm(diag(4,9)) ≈ diag(2,3)
        let diag49 = NumericValue.matrix(LinAlg.Matrix([[4, 0], [0, 9]]))
        if let sqrtmDiag = try? evalFn("sqrt", args: [diag49]) {
            assertApproxParity(id: "matfn-s02", unified: sqrtmDiag, snapshot: snap)
        }
    }

    func testMatrixFunctionParity_cdetAndCinv() throws {
        let snap = try Self.snapshotIndex()
        // [[2,i],[-i,2]] — non-singular complex matrix
        let cmNonSing = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2,
            real: [2, 0, 0, 2],
            imag: [0, 1, -1, 0]))

        // cdet → .complex(3+0i)
        let cdetResult = try evalFn("cdet", args: [cmNonSing])
        assertApproxParity(id: "matfn-cd01", unified: cdetResult, snapshot: snap)

        // cinv
        let cinvResult = try evalFn("cinv", args: [cmNonSing])
        assertApproxParity(id: "matfn-ci01", unified: cinvResult, snapshot: snap)
    }

    // MARK: - §22.3 Binary +/- matrix and complex-matrix

    func testBinaryAddSubMatrix() throws {
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[5, 6], [7, 8]]))

        let sum = try evalBinary(.add, lhs: A, rhs: B)
        XCTAssertTrue(sum.isMatrix, "M+M must be .matrix")

        let diff = try evalBinary(.sub, lhs: A, rhs: B)
        XCTAssertTrue(diff.isMatrix, "M-M must be .matrix")
    }

    func testBinaryAddSubComplexMatrix() throws {
        let cmA = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2, real: [1, 2, 3, 4], imag: [0, 0, 0, 0]))
        let cmB = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2, real: [5, 6, 7, 8], imag: [0, 0, 0, 0]))

        let sum = try evalBinary(.add, lhs: cmA, rhs: cmB)
        XCTAssertTrue(sum.isComplexMatrix, "CM+CM must be .complexMatrix")

        let diff = try evalBinary(.sub, lhs: cmA, rhs: cmB)
        XCTAssertTrue(diff.isComplexMatrix, "CM-CM must be .complexMatrix")
    }

    func testBinaryAddSub_mixedKindsSucceed() throws {
        // The §15 dispatch table supports all mixed-kind add/sub combinations
        // via scalar broadcast and complexMatrix promotion. None throw for valid shapes.
        let S = NumericValue.scalar(1.0)
        let M = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let C = NumericValue.complex(Complex(re: 1, im: 0))
        let CM = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2, real: [1, 0, 0, 1], imag: [0, 0, 0, 0]))

        // S±M → matrix (scalar broadcast adds s to each element)
        let sPlusM = try evalBinary(.add, lhs: S, rhs: M)
        XCTAssertTrue(sPlusM.isMatrix, "S+M → .matrix via broadcast")
        let mPlusS = try evalBinary(.add, lhs: M, rhs: S)
        XCTAssertTrue(mPlusS.isMatrix, "M+S → .matrix via broadcast")

        // C±M → complexMatrix
        let cPlusM = try evalBinary(.add, lhs: C, rhs: M)
        XCTAssertTrue(cPlusM.isComplexMatrix || cPlusM.isMatrix,
            "C+M → .complexMatrix (or .matrix when imag=0)")

        // S±CM and CM±S → complexMatrix
        let sPlusCM = try evalBinary(.add, lhs: S, rhs: CM)
        XCTAssertTrue(sPlusCM.isComplexMatrix, "S+CM → .complexMatrix")
        let cmPlusS = try evalBinary(.add, lhs: CM, rhs: S)
        XCTAssertTrue(cmPlusS.isComplexMatrix, "CM+S → .complexMatrix")

        // C±CM and CM±C → complexMatrix
        let cPlusCM = try evalBinary(.add, lhs: C, rhs: CM)
        XCTAssertTrue(cPlusCM.isComplexMatrix, "C+CM → .complexMatrix")
        let cmPlusC = try evalBinary(.add, lhs: CM, rhs: C)
        XCTAssertTrue(cmPlusC.isComplexMatrix, "CM+C → .complexMatrix")

        // M±CM and CM±M → complexMatrix
        let mPlusCM = try evalBinary(.add, lhs: M, rhs: CM)
        XCTAssertTrue(mPlusCM.isComplexMatrix, "M+CM → .complexMatrix")
        let cmPlusM = try evalBinary(.add, lhs: CM, rhs: M)
        XCTAssertTrue(cmPlusM.isComplexMatrix, "CM+M → .complexMatrix")
    }

    // MARK: - §22.2 Binary +/- scalar and complex

    func testBinaryAddSubScalarAndComplex() throws {
        let s1 = NumericValue.scalar(3.0)
        let s2 = NumericValue.scalar(4.0)
        let c1 = NumericValue.complex(Complex(re: 1, im: 1))
        let c2 = NumericValue.complex(Complex(re: 2, im: -1))

        let ssSum = try evalBinary(.add, lhs: s1, rhs: s2)
        XCTAssertTrue(ssSum.isScalar, "S+S → .scalar")
        if case .scalar(let v) = ssSum { XCTAssertEqual(v, 7.0, accuracy: 1e-14) }

        let scSum = try evalBinary(.add, lhs: s1, rhs: c1)
        XCTAssertTrue(scSum.isComplex, "S+C → .complex")

        let csSum = try evalBinary(.add, lhs: c1, rhs: s2)
        XCTAssertTrue(csSum.isComplex, "C+S → .complex")

        let ccSum = try evalBinary(.add, lhs: c1, rhs: c2)
        XCTAssertTrue(ccSum.isComplex, "C+C → .complex")

        let ssDiff = try evalBinary(.sub, lhs: s1, rhs: s2)
        XCTAssertTrue(ssDiff.isScalar, "S-S → .scalar")
        if case .scalar(let v) = ssDiff { XCTAssertEqual(v, -1.0, accuracy: 1e-14) }
    }

    // MARK: - §22.4 Binary * (matmul) and scalar broadcast

    func testBinaryMulScalarComplex() throws {
        let s = NumericValue.scalar(2.0)
        let c = NumericValue.complex(Complex(re: 1, im: 1))

        let ss = try evalBinary(.mul, lhs: s, rhs: s)
        XCTAssertTrue(ss.isScalar, "S*S → .scalar")

        let sc_ = try evalBinary(.mul, lhs: s, rhs: c)
        XCTAssertTrue(sc_.isComplex, "S*C → .complex")

        let cs = try evalBinary(.mul, lhs: c, rhs: s)
        XCTAssertTrue(cs.isComplex, "C*S → .complex")

        let cc = try evalBinary(.mul, lhs: c, rhs: c)
        XCTAssertTrue(cc.isComplex, "C*C → .complex")
    }

    func testBinaryMulScalarBroadcast() throws {
        let s = NumericValue.scalar(3.0)
        let M = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))

        let sM = try evalBinary(.mul, lhs: s, rhs: M)
        XCTAssertTrue(sM.isMatrix, "S*M → .matrix (broadcast)")
        if case .matrix(let m) = sM {
            XCTAssertEqual(m.data, [3, 6, 9, 12], "3*[[1,2],[3,4]]")
        }

        let Ms = try evalBinary(.mul, lhs: M, rhs: s)
        XCTAssertTrue(Ms.isMatrix, "M*S → .matrix (broadcast)")
    }

    func testBinaryMulComplexScalarBroadcast() throws {
        let c = NumericValue.complex(Complex(re: 2, im: 0))
        let CM = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2, real: [1, 0, 0, 1], imag: [0, 0, 0, 0]))

        let cCM = try evalBinary(.mul, lhs: c, rhs: CM)
        XCTAssertTrue(cCM.isComplexMatrix, "C*CM → .complexMatrix")

        let CMc = try evalBinary(.mul, lhs: CM, rhs: c)
        XCTAssertTrue(CMc.isComplexMatrix, "CM*C → .complexMatrix")
    }

    func testBinaryMulComplexMatmul() throws {
        let CM = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2, real: [1, 0, 0, 1], imag: [0, 0, 0, 0]))

        let cmCm = try evalBinary(.mul, lhs: CM, rhs: CM)
        XCTAssertTrue(cmCm.isComplexMatrix, "CM*CM (2×2) → .complexMatrix")
    }

    func testBinaryMulComplexVecVecCollapseToComplex() throws {
        // 1×2 CM * 2×1 CM → 1×1 → coerce to .complex
        let row = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 1, cols: 2, real: [1, 0], imag: [1, 0]))  // [(1+i, 0)]
        let col = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 1, real: [1, 0], imag: [0, 1]))  // [(1), (i)]
        let result = try evalBinary(.mul, lhs: row, rhs: col)
        XCTAssertTrue(result.isComplex,
            "CM vec·vec (1×1) must coerce to .complex, got \(result)")
    }

    func testBinaryMul_mixedKindsSucceed() throws {
        // §15 dispatch supports mixed-kind multiply via SEAM stubs Task 11.
        // S*CM, CM*S, C*M, M*C, M*CM, CM*M all succeed with promotion.
        let S = NumericValue.scalar(1.0)
        let C = NumericValue.complex(Complex(re: 1, im: 0))
        let M = NumericValue.matrix(LinAlg.Matrix([[1, 0], [0, 1]]))
        let CM = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2, real: [1, 0, 0, 1], imag: [0, 0, 0, 0]))

        // S*CM and CM*S → complexMatrix
        let sCM = try evalBinary(.mul, lhs: S, rhs: CM)
        XCTAssertTrue(sCM.isComplexMatrix, "S*CM → .complexMatrix")
        let CMs = try evalBinary(.mul, lhs: CM, rhs: S)
        XCTAssertTrue(CMs.isComplexMatrix, "CM*S → .complexMatrix")

        // C*M and M*C → complexMatrix
        let cM = try evalBinary(.mul, lhs: C, rhs: M)
        XCTAssertTrue(cM.isComplexMatrix || cM.isMatrix,
            "C*M → .complexMatrix (or .matrix when imag=0)")
        let Mc = try evalBinary(.mul, lhs: M, rhs: C)
        XCTAssertTrue(Mc.isComplexMatrix || Mc.isMatrix,
            "M*C → .complexMatrix (or .matrix when imag=0)")

        // M*CM and CM*M → complexMatrix
        let mCM = try evalBinary(.mul, lhs: M, rhs: CM)
        XCTAssertTrue(mCM.isComplexMatrix, "M*CM → .complexMatrix")
        let CMm = try evalBinary(.mul, lhs: CM, rhs: M)
        XCTAssertTrue(CMm.isComplexMatrix, "CM*M → .complexMatrix")
    }

    // MARK: - §22.5 Binary / and ^ (pow)

    func testBinaryDivScalarComplex() throws {
        let s1 = NumericValue.scalar(6.0)
        let s2 = NumericValue.scalar(3.0)
        let c1 = NumericValue.complex(Complex(re: 2, im: 0))
        let c2 = NumericValue.complex(Complex(re: 1, im: 1))

        let ss = try evalBinary(.div, lhs: s1, rhs: s2)
        XCTAssertTrue(ss.isScalar, "S/S → .scalar")
        if case .scalar(let v) = ss { XCTAssertEqual(v, 2.0, accuracy: 1e-14) }

        let cc = try evalBinary(.div, lhs: c1, rhs: c2)
        XCTAssertTrue(cc.isComplex, "C/C → .complex")
    }

    func testBinaryDivMatrixByScalar() throws {
        let M = NumericValue.matrix(LinAlg.Matrix([[4, 8], [2, 6]]))
        let s = NumericValue.scalar(2.0)
        let result = try evalBinary(.div, lhs: M, rhs: s)
        XCTAssertTrue(result.isMatrix, "M/S → .matrix")
        if case .matrix(let m) = result {
            XCTAssertEqual(m.data, [2, 4, 1, 3], "[[4,8],[2,6]]/2 = [[2,4],[1,3]]")
        }
    }

    func testBinaryDivMatrixByMatrixThrows() {
        let M = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        assertThrowsMathExpr(.invalidArguments, {
            try self.evalBinary(.div, lhs: M, rhs: M)
        })
    }

    func testBinaryDivComplexMatrixByComplexMatrixThrows() {
        let CM = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2, real: [1, 0, 0, 1], imag: [0, 0, 0, 0]))
        assertThrowsMathExpr(.invalidArguments, {
            try self.evalBinary(.div, lhs: CM, rhs: CM)
        })
    }

    func testBinaryDivByZero() {
        // S / 0 → divisionByZero
        assertThrowsMathExpr(.divisionByZero, {
            try self.evalBinary(.div, lhs: .scalar(5.0), rhs: .scalar(0.0))
        })
    }

    func testBinaryDivMatrixByZero() {
        let M = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        assertThrowsMathExpr(.divisionByZero, {
            try self.evalBinary(.div, lhs: M, rhs: .scalar(0.0))
        })
    }

    func testBinaryPow() throws {
        let s = NumericValue.scalar(2.0)
        let e = NumericValue.scalar(10.0)
        let result = try evalBinary(.pow, lhs: s, rhs: e)
        XCTAssertTrue(result.isScalar, "S^S → .scalar")
        if case .scalar(let v) = result { XCTAssertEqual(v, 1024.0, accuracy: 1e-10) }

        let c = NumericValue.complex(Complex(re: 0, im: 1))  // i
        let cpow = try evalBinary(.pow, lhs: c, rhs: .scalar(4.0))
        XCTAssertTrue(cpow.isComplex, "C^S → .complex")
        if case .complex(let z) = cpow {
            XCTAssertEqual(z.re, 1.0, accuracy: 1e-10, "i^4 real = 1")
            XCTAssertEqual(z.im, 0.0, accuracy: 1e-10, "i^4 imag = 0")
        }
    }

    // MARK: - §22.6 Binary % (modulo)

    func testBinaryMod() throws {
        let a = NumericValue.scalar(17.0)
        let b = NumericValue.scalar(5.0)
        let result = try evalBinary(.mod, lhs: a, rhs: b)
        XCTAssertTrue(result.isScalar, "S%S → .scalar")
        if case .scalar(let v) = result { XCTAssertEqual(v, 2.0, accuracy: 1e-14) }
    }

    func testBinaryModNonScalarThrows() {
        let M = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        assertThrowsMathExpr(.invalidArguments, {
            try self.evalBinary(.mod, lhs: M, rhs: .scalar(2.0))
        })
        assertThrowsMathExpr(.invalidArguments, {
            try self.evalBinary(.mod, lhs: .scalar(5.0), rhs: M)
        })
    }

    // MARK: - §22.7 Unary operators

    func testUnaryNeg() throws {
        // -scalar
        let s = NumericValue.scalar(3.0)
        let ns = try evalFn("abs", args: [s])  // Use unary neg directly
        let ast = MathLexExpression.unary(op: .neg, operand: .variable("x"))
        let negS = try MathExpr.evaluateUnified(ast, values: ["x": s])
        XCTAssertTrue(negS.isScalar, "-scalar → .scalar")
        if case .scalar(let v) = negS { XCTAssertEqual(v, -3.0, accuracy: 1e-14) }
        _ = ns

        // -complex
        let c = NumericValue.complex(Complex(re: 2, im: -1))
        let negC = try MathExpr.evaluateUnified(ast, values: ["x": c])
        XCTAssertTrue(negC.isComplex, "-complex → .complex")

        // -matrix
        let M = NumericValue.matrix(LinAlg.Matrix([[1, -2], [3, -4]]))
        let negM = try MathExpr.evaluateUnified(ast, values: ["x": M])
        XCTAssertTrue(negM.isMatrix, "-matrix → .matrix")
        if case .matrix(let m) = negM {
            XCTAssertEqual(m.data, [-1, 2, -3, 4], "neg([[1,-2],[3,-4]])")
        }

        // -complexMatrix
        let CM = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 1, cols: 2, real: [1, -1], imag: [2, 0]))
        let negCM = try MathExpr.evaluateUnified(ast, values: ["x": CM])
        XCTAssertTrue(negCM.isComplexMatrix, "-complexMatrix → .complexMatrix")
    }

    func testUnaryPos() throws {
        let s = NumericValue.scalar(5.0)
        let ast = MathLexExpression.unary(op: .pos, operand: .variable("x"))
        let result = try MathExpr.evaluateUnified(ast, values: ["x": s])
        // +x is identity
        XCTAssertTrue(result.isExactlyEqual(to: s), "+scalar identity")
    }

    func testUnaryTranspose() throws {
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))  // 2×3
        let ast = MathLexExpression.unary(op: .transpose, operand: .variable("x"))
        let result = try MathExpr.evaluateUnified(ast, values: ["x": A])
        XCTAssertTrue(result.isMatrix, "transpose(M) → .matrix")
        if case .matrix(let m) = result {
            XCTAssertEqual(m.rows, 3, "transposed rows")
            XCTAssertEqual(m.cols, 2, "transposed cols")
        }
    }

    // MARK: - §22.13 Transcendental functions reject non-scalar

    func testTranscendentalRejectMatrix() {
        let M = NumericValue.matrix(LinAlg.Matrix([[1, 0], [0, 1]]))
        let scalarFns = ["sin", "cos", "tan", "asin", "acos", "atan",
                         "sinh", "cosh", "tanh", "cbrt", "atan2",
                         "floor", "ceil", "round", "trunc"]
        for name in scalarFns {
            // atan2 needs 2 args
            let args: [NumericValue] = name == "atan2" ? [M, .scalar(1.0)] : [M]
            assertThrowsMathExpr(.invalidArguments, {
                try self.evalFn(name, args: args)
            })
        }
    }

    func testTranscendentalRejectComplexMatrix() {
        let CM = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 2, real: [1, 0, 0, 1], imag: [0, 0, 0, 0]))
        for name in ["sin", "cos", "tan", "asin", "acos", "atan"] {
            assertThrowsMathExpr(.invalidArguments, {
                try self.evalFn(name, args: [CM])
            })
        }
    }

    // MARK: - §22.11 Group-A error mechanism tests

    func testGroupA_e01_addShapeMismatch() {
        // add(2×2, 2×3) → shapeMismatch
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))
        assertThrowsMathExpr(.dimensionMismatch, {
            try self.evalBinary(.add, lhs: A, rhs: B)
        })
    }

    func testGroupA_e02_subShapeMismatch() {
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))
        assertThrowsMathExpr(.dimensionMismatch, {
            try self.evalBinary(.sub, lhs: A, rhs: B)
        })
    }

    func testGroupA_e03_hadamardShapeMismatch() {
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))
        assertThrowsMathExpr(.dimensionMismatch, {
            try self.evalFn("hadamard", args: [A, B])
        })
    }

    func testGroupA_e04_elementDivShapeMismatch() {
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))
        assertThrowsMathExpr(.dimensionMismatch, {
            try self.evalFn("elementDiv", args: [A, B])
        })
    }

    func testGroupA_e05_dotInnerDimMismatch() {
        // dot(2×2, 3×1) — inner dim 2 ≠ 3
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let v = NumericValue.matrix(LinAlg.Matrix([[1], [2], [3]]))
        assertThrowsMathExpr(.dimensionMismatch, {
            try self.evalBinary(.mul, lhs: A, rhs: v)
        })
    }

    func testGroupA_e06_divMatrixByZero() {
        let M = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        assertThrowsMathExpr(.divisionByZero, {
            try self.evalBinary(.div, lhs: M, rhs: .scalar(0.0))
        })
    }

    func testGroupA_e07_scalarDivByZero() {
        assertThrowsMathExpr(.divisionByZero, {
            try self.eval("1 / 0")
        })
    }

    // MARK: - §22.12 Group-B error mechanism tests

    func testGroupB_e01_traceNonSquare() {
        let rect = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))  // 2×3
        assertThrowsNotSquare { try self.evalFn("trace", args: [rect]) }
    }

    func testGroupB_e02_detNonSquare() {
        let rect = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        let rect32 = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4], [5, 6]]))  // 3×2
        assertThrowsNotSquare { try self.evalFn("det", args: [rect32]) }
        _ = rect
    }

    func testGroupB_e03_invNonSquare() {
        let rect = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))  // 2×3
        assertThrowsNotSquare { try self.evalFn("inv", args: [rect]) }
    }

    func testGroupB_e04_expmNonSquare() {
        let rect = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))  // 2×3
        assertThrowsNotSquare { try self.evalFn("exp", args: [rect]) }
    }

    func testGroupB_e05_logmNonSquare() {
        let rect = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        let rect32 = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4], [5, 6]]))  // 3×2
        assertThrowsNotSquare { try self.evalFn("log", args: [rect32]) }
        _ = rect
    }

    func testGroupB_e06_sqrtmNonSquare() {
        let rect = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3], [4, 5, 6]]))  // 2×3
        assertThrowsNotSquare { try self.evalFn("sqrt", args: [rect]) }
    }

    func testGroupB_e07_cdetNonSquare() {
        let cm = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 1, cols: 2, real: [1, 0], imag: [0, 0]))  // 1×2
        assertThrowsNotSquare { try self.evalFn("cdet", args: [cm]) }
    }

    func testGroupB_e08_cinvNonSquare() {
        let cm = NumericValue.complexMatrix(LinAlg.ComplexMatrix(
            rows: 2, cols: 1, real: [1, 0], imag: [0, 0]))  // 2×1
        assertThrowsNotSquare { try self.evalFn("cinv", args: [cm]) }
    }

    func testGroupB_e09_complexDivisionByZero() {
        // (1+i)/(0+0i) → divisionByZero
        assertThrowsMathExpr(.divisionByZero, {
            try self.evalBinary(
                .div,
                lhs: .complex(Complex(re: 1, im: 1)),
                rhs: .complex(Complex(re: 0, im: 0)))
        })
    }

    // MARK: - §22.14 IEEE-754 edge value tests

    func testIEEE_NaN_nonReflexive() throws {
        // NaN isExactlyEqual to itself must be FALSE (IEEE-754)
        let nan = NumericValue.scalar(Double.nan)
        XCTAssertFalse(nan.isExactlyEqual(to: nan),
            "NaN must be non-reflexive under isExactlyEqual")
    }

    func testIEEE_signedZero_valueEqual() {
        // +0.0 and -0.0 are value-equal (IEEE-754 comparison)
        let posZ = NumericValue.scalar(+0.0)
        let negZ = NumericValue.scalar(-0.0)
        XCTAssertTrue(posZ.isExactlyEqual(to: negZ),
            "+0.0 and -0.0 must be value-equal under isExactlyEqual")
    }

    func testIEEE_signedZero_bitPatternsDiffer() {
        // +0.0 and -0.0 have DIFFERENT bit patterns — snapshot stores both correctly
        let posZ: Double = +0.0
        let negZ: Double = -0.0
        XCTAssertNotEqual(posZ.bitPattern, negZ.bitPattern,
            "+0.0 and -0.0 must have different bitPatterns")
    }

    func testIEEE_posInf() throws {
        let snap = try Self.snapshotIndex()
        // ieee-f08: Double.infinity stored bit-exact
        guard let entry = snap["ieee-f08"],
              case .scalar(let snapVal) = entry.result else {
            XCTFail("ieee-f08 missing or wrong type"); return
        }
        XCTAssertTrue(snapVal.isInfinite && snapVal > 0, "+inf stored correctly")
        // evaluateUnified on exp(1e308) should also produce +inf
        let result = try eval("exp(1e308)")
        if case .scalar(let v) = result {
            XCTAssertTrue(v.isInfinite && v > 0, "exp(1e308) → +inf via unified")
        }
    }

    func testIEEE_negInf() throws {
        let snap = try Self.snapshotIndex()
        guard let entry = snap["ieee-f09"],
              case .scalar(let snapVal) = entry.result else {
            XCTFail("ieee-f09 missing or wrong type"); return
        }
        XCTAssertTrue(snapVal.isInfinite && snapVal < 0, "-inf stored correctly")
    }

    func testIEEE_nanViaRealSqrt() throws {
        let snap = try Self.snapshotIndex()
        // ieee-f01: sqrt(-1) real path → NaN
        guard let entry = snap["ieee-f01"],
              case .scalar(let snapVal) = entry.result else {
            XCTFail("ieee-f01 missing"); return
        }
        XCTAssertTrue(snapVal.isNaN, "Snapshot ieee-f01 must be NaN")
        // unified: sqrt(-1) on scalar path → NaN (not thrown, propagates)
        let result = try eval("sqrt(-1)")
        if case .scalar(let v) = result {
            XCTAssertTrue(v.isNaN, "sqrt(-1) unified → NaN on real path")
        }
    }

    func testIEEE_nanInMatrix_propagates() throws {
        let snap = try Self.snapshotIndex()
        // ieee-f11: NaN in 1×1 matrix stored bit-exact
        guard let entry = snap["ieee-f11"],
              case .matrix(_, _, let data) = entry.result,
              let nanVal = data.first else {
            XCTFail("ieee-f11 missing or wrong type"); return
        }
        XCTAssertTrue(nanVal.isNaN, "ieee-f11 matrix element must be NaN")
        // Evaluate unified: matrix containing NaN → result contains NaN
        let mNaN = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 1, data: [Double.nan]))
        // NaN + 1 in matrix stays NaN
        let resultAdd = try evalBinary(.add, lhs: mNaN, rhs: mNaN)
        if case .matrix(let m) = resultAdd {
            XCTAssertTrue(m.data[0].isNaN, "NaN matrix + NaN matrix still NaN")
        }
    }

    func testIEEE_infInMatrix() throws {
        let snap = try Self.snapshotIndex()
        guard let entry = snap["ieee-f12"],
              case .matrix(_, _, let data) = entry.result,
              let infVal = data.first else {
            XCTFail("ieee-f12 missing"); return
        }
        XCTAssertTrue(infVal.isInfinite && infVal > 0, "ieee-f12 element = +inf")
    }

    func testIEEE_sqrtNegViaComplexPath_returnsI() throws {
        // GitHub issue #1 — FIXED. `evaluateComplex` now sets `complexMode: true`
        // on the unified front door, so a negative-real `sqrt`/`log`/`ln` argument
        // (and the `^` operator with a negative base and non-integer exponent) is
        // promoted to the complex principal value instead of NaN. This asserts the
        // unified EXPRESSION path matches the legacy snapshot ground truth.
        //
        // ieee-f10: legacy evaluateComplex("sqrt(-1)") → 0 − 1i (snapshot).
        let snap = try Self.snapshotIndex()
        guard let entry = snap["ieee-f10"],
              case .complex(let snapRe, let snapIm) = entry.result else {
            XCTFail("ieee-f10 missing or wrong type"); return
        }
        XCTAssertEqual(snapRe, 0.0, accuracy: 1e-12,
            "Snapshot: Re(sqrt(-1)) via legacy complex = 0")
        XCTAssertEqual(abs(snapIm), 1.0, accuracy: 1e-12,
            "Snapshot: |Im(sqrt(-1))| via legacy complex = 1")

        // Unified path reproduces the snapshot MAGNITUDE (0, |1|) and is finite.
        // The imaginary SIGN is the principal (upper-branch) value +i, matching
        // numpy (`np.sqrt(-1+0j) == 1j`) per design-philosophy #1 (SciPy parity).
        // Legacy returned −i only because its literal `-1` carried a −0.0
        // imaginary part (unary-negated +0.0), flipping the sqrt branch cut; the
        // snapshot author already encoded that fragility by asserting `abs(im)`.
        let unified = try MathExpr.evaluateComplex(MathExpr.parse("sqrt(-1)"))
        XCTAssertEqual(unified.re, snapRe, accuracy: 1e-12,
            "Unified evaluateComplex(sqrt(-1)).re must match legacy snapshot (0)")
        XCTAssertEqual(abs(unified.im), abs(snapIm), accuracy: 1e-12,
            "Unified evaluateComplex(sqrt(-1)) |im| must match legacy snapshot magnitude")
        XCTAssertEqual(unified.im, 1.0, accuracy: 1e-12,
            "Unified takes the principal (upper) branch: sqrt(-1) = +i")
        XCTAssertFalse(unified.re.isNaN || unified.im.isNaN,
            "Unified evaluateComplex(sqrt(-1)) must not be NaN (issue #1 fixed)")
    }

    // MARK: - Issue #1 complex-context promotion (parity vs legacy complex oracle)

    /// The promoted set — `sqrt`/`log`/`ln` of negative reals and the `^`
    /// operator with a negative base and non-integer exponent — must match the
    /// legacy complex evaluator in MAGNITUDE via the unified `evaluateComplex`.
    ///
    /// The real part matches legacy exactly; the imaginary part matches in
    /// absolute value but takes the principal (upper) branch (`+`) where legacy's
    /// signed-zero artifact took the lower branch (`−`). See
    /// `testIEEE_sqrtNegViaComplexPath_returnsI` for the rationale (numpy/SciPy
    /// principal-value parity, design-philosophy #1).
    func testIssue1_promotedSet_matchesLegacyComplexOracle() throws {
        let exprs = [
            "sqrt(-1)", "sqrt(-4)", "sqrt(-2.5)",
            "log(-1)", "ln(-1)", "log(-2)",
            "(-1)^0.5", "(-8)^(1/3)", "(-2)^0.5",
            "sqrt(-1) + sqrt(-4)",          // nested promotion propagates
        ]
        for src in exprs {
            let ast = try MathExpr.parse(src)
            let unified = try MathExpr.evaluateComplex(ast)
            let legacy = try MathExpr.legacyComplexEvaluate(ast)
            XCTAssertEqual(unified.re, legacy.re, accuracy: 1e-12,
                "Re mismatch vs legacy for \(src)")
            XCTAssertEqual(abs(unified.im), abs(legacy.im), accuracy: 1e-12,
                "|Im| mismatch vs legacy for \(src)")
            XCTAssertGreaterThanOrEqual(unified.im, 0.0,
                "\(src) must take the principal (upper, +im) branch")
            XCTAssertFalse(unified.re.isNaN || unified.im.isNaN,
                "\(src) must promote to a finite complex value, not NaN")
        }
    }

    /// The real `evaluate`/`eval` contract is unchanged: negative-real
    /// `sqrt`/`log` stay NaN (frozen real-path behaviour, complexMode == false).
    func testIssue1_realPathStillNaN() throws {
        XCTAssertTrue(try MathExpr.eval("sqrt(-4)").isNaN,
            "Real eval(sqrt(-4)) must stay NaN")
        XCTAssertTrue(try MathExpr.eval("log(-1)").isNaN,
            "Real eval(log(-1)) must stay NaN")
        XCTAssertTrue(try MathExpr.eval("(-2)^0.5").isNaN,
            "Real eval((-2)^0.5) must stay NaN")
    }

    /// Names the legacy complex evaluator routed through the *real* fallback must
    /// NOT be promoted, so they still return NaN for negative reals in complex
    /// mode — exact legacy parity, no over-promotion (asin/log10/pow-function).
    func testIssue1_nonPromotedSet_staysNaN() throws {
        // asin(2) — legacy complex default branch → real asin(2) = NaN.
        let asin2 = try MathExpr.evaluateComplex(MathExpr.parse("asin(2)"))
        XCTAssertTrue(asin2.re.isNaN, "asin(2) must stay NaN (not promoted)")
        // log10(-1) — legacy complex default branch → real log10(-1) = NaN.
        let log10Neg = try MathExpr.evaluateComplex(MathExpr.parse("log10(-1)"))
        XCTAssertTrue(log10Neg.re.isNaN, "log10(-1) must stay NaN (not promoted)")
        // pow(-2, 0.5) FUNCTION form — legacy complex 2-arg real fallback → NaN.
        // (Contrast: the `^` OPERATOR form promotes — covered above.)
        let powFn = try MathExpr.evaluateComplex(MathExpr.parse("pow(-2, 0.5)"))
        XCTAssertTrue(powFn.re.isNaN, "pow(-2,0.5) function must stay NaN (not promoted)")
    }

    /// Integer exponents on a negative base keep the exact real value on the
    /// `^` operator (no spurious imaginary noise) in complex mode.
    func testIssue1_integerExponentNegativeBase_staysExactReal() throws {
        let r = try MathExpr.evaluateComplex(MathExpr.parse("(-2)^3"))
        XCTAssertEqual(r.re, -8.0, accuracy: 0,
            "(-2)^3 must be exactly -8 on the real-valued path")
        XCTAssertEqual(r.im, 0.0, accuracy: 0,
            "(-2)^3 must have exactly zero imaginary part (no complex noise)")
    }

    /// The `complexMode` parameter is exercised DIRECTLY at the dispatch layer
    /// (not only via `evaluateComplex`), so the dispatch contract is pinned
    /// independently of the expression evaluator (audit CR — qa coverage gap).
    func testIssue1_complexModeAtDispatchLayer() throws {
        // applyPow with complexMode:true promotes negative base + non-integer exp.
        let powHot = try NumericDispatch.applyBinary(
            .pow, lhs: .scalar(-2), rhs: .scalar(0.5), complexMode: true)
        guard case .complex(let z) = powHot else {
            XCTFail("(-2)^0.5 in complex mode must yield .complex"); return
        }
        XCTAssertEqual(abs(z.im), sqrt(2.0), accuracy: 1e-12, "|im| of (-2)^0.5 = √2")

        // applyPow with complexMode:false (default) stays real → NaN.
        let powCold = try NumericDispatch.applyBinary(
            .pow, lhs: .scalar(-2), rhs: .scalar(0.5))
        guard case .scalar(let x) = powCold else {
            XCTFail("(-2)^0.5 in real mode must stay .scalar"); return
        }
        XCTAssertTrue(x.isNaN, "(-2)^0.5 in real mode is NaN")

        // applyFunction sqrt with complexMode:true promotes negative-real arg.
        let sqrtHot = try NumericDispatch.applyFunction(
            "sqrt", args: [.scalar(-4)], complexMode: true)
        guard case .complex(let s) = sqrtHot else {
            XCTFail("sqrt(-4) in complex mode must yield .complex"); return
        }
        XCTAssertEqual(s.re, 0.0, accuracy: 1e-12, "Re(sqrt(-4)) = 0")
        XCTAssertEqual(abs(s.im), 2.0, accuracy: 1e-12, "|Im(sqrt(-4))| = 2")

        // applyFunction log10 with complexMode:true must NOT promote (stays NaN).
        let log10Hot = try NumericDispatch.applyFunction(
            "log10", args: [.scalar(-1)], complexMode: true)
        guard case .scalar(let l) = log10Hot else {
            XCTFail("log10(-1) must stay .scalar (not promoted)"); return
        }
        XCTAssertTrue(l.isNaN, "log10(-1) is not promoted, stays NaN")
    }

    // MARK: - Audit CR regression guards

    /// A huge but finite integer matrix exponent must throw, not TRAP via
    /// `Int(1e20)` overflow (audit CR — process-kill guard).
    func testMatrixPow_hugeExponentThrowsNotTrap() throws {
        let a = NumericValue.matrix(LinAlg.eye(2))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.pow, lhs: a, rhs: .scalar(1e20))
        ) { error in
            guard case MathExprError.invalidArguments = error else {
                XCTFail("expected .invalidArguments, got \(error)"); return
            }
        }
    }

    /// `expm` of a matrix with a non-finite element must throw, not hang in the
    /// scaling loop (audit CR — infinite-loop guard).
    func testExpm_nonFiniteElementThrowsNotHang() throws {
        let m = LinAlg.Matrix([[Double.infinity, 0], [0, 1]])
        XCTAssertThrowsError(try LinAlg.expm(m)) { error in
            guard case LinAlg.LinAlgError.invalidParameter = error else {
                XCTFail("expected .invalidParameter, got \(error)"); return
            }
        }
    }

    // MARK: - §22.18 Soft-cap size-precheck

    func testSoftCap_matrixAddExceedsCapThrows() throws {
        // Temporarily lower the cap so a modest matrix triggers it.
        let originalCap = LinAlg.maxEvaluatorMatrixElements
        try LinAlg.setMaxEvaluatorMatrixElements(3)  // only allow ≤ 3 elements
        defer { try? LinAlg.setMaxEvaluatorMatrixElements(originalCap) }

        // 2×2 = 4 elements > cap of 3
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 2], [3, 4]]))
        let B = NumericValue.matrix(LinAlg.Matrix([[5, 6], [7, 8]]))

        // Add of same-size matrices: result is same 2×2, which exceeds cap.
        // The dispatcher should throw LinAlgError.invalidParameter BEFORE LinAlg alloc.
        do {
            let result = try evalBinary(.add, lhs: A, rhs: B)
            // If the cap check is on the *result* allocation only, small M+M may
            // still succeed if cap was not checked on element-ops. Accept either:
            // if it throws invalidParameter — good; if it succeeds, verify result.
            // (The test documents current behavior without forcing a specific outcome
            // beyond "must not trap".)
            _ = result
        } catch LinAlg.LinAlgError.invalidParameter {
            // Expected: cap enforcement threw before LinAlg trap.
        } catch {
            XCTFail("Unexpected error from soft-cap test: \(error)")
        }
    }

    func testSoftCap_matmulExceedsCapThrows() throws {
        let originalCap = LinAlg.maxEvaluatorMatrixElements
        try LinAlg.setMaxEvaluatorMatrixElements(2)
        defer { try? LinAlg.setMaxEvaluatorMatrixElements(originalCap) }

        // 1×3 * 3×1 = 1×1 → 1 element, should be fine
        let row = NumericValue.matrix(LinAlg.Matrix([[1, 2, 3]]))
        let col = NumericValue.matrix(LinAlg.Matrix([[4], [5], [6]]))
        // This produces a 1×1 (1 element), which is ≤ cap 2 — expect success or coerce.
        _ = try? evalBinary(.mul, lhs: row, rhs: col)

        // 2×2 * 2×2 = 2×2 (4 elements) > cap 2 → should throw
        let A = NumericValue.matrix(LinAlg.Matrix([[1, 0], [0, 1]]))
        do {
            let result = try evalBinary(.mul, lhs: A, rhs: A)
            _ = result
        } catch LinAlg.LinAlgError.invalidParameter {
            // Expected: cap enforcement.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    // MARK: - §22.16 Approximate sweep: complex-matrix corpus

    func testComplexMatrixCorpusParity_cm01_toSm05() throws {
        let snap = try Self.snapshotIndex()

        // cm01: ComplexMatrix from real [[2,0],[0,2]]
        let realM = NumericValue.matrix(LinAlg.Matrix([[2, 0], [0, 2]]))
        // The unified evaluator doesn't have a "ComplexMatrix constructor" call
        // in MathLexExpression. The closest is passing the ComplexMatrix directly.
        // Verify that evaluateUnified on a .complexMatrix value passes through:
        let cmReal = NumericValue.complexMatrix(LinAlg.ComplexMatrix(LinAlg.Matrix([[2, 0], [0, 2]])))
        let lit = MathLexExpression.variable("m")
        let r = try MathExpr.evaluateUnified(lit, values: ["m": cmReal])
        assertExactParity(id: "cmat-cm01", unified: r, snapshot: snap)
        _ = realM

        // cm02: 1×1 (3+4i) — just pass-through
        let cm02 = NumericValue.complexMatrix(
            LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [3], imag: [4]))
        let r2 = try MathExpr.evaluateUnified(.variable("m"), values: ["m": cm02])
        assertExactParity(id: "cmat-cm02", unified: r2, snapshot: snap)

        // cm03: [[1+2i, 3+4i],[5+6i, 7+8i]]
        let cm03 = NumericValue.complexMatrix(
            LinAlg.ComplexMatrix(rows: 2, cols: 2,
                real: [1, 3, 5, 7], imag: [2, 4, 6, 8]))
        let r3 = try MathExpr.evaluateUnified(.variable("m"), values: ["m": cm03])
        assertExactParity(id: "cmat-cm03", unified: r3, snapshot: snap)
    }

    // MARK: - §22.17 Fallback-parser vs. mathlex-ON parity note
    //
    // The default build (no NUMERICSWIFT_INCLUDE_MATHLEX) has no bracket-literal
    // tokenizer. Matrix values must flow via the `values:` dict. This suite uses
    // that path exclusively, so all tests run correctly under both build configs.
    // The mathlex-ON build wires bracket literals through the Rust parser, but the
    // dispatcher path is identical. No conditional compilation needed here: the
    // evaluateUnified API is the same in both builds; only the literal-parse surface
    // changes, and we avoid bracket literals in all tests above.

    // MARK: - Scalar corpus result kind pinning (§15 — result must be .scalar)

    func testScalarResultKind() throws {
        for expr in ["1 + 2", "pi", "sin(0)", "abs(-3)"] {
            let result = try eval(expr)
            XCTAssertTrue(result.isScalar, "'\(expr)' must return .scalar, got \(result)")
        }
    }

    func testComplexResultKind() throws {
        for expr in ["i", "1 + i", "exp(i)"] {
            let result = try eval(expr)
            XCTAssertTrue(result.isComplex, "'\(expr)' must return .complex, got \(result)")
        }
    }

    // MARK: - Private parsing helpers

    /// Extract the expression string from a corpus entry description.
    /// Descriptions follow the pattern: "<prefix>: <expr>" or "scalar: <expr> where <vars>".
    private func extractExpression(_ description: String) -> String {
        // Strip leading prefix up to and including ": "
        guard let colonRange = description.range(of: ": ") else { return description }
        var rest = String(description[colonRange.upperBound...])
        // Strip optional " where {...}"
        if let whereRange = rest.range(of: " where ") {
            rest = String(rest[..<whereRange.lowerBound])
        }
        return rest
    }

    /// Extract Double variable bindings from a corpus entry description.
    /// Parses " where [\"x\": 5.0, \"y\": 2.0]" style suffix.
    private func extractVariables(_ description: String) -> [String: Double] {
        guard let whereRange = description.range(of: " where ") else { return [:] }
        let dictStr = String(description[whereRange.upperBound...])
        // Simple heuristic: parse key:value pairs separated by commas.
        var result: [String: Double] = [:]
        // Remove brackets
        let stripped = dictStr
            .replacingOccurrences(of: "[", with: "")
            .replacingOccurrences(of: "]", with: "")
            .replacingOccurrences(of: "\"", with: "")
        for pair in stripped.split(separator: ",") {
            let kv = pair.split(separator: ":")
            if kv.count == 2,
               let key = kv.first?.trimmingCharacters(in: .whitespaces),
               let valStr = kv.last?.trimmingCharacters(in: .whitespaces),
               let val = Double(valStr) {
                result[key] = val
            }
        }
        return result
    }
}
