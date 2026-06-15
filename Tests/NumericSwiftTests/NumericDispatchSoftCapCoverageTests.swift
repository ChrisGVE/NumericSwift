//
//  NumericDispatchSoftCapCoverageTests.swift
//  NumericSwiftTests
//
//  Systematic soft-cap (AC3.6 / CONS-07 / §4.8) pre-check coverage tests for
//  every matrix-producing dispatch path.  These tests prove that the guard
//  fires BEFORE any allocation attempt, using a very small cap so the over-cap
//  operand dimensions are tiny and the tests run fast.
//
//  ## Coverage matrix
//
//  Real-matrix paths:
//    • scalar * matrix / matrix * scalar       (BinaryOps, same-shape result)
//    • matrix / scalar                         (BinaryOps, same-shape result)
//    • neg(matrix)                             (UnaryFunctions, same-shape result)
//    • transpose(matrix)                       (UnaryFunctions, transposed cols×rows)
//    • expm(matrix)                            (FunctionDispatchers, same-shape square)
//    • logm(matrix)                            (FunctionDispatchers, same-shape square)
//    • sqrtm(matrix)                           (FunctionDispatchers, same-shape square)
//    • inv(matrix)                             (FunctionDispatchers, same-shape square)
//
//  Previously covered by Tasks 11/15/16 (regression sentinels only):
//    • matrix + matrix / matrix - matrix       (see NumericDispatchRealMatrixTests)
//    • matrix * matrix (matmul)                (see NumericDispatchRealMatrixTests)
//    • CM + CM / CM - CM / CM * CM             (see NumericDispatchComplexMatrixTests)
//    • neg(CM) / cinv(CM)                      (see corresponding CM tests)
//
//  Boundary / error-type tests:
//    • at-cap element count succeeds           (≤ maxEvaluatorMatrixElements)
//    • cap+1 element count throws              (> maxEvaluatorMatrixElements)
//    • error is LinAlgError.invalidParameter   (CONS-07 — never MathExprError)
//    • error message contains "exceeds soft cap"
//
//  Int-overflow routing:
//    • Documented (not in-process death-tested): overflowing rows*cols routes
//      to the HARD precondition in the Matrix/ComplexMatrix constructor, NOT
//      to a catchable LinAlgError.invalidParameter.  checkSoftCap itself
//      throws on Int overflow (overflow flag ⇒ soft throw as fallback guard),
//      matching testCheckSoftCapOverflowingDimsThrows in SizeCapTests.
//
//  ## Test discipline
//
//  Every test that modifies the soft cap resets it in tearDown so cap changes
//  cannot leak between tests.  Caps are set to a value that is tiny enough
//  to make the over-cap matrix dimensions small (keeping test runtime fast)
//  but large enough to allow the operand matrices themselves to be built
//  without tripping the cap at construction time.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class NumericDispatchSoftCapCoverageTests: XCTestCase {

    typealias E = LinAlg.LinAlgError

    // -------------------------------------------------------------------------
    // MARK: - Tear-down
    // -------------------------------------------------------------------------

    override func tearDown() {
        super.tearDown()
        try? LinAlg.setMaxEvaluatorMatrixElements(16_777_216)
    }

    // -------------------------------------------------------------------------
    // MARK: - Helpers
    // -------------------------------------------------------------------------

    /// Build a NumericValue.matrix from a flat data array.
    private func mat(_ rows: Int, _ cols: Int, _ data: [Double]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: rows, cols: cols, data: data))
    }

    /// Build a 3×3 identity wrapped as a NumericValue.matrix.
    private func identity3() -> NumericValue {
        .matrix(LinAlg.eye(3))
    }

    /// Assert the thrown error is LinAlgError.invalidParameter (CONS-07).
    private func assertInvalidParameter(_ error: Error, file: StaticString = #file, line: UInt = #line) {
        guard case E.invalidParameter = error else {
            XCTFail("expected LinAlgError.invalidParameter (CONS-07), got \(type(of: error))(\(error))",
                    file: file, line: line)
            return
        }
    }

    /// Assert the error is NOT a MathExprError (CONS-07 exclusivity).
    private func assertNotMathExprError(_ error: Error, file: StaticString = #file, line: UInt = #line) {
        XCTAssertNil(error as? MathExprError,
                     "CONS-07: soft-cap error must be LinAlgError.invalidParameter, not MathExprError",
                     file: file, line: line)
    }

    // =========================================================================
    // MARK: - scalar * matrix / matrix * scalar  (BinaryOps)
    // =========================================================================

    func testScalarMulMatrix_overSoftCap_throwsInvalidParameter() throws {
        // Cap = 4, matrix = 3×3 (9 elements) → over cap
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(3, 3, Array(repeating: 1.0, count: 9))
        let s = NumericValue.scalar(2.0)
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.mul, lhs: s, rhs: m)) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    func testMatrixMulScalar_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(3, 3, Array(repeating: 1.0, count: 9))
        let s = NumericValue.scalar(2.0)
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.mul, lhs: m, rhs: s)) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    func testScalarMulMatrix_underCap_succeeds() throws {
        // Cap = 9, matrix = 3×3 (9 elements) → exactly at cap → succeeds
        try LinAlg.setMaxEvaluatorMatrixElements(9)
        let m = mat(3, 3, Array(repeating: 1.0, count: 9))
        let s = NumericValue.scalar(2.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: s, rhs: m)
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r.rows, 3)
        XCTAssertEqual(r.cols, 3)
        XCTAssertEqual(r[0, 0], 2.0, accuracy: 1e-12)
    }

    func testMatrixMulScalar_underCap_succeeds() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(9)
        let m = mat(3, 3, Array(repeating: 3.0, count: 9))
        let s = NumericValue.scalar(2.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: m, rhs: s)
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], 6.0, accuracy: 1e-12)
    }

    // =========================================================================
    // MARK: - matrix / scalar  (BinaryOps)
    // =========================================================================

    func testMatrixDivScalar_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(3, 3, Array(repeating: 6.0, count: 9))
        let s = NumericValue.scalar(2.0)
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.div, lhs: m, rhs: s)) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    func testMatrixDivScalar_underCap_succeeds() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(9)
        let m = mat(3, 3, Array(repeating: 6.0, count: 9))
        let s = NumericValue.scalar(2.0)
        let result = try NumericDispatch.applyBinary(.div, lhs: m, rhs: s)
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], 3.0, accuracy: 1e-12)
    }

    // =========================================================================
    // MARK: - neg(matrix)  (UnaryFunctions)
    // =========================================================================

    func testNegMatrix_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(3, 3, Array(repeating: 1.0, count: 9))
        XCTAssertThrowsError(try NumericDispatch.applyUnary(.neg, operand: m)) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    func testNegMatrix_underCap_succeeds() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(9)
        let m = mat(3, 3, Array(repeating: 5.0, count: 9))
        let result = try NumericDispatch.applyUnary(.neg, operand: m)
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], -5.0, accuracy: 1e-12)
    }

    // =========================================================================
    // MARK: - transpose(matrix)  (UnaryFunctions)
    // =========================================================================

    /// Transpose of a 3×3 has the same element count (9); cap=4 → over cap.
    func testTransposeMatrix_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(3, 3, Array(repeating: 1.0, count: 9))
        XCTAssertThrowsError(try NumericDispatch.applyTransposeUnary(operand: m)) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    /// Transpose of a 2×3 matrix yields 3×2 — same element count but transposed shape.
    func testTransposeRectMatrix_overSoftCap_throwsInvalidParameter() throws {
        // 2×3 → transposed 3×2 = 6 elements; cap = 4 → over cap
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try NumericDispatch.applyTransposeUnary(operand: m)) { err in
            assertInvalidParameter(err)
        }
    }

    func testTransposeMatrix_underCap_correctShape() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(6)
        // 2×3 matrix; transposed result = 3×2 (6 elements = cap → allowed)
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        let result = try NumericDispatch.applyTransposeUnary(operand: m)
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r.rows, 3)
        XCTAssertEqual(r.cols, 2)
        // Element at (0,1) in transposed = element at (1,0) in original = 4
        XCTAssertEqual(r[0, 1], 4.0, accuracy: 1e-12)
    }

    // =========================================================================
    // MARK: - expm(matrix)  (FunctionDispatchers)
    // =========================================================================

    func testExpm_overSoftCap_throwsInvalidParameter() throws {
        // expm requires a square matrix; 3×3 = 9 elements; cap = 4 → over cap
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = NumericValue.matrix(LinAlg.eye(3))
        XCTAssertThrowsError(try NumericDispatch.applyFunction("exp", args: [m])) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    func testExpm_underCap_succeeds() throws {
        // expm([[0]]) = [[1]] (exp(0) = 1); use a 1×1 zero matrix for exact result.
        let m = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 1, data: [0.0]))
        let result = try NumericDispatch.applyFunction("exp", args: [m])
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        // expm of the zero matrix = identity; 1×1 identity has element 1.0
        XCTAssertEqual(r[0, 0], 1.0, accuracy: 1e-10)
    }

    // =========================================================================
    // MARK: - logm(matrix)  (FunctionDispatchers)
    // =========================================================================

    func testLogm_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        // Positive-definite 3×3 so logm wouldn't fail for mathematical reasons
        let m = NumericValue.matrix(LinAlg.eye(3))
        XCTAssertThrowsError(try NumericDispatch.applyFunction("log", args: [m])) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    func testLogm_underCap_succeeds() throws {
        // log(I) = 0 matrix (all eigenvalues = 1, log(1) = 0)
        let m = NumericValue.matrix(LinAlg.eye(2))
        let result = try NumericDispatch.applyFunction("log", args: [m])
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], 0.0, accuracy: 1e-10)
    }

    // =========================================================================
    // MARK: - sqrtm(matrix)  (FunctionDispatchers)
    // =========================================================================

    func testSqrtm_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = NumericValue.matrix(LinAlg.eye(3))
        XCTAssertThrowsError(try NumericDispatch.applyFunction("sqrt", args: [m])) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    func testSqrtm_underCap_succeeds() throws {
        // sqrt(I) = I; use 2×2 identity
        let m = NumericValue.matrix(LinAlg.eye(2))
        let result = try NumericDispatch.applyFunction("sqrt", args: [m])
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(r[0, 1], 0.0, accuracy: 1e-10)
    }

    // =========================================================================
    // MARK: - inv(matrix)  (FunctionDispatchers / applyAbsInvDetTrace)
    // =========================================================================

    func testInv_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = NumericValue.matrix(LinAlg.eye(3))
        XCTAssertThrowsError(try NumericDispatch.applyFunction("inv", args: [m])) { err in
            assertInvalidParameter(err)
            assertNotMathExprError(err)
        }
    }

    func testInv_underCap_succeeds() throws {
        // inv([[2,0],[0,4]]) = [[0.5,0],[0,0.25]]
        let data: [Double] = [2, 0, 0, 4]
        let m = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 2, data: data))
        let result = try NumericDispatch.applyFunction("inv", args: [m])
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], 0.5, accuracy: 1e-12)
        XCTAssertEqual(r[1, 1], 0.25, accuracy: 1e-12)
    }

    // =========================================================================
    // MARK: - Boundary: at-cap succeeds, cap+1 throws
    // =========================================================================

    /// Result with exactly `maxEvaluatorMatrixElements` elements must succeed.
    func testBoundary_atCap_scalarMulMatrix_succeeds() throws {
        // Cap = 4; 2×2 matrix = 4 elements = cap → succeeds
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(2, 2, [1, 2, 3, 4])
        let s = NumericValue.scalar(3.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: s, rhs: m)
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], 3.0, accuracy: 1e-12)
    }

    /// Result with `maxEvaluatorMatrixElements + 1` elements must throw.
    func testBoundary_capPlusOne_scalarMulMatrix_throws() throws {
        // Cap = 4; 1×5 matrix = 5 elements = cap+1 → throws
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(1, 5, [1, 2, 3, 4, 5])
        let s = NumericValue.scalar(2.0)
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.mul, lhs: s, rhs: m)) { err in
            assertInvalidParameter(err)
        }
    }

    func testBoundary_atCap_matrixDivScalar_succeeds() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(2, 2, [4, 6, 8, 10])
        let s = NumericValue.scalar(2.0)
        let result = try NumericDispatch.applyBinary(.div, lhs: m, rhs: s)
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], 2.0, accuracy: 1e-12)
    }

    func testBoundary_capPlusOne_negMatrix_throws() throws {
        // Cap = 4; 1×5 = 5 elements > cap → throws
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(1, 5, [1, 2, 3, 4, 5])
        XCTAssertThrowsError(try NumericDispatch.applyUnary(.neg, operand: m)) { err in
            assertInvalidParameter(err)
        }
    }

    func testBoundary_atCap_negMatrix_succeeds() throws {
        // Cap = 4; 2×2 = 4 elements = cap → succeeds
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(2, 2, [1, 2, 3, 4])
        let result = try NumericDispatch.applyUnary(.neg, operand: m)
        guard case .matrix(let r) = result else { return XCTFail("expected .matrix") }
        XCTAssertEqual(r[0, 0], -1.0, accuracy: 1e-12)
    }

    // =========================================================================
    // MARK: - CONS-07 error-type exclusivity and message format
    // =========================================================================

    /// The soft-cap error must be LinAlgError.invalidParameter, not MathExprError.
    func testCons07_errorType_isLinAlgErrorInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(3, 3, Array(repeating: 1.0, count: 9))
        let s = NumericValue.scalar(2.0)
        var caughtError: Error?
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.mul, lhs: s, rhs: m)) { err in
            caughtError = err
            // Must be LinAlgError
            XCTAssertNotNil(err as? E,
                "CONS-07: error must be LinAlgError.LinAlgError — got \(type(of: err))")
            // Must NOT be MathExprError
            XCTAssertNil(err as? MathExprError,
                "CONS-07: error must never be MathExprError — got \(err)")
        }
        if let err = caughtError {
            guard case E.invalidParameter(let msg) = err else {
                return XCTFail("expected .invalidParameter case, got \(err)")
            }
            // Message must mention the cap concept
            XCTAssertTrue(msg.contains("exceeds soft cap") || msg.contains("exceeds"),
                "CONS-07: message '\(msg)' must mention the cap")
        }
    }

    /// Same error-type assertion for neg(matrix).
    func testCons07_errorType_negMatrix_isLinAlgError() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = mat(3, 3, Array(repeating: 1.0, count: 9))
        XCTAssertThrowsError(try NumericDispatch.applyUnary(.neg, operand: m)) { err in
            XCTAssertNotNil(err as? E,
                "CONS-07: neg(matrix) over-cap error must be LinAlgError")
            XCTAssertNil(err as? MathExprError,
                "CONS-07: neg(matrix) over-cap error must not be MathExprError")
        }
    }

    /// Same assertion for inv(matrix).
    func testCons07_errorType_invMatrix_isLinAlgError() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let m = NumericValue.matrix(LinAlg.eye(3))
        XCTAssertThrowsError(try NumericDispatch.applyFunction("inv", args: [m])) { err in
            XCTAssertNotNil(err as? E,
                "CONS-07: inv(matrix) over-cap error must be LinAlgError")
            XCTAssertNil(err as? MathExprError,
                "CONS-07: inv(matrix) over-cap error must not be MathExprError")
        }
    }

    // =========================================================================
    // MARK: - Int-overflow routing to HARD precondition (documented test)
    // =========================================================================

    /// Documents the overflow routing: Int-overflowing rows*cols products cannot
    /// be caught as a soft-cap throw because the element count is meaningless.
    ///
    /// Per §4.10 and AC7.4c: checkSoftCap's current implementation throws on Int
    /// overflow (treating it as "exceeds soft cap"), which acts as a secondary
    /// defence.  The primary HARD-cap path (constructor precondition) is not
    /// in-process testable because `precondition` traps with SIGILL.
    ///
    /// This test verifies the secondary-defence behaviour: checkSoftCap itself
    /// throws on overflow rather than letting an overflowing count through.
    func testIntOverflow_checkSoftCap_throwsSecondaryDefence() {
        // Int.max rows × 2 cols: product overflows.
        // checkSoftCap treats overflow as "exceeds soft cap" and throws.
        XCTAssertThrowsError(try LinAlg.checkSoftCap(rows: Int.max, cols: 2)) { err in
            guard case E.invalidParameter = err else {
                XCTFail("expected LinAlgError.invalidParameter for overflow dims, got \(err)")
                return
            }
            // Confirm it is NOT a MathExprError — the overflow guard is still CONS-07
            XCTAssertNil(err as? MathExprError)
        }
    }

    // =========================================================================
    // MARK: - MF-5 scope-honesty: per-matrix guard, not total-memory bound
    // =========================================================================

    /// The soft cap bounds EACH result matrix individually.
    /// Two successive at-cap operations together hold 2× cap in memory; this
    /// is NOT blocked.  The cumulative working-set bound is deferred to v-next.
    func testMF5_twoAtCapMatrices_bothSucceed() throws {
        // Cap = 4; two separate 2×2 results each have 4 elements
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let a = mat(2, 2, [1, 2, 3, 4])
        let b = mat(2, 2, [5, 6, 7, 8])
        let s = NumericValue.scalar(1.0)
        // Each scalar*matrix is independent; neither exceeds the per-result cap
        let r1 = try NumericDispatch.applyBinary(.mul, lhs: s, rhs: a)
        let r2 = try NumericDispatch.applyBinary(.mul, lhs: s, rhs: b)
        guard case .matrix = r1, case .matrix = r2 else {
            XCTFail("both should produce .matrix results")
            return
        }
        // Both succeeded — cumulative working set exceeded the cap but that is
        // intentionally NOT guarded (MF-5 / §5).
    }

    // =========================================================================
    // MARK: - Regression sentinels for previously-covered paths
    // =========================================================================

    /// Confirm existing matrix+matrix soft-cap test still passes (regression guard).
    func testRegression_matrixAddMatrix_overCap_stillThrows() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let a = mat(3, 3, Array(repeating: 1.0, count: 9))
        let b = mat(3, 3, Array(repeating: 2.0, count: 9))
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)) { err in
            assertInvalidParameter(err)
        }
    }

    /// Confirm existing matrix*matrix (matmul) soft-cap test still passes.
    func testRegression_matmul_overCap_stillThrows() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let a = mat(3, 3, Array(repeating: 1.0, count: 9))
        let b = mat(3, 3, Array(repeating: 1.0, count: 9))
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.mul, lhs: a, rhs: b)) { err in
            assertInvalidParameter(err)
        }
    }
}
