//
//  ErrorModelEndToEndTests.swift
//  NumericSwiftTests
//
//  End-to-end validation of the two-mechanism error model (MF-1).
//
//  ## Contract under test
//
//  **Group-A operators** (add/sub/hadamard/elementDiv/dot/div(matrix,scalar)):
//    The unified dispatcher PRE-VALIDATES shape/divisor and THROWS a catchable
//    `MathExprError` BEFORE any `LinAlg` precondition can fire.  A `precondition`
//    failure issues SIGILL and terminates the process — XCTest cannot catch it,
//    so we cannot directly "prove no trap".  Instead, every Group-A test provides
//    inputs that WOULD trigger a LinAlg precondition if the guard were absent,
//    and asserts that a CATCHABLE `MathExprError` is thrown.  A successful
//    throw+catch conclusively proves the pre-validation gate fired first.
//
//  **Group-B functions** (trace/det/inv/expm/logm/sqrtm/cdet/cinv):
//    The dispatcher calls the throwing LinAlg operation with `try` and PROPAGATES
//    `LinAlgError.notSquare` (and other `LinAlgError` values) unchanged.  There
//    is NO dispatcher-side pre-validation.  Tests verify that `LinAlgError` (not
//    `MathExprError`) reaches the caller, which confirms neither wrapping nor
//    pre-validation occurs.
//
//  ## Scope notes
//
//  `trace(complexMatrix)` deliberately does NOT enforce squareness — it sums
//  `min(rows, cols)` diagonal elements following NumPy semantics (documented in
//  `evalTraceComplexMatrix`).  This is the correct design and is tested as such.
//
//  The `evaluateComplex` sqrt/log/pow negative-real value issue (GitHub #1) is
//  out of scope for this file — it is a value/domain issue, not an error-model
//  violation, and does not affect the throw-vs-trap boundary.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// swiftlint:disable:next type_body_length
final class ErrorModelEndToEndTests: XCTestCase {

    // -------------------------------------------------------------------------
    // MARK: - Private helpers
    // -------------------------------------------------------------------------

    /// Real matrix value.
    private func mat(_ rows: Int, _ cols: Int, _ flat: [Double]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: rows, cols: cols, data: flat))
    }

    /// Column-vector real matrix value.
    private func vec(_ values: [Double]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: values.count, cols: 1, data: values))
    }

    /// Complex matrix value (all-zero imaginary block).
    private func cmat(
        _ rows: Int, _ cols: Int,
        real: [Double], imag: [Double] = []
    ) -> NumericValue {
        let im = imag.isEmpty ? [Double](repeating: 0, count: rows * cols) : imag
        return .complexMatrix(LinAlg.ComplexMatrix(rows: rows, cols: cols, real: real, imag: im))
    }

    /// Assert that the error is `MathExprError.shapeMismatch`.
    private func assertShapeMismatch(_ error: Error, file: StaticString = #file, line: UInt = #line) {
        guard case MathExprError.shapeMismatch = error else {
            XCTFail(
                "Expected MathExprError.shapeMismatch, got \(type(of: error)): \(error)",
                file: file, line: line)
            return
        }
    }

    /// Assert that the error is `MathExprError.divisionByZero`.
    private func assertDivisionByZero(_ error: Error, file: StaticString = #file, line: UInt = #line) {
        guard case MathExprError.divisionByZero = error else {
            XCTFail(
                "Expected MathExprError.divisionByZero, got \(type(of: error)): \(error)",
                file: file, line: line)
            return
        }
    }

    /// Assert that the error is `LinAlg.LinAlgError.notSquare`.
    private func assertNotSquare(_ error: Error, file: StaticString = #file, line: UInt = #line) {
        guard let linAlgErr = error as? LinAlg.LinAlgError,
              case .notSquare = linAlgErr else {
            XCTFail(
                "Expected LinAlgError.notSquare, got \(type(of: error)): \(error)",
                file: file, line: line)
            return
        }
    }

    // =========================================================================
    // MARK: - Group-A: Real matrix operators
    //
    // Each test supplies inputs that WOULD trigger a LinAlg precondition if the
    // dispatcher's pre-validation were absent.  A successful throw+catch of
    // MathExprError proves the gate fired before any precondition.
    // =========================================================================

    // MARK: - 26.3 · add (real matrix)

    /// add: row mismatch → MathExprError.shapeMismatch, never a trap.
    func testGroupA_add_rowMismatch_throwsShapeMismatch() {
        let lhs = mat(2, 3, [1, 2, 3, 4, 5, 6])   // 2×3
        let rhs = mat(3, 3, Array(repeating: 1.0, count: 9))  // 3×3 — LinAlg.add would precondition
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.add, lhs: lhs, rhs: rhs)
        ) { assertShapeMismatch($0) }
    }

    /// add: column mismatch → MathExprError.shapeMismatch.
    func testGroupA_add_colMismatch_throwsShapeMismatch() {
        let lhs = mat(2, 3, Array(repeating: 1.0, count: 6))
        let rhs = mat(2, 4, Array(repeating: 1.0, count: 8))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.add, lhs: lhs, rhs: rhs)
        ) { assertShapeMismatch($0) }
    }

    /// add: compatible shapes succeed (regression guard).
    func testGroupA_add_compatible_succeeds() throws {
        let lhs = mat(2, 2, [1, 2, 3, 4])
        let rhs = mat(2, 2, [10, 20, 30, 40])
        let result = try NumericDispatch.applyBinary(.add, lhs: lhs, rhs: rhs)
        XCTAssertEqual(result.asMatrix?.data, [11, 22, 33, 44])
    }

    // MARK: - 26.4 · sub (real matrix)

    /// sub: row mismatch → MathExprError.shapeMismatch, never a trap.
    func testGroupA_sub_rowMismatch_throwsShapeMismatch() {
        let lhs = mat(2, 2, [1, 2, 3, 4])
        let rhs = mat(3, 2, Array(repeating: 0.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.sub, lhs: lhs, rhs: rhs)
        ) { assertShapeMismatch($0) }
    }

    /// sub: column mismatch → MathExprError.shapeMismatch.
    func testGroupA_sub_colMismatch_throwsShapeMismatch() {
        let lhs = mat(3, 2, Array(repeating: 1.0, count: 6))
        let rhs = mat(3, 4, Array(repeating: 1.0, count: 12))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.sub, lhs: lhs, rhs: rhs)
        ) { assertShapeMismatch($0) }
    }

    /// sub: compatible shapes succeed.
    func testGroupA_sub_compatible_succeeds() throws {
        let lhs = mat(2, 2, [10, 20, 30, 40])
        let rhs = mat(2, 2, [1, 2, 3, 4])
        let result = try NumericDispatch.applyBinary(.sub, lhs: lhs, rhs: rhs)
        XCTAssertEqual(result.asMatrix?.data, [9, 18, 27, 36])
    }

    // MARK: - 26.5 · hadamard (real matrix)

    /// hadamard: shape mismatch → MathExprError.shapeMismatch (Group-A pre-validation).
    func testGroupA_hadamard_shapeMismatch_throwsShapeMismatch() {
        let lhs = mat(2, 3, Array(repeating: 1.0, count: 6))
        let rhs = mat(2, 2, Array(repeating: 1.0, count: 4))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("hadamard", args: [lhs, rhs])
        ) { assertShapeMismatch($0) }
    }

    /// hadamard via applyBinary — same gate, confirmed via direct dispatcher call.
    func testGroupA_hadamard_viaApplyMultiArgFunction_rowMismatch() {
        let lhs = mat(3, 2, Array(repeating: 1.0, count: 6))
        let rhs = mat(2, 2, Array(repeating: 1.0, count: 4))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("hadamard", args: [lhs, rhs])
        ) { assertShapeMismatch($0) }
    }

    /// hadamard: compatible shapes produce element-wise product.
    func testGroupA_hadamard_compatible_succeeds() throws {
        let lhs = mat(2, 2, [2, 3, 4, 5])
        let rhs = mat(2, 2, [10, 10, 10, 10])
        let result = try NumericDispatch.applyFunction("hadamard", args: [lhs, rhs])
        XCTAssertEqual(result.asMatrix?.data, [20, 30, 40, 50])
    }

    // MARK: - 26.6 · elementDiv (real matrix)

    /// elementDiv: shape mismatch → MathExprError.shapeMismatch.
    func testGroupA_elementDiv_shapeMismatch_throwsShapeMismatch() {
        let lhs = mat(2, 3, Array(repeating: 1.0, count: 6))
        let rhs = mat(3, 2, Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("elementDiv", args: [lhs, rhs])
        ) { assertShapeMismatch($0) }
    }

    /// elementDiv: compatible shapes succeed.
    func testGroupA_elementDiv_compatible_succeeds() throws {
        let lhs = mat(2, 2, [10, 20, 30, 40])
        let rhs = mat(2, 2, [2, 4, 5, 8])
        let result = try NumericDispatch.applyFunction("elementDiv", args: [lhs, rhs])
        guard let data = result.asMatrix?.data else {
            return XCTFail("Expected matrix result")
        }
        XCTAssertEqual(data[0], 5, accuracy: 1e-12)
        XCTAssertEqual(data[1], 5, accuracy: 1e-12)
        XCTAssertEqual(data[2], 6, accuracy: 1e-12)
        XCTAssertEqual(data[3], 5, accuracy: 1e-12)
    }

    // MARK: - 26.7 · dot / matmul (real matrix)

    /// matmul via `*`: inner-dimension mismatch → MathExprError.shapeMismatch.
    /// A (2×3) * (4×2) would hit LinAlg.dot's precondition; the dispatcher catches it first.
    func testGroupA_dot_innerDimMismatch_throwsShapeMismatch() {
        let lhs = mat(2, 3, Array(repeating: 1.0, count: 6))   // 2×3
        let rhs = mat(4, 2, Array(repeating: 1.0, count: 8))   // 4×2 — inner: 3 ≠ 4
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.mul, lhs: lhs, rhs: rhs)
        ) { assertShapeMismatch($0) }
    }

    /// dotProduct function: inner-dimension mismatch.
    func testGroupA_dotProductFunction_innerDimMismatch_throwsShapeMismatch() {
        let lhs = mat(2, 3, Array(repeating: 1.0, count: 6))
        let rhs = mat(4, 1, Array(repeating: 1.0, count: 4))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("dotProduct", args: [lhs, rhs])
        ) { assertShapeMismatch($0) }
    }

    /// vec·vec via `*`: length mismatch → MathExprError.shapeMismatch.
    func testGroupA_dot_vecVecLengthMismatch_throwsShapeMismatch() {
        let lhsVec = vec([1, 2, 3])   // 3×1
        let rhsVec = vec([1, 2])      // 2×1 — mismatch
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.mul, lhs: lhsVec, rhs: rhsVec)
        ) { assertShapeMismatch($0) }
    }

    /// matmul: compatible shapes succeed.
    func testGroupA_dot_compatible_succeeds() throws {
        let lhs = mat(2, 3, [1, 2, 3, 4, 5, 6])   // 2×3
        let rhs = mat(3, 1, [1, 1, 1])              // 3×1
        let result = try NumericDispatch.applyBinary(.mul, lhs: lhs, rhs: rhs)
        // [1+2+3, 4+5+6] = [6, 15]
        XCTAssertEqual(result.asMatrix?.data, [6, 15])
    }

    // MARK: - 26.8 · div(matrix, scalar)

    /// matrix ÷ 0.0 → MathExprError.divisionByZero, never a LinAlg precondition trap.
    func testGroupA_divMatrixByZero_throwsDivisionByZero() {
        let m = mat(2, 2, [1, 2, 3, 4])
        let zero = NumericValue.scalar(0.0)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.div, lhs: m, rhs: zero)
        ) { assertDivisionByZero($0) }
    }

    /// matrix ÷ non-zero succeeds.
    func testGroupA_divMatrixByNonZero_succeeds() throws {
        let m = mat(2, 2, [2, 4, 6, 8])
        let two = NumericValue.scalar(2.0)
        let result = try NumericDispatch.applyBinary(.div, lhs: m, rhs: two)
        XCTAssertEqual(result.asMatrix?.data, [1, 2, 3, 4])
    }

    // =========================================================================
    // MARK: - Group-A: Complex matrix operators
    //
    // The complex-matrix paths use `validateCMSameShape` / `validateCMMatmulShape`
    // (private helpers in NumericDispatch+EvalStubs.swift) which throw
    // MathExprError.shapeMismatch before any element operation.
    // =========================================================================

    // MARK: - 26.18 · complexMatrix add (complex matrix)

    /// complexMatrix + complexMatrix: shape mismatch → MathExprError.shapeMismatch.
    func testGroupA_complexMatrix_add_shapeMismatch_throwsShapeMismatch() {
        let lhs = cmat(2, 3, real: Array(repeating: 1.0, count: 6))
        let rhs = cmat(3, 3, real: Array(repeating: 1.0, count: 9))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.add, lhs: lhs, rhs: rhs)
        ) { assertShapeMismatch($0) }
    }

    /// complexMatrix + complexMatrix: compatible shapes succeed.
    func testGroupA_complexMatrix_add_compatible_succeeds() throws {
        let lhs = cmat(2, 2, real: [1, 2, 3, 4], imag: [0, 0, 0, 0])
        let rhs = cmat(2, 2, real: [10, 20, 30, 40], imag: [1, 1, 1, 1])
        let result = try NumericDispatch.applyBinary(.add, lhs: lhs, rhs: rhs)
        guard let cm = result.asComplexMatrix else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(cm.real, [11, 22, 33, 44])
        XCTAssertEqual(cm.imag, [1, 1, 1, 1])
    }

    // MARK: - complexMatrix sub

    /// complexMatrix - complexMatrix: shape mismatch → MathExprError.shapeMismatch.
    func testGroupA_complexMatrix_sub_shapeMismatch_throwsShapeMismatch() {
        let lhs = cmat(2, 2, real: Array(repeating: 1.0, count: 4))
        let rhs = cmat(3, 2, real: Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.sub, lhs: lhs, rhs: rhs)
        ) { assertShapeMismatch($0) }
    }

    // MARK: - complexMatrix hadamard

    /// hadamard(CM, CM): shape mismatch → MathExprError.shapeMismatch.
    func testGroupA_complexMatrix_hadamard_shapeMismatch_throwsShapeMismatch() {
        let lhs = cmat(2, 3, real: Array(repeating: 1.0, count: 6))
        let rhs = cmat(2, 2, real: Array(repeating: 1.0, count: 4))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("hadamard", args: [lhs, rhs])
        ) { assertShapeMismatch($0) }
    }

    /// hadamard(CM, CM): compatible shapes succeed.
    func testGroupA_complexMatrix_hadamard_compatible_succeeds() throws {
        let lhs = cmat(1, 2, real: [2, 3], imag: [0, 0])
        let rhs = cmat(1, 2, real: [4, 5], imag: [1, 1])
        // (2+0i)(4+1i) = 8+2i ; (3+0i)(5+1i) = 15+3i
        let result = try NumericDispatch.applyFunction("hadamard", args: [lhs, rhs])
        guard let cm = result.asComplexMatrix else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(cm.real[0], 8, accuracy: 1e-12)
        XCTAssertEqual(cm.imag[0], 2, accuracy: 1e-12)
        XCTAssertEqual(cm.real[1], 15, accuracy: 1e-12)
        XCTAssertEqual(cm.imag[1], 3, accuracy: 1e-12)
    }

    // MARK: - complexMatrix dotProduct

    /// dotProduct(CM, CM): inner-dimension mismatch → MathExprError.shapeMismatch.
    func testGroupA_complexMatrix_dotProduct_innerDimMismatch_throwsShapeMismatch() {
        // (2×3) · (4×1): inner dims 3 ≠ 4
        let lhs = cmat(2, 3, real: Array(repeating: 1.0, count: 6))
        let rhs = cmat(4, 1, real: Array(repeating: 1.0, count: 4))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("dotProduct", args: [lhs, rhs])
        ) { assertShapeMismatch($0) }
    }

    /// dotProduct(CM, CM): vec·vec length mismatch → MathExprError.shapeMismatch.
    func testGroupA_complexMatrix_dotProduct_vecVecLengthMismatch_throwsShapeMismatch() {
        let lhs = cmat(3, 1, real: [1, 2, 3])
        let rhs = cmat(2, 1, real: [1, 2])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("dotProduct", args: [lhs, rhs])
        ) { assertShapeMismatch($0) }
    }

    /// complexMatrix * complexMatrix (matmul): inner-dimension mismatch.
    func testGroupA_complexMatrix_mul_innerDimMismatch_throwsShapeMismatch() {
        let lhs = cmat(2, 3, real: Array(repeating: 1.0, count: 6))
        let rhs = cmat(4, 2, real: Array(repeating: 1.0, count: 8))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.mul, lhs: lhs, rhs: rhs)
        ) { assertShapeMismatch($0) }
    }

    // =========================================================================
    // MARK: - 26.15 · Adversarial Group-A: inputs that LinAlg would trap on
    //
    // These tests explicitly document the precondition sites in LinAlg that the
    // dispatcher gates.  The names reference the internal site guarded.
    // =========================================================================

    /// LinAlg.add has a precondition on equal dimensions; the dispatcher catches
    /// this first via validateShapes(.equalDims).
    func testAdversarial_addShapeMismatch_gatesLinAlgAddPrecondition() {
        // A (1×2) + (2×1) would hit LinAlg.add's precondition on rows/cols equality.
        let a = mat(1, 2, [1, 2])
        let b = mat(2, 1, [3, 4])
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)
        ) { err in
            // The thrown error is catchable → precondition was NOT reached.
            assertShapeMismatch(err)
        }
    }

    /// LinAlg.hadamard has a precondition on equal dimensions; the dispatcher's
    /// validateShapes(.equalDims) intercepts first.
    func testAdversarial_hadamardShapeMismatch_gatesLinAlgHadamardPrecondition() {
        // A (2×2) hadamard (3×2) would hit LinAlg.hadamard's precondition.
        let a = mat(2, 2, [1, 2, 3, 4])
        let b = mat(3, 2, Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("hadamard", args: [a, b])
        ) { err in
            assertShapeMismatch(err)
        }
    }

    /// LinAlg.dot has a precondition on inner-dimension compatibility.
    /// The dispatcher's validateShapes(.dotProduct) fires first.
    func testAdversarial_dotInnerDimMismatch_gatesLinAlgDotPrecondition() {
        // A (3×4) * (3×1) has inner mismatch: 4 ≠ 3; LinAlg.dot would precondition.
        let a = mat(3, 4, Array(repeating: 1.0, count: 12))
        let b = mat(3, 1, Array(repeating: 1.0, count: 3))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.mul, lhs: a, rhs: b)
        ) { err in
            assertShapeMismatch(err)
        }
    }

    /// LinAlg.div(matrix, scalar) has a precondition on scalar != 0.
    /// The dispatcher's zero-check fires first.
    func testAdversarial_divByZero_gatesLinAlgDivPrecondition() {
        let m = mat(3, 3, Array(repeating: 1.0, count: 9))
        let zero = NumericValue.scalar(0.0)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.div, lhs: m, rhs: zero)
        ) { err in
            assertDivisionByZero(err)
        }
    }

    /// elementDiv shape mismatch gates LinAlg.elementDiv precondition.
    func testAdversarial_elementDivShapeMismatch_gatesLinAlgElementDivPrecondition() {
        // (2×3) ./ (2×2) — dims differ; LinAlg.elementDiv would precondition.
        let a = mat(2, 3, Array(repeating: 2.0, count: 6))
        let b = mat(2, 2, Array(repeating: 1.0, count: 4))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("elementDiv", args: [a, b])
        ) { err in
            assertShapeMismatch(err)
        }
    }

    // =========================================================================
    // MARK: - Group-B: Real matrix functions
    //
    // Group-B functions call LinAlg operations with `try` and PROPAGATE
    // LinAlgError.notSquare unmodified.  The dispatcher contains NO pre-guard
    // for squareness or singularity — LinAlg decides.
    // =========================================================================

    // MARK: - 26.9 · trace (real matrix)

    /// trace(nonSquareMatrix) propagates LinAlgError.notSquare — no dispatcher pre-guard.
    func testGroupB_trace_nonSquare_propagatesNotSquare() {
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])   // 2×3
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("trace", args: [.matrix(m.asMatrix!)])
        ) { assertNotSquare($0) }
    }

    /// trace(squareMatrix) succeeds with the correct diagonal sum.
    func testGroupB_trace_square_succeeds() throws {
        let m = mat(2, 2, [1, 2, 3, 4])
        let result = try NumericDispatch.applyFunction("trace", args: [m])
        // trace([[1,2],[3,4]]) = 1 + 4 = 5
        XCTAssertEqual(result.asScalar ?? .nan, 5, accuracy: 1e-12)
    }

    // MARK: - 26.10 · det (real matrix)

    /// det(nonSquareMatrix) propagates LinAlgError.notSquare.
    func testGroupB_det_nonSquare_propagatesNotSquare() {
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("det", args: [m])
        ) { assertNotSquare($0) }
    }

    /// det(squareMatrix) succeeds.
    func testGroupB_det_square_succeeds() throws {
        let m = mat(2, 2, [1, 2, 3, 4])
        let result = try NumericDispatch.applyFunction("det", args: [m])
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        XCTAssertEqual(result.asScalar!, -2, accuracy: 1e-10)
    }

    /// det(singularSquareMatrix) returns 0 (does not throw).
    func testGroupB_det_singularSquare_returnsZero() throws {
        // [[1,2],[2,4]] is singular; det = 0
        let m = mat(2, 2, [1, 2, 2, 4])
        let result = try NumericDispatch.applyFunction("det", args: [m])
        XCTAssertEqual(result.asScalar!, 0, accuracy: 1e-10)
    }

    // MARK: - 26.11 · inv (real matrix)

    /// inv(nonSquareMatrix) propagates LinAlgError.notSquare.
    func testGroupB_inv_nonSquare_propagatesNotSquare() {
        let m = mat(3, 2, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("inv", args: [m])
        ) { assertNotSquare($0) }
    }

    /// inv(singularMatrix) → MathExprError.invalidArguments (LinAlg returns nil).
    func testGroupB_inv_singularSquare_throwsInvalidArguments() {
        let singular = mat(2, 2, [1, 2, 2, 4])   // singular
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("inv", args: [singular])
        ) { error in
            guard case MathExprError.invalidArguments(let msg) = error else {
                return XCTFail("Expected invalidArguments, got \(error)")
            }
            XCTAssert(msg.contains("singular"), "Message should mention singularity: \(msg)")
        }
    }

    /// inv(nonSingularSquareMatrix) succeeds.
    func testGroupB_inv_nonSingularSquare_succeeds() throws {
        let m = mat(2, 2, [1, 2, 3, 4])
        let result = try NumericDispatch.applyFunction("inv", args: [m])
        guard let data = result.asMatrix?.data else { return XCTFail("Expected matrix") }
        // inv([[1,2],[3,4]]) = [[-2,1],[1.5,-0.5]]
        XCTAssertEqual(data[0], -2, accuracy: 1e-10)
        XCTAssertEqual(data[1],  1, accuracy: 1e-10)
        XCTAssertEqual(data[2],  1.5, accuracy: 1e-10)
        XCTAssertEqual(data[3], -0.5, accuracy: 1e-10)
    }

    // MARK: - 26.12 · expm (real matrix)

    /// expm(nonSquareMatrix) propagates LinAlgError.notSquare.
    func testGroupB_expm_nonSquare_propagatesNotSquare() {
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("exp", args: [m])
        ) { assertNotSquare($0) }
    }

    /// expm(squareMatrix) succeeds.
    func testGroupB_expm_squareZeroMatrix_returnsIdentity() throws {
        // exp(zero matrix) = identity
        let zero = mat(2, 2, [0, 0, 0, 0])
        let result = try NumericDispatch.applyFunction("exp", args: [zero])
        guard let data = result.asMatrix?.data else { return XCTFail("Expected matrix") }
        XCTAssertEqual(data[0], 1, accuracy: 1e-10)
        XCTAssertEqual(data[1], 0, accuracy: 1e-10)
        XCTAssertEqual(data[2], 0, accuracy: 1e-10)
        XCTAssertEqual(data[3], 1, accuracy: 1e-10)
    }

    // MARK: - 26.12 · logm (real matrix)

    /// logm(nonSquareMatrix) propagates LinAlgError.notSquare.
    func testGroupB_logm_nonSquare_propagatesNotSquare() {
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("log", args: [m])
        ) { assertNotSquare($0) }
    }

    /// logm(identityMatrix) ≈ zero matrix.
    func testGroupB_logm_identity_returnsZeroMatrix() throws {
        let eye = mat(2, 2, [1, 0, 0, 1])
        let result = try NumericDispatch.applyFunction("log", args: [eye])
        guard let data = result.asMatrix?.data else { return XCTFail("Expected matrix") }
        for v in data {
            XCTAssertEqual(v, 0, accuracy: 1e-10)
        }
    }

    // MARK: - 26.12 · sqrtm (real matrix)

    /// sqrtm(nonSquareMatrix) propagates LinAlgError.notSquare.
    func testGroupB_sqrtm_nonSquare_propagatesNotSquare() {
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("sqrt", args: [m])
        ) { assertNotSquare($0) }
    }

    /// sqrtm(identityMatrix)² ≈ identity.
    func testGroupB_sqrtm_identity_succeeds() throws {
        let eye = mat(2, 2, [1, 0, 0, 1])
        let result = try NumericDispatch.applyFunction("sqrt", args: [eye])
        guard let data = result.asMatrix?.data else { return XCTFail("Expected matrix") }
        // sqrt(I) = I
        XCTAssertEqual(data[0], 1, accuracy: 1e-10)
        XCTAssertEqual(data[1], 0, accuracy: 1e-10)
        XCTAssertEqual(data[2], 0, accuracy: 1e-10)
        XCTAssertEqual(data[3], 1, accuracy: 1e-10)
    }

    // =========================================================================
    // MARK: - Group-B: Complex matrix functions
    // =========================================================================

    // MARK: - 26.13 · cdet (complex matrix)

    /// cdet(nonSquareComplexMatrix) propagates LinAlgError.notSquare.
    func testGroupB_cdet_nonSquare_propagatesNotSquare() {
        let m = cmat(2, 3, real: Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("cdet", args: [m])
        ) { assertNotSquare($0) }
    }

    /// cdet(squareSingularComplexMatrix) returns (0+0i) — not an error (DOM-01).
    func testGroupB_cdet_singularSquare_returnsZeroComplex() throws {
        // [[1+0i, 2+0i],[2+0i, 4+0i]] is singular; det = 0
        let singular = cmat(2, 2, real: [1, 2, 2, 4])
        let result = try NumericDispatch.applyFunction("cdet", args: [singular])
        guard let z = result.asComplex else { return XCTFail("Expected complex result") }
        XCTAssertEqual(z.re, 0, accuracy: 1e-10)
        XCTAssertEqual(z.im, 0, accuracy: 1e-10)
    }

    /// cdet(nonSingularComplexMatrix) succeeds.
    func testGroupB_cdet_square_succeeds() throws {
        // [[1+0i, 0+1i],[0-1i, 1+0i]]: det = (1)(1) - (i)(-i) = 1 - 1 = 0? No:
        // det = 1*1 - (0+1i)*(0-1i) = 1 - (i*(-i)) = 1 - (1) = 0. Use simple case.
        // [[2+0i, 0],[0, 3+0i]]: det = 6+0i
        let m = cmat(2, 2, real: [2, 0, 0, 3])
        let result = try NumericDispatch.applyFunction("cdet", args: [m])
        guard let z = result.asComplex else { return XCTFail("Expected complex result") }
        XCTAssertEqual(z.re, 6, accuracy: 1e-10)
        XCTAssertEqual(z.im, 0, accuracy: 1e-10)
    }

    // MARK: - 26.14 · cinv (complex matrix)

    /// cinv(nonSquareComplexMatrix) propagates LinAlgError.notSquare.
    func testGroupB_cinv_nonSquare_propagatesNotSquare() {
        let m = cmat(3, 2, real: Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("cinv", args: [m])
        ) { assertNotSquare($0) }
    }

    /// cinv(singularComplexMatrix) → MathExprError.invalidArguments.
    func testGroupB_cinv_singularSquare_throwsInvalidArguments() {
        // [[1+0i, 2+0i],[2+0i, 4+0i]] is singular
        let singular = cmat(2, 2, real: [1, 2, 2, 4])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("cinv", args: [singular])
        ) { error in
            guard case MathExprError.invalidArguments(let msg) = error else {
                return XCTFail("Expected invalidArguments, got \(error)")
            }
            XCTAssert(msg.contains("singular"), "Message should mention singularity: \(msg)")
        }
    }

    /// cinv(nonSingularComplexMatrix) succeeds.
    func testGroupB_cinv_nonSingular_succeeds() throws {
        // Diagonal complex matrix [[2+0i, 0],[0, 4+0i]]; inv = [[0.5,0],[0,0.25]]
        let m = cmat(2, 2, real: [2, 0, 0, 4])
        let result = try NumericDispatch.applyFunction("cinv", args: [m])
        guard let cm = result.asComplexMatrix else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(cm.real[0], 0.5, accuracy: 1e-10)
        XCTAssertEqual(cm.real[3], 0.25, accuracy: 1e-10)
        for v in cm.imag { XCTAssertEqual(v, 0, accuracy: 1e-10) }
    }

    // MARK: - trace(complexMatrix) — documented NumPy-compat behaviour

    /// trace(complexMatrix) does NOT require squareness — it sums min(rows,cols)
    /// diagonal elements per NumPy semantics.  This is the correct design.
    func testGroupB_trace_complexMatrix_nonSquare_doesNotThrow() throws {
        // 2×3 complex matrix; trace = diagonal[0] + diagonal[1]
        // real = [[1,2,3],[4,5,6]] → diag re = 1+5 = 6; im = 0+0 = 0
        let m = cmat(2, 3, real: [1, 2, 3, 4, 5, 6], imag: [0, 0, 0, 0, 0, 0])
        let result = try NumericDispatch.applyFunction("trace", args: [m])
        guard let z = result.asComplex else { return XCTFail("Expected complex result") }
        XCTAssertEqual(z.re, 6, accuracy: 1e-12)   // 1+5
        XCTAssertEqual(z.im, 0, accuracy: 1e-12)
    }

    // =========================================================================
    // MARK: - 26.16 · Adversarial Group-B: no pre-validation in dispatcher
    //
    // Confirms that the dispatcher does NOT pre-validate squareness for Group-B.
    // LinAlgError (not MathExprError) reaching the caller proves pass-through.
    // =========================================================================

    /// The error type for non-square trace is LinAlgError, not MathExprError.
    /// This proves the dispatcher does not wrap or pre-validate.
    func testAdversarial_groupB_trace_noDispatcherPreValidation() {
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("trace", args: [m])
        ) { error in
            // Must be LinAlgError, not MathExprError (which would indicate pre-validation).
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Expected LinAlgError from LinAlg.trace; got \(type(of: error))")
            assertNotSquare(error)
        }
    }

    /// The error type for non-square det is LinAlgError, not MathExprError.
    func testAdversarial_groupB_det_noDispatcherPreValidation() {
        let m = mat(3, 2, Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("det", args: [m])
        ) { error in
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Expected LinAlgError from LinAlg.det; got \(type(of: error))")
            assertNotSquare(error)
        }
    }

    /// The error type for non-square inv is LinAlgError, not MathExprError.
    func testAdversarial_groupB_inv_noDispatcherPreValidation() {
        let m = mat(2, 3, Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("inv", args: [m])
        ) { error in
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Expected LinAlgError from LinAlg.inv; got \(type(of: error))")
            assertNotSquare(error)
        }
    }

    /// The error type for non-square expm is LinAlgError, not MathExprError.
    func testAdversarial_groupB_expm_noDispatcherPreValidation() {
        let m = mat(2, 3, Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("exp", args: [m])
        ) { error in
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Expected LinAlgError from LinAlg.expm; got \(type(of: error))")
            assertNotSquare(error)
        }
    }

    /// The error type for non-square logm is LinAlgError, not MathExprError.
    func testAdversarial_groupB_logm_noDispatcherPreValidation() {
        let m = mat(2, 3, Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("log", args: [m])
        ) { error in
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Expected LinAlgError from LinAlg.logm; got \(type(of: error))")
            assertNotSquare(error)
        }
    }

    /// The error type for non-square sqrtm is LinAlgError, not MathExprError.
    func testAdversarial_groupB_sqrtm_noDispatcherPreValidation() {
        let m = mat(2, 3, Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("sqrt", args: [m])
        ) { error in
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Expected LinAlgError from LinAlg.sqrtm; got \(type(of: error))")
            assertNotSquare(error)
        }
    }

    /// The error type for non-square cdet is LinAlgError, not MathExprError.
    func testAdversarial_groupB_cdet_noDispatcherPreValidation() {
        let m = cmat(2, 3, real: Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("cdet", args: [m])
        ) { error in
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Expected LinAlgError from LinAlg.cdet; got \(type(of: error))")
            assertNotSquare(error)
        }
    }

    /// The error type for non-square cinv is LinAlgError, not MathExprError.
    func testAdversarial_groupB_cinv_noDispatcherPreValidation() {
        let m = cmat(3, 2, real: Array(repeating: 1.0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("cinv", args: [m])
        ) { error in
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Expected LinAlgError from LinAlg.cinv; got \(type(of: error))")
            assertNotSquare(error)
        }
    }

    // =========================================================================
    // MARK: - 26.17 · Error-type boundary purity — mixed expressions
    //
    // Group-A errors remain MathExprError even when nested; Group-B errors remain
    // LinAlgError even through applyFunction dispatch.
    // =========================================================================

    /// A Group-A error (shapeMismatch) is MathExprError, not LinAlgError.
    func testErrorTypePurity_groupA_isNotLinAlgError() {
        let a = mat(2, 2, [1, 2, 3, 4])
        let b = mat(3, 3, Array(repeating: 1.0, count: 9))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)
        ) { error in
            XCTAssertTrue(error is MathExprError,
                          "Group-A error must be MathExprError, got \(type(of: error))")
            XCTAssertFalse(error is LinAlg.LinAlgError,
                           "Group-A error must NOT be LinAlgError")
        }
    }

    /// A Group-B error (notSquare) is LinAlgError, not MathExprError.
    func testErrorTypePurity_groupB_isNotMathExprError() {
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("trace", args: [m])
        ) { error in
            XCTAssertTrue(error is LinAlg.LinAlgError,
                          "Group-B error must be LinAlgError, got \(type(of: error))")
            XCTAssertFalse(error is MathExprError,
                           "Group-B error must NOT be MathExprError")
        }
    }

    // =========================================================================
    // MARK: - 26.19 · Contract-coverage matrix
    //
    // A summary table driving every (operator × shape-class) cell in the contract.
    // Each entry is verified to throw the right type for the right class.
    // =========================================================================

    /// Systematic walk of the full Group-A × {real, complex} × shape-mismatch table.
    func testContractMatrix_groupA_allCells_throwMathExprError() throws {
        typealias Case = (label: String, lhs: NumericValue, rhs: NumericValue, op: () throws -> NumericValue)

        let r2x2 = mat(2, 2, [1, 2, 3, 4])
        let r3x2 = mat(3, 2, Array(repeating: 1.0, count: 6))
        let r2x3 = mat(2, 3, Array(repeating: 1.0, count: 6))
        let c2x2 = cmat(2, 2, real: [1, 2, 3, 4])
        let c3x2 = cmat(3, 2, real: Array(repeating: 1.0, count: 6))
        let c2x3 = cmat(2, 3, real: Array(repeating: 1.0, count: 6))

        let cases: [(String, () throws -> NumericValue)] = [
            // Real matrix shape mismatches
            ("real-add-rows",       { try NumericDispatch.applyBinary(.add, lhs: r2x2, rhs: r3x2) }),
            ("real-sub-rows",       { try NumericDispatch.applyBinary(.sub, lhs: r2x2, rhs: r3x2) }),
            ("real-hadamard",       { try NumericDispatch.applyFunction("hadamard", args: [r2x2, r3x2]) }),
            ("real-elementDiv",     { try NumericDispatch.applyFunction("elementDiv", args: [r2x2, r3x2]) }),
            ("real-dot-inner",      { try NumericDispatch.applyBinary(.mul, lhs: r2x3, rhs: r3x2.asMatrix.map { m in
                                         NumericValue.matrix(LinAlg.Matrix(rows: 4, cols: 2, data: Array(repeating: 1.0, count: 8)))
                                     }!) }),
            ("real-div-by-zero",    { try NumericDispatch.applyBinary(.div, lhs: r2x2, rhs: .scalar(0.0)) }),
            // Complex matrix shape mismatches
            ("complex-add-rows",    { try NumericDispatch.applyBinary(.add, lhs: c2x2, rhs: c3x2) }),
            ("complex-sub-rows",    { try NumericDispatch.applyBinary(.sub, lhs: c2x2, rhs: c3x2) }),
            ("complex-hadamard",    { try NumericDispatch.applyFunction("hadamard", args: [c2x2, c3x2]) }),
            ("complex-dotProduct",  { try NumericDispatch.applyFunction("dotProduct", args: [c2x3, c3x2.asComplexMatrix.map { cm in
                                         NumericValue.complexMatrix(LinAlg.ComplexMatrix(rows: 4, cols: 2, real: Array(repeating: 1.0, count: 8), imag: Array(repeating: 0.0, count: 8)))
                                     }!]) }),
        ]

        for (label, op) in cases {
            XCTAssertThrowsError(
                try op(),
                "\(label) should throw MathExprError"
            ) { error in
                XCTAssertTrue(error is MathExprError,
                              "\(label): expected MathExprError, got \(type(of: error)): \(error)")
            }
        }
    }

    /// Systematic walk of the full Group-B × {real, complex} × non-square table.
    func testContractMatrix_groupB_allCells_propagateLinAlgError() throws {
        let r2x3 = mat(2, 3, Array(repeating: 1.0, count: 6))
        let c2x3 = cmat(2, 3, real: Array(repeating: 1.0, count: 6))

        let cases: [(String, () throws -> NumericValue)] = [
            ("trace-real-nonSquare",  { try NumericDispatch.applyFunction("trace", args: [r2x3]) }),
            ("det-real-nonSquare",    { try NumericDispatch.applyFunction("det",   args: [r2x3]) }),
            ("inv-real-nonSquare",    { try NumericDispatch.applyFunction("inv",   args: [r2x3]) }),
            ("expm-real-nonSquare",   { try NumericDispatch.applyFunction("exp",   args: [r2x3]) }),
            ("logm-real-nonSquare",   { try NumericDispatch.applyFunction("log",   args: [r2x3]) }),
            ("sqrtm-real-nonSquare",  { try NumericDispatch.applyFunction("sqrt",  args: [r2x3]) }),
            ("cdet-complex-nonSquare",{ try NumericDispatch.applyFunction("cdet",  args: [c2x3]) }),
            ("cinv-complex-nonSquare",{ try NumericDispatch.applyFunction("cinv",  args: [c2x3]) }),
        ]

        for (label, op) in cases {
            XCTAssertThrowsError(
                try op(),
                "\(label) should throw LinAlgError.notSquare"
            ) { error in
                XCTAssertTrue(error is LinAlg.LinAlgError,
                              "\(label): expected LinAlgError, got \(type(of: error)): \(error)")
                if let la = error as? LinAlg.LinAlgError, case .notSquare = la {
                    // correct
                } else {
                    XCTFail("\(label): error was LinAlgError but not .notSquare: \(error)")
                }
            }
        }
    }
}
