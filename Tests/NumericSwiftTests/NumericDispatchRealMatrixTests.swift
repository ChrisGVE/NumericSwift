//
//  NumericDispatchRealMatrixTests.swift
//  NumericSwiftTests
//
//  Tests for Task 11 real-matrix arithmetic cells in the unified dispatch core.
//
//  Coverage:
//    • Group-A pre-validation: shape mismatches THROW MathExprError.shapeMismatch
//      (never a LinAlg precondition trap).
//    • Correct results for add, sub, hadamard, elementDiv, dot (matmul),
//      matrix·vector, vector·matrix, vec·vec (→ scalar), matrix÷scalar.
//    • Soft-cap admission: over-limit result shape throws LinAlgError.invalidParameter.
//    • 1×1 coercion: matmul producing a 1×1 matrix collapses to .scalar.
//    • Unary negation via applyUnary(.neg, matrix).
//    • elementDiv registered as named function.
//    • Non-finite (inf/nan) propagation through arithmetic cells.
//    • MathExprError.shapeMismatch equality and description.
//    • Unsupported combos throw correct errors.
//
//  Reference values are computed from SciPy / scipy.linalg and embedded as
//  literals with inline provenance. All floating-point comparisons use
//  XCTAssertEqual(_, _, accuracy:) unless the value is exact.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class NumericDispatchRealMatrixTests: XCTestCase {

    // -------------------------------------------------------------------------
    // MARK: - Helpers
    // -------------------------------------------------------------------------

    private func mat(_ rows: Int, _ cols: Int, _ flat: [Double]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: rows, cols: cols, data: flat))
    }

    private func vec(_ values: [Double]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: values.count, cols: 1, data: values))
    }

    private func scalar(_ x: Double) -> NumericValue {
        .scalar(x)
    }

    /// Restore soft cap after each test so cap changes are isolated.
    override func tearDown() {
        super.tearDown()
        try? LinAlg.setMaxEvaluatorMatrixElements(16_777_216)
    }

    // -------------------------------------------------------------------------
    // MARK: - MathExprError.shapeMismatch — Task 11.1
    // -------------------------------------------------------------------------

    func testShapeMismatchEquality() {
        let e1 = MathExprError.shapeMismatch("add: shapes (2×3) and (3×2) must match")
        let e2 = MathExprError.shapeMismatch("add: shapes (2×3) and (3×2) must match")
        let e3 = MathExprError.shapeMismatch("different message")
        XCTAssertEqual(e1, e2)
        XCTAssertNotEqual(e1, e3)
    }

    func testShapeMismatchDescription_nonEmpty() {
        let err = MathExprError.shapeMismatch("add: shapes (2×3) and (3×2) must match")
        let desc = err.description
        XCTAssertFalse(desc.isEmpty)
        XCTAssert(desc.contains("shape mismatch"))
        XCTAssert(desc.contains("add"))
    }

    // -------------------------------------------------------------------------
    // MARK: - validateShapes helper — Task 11.3
    // -------------------------------------------------------------------------

    func testValidateShapes_equalDims_passes() {
        let a = LinAlg.Matrix(rows: 2, cols: 3, data: [Double](repeating: 0, count: 6))
        let b = LinAlg.Matrix(rows: 2, cols: 3, data: [Double](repeating: 0, count: 6))
        XCTAssertNoThrow(
            try NumericDispatch.validateShapes("op", lhs: a, rhs: b, rule: .equalDims))
    }

    func testValidateShapes_equalDims_rowMismatch_throws() {
        let a = LinAlg.Matrix(rows: 2, cols: 3, data: [Double](repeating: 0, count: 6))
        let b = LinAlg.Matrix(rows: 3, cols: 3, data: [Double](repeating: 0, count: 9))
        XCTAssertThrowsError(
            try NumericDispatch.validateShapes("add", lhs: a, rhs: b, rule: .equalDims)
        ) { err in
            if case MathExprError.shapeMismatch(let msg) = err {
                XCTAssert(msg.contains("add"))
            } else {
                XCTFail("Expected .shapeMismatch, got \(err)")
            }
        }
    }

    func testValidateShapes_dotProduct_matmul_passes() {
        // (2×3) * (3×4) — lhs.cols == rhs.rows == 3
        let a = LinAlg.Matrix(rows: 2, cols: 3, data: [Double](repeating: 0, count: 6))
        let b = LinAlg.Matrix(rows: 3, cols: 4, data: [Double](repeating: 0, count: 12))
        XCTAssertNoThrow(
            try NumericDispatch.validateShapes("*", lhs: a, rhs: b, rule: .dotProduct))
    }

    func testValidateShapes_dotProduct_matmul_fails() {
        // (2×3) * (2×3) — lhs.cols=3 ≠ rhs.rows=2
        let a = LinAlg.Matrix(rows: 2, cols: 3, data: [Double](repeating: 0, count: 6))
        let b = LinAlg.Matrix(rows: 2, cols: 3, data: [Double](repeating: 0, count: 6))
        XCTAssertThrowsError(
            try NumericDispatch.validateShapes("*", lhs: a, rhs: b, rule: .dotProduct)
        ) { err in
            if case MathExprError.shapeMismatch(let msg) = err {
                XCTAssert(msg.contains("*"))
                XCTAssert(msg.contains("3"))   // lhs.cols
                XCTAssert(msg.contains("2"))   // rhs.rows
            } else {
                XCTFail("Expected .shapeMismatch, got \(err)")
            }
        }
    }

    func testValidateShapes_dotProduct_vecVec_passes() {
        // Both cols==1, same row count → pass
        let a = LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3])
        let b = LinAlg.Matrix(rows: 3, cols: 1, data: [4, 5, 6])
        XCTAssertNoThrow(
            try NumericDispatch.validateShapes("dotProduct", lhs: a, rhs: b, rule: .dotProduct))
    }

    func testValidateShapes_dotProduct_vecVec_lengthMismatch_throws() {
        // Both cols==1, different row counts → shapeMismatch
        let a = LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3])
        let b = LinAlg.Matrix(rows: 2, cols: 1, data: [4, 5])
        XCTAssertThrowsError(
            try NumericDispatch.validateShapes("dotProduct", lhs: a, rhs: b, rule: .dotProduct)
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - matrix + matrix (Task 11.4)
    // -------------------------------------------------------------------------

    func testAddMatrixMatrix_correctResult() throws {
        // SciPy: np.array([[1,2],[3,4]]) + np.array([[5,6],[7,8]]) = [[6,8],[10,12]]
        let lhs = mat(2, 2, [1, 2, 3, 4])
        let rhs = mat(2, 2, [5, 6, 7, 8])
        let result = try NumericDispatch.applyBinary(.add, lhs: lhs, rhs: rhs)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 6.0)
        XCTAssertEqual(m[0, 1], 8.0)
        XCTAssertEqual(m[1, 0], 10.0)
        XCTAssertEqual(m[1, 1], 12.0)
    }

    func testAddMatrixMatrix_shapeMismatch_throwsShapeMismatch() {
        // Group-A: must throw MathExprError.shapeMismatch, NOT a precondition trap
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .add, lhs: mat(2, 3, .init(repeating: 0, count: 6)),
                rhs: mat(3, 2, .init(repeating: 0, count: 6)))
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testAddMatrixMatrix_shapeMismatch_message_names_operator() {
        // The error message should name the operator
        do {
            _ = try NumericDispatch.applyBinary(
                .add, lhs: mat(2, 3, .init(repeating: 0, count: 6)),
                rhs: mat(3, 2, .init(repeating: 0, count: 6)))
            XCTFail("Expected throw")
        } catch MathExprError.shapeMismatch(let msg) {
            XCTAssert(msg.contains("add"), "Message should name 'add'; got: \(msg)")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testAddVectorVector_correctResult() throws {
        // vec·vec add: both cols==1
        let lhs = vec([1, 2, 3])
        let rhs = vec([4, 5, 6])
        let result = try NumericDispatch.applyBinary(.add, lhs: lhs, rhs: rhs)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m[0, 0], 5.0)
        XCTAssertEqual(m[1, 0], 7.0)
        XCTAssertEqual(m[2, 0], 9.0)
    }

    // -------------------------------------------------------------------------
    // MARK: - matrix - matrix (Task 11.5)
    // -------------------------------------------------------------------------

    func testSubMatrixMatrix_correctResult() throws {
        // SciPy: np.array([[5,6],[7,8]]) - np.array([[1,2],[3,4]]) = [[4,4],[4,4]]
        let lhs = mat(2, 2, [5, 6, 7, 8])
        let rhs = mat(2, 2, [1, 2, 3, 4])
        let result = try NumericDispatch.applyBinary(.sub, lhs: lhs, rhs: rhs)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 4.0)
        XCTAssertEqual(m[0, 1], 4.0)
    }

    func testSubMatrixMatrix_shapeMismatch_throwsShapeMismatch() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .sub, lhs: mat(2, 3, .init(repeating: 0, count: 6)),
                rhs: mat(3, 2, .init(repeating: 0, count: 6)))
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testSubMatrixMatrix_shapeMismatch_message_names_operator() {
        do {
            _ = try NumericDispatch.applyBinary(
                .sub, lhs: mat(2, 3, .init(repeating: 0, count: 6)),
                rhs: mat(3, 2, .init(repeating: 0, count: 6)))
            XCTFail("Expected throw")
        } catch MathExprError.shapeMismatch(let msg) {
            XCTAssert(msg.contains("sub"), "Message should name 'sub'; got: \(msg)")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - hadamard (element-wise *) (Task 11.6)
    // -------------------------------------------------------------------------

    func testHadamard_correctResult() throws {
        // SciPy: np.array([[1,2],[3,4]]) * np.array([[5,6],[7,8]]) = [[5,12],[21,32]]
        let lhs = mat(2, 2, [1, 2, 3, 4])
        let rhs = mat(2, 2, [5, 6, 7, 8])
        let result = try NumericDispatch.applyFunction("hadamard", args: [lhs, rhs])
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 5.0)
        XCTAssertEqual(m[0, 1], 12.0)
        XCTAssertEqual(m[1, 0], 21.0)
        XCTAssertEqual(m[1, 1], 32.0)
    }

    func testHadamard_shapeMismatch_throwsShapeMismatch() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction(
                "hadamard", args: [mat(2, 2, .init(repeating: 0, count: 4)),
                                   mat(2, 3, .init(repeating: 0, count: 6))])
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testHadamard_scalarArgs_throwsInvalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("hadamard", args: [.scalar(2), .scalar(3)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected .invalidArguments, got \(err)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - elementDiv (element-wise /) (Task 11.7)
    // -------------------------------------------------------------------------

    func testElementDiv_correctResult() throws {
        // SciPy: np.array([[6,8],[10,12]]) / np.array([[2,4],[5,3]]) = [[3,2],[2,4]]
        let lhs = mat(2, 2, [6, 8, 10, 12])
        let rhs = mat(2, 2, [2, 4, 5, 3])
        let result = try NumericDispatch.applyFunction("elementDiv", args: [lhs, rhs])
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 3.0, accuracy: 1e-12)
        XCTAssertEqual(m[0, 1], 2.0, accuracy: 1e-12)
        XCTAssertEqual(m[1, 0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(m[1, 1], 4.0, accuracy: 1e-12)
    }

    func testElementDiv_shapeMismatch_throwsShapeMismatch() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction(
                "elementDiv", args: [mat(2, 2, .init(repeating: 1, count: 4)),
                                     mat(2, 3, .init(repeating: 1, count: 6))])
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testElementDiv_scalarArgs_throwsInvalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("elementDiv", args: [.scalar(4), .scalar(2)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected .invalidArguments, got \(err)")
        }
    }

    func testElementDiv_wrongArity_throwsInvalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("elementDiv", args: [mat(2, 2, [1, 2, 3, 4])])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected .invalidArguments, got \(err)")
        }
    }

    func testElementDiv_byZero_propagatesNaN_orInf() throws {
        // IEEE 754: non-zero / 0 = inf; 0 / 0 = nan — elementDiv passes through
        let lhs = mat(1, 2, [4.0, 0.0])
        let rhs = mat(1, 2, [0.0, 0.0])
        let result = try NumericDispatch.applyFunction("elementDiv", args: [lhs, rhs])
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssert(m[0, 0].isInfinite, "4/0 should be inf")
        XCTAssert(m[0, 1].isNaN,      "0/0 should be nan")
    }

    // -------------------------------------------------------------------------
    // MARK: - matrix * matrix = matmul (Task 11.8)
    // -------------------------------------------------------------------------

    func testMulMatrixMatrix_2x2_correctResult() throws {
        // SciPy: np.array([[1,2],[3,4]]) @ np.array([[5,6],[7,8]])
        //        = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let lhs = mat(2, 2, [1, 2, 3, 4])
        let rhs = mat(2, 2, [5, 6, 7, 8])
        let result = try NumericDispatch.applyBinary(.mul, lhs: lhs, rhs: rhs)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 19.0, accuracy: 1e-10)
        XCTAssertEqual(m[0, 1], 22.0, accuracy: 1e-10)
        XCTAssertEqual(m[1, 0], 43.0, accuracy: 1e-10)
        XCTAssertEqual(m[1, 1], 50.0, accuracy: 1e-10)
    }

    func testMulMatrixMatrix_rectangular_correctResult() throws {
        // SciPy: np.ones((2,3)) @ np.ones((3,4)) = 3*np.ones((2,4))
        let lhs = mat(2, 3, [Double](repeating: 1.0, count: 6))
        let rhs = mat(3, 4, [Double](repeating: 1.0, count: 12))
        let result = try NumericDispatch.applyBinary(.mul, lhs: lhs, rhs: rhs)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 4)
        for r in 0..<2 {
            for c in 0..<4 {
                XCTAssertEqual(m[r, c], 3.0, accuracy: 1e-10)
            }
        }
    }

    func testMulMatrixMatrix_innerDimMismatch_throwsShapeMismatch() {
        // (2×3) * (2×3) — inner dims 3≠2: must throw .shapeMismatch
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .mul, lhs: mat(2, 3, .init(repeating: 0, count: 6)),
                rhs: mat(2, 3, .init(repeating: 0, count: 6)))
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testMulMatrixMatrix_innerDimMismatch_messageNamesOp() {
        do {
            _ = try NumericDispatch.applyBinary(
                .mul, lhs: mat(2, 3, .init(repeating: 0, count: 6)),
                rhs: mat(2, 3, .init(repeating: 0, count: 6)))
            XCTFail("Expected throw")
        } catch MathExprError.shapeMismatch(let msg) {
            XCTAssert(msg.contains("*"), "Message should name '*'; got: \(msg)")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - matrix · vector (Task 11.9) and vector · matrix (Task 11.10)
    // -------------------------------------------------------------------------

    func testMulMatrixVector_correctResult() throws {
        // SciPy: np.array([[1,2],[3,4]]) @ np.array([[5],[6]])
        //        = [[1*5+2*6],[3*5+4*6]] = [[17],[39]]
        let m = mat(2, 2, [1, 2, 3, 4])
        let v = vec([5, 6])
        let result = try NumericDispatch.applyBinary(.mul, lhs: m, rhs: v)
        guard case .matrix(let r) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(r.rows, 2)
        XCTAssertEqual(r.cols, 1)
        XCTAssertEqual(r[0, 0], 17.0, accuracy: 1e-10)
        XCTAssertEqual(r[1, 0], 39.0, accuracy: 1e-10)
    }

    func testMulVectorMatrix_correctResult() throws {
        // SciPy: np.array([[1,2,3]]) @ np.array([[1,0],[0,1],[1,1]])
        //        = [[1*1+2*0+3*1, 1*0+2*1+3*1]] = [[4,5]]
        // row vector = matrix(1×3); rhs = matrix(3×2)
        let rowVec = mat(1, 3, [1, 2, 3])
        let rhs = mat(3, 2, [1, 0, 0, 1, 1, 1])
        let result = try NumericDispatch.applyBinary(.mul, lhs: rowVec, rhs: rhs)
        guard case .matrix(let r) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(r.rows, 1)
        XCTAssertEqual(r.cols, 2)
        XCTAssertEqual(r[0, 0], 4.0, accuracy: 1e-10)
        XCTAssertEqual(r[0, 1], 5.0, accuracy: 1e-10)
    }

    func testMulMatrixVector_dimMismatch_throwsShapeMismatch() {
        // (2×3) · vec(2) — lhs.cols=3 ≠ rhs.rows=2
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .mul, lhs: mat(2, 3, .init(repeating: 1, count: 6)),
                rhs: vec([1, 2]))
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - vec · vec → scalar coercion §4.3a (Tasks 11.8/11.9)
    // -------------------------------------------------------------------------

    func testMulVecVec_1x1_coercesToScalar() throws {
        // SciPy: np.array([3]) @ np.array([4]) = 12 (scalar)
        // vec * vec: col-vectors, lhs.cols=1 = rhs.rows=1, result is 1×1 → .scalar
        let u = vec([3])
        let v = vec([4])
        let result = try NumericDispatch.applyBinary(.mul, lhs: u, rhs: v)
        // 1×1 result must be coerced to .scalar per §4.3a
        guard case .scalar(let s) = result else {
            return XCTFail("Expected .scalar from vec·vec (§4.3a coercion), got \(result)")
        }
        XCTAssertEqual(s, 12.0, accuracy: 1e-10)
    }

    func testDotProduct_vecVec_coercesToScalar() throws {
        // SciPy: np.dot([1,2,3], [4,5,6]) = 32
        let u = vec([1, 2, 3])
        let v = vec([4, 5, 6])
        let result = try NumericDispatch.applyFunction("dotProduct", args: [u, v])
        guard case .scalar(let s) = result else {
            return XCTFail("Expected .scalar from dotProduct(vec,vec), got \(result)")
        }
        XCTAssertEqual(s, 32.0, accuracy: 1e-10)
    }

    func testDotProduct_matmul_correctResult() throws {
        // SciPy: (2×3) @ (3×2) = (2×2); each entry = sum of 3 ones = 3
        let lhs = mat(2, 3, [Double](repeating: 1.0, count: 6))
        let rhs = mat(3, 2, [Double](repeating: 1.0, count: 6))
        let result = try NumericDispatch.applyFunction("dotProduct", args: [lhs, rhs])
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
        for r in 0..<2 {
            for c in 0..<2 {
                XCTAssertEqual(m[r, c], 3.0, accuracy: 1e-10)
            }
        }
    }

    func testDotProduct_shapeMismatch_throwsShapeMismatch() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction(
                "dotProduct",
                args: [mat(2, 3, .init(repeating: 0, count: 6)),
                       mat(2, 3, .init(repeating: 0, count: 6))])
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - matrix ÷ scalar (Task 11.11)
    // -------------------------------------------------------------------------

    func testDivMatrixScalar_correctResult() throws {
        // SciPy: np.array([[6,8],[10,12]]) / 2 = [[3,4],[5,6]]
        let m = mat(2, 2, [6, 8, 10, 12])
        let result = try NumericDispatch.applyBinary(.div, lhs: m, rhs: .scalar(2.0))
        guard case .matrix(let r) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(r[0, 0], 3.0, accuracy: 1e-12)
        XCTAssertEqual(r[0, 1], 4.0, accuracy: 1e-12)
        XCTAssertEqual(r[1, 0], 5.0, accuracy: 1e-12)
        XCTAssertEqual(r[1, 1], 6.0, accuracy: 1e-12)
    }

    func testDivMatrixScalar_divByZero_throwsDivisionByZero() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .div, lhs: mat(2, 2, [1, 2, 3, 4]), rhs: .scalar(0.0))
        ) { err in
            if case MathExprError.divisionByZero = err { return }
            XCTFail("Expected .divisionByZero, got \(err)")
        }
    }

    func testDivMatrixMatrix_throwsInvalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .div, lhs: mat(2, 2, [1, 2, 3, 4]), rhs: mat(2, 2, [1, 2, 3, 4]))
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected .invalidArguments, got \(err)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - Unsupported Group-A combos (Task 11.12)
    // -------------------------------------------------------------------------

    func testDivScalarMatrix_throwsInvalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .div, lhs: .scalar(4), rhs: mat(2, 2, [1, 2, 3, 4]))
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected .invalidArguments, got \(err)")
        }
    }

    func testMulComplexMulComplexMatrix_implemented() throws {
        // complex * complexMatrix: (1+0i) * [[2+3i]] = [[2+3i]]
        let cm = LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [2.0], imag: [3.0])
        let result = try NumericDispatch.applyBinary(
            .mul,
            lhs: .complex(Complex(re: 1, im: 0)),
            rhs: .complexMatrix(cm))
        guard case .complexMatrix(let out) = result else {
            return XCTFail("Expected complexMatrix, got \(result)")
        }
        XCTAssertEqual(out.real[0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(out.imag[0], 3.0, accuracy: 1e-12)
    }

    // -------------------------------------------------------------------------
    // MARK: - Soft-cap result-shape check (Task 11.13)
    // -------------------------------------------------------------------------

    func testAddMatrixMatrix_overSoftCap_throwsInvalidParameter() throws {
        // Lower the soft cap to 4 elements, then try to add two 3×3 matrices
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let a = mat(3, 3, [Double](repeating: 1, count: 9))
        let b = mat(3, 3, [Double](repeating: 2, count: 9))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected .invalidParameter (soft-cap), got \(err)")
        }
    }

    func testSubMatrixMatrix_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let a = mat(3, 3, [Double](repeating: 1, count: 9))
        let b = mat(3, 3, [Double](repeating: 2, count: 9))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.sub, lhs: a, rhs: b)
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected .invalidParameter (soft-cap), got \(err)")
        }
    }

    func testHadamard_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let a = mat(3, 3, [Double](repeating: 1, count: 9))
        let b = mat(3, 3, [Double](repeating: 2, count: 9))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("hadamard", args: [a, b])
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected .invalidParameter (soft-cap), got \(err)")
        }
    }

    func testElementDiv_overSoftCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let a = mat(3, 3, [Double](repeating: 1, count: 9))
        let b = mat(3, 3, [Double](repeating: 2, count: 9))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("elementDiv", args: [a, b])
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected .invalidParameter (soft-cap), got \(err)")
        }
    }

    func testMulMatrixMatrix_overSoftCap_throwsInvalidParameter() throws {
        // (3×3) * (3×3) result = 9 elements; cap = 4 → throw
        try LinAlg.setMaxEvaluatorMatrixElements(4)
        let a = mat(3, 3, [Double](repeating: 1, count: 9))
        let b = mat(3, 3, [Double](repeating: 2, count: 9))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.mul, lhs: a, rhs: b)
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected .invalidParameter (soft-cap), got \(err)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - Non-finite propagation (Task 11.15)
    // -------------------------------------------------------------------------

    func testAddMatrix_nanPropagates() throws {
        let lhs = mat(1, 2, [Double.nan, 1.0])
        let rhs = mat(1, 2, [2.0, 3.0])
        let result = try NumericDispatch.applyBinary(.add, lhs: lhs, rhs: rhs)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssert(m[0, 0].isNaN, "nan + x should remain nan")
        XCTAssertEqual(m[0, 1], 4.0)
    }

    func testMulMatrix_infPropagates() throws {
        // inf * finite → inf in matmul result
        let lhs = mat(1, 1, [Double.infinity])
        let rhs = mat(1, 1, [2.0])
        let result = try NumericDispatch.applyBinary(.mul, lhs: lhs, rhs: rhs)
        // 1×1 result coerces to scalar
        guard case .scalar(let s) = result else { return XCTFail("Expected .scalar") }
        XCTAssert(s.isInfinite)
    }

    // -------------------------------------------------------------------------
    // MARK: - Unary negation via applyUnary (Task 11.16)
    // -------------------------------------------------------------------------

    func testApplyNegMatrix_correctResult() throws {
        // -[[1,2],[3,4]] = [[-1,-2],[-3,-4]]
        let m = mat(2, 2, [1, 2, 3, 4])
        let result = try NumericDispatch.applyUnary(.neg, operand: m)
        guard case .matrix(let r) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(r[0, 0], -1.0)
        XCTAssertEqual(r[0, 1], -2.0)
        XCTAssertEqual(r[1, 0], -3.0)
        XCTAssertEqual(r[1, 1], -4.0)
    }

    func testApplyNegVector_correctResult() throws {
        let v = vec([1, 2, 3])
        let result = try NumericDispatch.applyUnary(.neg, operand: v)
        guard case .matrix(let r) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(r[0, 0], -1.0)
        XCTAssertEqual(r[1, 0], -2.0)
        XCTAssertEqual(r[2, 0], -3.0)
    }

    // -------------------------------------------------------------------------
    // MARK: - Frozen corpus snapshot (Task 11.17/11.18)
    //
    // Reference values computed from SciPy:
    //   import numpy as np
    //   A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
    //   B = np.array([[9,8,7],[6,5,4],[3,2,1]], dtype=float)
    //   v = np.array([[1],[2],[3]])
    //   A+B → [[10,10,10],[10,10,10],[10,10,10]]
    //   A-B → [[-8,-6,-4],[-2,0,2],[4,6,8]]
    //   A@B → [[30,24,18],[84,69,54],[138,114,90]]
    //   A@v → [[14],[32],[50]]
    //   np.multiply(A,B) → [[9,16,21],[24,25,24],[21,16,9]]
    //   A/2 → [[0.5,1,1.5],[2,2.5,3],[3.5,4,4.5]]
    // -------------------------------------------------------------------------

    private let corpusA = [1.0, 2, 3, 4, 5, 6, 7, 8, 9]
    private let corpusB = [9.0, 8, 7, 6, 5, 4, 3, 2, 1]

    func testCorpus_add_3x3() throws {
        let a = mat(3, 3, corpusA)
        let b = mat(3, 3, corpusB)
        let result = try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        for r in 0..<3 { for c in 0..<3 { XCTAssertEqual(m[r, c], 10.0, accuracy: 1e-12) } }
    }

    func testCorpus_sub_3x3() throws {
        let a = mat(3, 3, corpusA)
        let b = mat(3, 3, corpusB)
        let result = try NumericDispatch.applyBinary(.sub, lhs: a, rhs: b)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        // A-B = [[-8,-6,-4],[-2,0,2],[4,6,8]]
        let expected: [[Double]] = [[-8, -6, -4], [-2, 0, 2], [4, 6, 8]]
        for r in 0..<3 { for c in 0..<3 {
            XCTAssertEqual(m[r, c], expected[r][c], accuracy: 1e-12)
        }}
    }

    func testCorpus_matmul_3x3() throws {
        let a = mat(3, 3, corpusA)
        let b = mat(3, 3, corpusB)
        let result = try NumericDispatch.applyBinary(.mul, lhs: a, rhs: b)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        // A@B = [[30,24,18],[84,69,54],[138,114,90]]
        let expected: [[Double]] = [[30, 24, 18], [84, 69, 54], [138, 114, 90]]
        for r in 0..<3 { for c in 0..<3 {
            XCTAssertEqual(m[r, c], expected[r][c], accuracy: 1e-10)
        }}
    }

    func testCorpus_matVec_3x3() throws {
        let a = mat(3, 3, corpusA)
        let v = vec([1, 2, 3])
        let result = try NumericDispatch.applyBinary(.mul, lhs: a, rhs: v)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        // A@v = [[14],[32],[50]]
        XCTAssertEqual(m[0, 0], 14.0, accuracy: 1e-10)
        XCTAssertEqual(m[1, 0], 32.0, accuracy: 1e-10)
        XCTAssertEqual(m[2, 0], 50.0, accuracy: 1e-10)
    }

    func testCorpus_hadamard_3x3() throws {
        let a = mat(3, 3, corpusA)
        let b = mat(3, 3, corpusB)
        let result = try NumericDispatch.applyFunction("hadamard", args: [a, b])
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        // np.multiply(A,B) = [[9,16,21],[24,25,24],[21,16,9]]
        let expected: [[Double]] = [[9, 16, 21], [24, 25, 24], [21, 16, 9]]
        for r in 0..<3 { for c in 0..<3 {
            XCTAssertEqual(m[r, c], expected[r][c], accuracy: 1e-12)
        }}
    }

    func testCorpus_divScalar_3x3() throws {
        let a = mat(3, 3, corpusA)
        let result = try NumericDispatch.applyBinary(.div, lhs: a, rhs: .scalar(2.0))
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        // A/2 = [[0.5,1,1.5],[2,2.5,3],[3.5,4,4.5]]
        let expected: [[Double]] = [[0.5, 1, 1.5], [2, 2.5, 3], [3.5, 4, 4.5]]
        for r in 0..<3 { for c in 0..<3 {
            XCTAssertEqual(m[r, c], expected[r][c], accuracy: 1e-12)
        }}
    }

    // -------------------------------------------------------------------------
    // MARK: - Group-A throw-vs-trap verification (Task 11.19)
    //
    // The process must NOT be terminated. If LinAlg's precondition fires instead
    // of the dispatcher's throw, the test suite itself would crash. A passing
    // test here means the dispatcher owns the error boundary.
    // -------------------------------------------------------------------------

    func testGroupA_add_throwsNotTraps() {
        // Mismatched shapes would crash if the precondition fires
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .add, lhs: mat(1, 3, [1, 2, 3]), rhs: mat(1, 2, [4, 5]))
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testGroupA_sub_throwsNotTraps() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .sub, lhs: mat(1, 3, [1, 2, 3]), rhs: mat(1, 2, [4, 5]))
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testGroupA_hadamard_throwsNotTraps() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction(
                "hadamard", args: [mat(1, 3, [1, 2, 3]), mat(1, 2, [4, 5])])
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testGroupA_elementDiv_throwsNotTraps() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction(
                "elementDiv", args: [mat(1, 3, [1, 2, 3]), mat(1, 2, [4, 5])])
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testGroupA_dot_throwsNotTraps() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .mul, lhs: mat(2, 3, .init(repeating: 0, count: 6)),
                rhs: mat(2, 3, .init(repeating: 0, count: 6)))
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testGroupA_divByZero_throwsNotTraps() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .div, lhs: mat(2, 2, [1, 2, 3, 4]), rhs: .scalar(0.0))
        ) { err in
            if case MathExprError.divisionByZero = err { return }
            XCTFail("Expected .divisionByZero, got \(err)")
        }
    }

    func testGroupA_dotProduct_function_throwsNotTraps() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction(
                "dotProduct",
                args: [mat(2, 3, .init(repeating: 0, count: 6)),
                       mat(2, 3, .init(repeating: 0, count: 6))])
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }
}
