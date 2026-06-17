//
//  NumericDispatchMatrixFunctionsTests.swift
//  NumericSwiftTests
//
//  Tests for Task 12: Group-B real matrix functions (trace/det/inv/expm/logm/sqrtm)
//  and real matrix integer power (matrix^n).
//
//  Coverage:
//    • 12.13 — Parity: trace/det values against frozen snapshot and known results
//    • 12.14 — Parity: inv against frozen snapshot; singular path throws
//    • 12.15 — Parity: expm/logm/sqrtm numerical correctness (≤1e-6)
//    • 12.16 — Error-propagation: non-square inputs → LinAlgError.notSquare
//               propagated (no pre-validation in dispatcher, no process trap)
//    • Matrix integer power (evalMatrixPow): positive, zero, negative exponents;
//               non-square and non-integer guards; singular negative-power throws
//
//  Group-B contract: the dispatcher calls the LinAlg throwing op inside `try`
//  and propagates `LinAlgError.notSquare` outward — it does NOT pre-validate
//  squareness itself (contrast with Group-A operators that pre-validate).
//
//  Reference values are computed from SciPy / NumPy and embedded as literals
//  with inline provenance.  All floating-point comparisons use
//  XCTAssertEqual(_, _, accuracy:) with tolerance specified per assertion.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - NumericDispatchMatrixFunctionsTests

final class NumericDispatchMatrixFunctionsTests: XCTestCase {

    // -------------------------------------------------------------------------
    // MARK: - Helpers
    // -------------------------------------------------------------------------

    /// Wrap a `[[Double]]` into a `NumericValue.matrix`.
    private func mat(_ rows: [[Double]]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows))
    }

    /// Wrap flat data into a `NumericValue.matrix`.
    private func mat(_ r: Int, _ c: Int, _ flat: [Double]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: r, cols: c, data: flat))
    }

    /// Call `NumericDispatch.applyFunction(_:args:)` with a single argument.
    private func callFn(_ name: String, _ arg: NumericValue) throws -> NumericValue {
        try NumericDispatch.applyFunction(name, args: [arg])
    }

    /// Assert two matrices are element-wise equal within `tolerance`.
    private func assertMatrixEqual(
        _ result: NumericValue,
        rows: Int, cols: Int, data: [Double],
        tolerance: Double = 1e-10,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard case .matrix(let m) = result else {
            XCTFail("Expected .matrix, got \(result)", file: file, line: line)
            return
        }
        XCTAssertEqual(m.rows, rows, "row count", file: file, line: line)
        XCTAssertEqual(m.cols, cols, "col count", file: file, line: line)
        for i in 0..<min(data.count, m.data.count) {
            XCTAssertEqual(m.data[i], data[i], accuracy: tolerance,
                "data[\(i)] mismatch", file: file, line: line)
        }
    }

    /// Unwrap a `NumericValue.scalar`, failing the test if the kind is wrong.
    @discardableResult
    private func scalar(
        _ v: NumericValue,
        accuracy: Double = 0,
        expected: Double? = nil,
        file: StaticString = #file,
        line: UInt = #line
    ) -> Double {
        guard case .scalar(let x) = v else {
            XCTFail("Expected .scalar, got \(v)", file: file, line: line)
            return .nan
        }
        if let exp = expected {
            XCTAssertEqual(x, exp, accuracy: accuracy, file: file, line: line)
        }
        return x
    }

    // -------------------------------------------------------------------------
    // MARK: - 12.13  Parity: trace / det
    // -------------------------------------------------------------------------

    // SciPy provenance:
    //   import numpy as np
    //   A = np.array([[1,2],[3,4]])
    //   np.trace(A)   → 5
    //   np.linalg.det(A) → -2

    func testTrace_2x2_equalsSum_of_diagonal() throws {
        // trace([[1,2],[3,4]]) = 1 + 4 = 5
        let result = try callFn("trace", mat([[1, 2], [3, 4]]))
        XCTAssertEqual(scalar(result), 5.0, accuracy: 1e-12)
    }

    func testTrace_3x3_identity_equals_3() throws {
        let result = try callFn("trace", .matrix(LinAlg.eye(3)))
        XCTAssertEqual(scalar(result), 3.0, accuracy: 1e-12)
    }

    func testTrace_diagonal_matrix() throws {
        // trace(diag(1,2,3)) = 6  (parity entry matfn-t02)
        let result = try callFn("trace", .matrix(LinAlg.diag([1, 2, 3])))
        XCTAssertEqual(scalar(result), 6.0, accuracy: 1e-12)
    }

    func testDet_2x2_nonsingular() throws {
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2  (parity entry matfn-d01)
        let result = try callFn("det", mat([[1, 2], [3, 4]]))
        XCTAssertEqual(scalar(result), -2.0, accuracy: 1e-10)
    }

    func testDet_singular_matrix_returns_zero() throws {
        // det([[1,2],[2,4]]) = 0  (parity entry matfn-d02)
        let result = try callFn("det", mat([[1, 2], [2, 4]]))
        XCTAssertEqual(scalar(result), 0.0, accuracy: 1e-10)
    }

    func testDet_3x3_identity_equals_one() throws {
        let result = try callFn("det", .matrix(LinAlg.eye(3)))
        XCTAssertEqual(scalar(result), 1.0, accuracy: 1e-12)
    }

    // SciPy: np.linalg.det(np.array([[2,1],[1,3]])) → 5
    func testDet_2x2_positive_det() throws {
        let result = try callFn("det", mat([[2, 1], [1, 3]]))
        XCTAssertEqual(scalar(result), 5.0, accuracy: 1e-10)
    }

    // -------------------------------------------------------------------------
    // MARK: - 12.14  Parity: inv; singular path throws
    // -------------------------------------------------------------------------

    // SciPy:
    //   np.linalg.inv([[1,2],[3,4]])
    //   → [[-2., 1.], [1.5, -0.5]]

    func testInv_2x2_nonsingular_parity() throws {
        // Parity entry matfn-i01
        let result = try callFn("inv", mat([[1, 2], [3, 4]]))
        // Expected: [[-2, 1], [1.5, -0.5]]
        assertMatrixEqual(result,
            rows: 2, cols: 2,
            data: [-2.0, 1.0, 1.5, -0.5],
            tolerance: 1e-10)
    }

    func testInv_identity_returns_identity() throws {
        let result = try callFn("inv", .matrix(LinAlg.eye(3)))
        assertMatrixEqual(result,
            rows: 3, cols: 3,
            data: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            tolerance: 1e-10)
    }

    // SciPy: np.linalg.inv([[2,0],[0,4]]) → [[0.5, 0], [0, 0.25]]
    func testInv_diagonal_matrix() throws {
        let result = try callFn("inv", mat([[2, 0], [0, 4]]))
        assertMatrixEqual(result,
            rows: 2, cols: 2,
            data: [0.5, 0, 0, 0.25],
            tolerance: 1e-10)
    }

    /// Singular matrix inv → MathExprError.invalidArguments (Group-B semantics:
    /// LinAlg.inv returns nil for singular; dispatcher converts nil → throw).
    func testInv_singular_throws_invalidArguments() throws {
        // [[1,2],[2,4]] is singular (det = 0)
        XCTAssertThrowsError(
            try callFn("inv", mat([[1, 2], [2, 4]]))
        ) { err in
            guard case MathExprError.invalidArguments(let msg) = err else {
                XCTFail("Expected MathExprError.invalidArguments, got \(err)")
                return
            }
            XCTAssert(msg.lowercased().contains("singular"),
                "Error message should mention 'singular': \(msg)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - 12.15  Parity: expm / logm / sqrtm numerical correctness
    // -------------------------------------------------------------------------

    // SciPy:
    //   from scipy.linalg import expm, logm, sqrtm
    //   expm(np.zeros((2,2))) → [[1,0],[0,1]]
    //   expm(np.diag([1,2]))  → diag(e, e²)
    //   logm(np.eye(2))       → [[0,0],[0,0]]
    //   sqrtm(np.eye(2))      → [[1,0],[0,1]]
    //   sqrtm(np.diag([4,9])) → [[2,0],[0,3]]

    func testExpm_zeros_matrix_returns_identity() throws {
        // parity entry matfn-e01
        let result = try callFn("exp", .matrix(LinAlg.zeros(2, 2)))
        assertMatrixEqual(result,
            rows: 2, cols: 2,
            data: [1, 0, 0, 1],
            tolerance: 1e-6)
    }

    func testExpm_diagonal_parity() throws {
        // parity entry matfn-e02: expm(diag(1,2)) = diag(e, e²)
        let result = try callFn("exp", mat([[1, 0], [0, 2]]))
        guard case .matrix(let m) = result else {
            XCTFail("Expected .matrix"); return
        }
        XCTAssertEqual(m[0, 0], exp(1.0), accuracy: 1e-11,
            "expm([[1,0],[0,2]])[0,0] should be e")
        XCTAssertEqual(m[1, 1], exp(2.0), accuracy: 1e-11,
            "expm([[1,0],[0,2]])[1,1] should be e²")
        XCTAssertEqual(m[0, 1], 0.0, accuracy: 1e-10)
        XCTAssertEqual(m[1, 0], 0.0, accuracy: 1e-10)
    }

    func testLogm_identity_returns_zero_matrix() throws {
        // parity entry matfn-l01
        let result = try callFn("log", .matrix(LinAlg.eye(2)))
        assertMatrixEqual(result,
            rows: 2, cols: 2,
            data: [0, 0, 0, 0],
            tolerance: 1e-10)
    }

    func testLogm_positive_diagonal() throws {
        // logm(diag(e, e²)) = diag(1, 2)
        let d = LinAlg.diag([exp(1.0), exp(2.0)])
        let result = try callFn("log", .matrix(d))
        guard case .matrix(let m) = result else {
            XCTFail("Expected .matrix"); return
        }
        XCTAssertEqual(m[0, 0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(m[1, 1], 2.0, accuracy: 1e-10)
        XCTAssertEqual(m[0, 1], 0.0, accuracy: 1e-10)
    }

    /// logm with negative eigenvalues → LinAlg.logm returns nil
    /// → dispatcher throws MathExprError.invalidArguments.
    func testLogm_negative_eigenvalue_throws() throws {
        // diag(-1, 2) has negative eigenvalue → logm → nil → throws
        let d = LinAlg.diag([-1.0, 2.0])
        XCTAssertThrowsError(
            try callFn("log", .matrix(d))
        ) { err in
            guard case MathExprError.invalidArguments(let msg) = err else {
                XCTFail("Expected MathExprError.invalidArguments, got \(err)")
                return
            }
            XCTAssert(msg.contains("diagonalizable") || msg.contains("logarithm") || msg.contains("eigenvalues"),
                "Error message should describe the limitation: \(msg)")
        }
    }

    func testSqrtm_identity_returns_identity() throws {
        // parity entry matfn-s01
        let result = try callFn("sqrt", .matrix(LinAlg.eye(2)))
        assertMatrixEqual(result,
            rows: 2, cols: 2,
            data: [1, 0, 0, 1],
            tolerance: 1e-10)
    }

    func testSqrtm_diagonal_parity() throws {
        // parity entry matfn-s02: sqrtm(diag(4,9)) = diag(2,3)
        let result = try callFn("sqrt", mat([[4, 0], [0, 9]]))
        guard case .matrix(let m) = result else {
            XCTFail("Expected .matrix"); return
        }
        XCTAssertEqual(m[0, 0], 2.0, accuracy: 1e-10)
        XCTAssertEqual(m[1, 1], 3.0, accuracy: 1e-10)
        XCTAssertEqual(m[0, 1], 0.0, accuracy: 1e-10)
    }

    /// sqrtm with negative eigenvalues → LinAlg.sqrtm returns nil
    /// → dispatcher throws MathExprError.invalidArguments.
    func testSqrtm_negative_eigenvalue_throws() throws {
        let d = LinAlg.diag([-4.0, 1.0])
        XCTAssertThrowsError(
            try callFn("sqrt", .matrix(d))
        ) { err in
            guard case MathExprError.invalidArguments(let msg) = err else {
                XCTFail("Expected MathExprError.invalidArguments, got \(err)")
                return
            }
            XCTAssert(msg.contains("eigenvalue") || msg.contains("negative") || msg.contains("square root"),
                "Error message should describe the limitation: \(msg)")
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - 12.16  Error propagation: non-square → LinAlgError.notSquare
    // -------------------------------------------------------------------------
    //
    // Group-B contract: the dispatcher does NOT pre-validate squareness.
    // It calls the LinAlg op with `try`, so `LinAlgError.notSquare` propagates
    // from inside LinAlg directly to the caller — no process trap, no pre-guard.

    private func assertThrowsNotSquare(
        _ name: String, rows: Int, cols: Int,
        file: StaticString = #file, line: UInt = #line
    ) {
        let m = mat(rows, cols, [Double](repeating: 1.0, count: rows * cols))
        XCTAssertThrowsError(
            try callFn(name, m),
            "\(name)(\(rows)×\(cols)) must throw",
            file: file, line: line
        ) { err in
            switch err {
            case LinAlg.LinAlgError.notSquare:
                break   // correct — Group-B propagation
            default:
                XCTFail("\(name)(\(rows)×\(cols)): expected notSquare, got \(err)",
                    file: file, line: line)
            }
        }
    }

    func testTrace_nonSquare_propagatesNotSquare() {
        assertThrowsNotSquare("trace", rows: 2, cols: 3)   // groupB-e01
    }

    func testDet_nonSquare_propagatesNotSquare() {
        assertThrowsNotSquare("det", rows: 3, cols: 2)     // groupB-e02
    }

    func testInv_nonSquare_propagatesNotSquare() {
        assertThrowsNotSquare("inv", rows: 2, cols: 3)     // groupB-e03
    }

    func testExpm_nonSquare_propagatesNotSquare() {
        assertThrowsNotSquare("exp", rows: 2, cols: 3)     // groupB-e04
    }

    func testLogm_nonSquare_propagatesNotSquare() {
        assertThrowsNotSquare("log", rows: 3, cols: 2)     // groupB-e05
    }

    func testSqrtm_nonSquare_propagatesNotSquare() {
        assertThrowsNotSquare("sqrt", rows: 2, cols: 3)    // groupB-e06
    }

    /// Verify the dispatcher does NOT pre-validate squareness for Group-B
    /// (i.e. the error comes FROM LinAlg, not from a guard in the dispatcher).
    /// This is the contract distinguishing Group-B from Group-A.
    func testTrace_nonSquare_error_originates_in_linAlg() {
        // If the dispatcher pre-validated, it would throw MathExprError.shapeMismatch.
        // Group-B contract: must throw LinAlgError.notSquare, never shapeMismatch.
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try callFn("trace", m)) { err in
            if case MathExprError.shapeMismatch = err {
                XCTFail("Dispatcher must NOT pre-validate for Group-B functions; "
                    + "got shapeMismatch instead of notSquare")
            }
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - 12.10  Arity: Group-B functions require exactly 1 argument
    // -------------------------------------------------------------------------

    func testTrace_wrongArity_throws() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("trace", args: [])
        )
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction(
                "trace",
                args: [mat([[1, 0], [0, 1]]), mat([[1, 0], [0, 1]])])
        )
    }

    func testDet_wrongArity_throws() {
        XCTAssertThrowsError(try NumericDispatch.applyFunction("det", args: []))
    }

    func testInv_wrongArity_throws() {
        XCTAssertThrowsError(try NumericDispatch.applyFunction("inv", args: []))
    }

    // -------------------------------------------------------------------------
    // MARK: - 12.11  Kind rejection: trace/det/inv reject scalar/complex
    // -------------------------------------------------------------------------

    func testTrace_scalar_throws_invalidArguments() {
        XCTAssertThrowsError(try callFn("trace", .scalar(5.0))) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("Expected invalidArguments, got \(err)"); return
            }
        }
    }

    func testDet_scalar_throws_invalidArguments() {
        XCTAssertThrowsError(try callFn("det", .scalar(3.0))) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("Expected invalidArguments, got \(err)"); return
            }
        }
    }

    func testInv_scalar_throws_invalidArguments() {
        XCTAssertThrowsError(try callFn("inv", .scalar(2.0))) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("Expected invalidArguments, got \(err)"); return
            }
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - 12.17  Scalar result construction: trace / det → NumericValue.scalar
    // -------------------------------------------------------------------------

    func testTrace_returns_scalar_kind() throws {
        let result = try callFn("trace", .matrix(LinAlg.eye(2)))
        XCTAssertEqual(result.kind, .scalar, "trace result must be .scalar")
    }

    func testDet_returns_scalar_kind() throws {
        let result = try callFn("det", .matrix(LinAlg.eye(2)))
        XCTAssertEqual(result.kind, .scalar, "det result must be .scalar")
    }

    // -------------------------------------------------------------------------
    // MARK: - 12.18  Matrix result construction: inv/expm/logm/sqrtm → .matrix
    // -------------------------------------------------------------------------

    func testInv_returns_matrix_kind() throws {
        let result = try callFn("inv", .matrix(LinAlg.eye(2)))
        XCTAssertEqual(result.kind, .matrix, "inv result must be .matrix")
    }

    func testExpm_returns_matrix_kind() throws {
        let result = try callFn("exp", .matrix(LinAlg.eye(2)))
        XCTAssertEqual(result.kind, .matrix, "expm result must be .matrix")
    }

    func testLogm_returns_matrix_kind() throws {
        let result = try callFn("log", .matrix(LinAlg.eye(2)))
        XCTAssertEqual(result.kind, .matrix, "logm result must be .matrix")
    }

    func testSqrtm_returns_matrix_kind() throws {
        let result = try callFn("sqrt", .matrix(LinAlg.eye(2)))
        XCTAssertEqual(result.kind, .matrix, "sqrtm result must be .matrix")
    }

    // -------------------------------------------------------------------------
    // MARK: - Matrix integer power (evalMatrixPow via applyBinary(.pow, …))
    // -------------------------------------------------------------------------
    //
    // SciPy provenance:
    //   import numpy as np
    //   A = np.array([[1,1],[0,1]])
    //   np.linalg.matrix_power(A, 0) → [[1,0],[0,1]]
    //   np.linalg.matrix_power(A, 1) → [[1,1],[0,1]]
    //   np.linalg.matrix_power(A, 2) → [[1,2],[0,1]]
    //   np.linalg.matrix_power(A, 3) → [[1,3],[0,1]]
    //   np.linalg.matrix_power(A,-1) → [[1,-1],[0,1]]
    //   np.linalg.matrix_power(A,-2) → [[1,-2],[0,1]]

    private func matPow(_ m: NumericValue, _ e: Double) throws -> NumericValue {
        try NumericDispatch.applyBinary(.pow, lhs: m, rhs: .scalar(e))
    }

    func testMatrixPow_zero_exponent_returns_identity() throws {
        // A⁰ = I for any square matrix, even singular
        let A = mat([[1, 1], [0, 1]])
        let result = try matPow(A, 0)
        assertMatrixEqual(result, rows: 2, cols: 2, data: [1, 0, 0, 1])
    }

    func testMatrixPow_zero_exponent_singular_also_identity() throws {
        // Even a singular matrix: A⁰ = I by convention
        let singular = mat([[1, 2], [2, 4]])
        let result = try matPow(singular, 0)
        assertMatrixEqual(result, rows: 2, cols: 2, data: [1, 0, 0, 1])
    }

    func testMatrixPow_exponent_one_returns_original() throws {
        let A = mat([[1, 1], [0, 1]])
        let result = try matPow(A, 1)
        assertMatrixEqual(result, rows: 2, cols: 2, data: [1, 1, 0, 1])
    }

    func testMatrixPow_exponent_two_upper_triangular() throws {
        // [[1,1],[0,1]]^2 = [[1,2],[0,1]]
        let A = mat([[1, 1], [0, 1]])
        let result = try matPow(A, 2)
        assertMatrixEqual(result, rows: 2, cols: 2, data: [1, 2, 0, 1],
            tolerance: 1e-12)
    }

    func testMatrixPow_exponent_three_upper_triangular() throws {
        // [[1,1],[0,1]]^3 = [[1,3],[0,1]]
        let A = mat([[1, 1], [0, 1]])
        let result = try matPow(A, 3)
        assertMatrixEqual(result, rows: 2, cols: 2, data: [1, 3, 0, 1],
            tolerance: 1e-12)
    }

    func testMatrixPow_large_exponent_correct() throws {
        // For diagonal matrix diag(2,3)^5 = diag(32, 243)
        // SciPy: np.linalg.matrix_power(np.diag([2.,3.]), 5) → [[32,0],[0,243]]
        let D = mat([[2, 0], [0, 3]])
        let result = try matPow(D, 5)
        assertMatrixEqual(result,
            rows: 2, cols: 2,
            data: [32, 0, 0, 243],
            tolerance: 1e-10)
    }

    func testMatrixPow_negative_exponent_invertible() throws {
        // [[1,1],[0,1]]^(-1) = [[1,-1],[0,1]]
        let A = mat([[1, 1], [0, 1]])
        let result = try matPow(A, -1)
        assertMatrixEqual(result, rows: 2, cols: 2, data: [1, -1, 0, 1],
            tolerance: 1e-10)
    }

    func testMatrixPow_negative_two_upper_triangular() throws {
        // [[1,1],[0,1]]^(-2) = [[1,-2],[0,1]]
        let A = mat([[1, 1], [0, 1]])
        let result = try matPow(A, -2)
        assertMatrixEqual(result, rows: 2, cols: 2, data: [1, -2, 0, 1],
            tolerance: 1e-10)
    }

    func testMatrixPow_identity_any_power_is_identity() throws {
        let I = NumericValue.matrix(LinAlg.eye(3))
        for e: Double in [1, 2, 5, 10, -1, -3] {
            let result = try matPow(I, e)
            assertMatrixEqual(result,
                rows: 3, cols: 3,
                data: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                tolerance: 1e-10)
        }
    }

    func testMatrixPow_nonSquare_throws_invalidArguments() {
        // Caller pre-validates squareness and throws MathExprError.invalidArguments
        let m = mat(2, 3, [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(try matPow(m, 2)) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("Expected invalidArguments for non-square matrix power, got \(err)")
                return
            }
        }
    }

    func testMatrixPow_fractionalExponent_throws_invalidArguments() {
        // Non-integer exponent for matrix power is not supported
        let A = mat([[1, 0], [0, 1]])
        XCTAssertThrowsError(try matPow(A, 1.5)) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("Expected invalidArguments for fractional exponent, got \(err)")
                return
            }
        }
    }

    func testMatrixPow_singular_negative_exponent_throws() {
        // Singular matrix with negative exponent: A^(-1) fails because A is not invertible
        let singular = mat([[1, 2], [2, 4]])
        XCTAssertThrowsError(try matPow(singular, -1)) { err in
            guard case MathExprError.invalidArguments(let msg) = err else {
                XCTFail("Expected invalidArguments for singular matrix^(-1), got \(err)")
                return
            }
            XCTAssert(msg.lowercased().contains("singular"),
                "Error message should mention singular: \(msg)")
        }
    }

    func testMatrixPow_result_is_matrix_kind() throws {
        let A = mat([[2, 0], [0, 3]])
        let result = try matPow(A, 2)
        XCTAssertEqual(result.kind, .matrix, "matrix^n result must be .matrix")
    }

    // Round-trip: A^2 * A^(-2) ≈ I
    func testMatrixPow_roundTrip_positive_negative() throws {
        let A = mat([[2, 1], [1, 3]])
        let fwd = try matPow(A, 3)
        let back = try matPow(A, -3)
        guard case .matrix(let mF) = fwd, case .matrix(let mB) = back else {
            XCTFail("Both results must be .matrix"); return
        }
        let product = LinAlg.dot(mF, mB)
        // product should be ≈ identity
        XCTAssertEqual(product[0, 0], 1.0, accuracy: 1e-8)
        XCTAssertEqual(product[1, 1], 1.0, accuracy: 1e-8)
        XCTAssertEqual(product[0, 1], 0.0, accuracy: 1e-8)
        XCTAssertEqual(product[1, 0], 0.0, accuracy: 1e-8)
    }
}
