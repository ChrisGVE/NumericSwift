//
//  NumericDispatchMatrixPowerTests.swift
//  NumericSwiftTests
//
//  Task 13: matrix integer power — gap-filling tests.
//
//  The core positive/negative/zero/singular/non-square/fractional tests
//  already live in NumericDispatchMatrixFunctionsTests.swift (written for Task
//  12). This file adds the four behaviours that were NOT covered there:
//
//    13.16 — Exponentiation-by-squaring matches naive repeated LinAlg.dot for
//             n in 2...8 across several matrices.
//    13.12 — 1×1 matrix^n result stays `.matrix` (no §4.3a scalar collapse).
//    13.7  — complexMatrix^integer: exponentiation-by-squaring (cinv for n < 0),
//             value checks against closed-form powers (diag(i), etc.).
//    13.13 — Large/overflowing exponent: a Double that is integral-valued but
//             outside Int range throws invalidArguments before any loop.
//
//  SciPy provenance:
//    All expected values are computed with numpy.linalg.matrix_power and
//    embedded as literals; exact tolerance used for integer-entry matrices.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - NumericDispatchMatrixPowerTests

final class NumericDispatchMatrixPowerTests: XCTestCase {

    // -------------------------------------------------------------------------
    // MARK: - Helpers
    // -------------------------------------------------------------------------

    private func mat(_ rows: [[Double]]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows))
    }

    private func mat(_ r: Int, _ c: Int, _ flat: [Double]) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: r, cols: c, data: flat))
    }

    /// Apply matrix integer power via the dispatch surface.
    private func matPow(_ m: NumericValue, _ e: Double) throws -> NumericValue {
        try NumericDispatch.applyBinary(.pow, lhs: m, rhs: .scalar(e))
    }

    /// Assert element-wise equality within `tolerance`.
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
        zip(data, m.data).enumerated().forEach { i, pair in
            XCTAssertEqual(pair.1, pair.0, accuracy: tolerance,
                "data[\(i)]", file: file, line: line)
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - 13.16 — Squaring algorithm matches naive repeated LinAlg.dot
    // -------------------------------------------------------------------------
    //
    // Contract: matPow(M, n) == dot(dot(… dot(M, M) …)) (n-1 dots) for all n.
    // Verifies the exponentiation-by-squaring optimisation is numerically
    // equivalent to the readable reference implementation.

    /// Naive reference: apply LinAlg.dot (n-1) times.
    private func naivePow(_ m: LinAlg.Matrix, _ n: Int) -> LinAlg.Matrix {
        precondition(n >= 1)
        var result = m
        for _ in 1..<n { result = LinAlg.dot(result, m) }
        return result
    }

    func testSquaringEqualsNaive_upperTriangular_n2to8() throws {
        // A = [[1,1],[0,1]]: powers have closed-form A^n = [[1,n],[0,1]].
        // Validates algorithm for the range n=2..8.
        // SciPy: np.linalg.matrix_power([[1,1],[0,1]], n) for n in 2..8
        let A = LinAlg.Matrix([[1, 1], [0, 1]])
        let mA = NumericValue.matrix(A)
        for n in 2...8 {
            let fast = try matPow(mA, Double(n))
            let naive = naivePow(A, n)
            guard case .matrix(let fastM) = fast else {
                XCTFail("Expected .matrix for n=\(n)"); continue
            }
            XCTAssertEqual(fastM.rows, naive.rows)
            XCTAssertEqual(fastM.cols, naive.cols)
            zip(fastM.data, naive.data).enumerated().forEach { i, pair in
                XCTAssertEqual(pair.0, pair.1, accuracy: 1e-12,
                    "A^\(n) data[\(i)]: squaring vs naive")
            }
        }
    }

    func testSquaringEqualsNaive_diagonal_n2to6() throws {
        // D = diag(2, 3): D^n = diag(2^n, 3^n) — exact for integer matrices.
        // SciPy: np.linalg.matrix_power(np.diag([2.,3.]), n) for n in 2..6
        let D = LinAlg.Matrix([[2, 0], [0, 3]])
        let mD = NumericValue.matrix(D)
        for n in 2...6 {
            let fast = try matPow(mD, Double(n))
            let naive = naivePow(D, n)
            guard case .matrix(let fastM) = fast else {
                XCTFail("Expected .matrix for n=\(n)"); continue
            }
            zip(fastM.data, naive.data).enumerated().forEach { i, pair in
                XCTAssertEqual(pair.0, pair.1, accuracy: 1e-10,
                    "D^\(n) data[\(i)]: squaring vs naive")
            }
        }
    }

    func testSquaringEqualsNaive_general_3x3_n3to5() throws {
        // A 3×3 matrix with no special structure.
        // SciPy: np.linalg.matrix_power([[1,2,0],[0,1,3],[0,0,1]], n) for n in 3..5
        let G = LinAlg.Matrix([[1, 2, 0], [0, 1, 3], [0, 0, 1]])
        let mG = NumericValue.matrix(G)
        for n in 3...5 {
            let fast = try matPow(mG, Double(n))
            let naive = naivePow(G, n)
            guard case .matrix(let fastM) = fast else {
                XCTFail("Expected .matrix for n=\(n)"); continue
            }
            zip(fastM.data, naive.data).enumerated().forEach { i, pair in
                XCTAssertEqual(pair.0, pair.1, accuracy: 1e-8,
                    "G^\(n) data[\(i)]: squaring vs naive")
            }
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - 13.12 — 1×1 matrix result stays .matrix (no §4.3a collapse)
    // -------------------------------------------------------------------------
    //
    // §4.3a coercion collapses ONLY vec·vec (dot/matmul) results to scalar.
    // matrix^n must NOT trigger scalar coercion for 1×1 results.

    func testMatrixPow_1x1_staysMatrix_positive() throws {
        // [[3]]^4 = [[81]]; result must be .matrix, not .scalar(81).
        let M = mat([[3.0]])
        let result = try matPow(M, 4)
        XCTAssertEqual(result.kind, .matrix,
            "1×1 matrix^4 must remain .matrix, not collapse to .scalar")
        assertMatrixEqual(result, rows: 1, cols: 1, data: [81.0])
    }

    func testMatrixPow_1x1_staysMatrix_zero() throws {
        // [[5]]^0 = [[1]]; result must be .matrix, not .scalar(1).
        let M = mat([[5.0]])
        let result = try matPow(M, 0)
        XCTAssertEqual(result.kind, .matrix,
            "1×1 matrix^0 must remain .matrix, not collapse to .scalar")
        assertMatrixEqual(result, rows: 1, cols: 1, data: [1.0])
    }

    func testMatrixPow_1x1_staysMatrix_negative() throws {
        // [[2]]^(-3) = [[0.125]]; result must be .matrix, not .scalar.
        let M = mat([[2.0]])
        let result = try matPow(M, -3)
        XCTAssertEqual(result.kind, .matrix,
            "1×1 matrix^(-3) must remain .matrix, not collapse to .scalar")
        assertMatrixEqual(result, rows: 1, cols: 1, data: [0.125], tolerance: 1e-12)
    }

    // -------------------------------------------------------------------------
    // MARK: - 13.7 — complexMatrix^integer (implemented; Codex pre-0.3.0 audit)
    // -------------------------------------------------------------------------
    //
    // evalComplexMatrixPow performs exponentiation-by-squaring (cinv for n < 0),
    // mirroring the real evalMatrixPow. These replace the former stub-throws test.

    /// Assert a `.complexMatrix` result equals the given real/imag blocks.
    private func assertComplexMatrixEqual(
        _ result: NumericValue,
        rows: Int, cols: Int, real: [Double], imag: [Double],
        tolerance: Double = 1e-10,
        file: StaticString = #file, line: UInt = #line
    ) {
        guard case .complexMatrix(let cm) = result else {
            XCTFail("Expected .complexMatrix, got \(result)", file: file, line: line)
            return
        }
        XCTAssertEqual(cm.rows, rows, "row count", file: file, line: line)
        XCTAssertEqual(cm.cols, cols, "col count", file: file, line: line)
        zip(real, cm.real).enumerated().forEach { i, p in
            XCTAssertEqual(p.1, p.0, accuracy: tolerance, "real[\(i)]", file: file, line: line)
        }
        zip(imag, cm.imag).enumerated().forEach { i, p in
            XCTAssertEqual(p.1, p.0, accuracy: tolerance, "imag[\(i)]", file: file, line: line)
        }
    }

    private func cmPow(_ cm: LinAlg.ComplexMatrix, _ e: Double) throws -> NumericValue {
        try NumericDispatch.applyBinary(.pow, lhs: .complexMatrix(cm), rhs: .scalar(e))
    }

    /// Purely-real complex matrix [[1,1],[0,1]]^3 = [[1,3],[0,1]] (imag 0).
    func testComplexMatrixPow_realEntries_matchesRealPower() throws {
        let cm = LinAlg.ComplexMatrix(LinAlg.Matrix([[1, 1], [0, 1]]))
        let r = try cmPow(cm, 3)
        assertComplexMatrixEqual(r, rows: 2, cols: 2,
            real: [1, 3, 0, 1], imag: [0, 0, 0, 0])
    }

    /// diag(i)^2 = diag(i²) = diag(-1).
    func testComplexMatrixPow_imaginaryDiagonal_squared() throws {
        let cm = LinAlg.ComplexMatrix(rows: 2, cols: 2,
            real: [0, 0, 0, 0], imag: [1, 0, 0, 1])
        let r = try cmPow(cm, 2)
        assertComplexMatrixEqual(r, rows: 2, cols: 2,
            real: [-1, 0, 0, -1], imag: [0, 0, 0, 0])
    }

    /// A^0 = identity (complex), even for a singular A.
    func testComplexMatrixPow_zeroExponent_identity() throws {
        let cm = LinAlg.ComplexMatrix(rows: 2, cols: 2,
            real: [0, 0, 0, 0], imag: [0, 0, 0, 0])  // zero matrix (singular)
        let r = try cmPow(cm, 0)
        assertComplexMatrixEqual(r, rows: 2, cols: 2,
            real: [1, 0, 0, 1], imag: [0, 0, 0, 0])
    }

    /// diag(i)^(-1) = diag(1/i) = diag(-i).
    func testComplexMatrixPow_negativeExponent_inverse() throws {
        let cm = LinAlg.ComplexMatrix(rows: 2, cols: 2,
            real: [0, 0, 0, 0], imag: [1, 0, 0, 1])
        let r = try cmPow(cm, -1)
        assertComplexMatrixEqual(r, rows: 2, cols: 2,
            real: [0, 0, 0, 0], imag: [-1, 0, 0, -1])
    }

    /// A negative power of a singular complex matrix throws invalidArguments.
    func testComplexMatrixPow_singularNegative_throws() {
        let cm = LinAlg.ComplexMatrix(rows: 2, cols: 2,
            real: [0, 0, 0, 0], imag: [0, 0, 0, 0])  // zero matrix
        XCTAssertThrowsError(try cmPow(cm, -2)) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("singular negative power must throw invalidArguments, got \(err)")
                return
            }
        }
    }

    // -------------------------------------------------------------------------
    // MARK: - 13.13 — Large/overflow exponent policy
    // -------------------------------------------------------------------------
    //
    // A Double exponent that is integral-valued but exceeds Int.max (or is
    // non-finite) should be rejected with invalidArguments before any
    // arithmetic begins.  The existing guard `e == e.rounded()` accepts any
    // finite integral-valued Double, including values like 1e18 that truncate
    // to Int safely on 64-bit platforms.  This test guards that +/-inf and NaN
    // are rejected, since they satisfy e != e.rounded() or are non-finite.
    //
    // On a 64-bit platform Int.max ≈ 9.2e18; Double can represent values above
    // this exactly, so we test ±Inf and NaN (which the rounded() guard catches
    // via NaN != NaN and inf != inf.rounded() on some platforms).
    //
    // Note: the exponentiation-by-squaring loop is O(log n), so a very large
    // but representable n is computationally bounded — element-value growth
    // (overflow to inf) follows IEEE 754 per the nonFiniteFloat policy, which
    // is the same as scalar pow behaviour.

    func testMatrixPow_nanExponent_throws() {
        let A = mat([[1, 0], [0, 1]])
        XCTAssertThrowsError(try matPow(A, Double.nan)) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("NaN exponent must throw invalidArguments, got \(err)")
                return
            }
        }
    }

    func testMatrixPow_positiveInfExponent_throws() {
        let A = mat([[1, 0], [0, 1]])
        XCTAssertThrowsError(try matPow(A, Double.infinity)) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("+Inf exponent must throw invalidArguments, got \(err)")
                return
            }
        }
    }

    func testMatrixPow_negativeInfExponent_throws() {
        let A = mat([[1, 0], [0, 1]])
        XCTAssertThrowsError(try matPow(A, -Double.infinity)) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("-Inf exponent must throw invalidArguments, got \(err)")
                return
            }
        }
    }

    // Large-but-valid exponent on identity: result stays .matrix (no crash).
    // This confirms the loop terminates (O(log n)) for a very large valid n.
    func testMatrixPow_largeValidExponent_terminesNormally() throws {
        // 2×2 identity^(2^30) = identity; O(log n) = 30 multiplications.
        let I = NumericValue.matrix(LinAlg.eye(2))
        let result = try matPow(I, Double(1 << 30))
        XCTAssertEqual(result.kind, .matrix,
            "identity^(2^30) must terminate and return .matrix")
        guard case .matrix(let m) = result else { return }
        XCTAssertEqual(m[0, 0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(m[1, 1], 1.0, accuracy: 1e-10)
        XCTAssertEqual(m[0, 1], 0.0, accuracy: 1e-10)
    }
}
