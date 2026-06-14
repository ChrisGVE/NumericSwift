//
//  NumericDispatchCoercionTests.swift
//  NumericSwiftTests
//
//  Coercion-lattice tests for the unified numeric pipeline (Task 14).
//
//  Coverage strategy follows the §15 truth table and §4.3a specification:
//
//  Group A — Scalar/complex promotion (14.14)
//    • scalar+scalar → scalar (no promotion, baseline)
//    • scalar+complex → complex (S→C promotion)
//    • complex+scalar → complex (S→C promotion, commuted)
//    • complex+complex → complex (no promotion, baseline)
//    Verifies scalar operand is promoted to complex, not the reverse.
//
//  Group B — Real-matrix / complex-matrix promotion (14.15)
//    Helpers: promoteScalarToComplex, promoteToComplexMatrix
//    • promoteScalarToComplex wraps Double with zero im
//    • promoteToComplexMatrix lifts Matrix to ComplexMatrix with zero im arrays
//    Shape preservation is checked; data fidelity verified element-wise.
//
//  Group C — 1×1 → scalar collapse POSITIVE tests at the dot site (14.16)
//    Both the `*` (M*M matmul) and `dotProduct` function paths are tested:
//    • row-vector * col-vector → scalar (1×3 · 3×1 = dot = 14.0)
//    • col-vector * row-vector: NOT a vec·vec in the dot sense (produces 3×3 matrix)
//    • `dotProduct(M, M)` vec·vec → scalar
//    • `*` (M*M) non-vec-vec but producing 1×1: also coerces per §15 ("1×1 → §4.3a")
//    Exact values verified against manual computation.
//
//  Group D — 1×1 NO-collapse NEGATIVE tests (14.17)
//    Proves the collapse ONLY fires at the dot/matmul result site.
//    • Explicit 1×1 matrix operand in add → stays .matrix
//    • 1×1 result of hadamard → stays .matrix
//    • 1×1 result of matrix+matrix (1×1 + 1×1) → stays .matrix
//    • 1×1 result of matrix-matrix → stays .matrix
//    • transpose of 1×1 → stays .matrix
//    • neg of 1×1 → stays .matrix
//    • inv of 1×1 → stays .matrix (not coerced)
//    • det of 1×1 → returns scalar (this IS by design: det always returns Double)
//    The negative tests are grouped with assertions on the kind token, not value.
//
//  Group E — Cross-axis no-coercion routing (14.18)
//    • scalar * matrix → matrix (not complexMatrix)
//    • matrix * scalar → matrix (not complexMatrix)
//    • complex * complex → complex (not scalar)
//    • matrix * matrix (non-1×1 result) → matrix (not scalar)
//    Ensures kind routing doesn't over-promote.
//
//  Group F — coerce1x1 helper unit tests (14.7)
//    Direct tests of the static helper to ensure:
//    • 1×1 .matrix → .scalar with correct value
//    • 2×2 .matrix → unchanged .matrix
//    • .scalar → unchanged .scalar
//    • .complex → unchanged .complex
//    • 1×1 .complexMatrix is NOT touched (coerce1x1Complex's domain)
//
//  Group G — coerce1x1Complex helper unit tests (14.8)
//    Direct tests of the static helper:
//    • 1×1 .complexMatrix → .complex with correct re and im
//    • 2×2 .complexMatrix → unchanged .complexMatrix
//    • .scalar → unchanged .scalar
//    • .matrix → unchanged .matrix
//    • 1×1 .matrix is NOT touched (coerce1x1's domain)
//
//  Group H — promoteToComplexMatrix unit tests (14.3)
//    • Shape preservation: rows and cols match input
//    • Real data fidelity: real array equals input matrix data
//    • Zero imaginary: im array is all zeros
//    • Multiple sizes including non-square
//
//  Group I — promoteScalarToComplex unit tests (14.2)
//    • re == input, im == 0 for positive, negative, zero, inf, nan
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - NumericDispatchCoercionTests

final class NumericDispatchCoercionTests: XCTestCase {

    // MARK: - Helpers

    private func makeMatrix(
        _ rows: Int, _ cols: Int, data: [Double]? = nil, value: Double = 1.0
    ) -> NumericValue {
        let d = data ?? [Double](repeating: value, count: rows * cols)
        return .matrix(LinAlg.Matrix(rows: rows, cols: cols, data: d))
    }

    private func makeCM(
        _ rows: Int, _ cols: Int,
        real: [Double]? = nil, imag: [Double]? = nil,
        re: Double = 1.0, im: Double = 0.0
    ) -> NumericValue {
        let n = rows * cols
        let r = real ?? [Double](repeating: re, count: n)
        let i = imag ?? [Double](repeating: im, count: n)
        return .complexMatrix(LinAlg.ComplexMatrix(rows: rows, cols: cols, real: r, imag: i))
    }

    // MARK: - Group A: scalar / complex promotion (14.14)

    func testScalarPlusScalar_isScalar() throws {
        // Baseline: no promotion when both are scalar
        let result = try NumericDispatch.applyBinary(.add, lhs: .scalar(3), rhs: .scalar(5))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar; got \(result)") }
        XCTAssertEqual(v, 8.0)
    }

    func testScalarPlusComplex_isComplex() throws {
        // scalar + complex → complex; scalar is promoted S→C
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .scalar(2.0), rhs: .complex(Complex(re: 0, im: 3)))
        guard case .complex(let z) = result else { return XCTFail("Expected complex; got \(result)") }
        XCTAssertEqual(z.re, 2.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, 3.0, accuracy: 1e-14)
    }

    func testComplexPlusScalar_isComplex() throws {
        // complex + scalar → complex (commuted); scalar is promoted S→C
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .complex(Complex(re: 1, im: 4)), rhs: .scalar(3.0))
        guard case .complex(let z) = result else { return XCTFail("Expected complex; got \(result)") }
        XCTAssertEqual(z.re, 4.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, 4.0, accuracy: 1e-14)
    }

    func testComplexPlusComplex_isComplex() throws {
        // Baseline: no promotion, both complex
        let a = Complex(re: 1, im: 2), b = Complex(re: 3, im: 4)
        let result = try NumericDispatch.applyBinary(.add, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = result else { return XCTFail("Expected complex; got \(result)") }
        XCTAssertEqual(z.re, 4.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, 6.0, accuracy: 1e-14)
    }

    func testScalarMulComplex_isComplex() throws {
        // scalar * complex → complex
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .scalar(2.0), rhs: .complex(Complex(re: 3, im: 1)))
        guard case .complex(let z) = result else { return XCTFail("Expected complex; got \(result)") }
        XCTAssertEqual(z.re, 6.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, 2.0, accuracy: 1e-14)
    }

    func testComplexMulScalar_isComplex() throws {
        // complex * scalar → complex (commuted)
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complex(Complex(re: 3, im: 1)), rhs: .scalar(2.0))
        guard case .complex(let z) = result else { return XCTFail("Expected complex; got \(result)") }
        XCTAssertEqual(z.re, 6.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, 2.0, accuracy: 1e-14)
    }

    func testScalarSubComplex_promotesResult() throws {
        // scalar - complex → complex
        let result = try NumericDispatch.applyBinary(
            .sub, lhs: .scalar(5.0), rhs: .complex(Complex(re: 2, im: 3)))
        guard case .complex(let z) = result else { return XCTFail("Expected complex; got \(result)") }
        XCTAssertEqual(z.re, 3.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, -3.0, accuracy: 1e-14)
    }

    // MARK: - Group B: M → CM promotion helpers (14.15)

    func testPromoteToComplexMatrix_shapePreserved() {
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let cm = NumericDispatch.promoteToComplexMatrix(m)
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 3)
    }

    func testPromoteToComplexMatrix_realDataFidelity() {
        let data: [Double] = [1, 2, 3, 4, 5, 6]
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: data)
        let cm = NumericDispatch.promoteToComplexMatrix(m)
        // Real part must exactly match the input data
        for (i, expected) in data.enumerated() {
            XCTAssertEqual(cm.real[i], expected, accuracy: 1e-14,
                           "real[\(i)] mismatch after promotion")
        }
    }

    func testPromoteToComplexMatrix_imagZero() {
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let cm = NumericDispatch.promoteToComplexMatrix(m)
        // Imaginary part must be all zeros
        XCTAssertTrue(cm.imag.allSatisfy { $0 == 0 },
                      "imaginary part should be all zeros after real→complex promotion")
    }

    func testPromoteToComplexMatrix_nonSquare() {
        // Verify non-square matrix promotions work
        let m = LinAlg.Matrix(rows: 3, cols: 1, data: [7, 8, 9])
        let cm = NumericDispatch.promoteToComplexMatrix(m)
        XCTAssertEqual(cm.rows, 3)
        XCTAssertEqual(cm.cols, 1)
        XCTAssertEqual(cm.real, [7, 8, 9])
        XCTAssertTrue(cm.imag.allSatisfy { $0 == 0 })
    }

    func testPromoteToComplexMatrix_1x1() {
        // A 1×1 matrix stays a 1×1 complexMatrix after promotion
        // (no dimensional collapse occurs at the promotion site)
        let m = LinAlg.Matrix(rows: 1, cols: 1, data: [42.0])
        let cm = NumericDispatch.promoteToComplexMatrix(m)
        XCTAssertEqual(cm.rows, 1)
        XCTAssertEqual(cm.cols, 1)
        XCTAssertEqual(cm.real, [42.0])
        XCTAssertEqual(cm.imag, [0.0])
    }

    // MARK: - Group I: promoteScalarToComplex helper (14.2)

    func testPromoteScalarToComplex_positiveReal() {
        let z = NumericDispatch.promoteScalarToComplex(3.14)
        XCTAssertEqual(z.re, 3.14, accuracy: 1e-14)
        XCTAssertEqual(z.im, 0.0)
    }

    func testPromoteScalarToComplex_negativeReal() {
        let z = NumericDispatch.promoteScalarToComplex(-7.5)
        XCTAssertEqual(z.re, -7.5, accuracy: 1e-14)
        XCTAssertEqual(z.im, 0.0)
    }

    func testPromoteScalarToComplex_zero() {
        let z = NumericDispatch.promoteScalarToComplex(0.0)
        XCTAssertEqual(z.re, 0.0)
        XCTAssertEqual(z.im, 0.0)
    }

    func testPromoteScalarToComplex_infinity() {
        let z = NumericDispatch.promoteScalarToComplex(Double.infinity)
        XCTAssertTrue(z.re.isInfinite)
        XCTAssertEqual(z.im, 0.0)
    }

    func testPromoteScalarToComplex_nan() {
        let z = NumericDispatch.promoteScalarToComplex(Double.nan)
        XCTAssertTrue(z.re.isNaN)
        XCTAssertEqual(z.im, 0.0)
    }

    // MARK: - Group F: coerce1x1 real helper unit tests (14.7)

    func testCoerce1x1_real_1x1Matrix_collapses() {
        // A 1×1 real matrix collapses to .scalar with the correct value
        let m = LinAlg.Matrix(rows: 1, cols: 1, data: [7.5])
        let result = NumericDispatch.coerce1x1(.matrix(m))
        guard case .scalar(let v) = result else {
            return XCTFail("coerce1x1 should collapse 1×1 matrix to scalar; got \(result)")
        }
        XCTAssertEqual(v, 7.5, accuracy: 1e-14)
    }

    func testCoerce1x1_real_2x2Matrix_unchanged() {
        // A 2×2 matrix must NOT be coerced
        let m = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let input: NumericValue = .matrix(m)
        let result = NumericDispatch.coerce1x1(input)
        guard case .matrix(let out) = result else {
            return XCTFail("coerce1x1 should not coerce a 2×2 matrix; got \(result)")
        }
        XCTAssertEqual(out.rows, 2)
        XCTAssertEqual(out.cols, 2)
    }

    func testCoerce1x1_scalar_unchanged() {
        // .scalar passthrough — coerce1x1 must not touch scalars
        let input: NumericValue = .scalar(42.0)
        let result = NumericDispatch.coerce1x1(input)
        guard case .scalar(let v) = result else {
            return XCTFail("coerce1x1 should leave .scalar unchanged; got \(result)")
        }
        XCTAssertEqual(v, 42.0)
    }

    func testCoerce1x1_complex_unchanged() {
        // .complex passthrough
        let input: NumericValue = .complex(Complex(re: 1, im: 2))
        let result = NumericDispatch.coerce1x1(input)
        guard case .complex(let z) = result else {
            return XCTFail("coerce1x1 should leave .complex unchanged; got \(result)")
        }
        XCTAssertEqual(z.re, 1.0)
        XCTAssertEqual(z.im, 2.0)
    }

    func testCoerce1x1_complexMatrix1x1_isNotTouched() {
        // 1×1 complexMatrix must NOT be coerced by coerce1x1 — that is
        // coerce1x1Complex's domain. Prove it stays .complexMatrix.
        let cm = LinAlg.ComplexMatrix(rows: 1, cols: 1,
                                       real: [5.0], imag: [3.0])
        let input: NumericValue = .complexMatrix(cm)
        let result = NumericDispatch.coerce1x1(input)
        guard case .complexMatrix = result else {
            return XCTFail(
                "coerce1x1 must not touch .complexMatrix; got \(result)")
        }
    }

    // MARK: - Group G: coerce1x1Complex helper unit tests (14.8)

    func testCoerce1x1Complex_1x1CM_collapsesToComplex() {
        // A 1×1 complexMatrix collapses to .complex with correct re and im
        let cm = LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [4.0], imag: [-2.0])
        let result = NumericDispatch.coerce1x1Complex(.complexMatrix(cm))
        guard case .complex(let z) = result else {
            return XCTFail("coerce1x1Complex should collapse 1×1 CM to complex; got \(result)")
        }
        XCTAssertEqual(z.re, 4.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, -2.0, accuracy: 1e-14)
    }

    func testCoerce1x1Complex_2x2CM_unchanged() {
        // A 2×2 complexMatrix must NOT be coerced
        let cm = LinAlg.ComplexMatrix(rows: 2, cols: 2,
                                       real: [1, 2, 3, 4], imag: [0, 0, 0, 0])
        let input: NumericValue = .complexMatrix(cm)
        let result = NumericDispatch.coerce1x1Complex(input)
        guard case .complexMatrix(let out) = result else {
            return XCTFail("coerce1x1Complex should not coerce a 2×2 CM; got \(result)")
        }
        XCTAssertEqual(out.rows, 2)
        XCTAssertEqual(out.cols, 2)
    }

    func testCoerce1x1Complex_scalar_unchanged() {
        let input: NumericValue = .scalar(9.0)
        let result = NumericDispatch.coerce1x1Complex(input)
        guard case .scalar(let v) = result else {
            return XCTFail("coerce1x1Complex should leave .scalar unchanged; got \(result)")
        }
        XCTAssertEqual(v, 9.0)
    }

    func testCoerce1x1Complex_complex_unchanged() {
        let input: NumericValue = .complex(Complex(re: 3, im: 4))
        let result = NumericDispatch.coerce1x1Complex(input)
        guard case .complex(let z) = result else {
            return XCTFail("coerce1x1Complex should leave .complex unchanged; got \(result)")
        }
        XCTAssertEqual(z.re, 3.0)
        XCTAssertEqual(z.im, 4.0)
    }

    func testCoerce1x1Complex_realMatrix1x1_isNotTouched() {
        // 1×1 real matrix must NOT be coerced by coerce1x1Complex — that is
        // coerce1x1's domain. Prove it stays .matrix.
        let m = LinAlg.Matrix(rows: 1, cols: 1, data: [6.0])
        let input: NumericValue = .matrix(m)
        let result = NumericDispatch.coerce1x1Complex(input)
        guard case .matrix = result else {
            return XCTFail(
                "coerce1x1Complex must not touch .matrix; got \(result)")
        }
    }

    // MARK: - Group C: 1×1 → scalar collapse POSITIVE tests (14.16)

    func testMul_rowVecTimesColVec_collapses1x1ToScalar() throws {
        // (1×3) * (3×1) via `*` (matmul) → 1×1 → .scalar
        // [1, 2, 3] · [1, 2, 3]ᵀ = 1+4+9 = 14
        let l = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 3, data: [1, 2, 3]))
        let r = NumericValue.matrix(LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3]))
        let result = try NumericDispatch.applyBinary(.mul, lhs: l, rhs: r)
        guard case .scalar(let v) = result else {
            return XCTFail("(1×3)*(3×1) must collapse to .scalar; got \(result)")
        }
        XCTAssertEqual(v, 14.0, accuracy: 1e-12)
    }

    func testMul_colVecTimesRowVec_produces3x3Matrix() throws {
        // (3×1) * (1×3) → 3×3 — NOT a 1×1, must stay .matrix
        let l = NumericValue.matrix(LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3]))
        let r = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 3, data: [4, 5, 6]))
        let result = try NumericDispatch.applyBinary(.mul, lhs: l, rhs: r)
        guard case .matrix(let m) = result else {
            return XCTFail("(3×1)*(1×3) must stay .matrix (outer product); got \(result)")
        }
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 3)
        // [1,2,3]ᵀ * [4,5,6] = outer product; [0,0] = 1*4 = 4
        XCTAssertEqual(m[0, 0], 4.0, accuracy: 1e-12)
    }

    func testMul_1x2_times_2x1_collapses1x1ToScalar() throws {
        // (1×2) * (2×1) → 1×1 → .scalar
        // [3, 4] · [1, 2] = 3+8 = 11
        let l = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 2, data: [3, 4]))
        let r = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 1, data: [1, 2]))
        let result = try NumericDispatch.applyBinary(.mul, lhs: l, rhs: r)
        guard case .scalar(let v) = result else {
            return XCTFail("(1×2)*(2×1) must collapse to .scalar; got \(result)")
        }
        XCTAssertEqual(v, 11.0, accuracy: 1e-12)
    }

    func testDotProductFunction_vecVec_collapses1x1ToScalar() throws {
        // dotProduct([1,2,3]ᵀ, [4,5,6]ᵀ) = 1*4+2*5+3*6 = 4+10+18 = 32
        // Both are column vectors (cols==1), so LinAlg.dot detects vec·vec.
        let l = NumericValue.matrix(LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3]))
        let r = NumericValue.matrix(LinAlg.Matrix(rows: 3, cols: 1, data: [4, 5, 6]))
        let result = try NumericDispatch.applyFunction("dotProduct", args: [l, r])
        guard case .scalar(let v) = result else {
            return XCTFail("dotProduct(vec, vec) must collapse to .scalar; got \(result)")
        }
        XCTAssertEqual(v, 32.0, accuracy: 1e-12)
    }

    func testDotProductFunction_matVec_staysMatrix() throws {
        // dotProduct(2×3 mat, 3×1 vec) → 2×1 matrix (not 1×1)
        let mat = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 3,
                                                     data: [1, 0, 0, 0, 1, 0]))
        let vec = NumericValue.matrix(LinAlg.Matrix(rows: 3, cols: 1, data: [5, 6, 7]))
        let result = try NumericDispatch.applyFunction("dotProduct", args: [mat, vec])
        guard case .matrix(let m) = result else {
            return XCTFail("dotProduct(2×3, 3×1) must be .matrix; got \(result)")
        }
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 1)
    }

    // MARK: - Group D: 1×1 NO-collapse negative tests (14.17)

    func testAdd_1x1Plus1x1_stays1x1Matrix() throws {
        // 1×1 + 1×1 via add → 1×1 .matrix (no collapse)
        let a = makeMatrix(1, 1, value: 3.0)
        let b = makeMatrix(1, 1, value: 4.0)
        let result = try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)
        guard case .matrix(let m) = result else {
            return XCTFail("add(1×1, 1×1) must stay .matrix; got \(result)")
        }
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m[0, 0], 7.0, accuracy: 1e-12)
    }

    func testSub_1x1Minus1x1_stays1x1Matrix() throws {
        // 1×1 - 1×1 via sub → 1×1 .matrix (no collapse)
        let a = makeMatrix(1, 1, value: 10.0)
        let b = makeMatrix(1, 1, value: 6.0)
        let result = try NumericDispatch.applyBinary(.sub, lhs: a, rhs: b)
        guard case .matrix(let m) = result else {
            return XCTFail("sub(1×1, 1×1) must stay .matrix; got \(result)")
        }
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m[0, 0], 4.0, accuracy: 1e-12)
    }

    func testHadamard_1x1_stays1x1Matrix() throws {
        // hadamard(1×1, 1×1) → 1×1 .matrix (no collapse)
        let a = makeMatrix(1, 1, value: 5.0)
        let b = makeMatrix(1, 1, value: 3.0)
        let result = try NumericDispatch.applyFunction("hadamard", args: [a, b])
        guard case .matrix(let m) = result else {
            return XCTFail("hadamard(1×1, 1×1) must stay .matrix; got \(result)")
        }
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m[0, 0], 15.0, accuracy: 1e-12)
    }

    func testTranspose_1x1_stays1x1Matrix() throws {
        // transpose(1×1) → 1×1 .matrix (no collapse)
        let a = makeMatrix(1, 1, value: 9.0)
        let result = try NumericDispatch.applyFunction("transpose", args: [a])
        guard case .matrix(let m) = result else {
            return XCTFail("transpose(1×1) must stay .matrix; got \(result)")
        }
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m[0, 0], 9.0, accuracy: 1e-12)
    }

    func testNeg_1x1Matrix_stays1x1Matrix() throws {
        // neg(1×1 matrix) → 1×1 .matrix (no collapse)
        let a = makeMatrix(1, 1, value: 7.0)
        let result = try NumericDispatch.applyUnary(.neg, operand: a)
        guard case .matrix(let m) = result else {
            return XCTFail("neg(1×1 matrix) must stay .matrix; got \(result)")
        }
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m[0, 0], -7.0, accuracy: 1e-12)
    }

    func testExplicit1x1MatrixOperandInMul_scalar_staysMatrix() throws {
        // scalar * 1×1 matrix → 1×1 .matrix (no collapse: this is scalar*matrix, not matmul)
        let s: NumericValue = .scalar(2.0)
        let m = makeMatrix(1, 1, value: 5.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: s, rhs: m)
        guard case .matrix(let out) = result else {
            return XCTFail("scalar*(1×1 matrix) must stay .matrix; got \(result)")
        }
        XCTAssertEqual(out.rows, 1)
        XCTAssertEqual(out.cols, 1)
        XCTAssertEqual(out[0, 0], 10.0, accuracy: 1e-12)
    }

    func testInv_1x1_staysMatrix() throws {
        // inv(1×1 matrix) → 1×1 .matrix (Group-B, no collapse)
        let a = makeMatrix(1, 1, value: 4.0)
        let result = try NumericDispatch.applyFunction("inv", args: [a])
        guard case .matrix(let m) = result else {
            return XCTFail("inv(1×1 matrix) must stay .matrix; got \(result)")
        }
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m[0, 0], 0.25, accuracy: 1e-12)
    }

    func testDet_1x1_returnsScalar_byDesign() throws {
        // det(1×1) always returns .scalar — this is LinAlg.det's design, not §4.3a coercion.
        // Document: det is a scalar-valued function, not a matrix operation with a 1×1 result.
        let a = makeMatrix(1, 1, value: 5.0)
        let result = try NumericDispatch.applyFunction("det", args: [a])
        guard case .scalar(let v) = result else {
            return XCTFail("det(1×1) must be .scalar (det is always scalar); got \(result)")
        }
        XCTAssertEqual(v, 5.0, accuracy: 1e-12)
    }

    // MARK: - Group E: cross-axis no-coercion routing (14.18)

    func testScalarMulMatrix_isMatrix_notComplexMatrix() throws {
        // scalar * matrix → .matrix (widening to CM should NOT happen here)
        let s: NumericValue = .scalar(3.0)
        let m = makeMatrix(2, 2, value: 1.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: s, rhs: m)
        guard case .matrix = result else {
            return XCTFail("scalar*matrix must be .matrix (not CM); got \(result)")
        }
    }

    func testMatrixMulScalar_isMatrix_notComplexMatrix() throws {
        // matrix * scalar → .matrix (not CM)
        let m = makeMatrix(2, 2, value: 1.0)
        let s: NumericValue = .scalar(3.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: m, rhs: s)
        guard case .matrix = result else {
            return XCTFail("matrix*scalar must be .matrix (not CM); got \(result)")
        }
    }

    func testComplexTimesComplex_isComplex_notScalar() throws {
        // complex * complex → .complex (not scalar, no narrowing)
        let a: NumericValue = .complex(Complex(re: 2, im: 0))
        let b: NumericValue = .complex(Complex(re: 3, im: 0))
        let result = try NumericDispatch.applyBinary(.mul, lhs: a, rhs: b)
        guard case .complex = result else {
            return XCTFail("complex*complex must stay .complex; got \(result)")
        }
    }

    func testMatrixMul_2x2_isMatrix_notScalar() throws {
        // matrix*matrix (2×2 result) → .matrix (not scalar)
        let a = makeMatrix(2, 2, value: 1.0)
        let b = makeMatrix(2, 2, value: 1.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: a, rhs: b)
        guard case .matrix(let m) = result else {
            return XCTFail("(2×2)*(2×2) must be .matrix; got \(result)")
        }
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
    }

    func testMatrixAdd_isMatrix_notScalar() throws {
        // matrix + matrix → .matrix (never scalar regardless of size)
        let a = makeMatrix(2, 2, value: 1.0)
        let b = makeMatrix(2, 2, value: 2.0)
        let result = try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)
        guard case .matrix = result else {
            return XCTFail("matrix+matrix must be .matrix; got \(result)")
        }
    }

    // MARK: - is1x1 property (§4.3a gate, NumericValue+Accessors)

    func testIs1x1_realMatrix() {
        // The is1x1 predicate is the §4.3a gate used by coerce1x1
        let v = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 1, data: [3.0]))
        XCTAssertTrue(v.is1x1, "1×1 real matrix should be is1x1")
    }

    func testIs1x1_realMatrix_2x1() {
        let v = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 1, data: [1, 2]))
        XCTAssertFalse(v.is1x1, "2×1 matrix should not be is1x1")
    }

    func testIs1x1_complexMatrix() {
        let cm = LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [1.0], imag: [0.0])
        let v = NumericValue.complexMatrix(cm)
        XCTAssertTrue(v.is1x1, "1×1 complexMatrix should be is1x1")
    }

    func testIs1x1_scalar_isFalse() {
        // Scalars are NOT 1×1 by the §4.3a gate definition (they are already scalars)
        XCTAssertFalse(NumericValue.scalar(1.0).is1x1)
    }

    func testIs1x1_complex_isFalse() {
        XCTAssertFalse(NumericValue.complex(Complex(re: 1, im: 0)).is1x1)
    }

    // MARK: - Coercion site isolation: matmul vs add (14.9 negative guard)

    func testCoercionFiresOnlyAtDotSite_not_at_scaledMatrix() throws {
        // Verify: scalar * 1×1 matrix takes the scalar-broadcast path
        // (LinAlg.mul(scalar, m)), NOT the matmul path, so no 1×1 collapse.
        let s: NumericValue = .scalar(5.0)
        let m = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 1, data: [2.0]))
        let result = try NumericDispatch.applyBinary(.mul, lhs: s, rhs: m)
        // Result must be .matrix(1×1), not .scalar(10.0)
        guard case .matrix(let out) = result else {
            return XCTFail(
                "scalar*(1×1 matrix) must NOT collapse: result should be .matrix; got \(result)")
        }
        XCTAssertEqual(out[0, 0], 10.0, accuracy: 1e-12)
    }

    func testCoercionFiresOnlyAtDotSite_colVecColVec_isVecDot() throws {
        // Two col-vectors produce a 1×1 via the vec·vec branch of LinAlg.dot,
        // which IS the coercion site. Verify collapse fires.
        let u = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 1, data: [3, 4]))
        let v = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 1, data: [1, 2]))
        // Using `*` operator path
        let resultMul = try NumericDispatch.applyBinary(.mul, lhs: u, rhs: v)
        // 3*1+4*2 = 3+8 = 11 — BUT wait: (2×1)*(2×1) inner dims 1≠2, shape error!
        // Actually col*col doesn't work via matrix mul — this should throw shapeMismatch.
        // The vec·vec path in LinAlg.dot requires BOTH cols==1 AND rows to match,
        // but shapes are checked by validateShapes first.
        // (2×1) * (2×1): lhs.cols=1, rhs.cols=1 → vec·vec branch, guard lhs.rows==rhs.rows
        // → 2==2 ✓ → succeeds and returns 1×1 from LinAlg.dot
        guard case .scalar(let v11) = resultMul else {
            return XCTFail("col·col vec dot via * must collapse to .scalar; got \(resultMul)")
        }
        XCTAssertEqual(v11, 11.0, accuracy: 1e-12)
    }
}
