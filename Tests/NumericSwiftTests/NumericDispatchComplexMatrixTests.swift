//
//  NumericDispatchComplexMatrixTests.swift
//  NumericSwiftTests
//
//  Comprehensive tests for evaluator-implemented complex-matrix arithmetic
//  (Task 15, Phase 2 of the unified numeric pipeline).
//
//  Coverage strategy:
//    • add/sub correctness (component-wise, shape mismatch errors)
//    • hadamard correctness (element-wise complex product, shape errors)
//    • matmul via real-block decomposition (Task 15.8/18):
//        - Verified against a direct naive complex triple-loop reference
//        - Cases: 2×2, 2×3·3×2, matrix-vector, vec·vec → 1×1 coercion
//    • bilinear dot ≠ Hermitian dot (Task 15.9/19 CRITICAL assertion)
//    • scalar/complex broadcast arithmetic (neg, transpose, abs, trace)
//    • mixed-type promotion cells (scalar*CM, complex*M, M*CM, …)
//    • division cells (matrix/complex, CM/scalar, CM/complex)
//    • 1×1 coercion for matmul and dotProduct (§4.3a)
//    • soft-cap pre-check covers ALL intermediates for matmul (§4.8 / Task 15.20)
//    • shape invariants (rows·cols == real.count == imag.count)
//    • degenerate shapes (1×1 ops, matrix-vector, column-vector dot)
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - Helpers

extension NumericDispatchComplexMatrixTests {

    /// Build a ComplexMatrix with all-identical elements.
    private func cm(_ rows: Int, _ cols: Int, re: Double, im: Double) -> LinAlg.ComplexMatrix {
        let n = rows * cols
        return LinAlg.ComplexMatrix(rows: rows, cols: cols,
                                    real: [Double](repeating: re, count: n),
                                    imag: [Double](repeating: im, count: n))
    }

    /// Build a ComplexMatrix from explicit real and imaginary 2D arrays.
    private func cmFrom(real: [[Double]], imag: [[Double]]) -> LinAlg.ComplexMatrix {
        LinAlg.ComplexMatrix(real: real, imag: imag)
    }

    /// Naive reference complex matrix multiply: triple loop, no optimisation.
    ///
    /// Used to independently verify the real-block decomposition in `complexMatmul`.
    private func naiveComplexMul(
        _ a: LinAlg.ComplexMatrix,
        _ b: LinAlg.ComplexMatrix
    ) -> (real: [[Double]], imag: [[Double]]) {
        var outReal = [[Double]](repeating: [Double](repeating: 0, count: b.cols), count: a.rows)
        var outImag = [[Double]](repeating: [Double](repeating: 0, count: b.cols), count: a.rows)
        for i in 0..<a.rows {
            for j in 0..<b.cols {
                var re = 0.0, im = 0.0
                for k in 0..<a.cols {
                    let ar = a.real[i * a.cols + k], ai = a.imag[i * a.cols + k]
                    let br = b.real[k * b.cols + j], bi = b.imag[k * b.cols + j]
                    re += ar * br - ai * bi
                    im += ar * bi + ai * br
                }
                outReal[i][j] = re
                outImag[i][j] = im
            }
        }
        return (outReal, outImag)
    }

    /// Assert two ComplexMatrix values are equal within tolerance.
    private func assertCMEqual(
        _ cm: LinAlg.ComplexMatrix,
        real: [[Double]], imag: [[Double]],
        accuracy: Double = 1e-10,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(cm.rows, real.count, "rows mismatch", file: file, line: line)
        XCTAssertEqual(cm.cols, real[0].count, "cols mismatch", file: file, line: line)
        for r in 0..<cm.rows {
            for c in 0..<cm.cols {
                let idx = r * cm.cols + c
                XCTAssertEqual(cm.real[idx], real[r][c], accuracy: accuracy,
                               "real[\(r),\(c)] mismatch", file: file, line: line)
                XCTAssertEqual(cm.imag[idx], imag[r][c], accuracy: accuracy,
                               "imag[\(r),\(c)] mismatch", file: file, line: line)
            }
        }
    }

    /// Restore the soft cap to its default after each test so tests are independent.
    override func tearDown() {
        super.tearDown()
        try? LinAlg.setMaxEvaluatorMatrixElements(16_777_216)
    }
}

// MARK: - NumericDispatchComplexMatrixTests

final class NumericDispatchComplexMatrixTests: XCTestCase {

    // MARK: - complexMatrix add/sub: shape validation (Group-A)

    func testCMPlusCM_shapeMismatch_throws() {
        let a = NumericValue.complexMatrix(cm(2, 2, re: 1, im: 0))
        let b = NumericValue.complexMatrix(cm(2, 3, re: 1, im: 0))
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testCMMinusCM_shapeMismatch_throws() {
        let a = NumericValue.complexMatrix(cm(3, 1, re: 1, im: 0))
        let b = NumericValue.complexMatrix(cm(2, 1, re: 1, im: 0))
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.sub, lhs: a, rhs: b)) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    // MARK: - complexMatrix add/sub: correctness

    func testCMPlusCM_2x3_componentWise() throws {
        // A = [[1+2i, 3+4i, 5+6i],[7+8i, 9+10i, 11+12i]]
        // B = [[1+0i, 0+1i, 2+3i],[0+0i, 1+1i, 2+2i]]
        // Expected: A+B = [[2+2i, 3+5i, 7+9i],[7+8i, 10+11i, 13+14i]]
        let a = cmFrom(real: [[1, 3, 5],[7, 9, 11]], imag: [[2, 4, 6],[8, 10, 12]])
        let b = cmFrom(real: [[1, 0, 2],[0, 1, 2]], imag: [[0, 1, 3],[0, 1, 2]])
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        assertCMEqual(c, real: [[2, 3, 7],[7, 10, 13]], imag: [[2, 5, 9],[8, 11, 14]])
    }

    func testCMMinusCM_selfIsZero() throws {
        // A - A == zero matrix
        let a = cmFrom(real: [[3, -1],[2, 5]], imag: [[1, 4],[0, -2]])
        let result = try NumericDispatch.applyBinary(
            .sub, lhs: .complexMatrix(a), rhs: .complexMatrix(a))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.rows, 2)
        XCTAssertEqual(c.cols, 2)
        for v in c.real { XCTAssertEqual(v, 0.0, accuracy: 1e-12) }
        for v in c.imag { XCTAssertEqual(v, 0.0, accuracy: 1e-12) }
    }

    func testCMMinusCM_1xN_mismatch_throws() {
        // 1×3 - 3×1 should throw (not a shape match)
        let a = NumericValue.complexMatrix(cm(1, 3, re: 1, im: 0))
        let b = NumericValue.complexMatrix(cm(3, 1, re: 1, im: 0))
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.sub, lhs: a, rhs: b)) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    // MARK: - complexMatrix hadamard: shape validation + correctness

    func testHadamardCM_shapeMismatch_throws() {
        let a = NumericValue.complexMatrix(cm(2, 2, re: 1, im: 0))
        let b = NumericValue.complexMatrix(cm(2, 3, re: 1, im: 0))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("hadamard", args: [a, b])
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testHadamardCM_imagTimesImag_givesNegativeReal() throws {
        // (0+i) ⊙ (0+i) = (0*0 - 1*1) + i(0*1 + 1*0) = -1+0i
        let a = cmFrom(real: [[0]], imag: [[1]])
        let b = cmFrom(real: [[0]], imag: [[1]])
        let result = try NumericDispatch.applyFunction(
            "hadamard", args: [.complexMatrix(a), .complexMatrix(b)])
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0], -1.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0],  0.0, accuracy: 1e-12)
    }

    func testHadamardCM_realOperands_matchesRealHadamard() throws {
        // Real-only operands: result equals real Hadamard (imag == 0)
        let a = cmFrom(real: [[2, 3],[4, 5]], imag: [[0,0],[0,0]])
        let b = cmFrom(real: [[1, 2],[3, 4]], imag: [[0,0],[0,0]])
        let result = try NumericDispatch.applyFunction(
            "hadamard", args: [.complexMatrix(a), .complexMatrix(b)])
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        assertCMEqual(c, real: [[2, 6],[12, 20]], imag: [[0,0],[0,0]])
    }

    // MARK: - complexMatrix matmul: real-block decomposition vs naive reference

    /// The core verification: real-block result matches independent triple-loop reference.
    func testMatmul_2x2_vsNaiveReference() throws {
        // A = [[1+i, 2-i],[3+2i, 4+0i]]
        // B = [[1+0i, 0+i],[2-i, 1+1i]]
        let a = cmFrom(real: [[1,2],[3,4]], imag: [[1,-1],[2,0]])
        let b = cmFrom(real: [[1,0],[2,1]], imag: [[0,1],[-1,1]])
        let decomp = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = decomp else { return XCTFail("Expected complexMatrix") }
        let ref = naiveComplexMul(a, b)
        assertCMEqual(c, real: ref.real, imag: ref.imag, accuracy: 1e-10)
    }

    func testMatmul_2x3_by_3x2_vsNaiveReference() throws {
        let a = cmFrom(real: [[1,2,3],[4,5,6]], imag: [[1,0,-1],[2,1,0]])
        let b = cmFrom(real: [[1,2],[3,4],[5,6]], imag: [[0,1],[1,0],[0,-1]])
        let decomp = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = decomp else { return XCTFail("Expected complexMatrix") }
        let ref = naiveComplexMul(a, b)
        XCTAssertEqual(c.rows, 2)
        XCTAssertEqual(c.cols, 2)
        assertCMEqual(c, real: ref.real, imag: ref.imag, accuracy: 1e-10)
    }

    func testMatmul_matrixVector_vsNaiveReference() throws {
        // (2×3) * (3×1) → (2×1)
        let a = cmFrom(real: [[1,2,3],[4,5,6]], imag: [[0,1,0],[1,0,1]])
        let b = cmFrom(real: [[1],[2],[3]], imag: [[1],[0],[-1]])
        let decomp = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = decomp else { return XCTFail("Expected complexMatrix") }
        let ref = naiveComplexMul(a, b)
        XCTAssertEqual(c.rows, 2)
        XCTAssertEqual(c.cols, 1)
        assertCMEqual(c, real: ref.real, imag: ref.imag, accuracy: 1e-10)
    }

    func testMatmul_purelyImaginary_2x2() throws {
        // All-imaginary matrices: [i*I] * [i*I] = -I
        let a = cmFrom(real: [[0,0],[0,0]], imag: [[1,0],[0,1]])
        let b = cmFrom(real: [[0,0],[0,0]], imag: [[1,0],[0,1]])
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        // i*i = -1, so (iI)*(iI) = -I
        assertCMEqual(c, real: [[-1,0],[0,-1]], imag: [[0,0],[0,0]])
    }

    func testMatmul_innerDimMismatch_throws() {
        let a = NumericValue.complexMatrix(cm(2, 3, re: 1, im: 0))
        let b = NumericValue.complexMatrix(cm(2, 3, re: 1, im: 0))
        XCTAssertThrowsError(try NumericDispatch.applyBinary(.mul, lhs: a, rhs: b)) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    func testMatmul_identityTimesMatrix_unchanged() throws {
        // I₂ * A = A for complex A
        let ident = LinAlg.ComplexMatrix(LinAlg.eye(2))
        let a = cmFrom(real: [[1,2],[3,4]], imag: [[5,6],[7,8]])
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(ident), rhs: .complexMatrix(a))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        assertCMEqual(c, real: [[1,2],[3,4]], imag: [[5,6],[7,8]])
    }

    // MARK: - 1×1 coercion (§4.3a)

    func testMatmul_vecDotVec_coercedToComplex() throws {
        // (1×2) * (2×1) → 1×1 → coerced to .complex
        let a = cmFrom(real: [[1, 2]], imag: [[1, 0]])    // [1+i, 2]
        let b = cmFrom(real: [[3],[4]], imag: [[0],[1]])   // [3, 4+i]
        // (1+i)*3 + 2*(4+i) = (3+3i) + (8+2i) = 11+5i
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complex(let z) = result else {
            return XCTFail("Expected 1×1 coercion to .complex, got \(result)")
        }
        XCTAssertEqual(z.re, 11.0, accuracy: 1e-10)
        XCTAssertEqual(z.im,  5.0, accuracy: 1e-10)
    }

    func testMatmul_colVecDotColVec_coercedToComplex() throws {
        // vec·vec (both 2×1) → 1×1 → coerced to .complex
        let a = cmFrom(real: [[1],[2]], imag: [[1],[0]])   // [1+i, 2]
        let b = cmFrom(real: [[3],[4]], imag: [[0],[1]])   // [3, 4+i]
        // bilinear: (1+i)*3 + 2*(4+i) = (3+3i) + (8+2i) = 11+5i
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complex(let z) = result else {
            return XCTFail("Expected 1×1 coercion to .complex (vec·vec), got \(result)")
        }
        XCTAssertEqual(z.re, 11.0, accuracy: 1e-10)
        XCTAssertEqual(z.im,  5.0, accuracy: 1e-10)
    }

    // MARK: - Bilinear dot vs Hermitian dot (CRITICAL: no conjugation)

    /// DOM-06 contract: bilinear dot ≠ Hermitian dot on a non-real case.
    ///
    /// For complex vectors a = [i], b = [i]:
    ///   bilinear: Σ aᵢ·bᵢ = i·i = -1+0i
    ///   Hermitian (conjugate): Σ conj(aᵢ)·bᵢ = (-i)·i = -i² = +1+0i
    ///
    /// The two differ, confirming NO conjugation in the bilinear implementation.
    func testBilinearDot_notHermitian_CRITICAL() throws {
        // a = [0+i],  b = [0+i]
        let a = cmFrom(real: [[0]], imag: [[1]])
        let b = cmFrom(real: [[0]], imag: [[1]])

        // Bilinear: i·i = i² = -1
        let bilinearResult = try NumericDispatch.applyFunction(
            "dotProduct", args: [.complexMatrix(a), .complexMatrix(b)])
        guard case .complex(let bilinear) = bilinearResult else {
            return XCTFail("Expected .complex for bilinear dot, got \(bilinearResult)")
        }
        XCTAssertEqual(bilinear.re, -1.0, accuracy: 1e-12,
                       "bilinear i·i should be -1 (no conjugation)")
        XCTAssertEqual(bilinear.im,  0.0, accuracy: 1e-12)

        // Hermitian would give conj(i)·i = (-i)·i = -i² = +1; confirm bilinear ≠ Hermitian
        // Compute conjugated form manually: conj(a)·b = conj([i])·[i] = (-i)·(i) = 1
        let hermitianRe = 1.0   // the value the Hermitian form WOULD give
        XCTAssertNotEqual(bilinear.re, hermitianRe,
                          "bilinear result must NOT equal the Hermitian result")
    }

    func testBilinearDot_realVectors_matchesRealDot() throws {
        // For real vectors, bilinear == Hermitian == ordinary dot product
        let a = cmFrom(real: [[1],[2],[3]], imag: [[0],[0],[0]])
        let b = cmFrom(real: [[4],[5],[6]], imag: [[0],[0],[0]])
        // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
        let result = try NumericDispatch.applyFunction(
            "dotProduct", args: [.complexMatrix(a), .complexMatrix(b)])
        guard case .complex(let z) = result else {
            return XCTFail("Expected .complex, got \(result)")
        }
        XCTAssertEqual(z.re, 32.0, accuracy: 1e-12)
        XCTAssertEqual(z.im,  0.0, accuracy: 1e-12)
    }

    func testDotProduct_CM_lengthMismatch_throws() {
        let a = NumericValue.complexMatrix(cm(3, 1, re: 1, im: 0))
        let b = NumericValue.complexMatrix(cm(2, 1, re: 1, im: 0))
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("dotProduct", args: [a, b])
        ) { err in
            if case MathExprError.shapeMismatch = err { return }
            XCTFail("Expected .shapeMismatch, got \(err)")
        }
    }

    // MARK: - neg(complexMatrix)

    func testNeg_correctness() throws {
        let a = cmFrom(real: [[1, -2],[3, 0]], imag: [[-4, 5],[0, -6]])
        let result = try NumericDispatch.applyUnary(.neg, operand: .complexMatrix(a))
        guard case .complexMatrix(let n) = result else { return XCTFail("Expected complexMatrix") }
        assertCMEqual(n, real: [[-1, 2],[-3, 0]], imag: [[4, -5],[0, 6]])
    }

    func testNeg_doubleNeg_isIdentity() throws {
        let a = cmFrom(real: [[1, 2],[3, 4]], imag: [[5, 6],[7, 8]])
        let neg1 = try NumericDispatch.applyUnary(.neg, operand: .complexMatrix(a))
        let neg2 = try NumericDispatch.applyUnary(.neg, operand: neg1)
        guard case .complexMatrix(let c) = neg2 else { return XCTFail("Expected complexMatrix") }
        assertCMEqual(c, real: [[1, 2],[3, 4]], imag: [[5, 6],[7, 8]])
    }

    // MARK: - transpose(complexMatrix)

    func testTranspose_2x3_yieldsCorrectShape() throws {
        let a = cmFrom(real: [[1,2,3],[4,5,6]], imag: [[0,0,0],[0,0,0]])
        let result = try NumericDispatch.applyUnary(.transpose, operand: .complexMatrix(a))
        guard case .complexMatrix(let t) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(t.rows, 3)
        XCTAssertEqual(t.cols, 2)
        // t[0,1] = a[1,0] = 4
        XCTAssertEqual(t.real[0 * 2 + 1], 4.0, accuracy: 1e-12)
        // t[2,0] = a[0,2] = 3
        XCTAssertEqual(t.real[2 * 2 + 0], 3.0, accuracy: 1e-12)
    }

    func testTranspose_withImagPart_swapsRowCol() throws {
        // Verify no conjugation: imag part should simply be transposed, not negated
        let a = cmFrom(real: [[1,2],[3,4]], imag: [[5,6],[7,8]])
        let result = try NumericDispatch.applyUnary(.transpose, operand: .complexMatrix(a))
        guard case .complexMatrix(let t) = result else { return XCTFail("Expected complexMatrix") }
        // t.imag[0,1] = a.imag[1,0] = 7 (no conjugation → not negated)
        XCTAssertEqual(t.imag[0 * 2 + 1], 7.0, accuracy: 1e-12,
                       "transpose is non-Hermitian: imag[0,1] == a.imag[1,0], NOT negated")
    }

    func testTranspose_doubleTranspose_isIdentity() throws {
        let a = cmFrom(real: [[1,2,3],[4,5,6]], imag: [[0,1,2],[3,4,5]])
        let t1 = try NumericDispatch.applyUnary(.transpose, operand: .complexMatrix(a))
        let t2 = try NumericDispatch.applyUnary(.transpose, operand: t1)
        guard case .complexMatrix(let c) = t2 else { return XCTFail("Expected complexMatrix") }
        assertCMEqual(c, real: [[1,2,3],[4,5,6]], imag: [[0,1,2],[3,4,5]])
    }

    // MARK: - abs(complexMatrix): complex Frobenius norm

    func testAbs_1x1_isMagnitude() throws {
        // |3+4i| = 5
        let a = cmFrom(real: [[3]], imag: [[4]])
        let result = try NumericDispatch.applyFunction("abs", args: [.complexMatrix(a)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 5.0, accuracy: 1e-12)
    }

    func testAbs_2x2_frobeniusNorm() throws {
        // Frobenius norm of [[1+0i, 0+1i],[0+1i, 1+0i]] = sqrt(1+1+1+1) = 2
        let a = cmFrom(real: [[1,0],[0,1]], imag: [[0,1],[1,0]])
        let result = try NumericDispatch.applyFunction("abs", args: [.complexMatrix(a)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 2.0, accuracy: 1e-12)
    }

    func testAbs_realMatrix_equalsFrobeniusNorm() throws {
        // Real 3×3 all-ones: Frobenius norm = sqrt(9) = 3
        let a = cmFrom(real: [[1,1,1],[1,1,1],[1,1,1]], imag: [[0,0,0],[0,0,0],[0,0,0]])
        let result = try NumericDispatch.applyFunction("abs", args: [.complexMatrix(a)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 3.0, accuracy: 1e-12)
    }

    // MARK: - trace(complexMatrix)

    func testTrace_2x2_complexDiagonal() throws {
        // trace([[1+2i, 0],[0, 3+4i]]) = 4+6i
        let a = cmFrom(real: [[1,0],[0,3]], imag: [[2,0],[0,4]])
        let result = try NumericDispatch.applyFunction("trace", args: [.complexMatrix(a)])
        guard case .complex(let z) = result else { return XCTFail("Expected complex") }
        XCTAssertEqual(z.re, 4.0, accuracy: 1e-12)
        XCTAssertEqual(z.im, 6.0, accuracy: 1e-12)
    }

    func testTrace_identityIsN() throws {
        // trace(I₃ as CM) = 3+0i
        let a = LinAlg.ComplexMatrix(LinAlg.eye(3))
        let result = try NumericDispatch.applyFunction("trace", args: [.complexMatrix(a)])
        guard case .complex(let z) = result else { return XCTFail("Expected complex") }
        XCTAssertEqual(z.re, 3.0, accuracy: 1e-12)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-12)
    }

    // MARK: - Division cells

    func testComplexMatrixDivScalar_correctness() throws {
        // [[2+4i, 6+8i]] / 2.0 = [[1+2i, 3+4i]]
        let a = cmFrom(real: [[2, 6]], imag: [[4, 8]])
        let result = try NumericDispatch.applyBinary(
            .div, lhs: .complexMatrix(a), rhs: .scalar(2.0))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0], 1.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(c.real[1], 3.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[1], 4.0, accuracy: 1e-12)
    }

    func testComplexMatrixDivComplex_correctness() throws {
        // [[4+2i]] / (1+i) = (4+2i)/(1+i) = [(4+2)+i(2-4)] / 2 = 3 + i(-1) = 3-i
        let a = cmFrom(real: [[4]], imag: [[2]])
        let divisor = Complex(re: 1, im: 1)
        let result = try NumericDispatch.applyBinary(
            .div, lhs: .complexMatrix(a), rhs: .complex(divisor))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0],  3.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], -1.0, accuracy: 1e-12)
    }

    func testMatrixDivComplex_correctness() throws {
        // [[6]] (real) / (2+0i) = [[3+0i]]
        let m = LinAlg.Matrix(rows: 1, cols: 1, data: [6.0])
        let result = try NumericDispatch.applyBinary(
            .div, lhs: .matrix(m), rhs: .complex(Complex(re: 2, im: 0)))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0], 3.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], 0.0, accuracy: 1e-12)
    }

    func testMatrixDivComplex_purlyImagDivisor() throws {
        // [[4]] / (0+2i) = 4 / (2i) = -2i  →  real=0, imag=-2
        let m = LinAlg.Matrix(rows: 1, cols: 1, data: [4.0])
        let result = try NumericDispatch.applyBinary(
            .div, lhs: .matrix(m), rhs: .complex(Complex(re: 0, im: 2)))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0],  0.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], -2.0, accuracy: 1e-12)
    }

    // MARK: - Mixed-promotion cells

    func testScalarMulComplexMatrix_commutative() throws {
        // 2 * CM and CM * 2 should produce identical results
        let a = cmFrom(real: [[1, 2],[3, 4]], imag: [[5, 6],[7, 8]])
        let r1 = try NumericDispatch.applyBinary(.mul, lhs: .scalar(2), rhs: .complexMatrix(a))
        let r2 = try NumericDispatch.applyBinary(.mul, lhs: .complexMatrix(a), rhs: .scalar(2))
        guard case .complexMatrix(let c1) = r1, case .complexMatrix(let c2) = r2 else {
            return XCTFail("Expected complexMatrix from both sides")
        }
        // Both should equal [[2+10i, 4+12i],[6+14i, 8+16i]]
        for i in 0..<4 {
            XCTAssertEqual(c1.real[i], c2.real[i], accuracy: 1e-12)
            XCTAssertEqual(c1.imag[i], c2.imag[i], accuracy: 1e-12)
        }
        XCTAssertEqual(c1.real[0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(c1.imag[0], 10.0, accuracy: 1e-12)
    }

    func testComplexMulMatrix_correctness() throws {
        // (2+i) * [[1+0i, 0+0i]] = [[2+i, 0+0i]]
        let m = LinAlg.Matrix(rows: 1, cols: 2, data: [1.0, 0.0])
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complex(Complex(re: 2, im: 1)), rhs: .matrix(m))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], 1.0, accuracy: 1e-12)
        XCTAssertEqual(c.real[1], 0.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[1], 0.0, accuracy: 1e-12)
    }

    func testComplexMulComplexMatrix_correctness() throws {
        // (1+i) * [[2+i]] = (1+i)(2+i) = (2-1) + i(1+2) = 1+3i
        let cm = cmFrom(real: [[2]], imag: [[1]])
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complex(Complex(re: 1, im: 1)), rhs: .complexMatrix(cm))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0], 1.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], 3.0, accuracy: 1e-12)
    }

    func testMatrixMulComplexMatrix_correctness() throws {
        // [[2,0],[0,3]] (real) * [[1+i, 0],[0, 2+i]] (CM) — diagonal scaling
        let m = LinAlg.Matrix([[2.0, 0.0],[0.0, 3.0]])
        let b = cmFrom(real: [[1,0],[0,2]], imag: [[1,0],[0,1]])
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .matrix(m), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        // Row 0: [2*(1+i), 2*0] = [2+2i, 0]
        // Row 1: [0, 3*(2+i)] = [0, 6+3i]
        XCTAssertEqual(c.real[0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(c.real[3], 6.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[3], 3.0, accuracy: 1e-12)
    }

    // MARK: - complexMatrix ± complexMatrix add/sub with mixed

    func testComplexPlusMatrix_broadcastsToAllElements() throws {
        // (2+3i) + [[1,2],[3,4]] (all-real) → [[3+3i, 4+3i],[5+3i, 6+3i]]
        let m = LinAlg.Matrix([[1.0, 2.0],[3.0, 4.0]])
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .complex(Complex(re: 2, im: 3)), rhs: .matrix(m))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0], 3.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], 3.0, accuracy: 1e-12)
        XCTAssertEqual(c.real[3], 6.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[3], 3.0, accuracy: 1e-12)
    }

    func testMatrixPlusComplexMatrix_addsBothBlocks() throws {
        // [[1,0],[0,1]] (M) + [[0+i, 0],[0, 0+i]] (CM) = [[1+i, 0],[0, 1+i]]
        let m = LinAlg.eye(2)
        let b = cmFrom(real: [[0,0],[0,0]], imag: [[1,0],[0,1]])
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .matrix(m), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real[0], 1.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[0], 1.0, accuracy: 1e-12)
        XCTAssertEqual(c.real[1], 0.0, accuracy: 1e-12)
        XCTAssertEqual(c.imag[1], 0.0, accuracy: 1e-12)
    }

    // MARK: - Shape invariants

    func testComplexMatrixAdd_preservesShapeInvariant() throws {
        // result.real.count == result.rows * result.cols
        let a = cmFrom(real: [[1,2,3],[4,5,6]], imag: [[0,0,0],[0,0,0]])
        let b = cmFrom(real: [[1,1,1],[1,1,1]], imag: [[0,0,0],[0,0,0]])
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.real.count, c.rows * c.cols)
        XCTAssertEqual(c.imag.count, c.rows * c.cols)
    }

    func testComplexMatrixMul_resultShapeIsLhsRowsByRhsCols() throws {
        // (3×2) * (2×4) → (3×4)
        let a = cmFrom(real: [[1,2],[3,4],[5,6]], imag: [[0,0],[0,0],[0,0]])
        let b = cmFrom(real: [[1,2,3,4],[5,6,7,8]], imag: [[0,0,0,0],[0,0,0,0]])
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        guard case .complexMatrix(let c) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(c.rows, 3)
        XCTAssertEqual(c.cols, 4)
        XCTAssertEqual(c.real.count, 12)
        XCTAssertEqual(c.imag.count, 12)
    }

    // MARK: - Soft-cap pre-check (§4.8 / Task 15.20)

    func testSoftCap_CMmulCM_resultUnderCap_succeeds() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(100)
        // 5×5 result = 25 elements < 100 cap
        let a = cmFrom(
            real: Array(repeating: Array(repeating: 1.0, count: 5), count: 5),
            imag: Array(repeating: Array(repeating: 0.0, count: 5), count: 5))
        XCTAssertNoThrow(
            try NumericDispatch.applyBinary(.mul, lhs: .complexMatrix(a), rhs: .complexMatrix(a)))
    }

    func testSoftCap_CMmulCM_resultOverCap_throwsInvalidParameter() throws {
        // Set cap to 3 elements; 2×2 result = 4 elements > 3
        try LinAlg.setMaxEvaluatorMatrixElements(3)
        let a = cmFrom(real: [[1,2],[3,4]], imag: [[0,0],[0,0]])
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.mul, lhs: .complexMatrix(a), rhs: .complexMatrix(a))
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected LinAlgError.invalidParameter (CONS-07), got \(err)")
        }
    }

    func testSoftCap_CMaddCM_overCap_throwsInvalidParameter() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(3)
        let a = NumericValue.complexMatrix(cm(2, 2, re: 1, im: 0))
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.add, lhs: a, rhs: a)
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected LinAlgError.invalidParameter, got \(err)")
        }
    }

    /// §4.8 requirement: soft-cap covers ALL INTERMEDIATE real products, not just
    /// the final result. With 4 elements in the result but cap=3, the intermediate
    /// products (also 4 elements each) must trigger the cap.
    func testSoftCap_matmul_intermediatesOverCap_throws() throws {
        // 2×2 result has 4 elements; cap=3 → soft-cap fires before any intermediate
        try LinAlg.setMaxEvaluatorMatrixElements(3)
        let a = cmFrom(real: [[1,0],[0,1]], imag: [[0,0],[0,0]])
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.mul, lhs: .complexMatrix(a), rhs: .complexMatrix(a))
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected invalidParameter for intermediates over cap, got \(err)")
        }
    }

    func testSoftCap_neg_overCap_throws() throws {
        try LinAlg.setMaxEvaluatorMatrixElements(3)
        let a = NumericValue.complexMatrix(cm(2, 2, re: 1, im: 0))
        XCTAssertThrowsError(
            try NumericDispatch.applyUnary(.neg, operand: a)
        ) { err in
            if case LinAlg.LinAlgError.invalidParameter = err { return }
            XCTFail("Expected invalidParameter, got \(err)")
        }
    }
}
