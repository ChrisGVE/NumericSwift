//
//  NumericDispatchTests.swift
//  NumericSwiftTests
//
//  Tests for NumericDispatch — the central routing surface of the unified
//  numeric pipeline (Sources/NumericSwift/NumericDispatch.swift).
//
//  Coverage strategy:
//    • Known-working paths: check that result KIND is correct and value is sane.
//    • Stub paths (EVAL seams): verify .unsupportedNode is thrown with the
//      expected prefix "not yet implemented:".
//    • Error paths: verify correct MathExprError type and message fragment.
//    • Group-A pre-validation: shape mismatch → LinAlgError thrown before LinAlg
//      internal precondition fires.
//    • Group-B propagation: non-square input → LinAlgError.notSquare propagated.
//    • 1×1 coercion: matrix*matrix producing 1×1 result collapses to .scalar.
//    • Unknown function: → .unknownFunction.
//    • Size cap: checkSoftCap fires for oversized spec.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - Helpers

extension NumericDispatchTests {

    private func makeMatrix(_ rows: Int, _ cols: Int, value: Double = 1.0) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: rows, cols: cols,
                              data: [Double](repeating: value, count: rows * cols)))
    }

    private func makeCM(_ rows: Int, _ cols: Int, re: Double = 1.0, im: Double = 0.0) -> NumericValue {
        let n = rows * cols
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: rows, cols: cols,
            real: [Double](repeating: re, count: n),
            imag: [Double](repeating: im, count: n)))
    }

    private func assertStub(_ result: Result<NumericValue, Error>, taskTag: String,
                            file: StaticString = #filePath, line: UInt = #line) {
        guard case .failure(let err) = result else {
            XCTFail("Expected .unsupportedNode stub to throw, but got success", file: file, line: line)
            return
        }
        guard case MathExprError.unsupportedNode(let msg) = err else {
            XCTFail("Expected .unsupportedNode, got \(err)", file: file, line: line)
            return
        }
        XCTAssertTrue(msg.contains("not yet implemented"),
                      "Stub message should start with 'not yet implemented:', got: \(msg)",
                      file: file, line: line)
        XCTAssertTrue(msg.contains(taskTag),
                      "Stub message should reference \(taskTag), got: \(msg)",
                      file: file, line: line)
    }

    private func run(_ block: () throws -> NumericValue) -> Result<NumericValue, Error> {
        do { return .success(try block()) }
        catch { return .failure(error) }
    }
}

// MARK: - NumericDispatchTests

final class NumericDispatchTests: XCTestCase {

    // MARK: - applyBinary: add / sub — scalar paths

    func testAddScalarScalar() throws {
        let result = try NumericDispatch.applyBinary(.add, lhs: .scalar(3), rhs: .scalar(4))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 7.0)
    }

    func testSubScalarScalar() throws {
        let result = try NumericDispatch.applyBinary(.sub, lhs: .scalar(10), rhs: .scalar(4))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 6.0)
    }

    func testAddScalarComplex() throws {
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .scalar(1.0), rhs: .complex(Complex(re: 0, im: 1)))
        guard case .complex(let z) = result else { return XCTFail("Expected complex") }
        XCTAssertEqual(z.re, 1.0)
        XCTAssertEqual(z.im, 1.0)
    }

    func testAddComplexComplex() throws {
        let a = Complex(re: 1, im: 2), b = Complex(re: 3, im: 4)
        let result = try NumericDispatch.applyBinary(.add, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = result else { return XCTFail("Expected complex") }
        XCTAssertEqual(z.re, 4.0)
        XCTAssertEqual(z.im, 6.0)
    }

    // MARK: - applyBinary: add — matrix paths

    func testAddMatrixMatrix_sameShape() throws {
        let result = try NumericDispatch.applyBinary(
            .add, lhs: makeMatrix(2, 2, value: 1.0), rhs: makeMatrix(2, 2, value: 2.0))
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
        XCTAssertEqual(m[0, 0], 3.0)
    }

    func testAddMatrixMatrix_shapeMismatch_throws() {
        let result = run {
            try NumericDispatch.applyBinary(
                .add, lhs: self.makeMatrix(2, 3), rhs: self.makeMatrix(3, 2))
        }
        guard case .failure(let err) = result else {
            return XCTFail("Expected failure for shape mismatch")
        }
        // Must be MathExprError.shapeMismatch (Group-A pre-validation throws before LinAlg precondition)
        if case MathExprError.shapeMismatch = err { return }
        XCTFail("Expected .shapeMismatch, got \(err)")
    }

    func testSubMatrixMatrix() throws {
        let lhsData = [Double](repeating: 5.0, count: 4)
        let rhsData = [Double](repeating: 2.0, count: 4)
        let lhs = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 2, data: lhsData))
        let rhs = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 2, data: rhsData))
        let result = try NumericDispatch.applyBinary(.sub, lhs: lhs, rhs: rhs)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 3.0)
    }

    // MARK: - applyBinary: add — EVAL implemented paths

    func testAddScalarMatrix() throws {
        // scalar + M: broadcasts scalar to every element
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .scalar(5), rhs: makeMatrix(2, 2, value: 1.0))
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 6.0, accuracy: 1e-12)
        XCTAssertEqual(m[1, 1], 6.0, accuracy: 1e-12)
    }

    func testAddMatrixScalar() throws {
        // M + scalar: commutative
        let result = try NumericDispatch.applyBinary(
            .add, lhs: makeMatrix(2, 2, value: 3.0), rhs: .scalar(2))
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 5.0, accuracy: 1e-12)
    }

    func testAddComplexPlusMatrix() throws {
        // complex + matrix: promotes M → CM, adds real part only
        let result = try NumericDispatch.applyBinary(
            .add, lhs: .complex(Complex(re: 1, im: 2)), rhs: makeMatrix(2, 2, value: 3.0))
        guard case .complexMatrix(let cm) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 2)
        // Each element should be (3+1) + 2i = 4+2i
        XCTAssertEqual(cm.real[0], 4.0, accuracy: 1e-12)
        XCTAssertEqual(cm.imag[0], 2.0, accuracy: 1e-12)
    }

    func testAddComplexMatrixPlusComplexMatrix() throws {
        // CM + CM: element-wise on real+imag blocks
        let a = makeCM(2, 2, re: 1.0, im: 2.0)
        let b = makeCM(2, 2, re: 3.0, im: 4.0)
        let result = try NumericDispatch.applyBinary(.add, lhs: a, rhs: b)
        guard case .complexMatrix(let cm) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(cm.real[0], 4.0, accuracy: 1e-12)
        XCTAssertEqual(cm.imag[0], 6.0, accuracy: 1e-12)
    }

    // MARK: - applyBinary: mul — scalar paths

    func testMulScalarScalar() throws {
        let result = try NumericDispatch.applyBinary(.mul, lhs: .scalar(3), rhs: .scalar(4))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 12.0)
    }

    func testMulScalarMatrix() throws {
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .scalar(2.0), rhs: makeMatrix(2, 2, value: 3.0))
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 6.0)
    }

    func testMulMatrixScalar() throws {
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: makeMatrix(2, 2, value: 3.0), rhs: .scalar(2.0))
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 6.0)
    }

    // MARK: - applyBinary: mul — matrix*matrix and 1×1 coercion

    func testMulMatrixMatrix_innerDimMismatch_throws() {
        let result = run {
            // (2×3) * (2×3) — inner dims 3≠2 mismatch
            try NumericDispatch.applyBinary(
                .mul, lhs: self.makeMatrix(2, 3), rhs: self.makeMatrix(2, 3))
        }
        guard case .failure(let err) = result else {
            return XCTFail("Expected failure for inner-dim mismatch")
        }
        if case MathExprError.shapeMismatch = err { return }
        XCTFail("Expected .shapeMismatch, got \(err)")
    }

    func testMulMatrixMatrix_produces2x2() throws {
        // (2×3) * (3×2) → 2×2
        let l = makeMatrix(2, 3, value: 1.0)
        let r = makeMatrix(3, 2, value: 1.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: l, rhs: r)
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
    }

    func testMulMatrixMatrix_1x1_coercedToScalar() throws {
        // (1×3) * (3×1) → 1×1 matrix → coerced to .scalar
        // Each element is 1.0, so dot product = 3.0
        let l = makeMatrix(1, 3, value: 1.0)
        let r = makeMatrix(3, 1, value: 1.0)
        let result = try NumericDispatch.applyBinary(.mul, lhs: l, rhs: r)
        guard case .scalar(let v) = result else {
            return XCTFail("Expected 1×1 coercion to .scalar, got \(result)")
        }
        XCTAssertEqual(v, 3.0, accuracy: 1e-12)
    }

    func testMulScalarComplexMatrix() throws {
        // scalar * CM: broadcasts scalar over both real and imag blocks
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .scalar(2.0), rhs: makeCM(2, 2, re: 3.0, im: 1.0))
        guard case .complexMatrix(let cm) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(cm.real[0], 6.0, accuracy: 1e-12)
        XCTAssertEqual(cm.imag[0], 2.0, accuracy: 1e-12)
    }

    func testMulComplexMatrix_producesComplexMatrix() throws {
        // I₂ * I₂ as CM should yield I₂ as CM (1+0i per diagonal)
        let ident = LinAlg.ComplexMatrix(LinAlg.eye(2))
        let result = try NumericDispatch.applyBinary(
            .mul, lhs: .complexMatrix(ident), rhs: .complexMatrix(ident))
        guard case .complexMatrix(let cm) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 2)
        XCTAssertEqual(cm.real[0], 1.0, accuracy: 1e-12)  // (0,0)
        XCTAssertEqual(cm.imag[0], 0.0, accuracy: 1e-12)
        XCTAssertEqual(cm.real[1], 0.0, accuracy: 1e-12)  // (0,1)
        XCTAssertEqual(cm.real[3], 1.0, accuracy: 1e-12)  // (1,1)
    }

    // MARK: - applyBinary: div

    func testDivScalarScalar() throws {
        let result = try NumericDispatch.applyBinary(.div, lhs: .scalar(9), rhs: .scalar(3))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 3.0)
    }

    func testDivScalarZero_throws() {
        let result = run {
            try NumericDispatch.applyBinary(.div, lhs: .scalar(1), rhs: .scalar(0))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.divisionByZero = err { return }
        XCTFail("Expected .divisionByZero, got \(err)")
    }

    func testDivMatrixScalar() throws {
        let result = try NumericDispatch.applyBinary(
            .div, lhs: makeMatrix(2, 2, value: 4.0), rhs: .scalar(2.0))
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 2.0)
    }

    func testDivMatrixZeroScalar_throws() {
        let result = run {
            try NumericDispatch.applyBinary(
                .div, lhs: self.makeMatrix(2, 2, value: 1.0), rhs: .scalar(0.0))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.divisionByZero = err { return }
        XCTFail("Expected .divisionByZero, got \(err)")
    }

    func testDivScalarMatrix_throws() {
        let result = run {
            try NumericDispatch.applyBinary(
                .div, lhs: .scalar(1), rhs: self.makeMatrix(2, 2))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    func testDivMatrixMatrix_throws() {
        let result = run {
            try NumericDispatch.applyBinary(
                .div, lhs: self.makeMatrix(2, 2), rhs: self.makeMatrix(2, 2))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    func testDivMatrixComplex() throws {
        // M / (1+0i): same as M / 1.0, real stays real, imag = 0
        let result = try NumericDispatch.applyBinary(
            .div, lhs: makeMatrix(2, 2, value: 4.0), rhs: .complex(Complex(re: 2, im: 0)))
        guard case .complexMatrix(let cm) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(cm.real[0], 2.0, accuracy: 1e-12)
        XCTAssertEqual(cm.imag[0], 0.0, accuracy: 1e-12)
    }

    // MARK: - applyBinary: pow

    func testPowScalarScalar() throws {
        let result = try NumericDispatch.applyBinary(.pow, lhs: .scalar(2), rhs: .scalar(3))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 8.0, accuracy: 1e-12)
    }

    func testPowScalarComplex() throws {
        let result = try NumericDispatch.applyBinary(
            .pow, lhs: .scalar(1.0), rhs: .complex(Complex(re: 0, im: 1)))
        guard case .complex = result else { return XCTFail("Expected complex") }
    }

    func testPowMatrixScalar_implemented() throws {
        // [[1,1],[1,1]]^2 = [[2,2],[2,2]]  (exponentiation-by-squaring)
        let result = try NumericDispatch.applyBinary(
            .pow, lhs: self.makeMatrix(2, 2), rhs: .scalar(2.0))
        guard case .matrix(let m) = result else {
            return XCTFail("Expected .matrix result for matrix^2")
        }
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
        for i in 0..<4 {
            XCTAssertEqual(m.data[i], 2.0, accuracy: 1e-12,
                "[[1,1],[1,1]]^2 should be all-2 matrix, data[\(i)] mismatch")
        }
    }

    func testPowScalarMatrix_throws() {
        let result = run {
            try NumericDispatch.applyBinary(
                .pow, lhs: .scalar(2), rhs: self.makeMatrix(2, 2))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    func testPowMatrixMatrix_throws() {
        let result = run {
            try NumericDispatch.applyBinary(
                .pow, lhs: self.makeMatrix(2, 2), rhs: self.makeMatrix(2, 2))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    // MARK: - applyBinary: mod

    func testModScalarScalar() throws {
        let result = try NumericDispatch.applyBinary(.mod, lhs: .scalar(7), rhs: .scalar(3))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 1.0, accuracy: 1e-12)
    }

    func testModNonScalar_throws() {
        let result = run {
            try NumericDispatch.applyBinary(
                .mod, lhs: self.makeMatrix(2, 2), rhs: .scalar(2))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.unsupportedNode(let msg) = err {
            XCTAssertTrue(msg.contains("modulo requires scalar operands"))
            return
        }
        XCTFail("Expected .unsupportedNode for modulo, got \(err)")
    }

    // MARK: - applyBinary: plusMinus / minusPlus

    func testPlusMinusThrows() {
        let result = run {
            try NumericDispatch.applyBinary(
                .plusMinus, lhs: .scalar(1), rhs: .scalar(2))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.unsupportedNode(let msg) = err {
            XCTAssertTrue(msg.contains("display-only"))
            return
        }
        XCTFail("Expected .unsupportedNode, got \(err)")
    }

    func testMinusPlusThrows() {
        let result = run {
            try NumericDispatch.applyBinary(
                .minusPlus, lhs: .scalar(1), rhs: .scalar(2))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.unsupportedNode = err { return }
        XCTFail("Expected .unsupportedNode, got \(err)")
    }

    // MARK: - applyUnary: neg

    func testNegScalar() throws {
        let result = try NumericDispatch.applyUnary(.neg, operand: .scalar(3.5))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, -3.5)
    }

    func testNegComplex() throws {
        let result = try NumericDispatch.applyUnary(
            .neg, operand: .complex(Complex(re: 1, im: -2)))
        guard case .complex(let z) = result else { return XCTFail("Expected complex") }
        XCTAssertEqual(z.re, -1.0)
        XCTAssertEqual(z.im, 2.0)
    }

    func testNegMatrix() throws {
        let result = try NumericDispatch.applyUnary(.neg, operand: makeMatrix(2, 2, value: 5.0))
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], -5.0)
    }

    func testNegComplexMatrix() throws {
        // neg(CM): element-wise negate both real and imag
        let cm = LinAlg.ComplexMatrix(rows: 2, cols: 2,
                                      real: [1.0, -2.0, 3.0, -4.0],
                                      imag: [5.0, -6.0, 7.0, -8.0])
        let result = try NumericDispatch.applyUnary(.neg, operand: .complexMatrix(cm))
        guard case .complexMatrix(let neg) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(neg.real[0], -1.0, accuracy: 1e-12)
        XCTAssertEqual(neg.imag[0], -5.0, accuracy: 1e-12)
        XCTAssertEqual(neg.real[1],  2.0, accuracy: 1e-12)
        XCTAssertEqual(neg.imag[3],  8.0, accuracy: 1e-12)
    }

    // MARK: - applyUnary: pos (identity)

    func testPosScalar() throws {
        let result = try NumericDispatch.applyUnary(.pos, operand: .scalar(7.0))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 7.0)
    }

    func testPosMatrix() throws {
        let m = makeMatrix(2, 2, value: 3.0)
        let result = try NumericDispatch.applyUnary(.pos, operand: m)
        guard case .matrix = result else { return XCTFail("Expected matrix") }
    }

    // MARK: - applyUnary: factorial

    func testFactorialScalar() throws {
        let result = try NumericDispatch.applyUnary(.factorial, operand: .scalar(5.0))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 120.0, accuracy: 1e-10)
    }

    func testFactorialNegative_throws() {
        let result = run {
            try NumericDispatch.applyUnary(.factorial, operand: .scalar(-1.0))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    func testFactorialMatrix_throws() {
        let result = run {
            try NumericDispatch.applyUnary(.factorial, operand: self.makeMatrix(2, 2))
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    // MARK: - applyUnary: transpose

    func testTransposeScalar_identity() throws {
        let result = try NumericDispatch.applyUnary(.transpose, operand: .scalar(9.0))
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 9.0)
    }

    func testTransposeMatrix() throws {
        let m = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let result = try NumericDispatch.applyUnary(.transpose, operand: .matrix(m))
        guard case .matrix(let t) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(t.rows, 2)
        XCTAssertEqual(t.cols, 2)
        // After transpose: [0,1] element should be what was [1,0]
        XCTAssertEqual(t[0, 1], 3.0)
        XCTAssertEqual(t[1, 0], 2.0)
    }

    func testTransposeComplexMatrix() throws {
        // Plain (non-Hermitian) transpose: shape (2×3) → (3×2)
        let cm = LinAlg.ComplexMatrix(rows: 2, cols: 3,
                                      real: [1,2,3, 4,5,6],
                                      imag: [0,0,0, 0,0,0])
        let result = try NumericDispatch.applyUnary(.transpose, operand: .complexMatrix(cm))
        guard case .complexMatrix(let t) = result else { return XCTFail("Expected complexMatrix") }
        XCTAssertEqual(t.rows, 3)
        XCTAssertEqual(t.cols, 2)
        // Element (0,1) of transpose = element (1,0) of original = 4
        XCTAssertEqual(t.real[0 * 2 + 1], 4.0, accuracy: 1e-12)
        // Element (1,0) of transpose = element (0,1) of original = 2
        XCTAssertEqual(t.real[1 * 2 + 0], 2.0, accuracy: 1e-12)
    }

    // MARK: - applyFunction: trig (scalar)

    func testSinScalar() throws {
        let result = try NumericDispatch.applyFunction("sin", args: [.scalar(0.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 0.0, accuracy: 1e-15)
    }

    func testCosScalar() throws {
        let result = try NumericDispatch.applyFunction("cos", args: [.scalar(0.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 1.0, accuracy: 1e-15)
    }

    func testTanScalar() throws {
        let result = try NumericDispatch.applyFunction("tan", args: [.scalar(0.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 0.0, accuracy: 1e-15)
    }

    func testAsinScalar() throws {
        let result = try NumericDispatch.applyFunction("asin", args: [.scalar(1.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, .pi / 2, accuracy: 1e-12)
    }

    func testSinhScalar() throws {
        let result = try NumericDispatch.applyFunction("sinh", args: [.scalar(0.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 0.0, accuracy: 1e-15)
    }

    func testAsinhScalar() throws {
        let result = try NumericDispatch.applyFunction("asinh", args: [.scalar(0.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 0.0, accuracy: 1e-15)
    }

    // MARK: - applyFunction: trig (complex)

    func testSinComplex() throws {
        let result = try NumericDispatch.applyFunction(
            "sin", args: [.complex(Complex(re: 0, im: 0))])
        guard case .complex(let z) = result else { return XCTFail("Expected complex") }
        XCTAssertEqual(z.re, 0.0, accuracy: 1e-15)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-15)
    }

    // MARK: - applyFunction: trig (matrix) throws

    func testSinMatrix_throws() {
        let result = run {
            try NumericDispatch.applyFunction("sin", args: [self.makeMatrix(2, 2)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments(let msg) = err {
            XCTAssertTrue(msg.contains("matrices"))
            return
        }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    // MARK: - applyFunction: exp / log / sqrt — scalar

    func testExpScalar() throws {
        let result = try NumericDispatch.applyFunction("exp", args: [.scalar(0.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 1.0, accuracy: 1e-15)
    }

    func testLogScalar() throws {
        let result = try NumericDispatch.applyFunction("log", args: [.scalar(1.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 0.0, accuracy: 1e-15)
    }

    func testSqrtScalar() throws {
        let result = try NumericDispatch.applyFunction("sqrt", args: [.scalar(4.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 2.0, accuracy: 1e-12)
    }

    func testExpComplex() throws {
        let result = try NumericDispatch.applyFunction(
            "exp", args: [.complex(Complex(re: 0, im: 0))])
        guard case .complex(let z) = result else { return XCTFail("Expected complex") }
        XCTAssertEqual(z.re, 1.0, accuracy: 1e-12)
    }

    func testExpMatrix_returnsMatrix() throws {
        // exp of 0-matrix should be identity
        let result = try NumericDispatch.applyFunction(
            "exp", args: [makeMatrix(2, 2, value: 0.0)])
        guard case .matrix = result else { return XCTFail("Expected matrix from expm") }
    }

    func testExpComplexMatrix_throws() {
        let result = run {
            try NumericDispatch.applyFunction("exp", args: [self.makeCM(2, 2)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    // MARK: - applyFunction: exp/log — Group-B notSquare propagation

    func testLogMatrix_nonSquare_propagatesNotSquare() {
        let result = run {
            // 2×3 matrix: logm throws .notSquare
            try NumericDispatch.applyFunction("log", args: [self.makeMatrix(2, 3)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case LinAlg.LinAlgError.notSquare = err { return }
        XCTFail("Expected .notSquare propagated from logm, got \(err)")
    }

    func testExpMatrix_nonSquare_propagatesNotSquare() {
        let result = run {
            try NumericDispatch.applyFunction("exp", args: [self.makeMatrix(2, 3)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case LinAlg.LinAlgError.notSquare = err { return }
        XCTFail("Expected .notSquare propagated from expm, got \(err)")
    }

    // MARK: - applyFunction: abs

    func testAbsScalar() throws {
        let result = try NumericDispatch.applyFunction("abs", args: [.scalar(-3.5)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 3.5)
    }

    func testAbsComplex() throws {
        let result = try NumericDispatch.applyFunction(
            "abs", args: [.complex(Complex(re: 3, im: 4))])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar (modulus)") }
        XCTAssertEqual(v, 5.0, accuracy: 1e-12)
    }

    func testAbsMatrix_returnsFrobeniusNorm() throws {
        // [[3,4],[0,0]] → Frobenius = sqrt(9+16) = 5
        let m = LinAlg.Matrix([[3.0, 4.0], [0.0, 0.0]])
        let result = try NumericDispatch.applyFunction("abs", args: [.matrix(m)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar (Frobenius)") }
        XCTAssertEqual(v, 5.0, accuracy: 1e-12)
    }

    func testAbsComplexMatrix_frobeniusNorm() throws {
        // abs(CM) = complex Frobenius norm = sqrt(Σ|z_ij|²)
        // Single element: |3+4i| = 5  → Frobenius norm of 1×1 = 5
        let cm = LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [3.0], imag: [4.0])
        let result = try NumericDispatch.applyFunction("abs", args: [.complexMatrix(cm)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar Frobenius norm") }
        XCTAssertEqual(v, 5.0, accuracy: 1e-12)
    }

    // MARK: - applyFunction: inv

    func testInvMatrix_square() throws {
        // Identity matrix — inv should be identity
        let ident = LinAlg.eye(2)
        let result = try NumericDispatch.applyFunction("inv", args: [.matrix(ident)])
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(m[1, 1], 1.0, accuracy: 1e-10)
    }

    func testInvMatrix_nonSquare_propagatesNotSquare() {
        let result = run {
            try NumericDispatch.applyFunction("inv", args: [self.makeMatrix(2, 3)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case LinAlg.LinAlgError.notSquare = err { return }
        XCTFail("Expected .notSquare, got \(err)")
    }

    func testInvScalar_throws() {
        let result = run {
            try NumericDispatch.applyFunction("inv", args: [.scalar(2.0)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    // MARK: - applyFunction: det

    func testDetMatrix_2x2() throws {
        // [[1,2],[3,4]] → det = 1*4 - 2*3 = -2
        let m = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let result = try NumericDispatch.applyFunction("det", args: [.matrix(m)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, -2.0, accuracy: 1e-10)
    }

    func testDetMatrix_nonSquare_propagatesNotSquare() {
        let result = run {
            try NumericDispatch.applyFunction("det", args: [self.makeMatrix(2, 3)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case LinAlg.LinAlgError.notSquare = err { return }
        XCTFail("Expected .notSquare, got \(err)")
    }

    func testDetScalar_throws() {
        let result = run {
            try NumericDispatch.applyFunction("det", args: [.scalar(5.0)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    func testDetComplexMatrix_identityGivesComplex1() throws {
        // det of 2×2 identity = 1+0i
        let cm = LinAlg.ComplexMatrix(LinAlg.eye(2))
        let result = try NumericDispatch.applyFunction("det", args: [.complexMatrix(cm)])
        guard case .complex(let z) = result else { return XCTFail("Expected complex") }
        XCTAssertEqual(z.re, 1.0, accuracy: 1e-10)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-10)
    }

    // MARK: - applyFunction: trace

    func testTraceMatrix() throws {
        // [[1,2],[3,4]] → trace = 5
        let m = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let result = try NumericDispatch.applyFunction("trace", args: [.matrix(m)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 5.0, accuracy: 1e-12)
    }

    func testTraceMatrix_nonSquare_propagatesNotSquare() {
        let result = run {
            try NumericDispatch.applyFunction("trace", args: [self.makeMatrix(2, 3)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case LinAlg.LinAlgError.notSquare = err { return }
        XCTFail("Expected .notSquare, got \(err)")
    }

    func testTraceScalar_throws() {
        let result = run {
            try NumericDispatch.applyFunction("trace", args: [.scalar(7.0)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    func testTraceComplexMatrix() throws {
        // trace([[1+2i, 0],[0, 3+4i]]) = (1+2i)+(3+4i) = 4+6i
        let cm = LinAlg.ComplexMatrix(rows: 2, cols: 2,
                                      real: [1, 0, 0, 3],
                                      imag: [2, 0, 0, 4])
        let result = try NumericDispatch.applyFunction("trace", args: [.complexMatrix(cm)])
        guard case .complex(let z) = result else { return XCTFail("Expected complex trace") }
        XCTAssertEqual(z.re, 4.0, accuracy: 1e-12)
        XCTAssertEqual(z.im, 6.0, accuracy: 1e-12)
    }

    // MARK: - applyFunction: transpose (function form)

    func testTransposeFunctionScalar_identity() throws {
        let result = try NumericDispatch.applyFunction("transpose", args: [.scalar(4.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 4.0)
    }

    func testTransposeFunctionMatrix() throws {
        let m = LinAlg.Matrix([[1.0, 2.0, 3.0]])
        let result = try NumericDispatch.applyFunction("transpose", args: [.matrix(m)])
        guard case .matrix(let t) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(t.rows, 3)
        XCTAssertEqual(t.cols, 1)
    }

    // MARK: - applyFunction: dotProduct

    func testDotProduct_matrix_collapses1x1() throws {
        // (1×3) · (3×1) = scalar 3.0
        let l = makeMatrix(1, 3, value: 1.0)
        let r = makeMatrix(3, 1, value: 1.0)
        let result = try NumericDispatch.applyFunction("dotProduct", args: [l, r])
        guard case .scalar(let v) = result else {
            return XCTFail("Expected 1×1 coercion to scalar, got \(result)")
        }
        XCTAssertEqual(v, 3.0, accuracy: 1e-12)
    }

    func testDotProduct_innerDimMismatch_throws() {
        let result = run {
            try NumericDispatch.applyFunction(
                "dotProduct", args: [self.makeMatrix(2, 3), self.makeMatrix(2, 3)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.shapeMismatch = err { return }
        XCTFail("Expected .shapeMismatch, got \(err)")
    }

    func testDotProduct_scalarArgs_throws() {
        let result = run {
            try NumericDispatch.applyFunction(
                "dotProduct", args: [.scalar(1), .scalar(2)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    func testDotProduct_complexMatrix_bilinear() throws {
        // dotProduct(CM, CM): CM here are column vectors (2×1)
        // [1+i, 2+0i] · [3+0i, 4+i] = (1+i)(3+0i) + (2)(4+i)
        //                             = (3+3i) + (8+2i) = 11+5i
        let a = LinAlg.ComplexMatrix(rows: 2, cols: 1, real: [1, 2], imag: [1, 0])
        let b = LinAlg.ComplexMatrix(rows: 2, cols: 1, real: [3, 4], imag: [0, 1])
        let result = try NumericDispatch.applyFunction("dotProduct",
                                                       args: [.complexMatrix(a),
                                                              .complexMatrix(b)])
        guard case .complex(let z) = result else {
            return XCTFail("Expected complex (1×1 coercion), got \(result)")
        }
        XCTAssertEqual(z.re, 11.0, accuracy: 1e-12)
        XCTAssertEqual(z.im,  5.0, accuracy: 1e-12)
    }

    // MARK: - applyFunction: hadamard

    func testHadamardMatrix() throws {
        let l = makeMatrix(2, 2, value: 2.0)
        let r = makeMatrix(2, 2, value: 3.0)
        let result = try NumericDispatch.applyFunction("hadamard", args: [l, r])
        guard case .matrix(let m) = result else { return XCTFail("Expected matrix") }
        XCTAssertEqual(m[0, 0], 6.0)
    }

    func testHadamardMatrix_shapeMismatch_throws() {
        let result = run {
            try NumericDispatch.applyFunction(
                "hadamard", args: [self.makeMatrix(2, 2), self.makeMatrix(2, 3)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.shapeMismatch = err { return }
        XCTFail("Expected .shapeMismatch, got \(err)")
    }

    func testHadamardComplexMatrix() throws {
        // hadamard([[i, 0],[0, 1+i]], [[2, 1],[1, 1]])
        // (0+i)(2+0i)=0+2i,  (0+0)(1+0)=0,  (0+0)(1+0)=0,  (1+i)(1+0)=1+i
        let a = LinAlg.ComplexMatrix(rows: 2, cols: 2,
                                     real: [0, 0, 0, 1], imag: [1, 0, 0, 1])
        let b = LinAlg.ComplexMatrix(rows: 2, cols: 2,
                                     real: [2, 1, 1, 1], imag: [0, 0, 0, 0])
        let result = try NumericDispatch.applyFunction("hadamard",
                                                       args: [.complexMatrix(a),
                                                              .complexMatrix(b)])
        guard case .complexMatrix(let cm) = result else { return XCTFail("Expected complexMatrix") }
        // (0+i)*(2+0i) = 0*2-1*0 + i*(0*0+1*2) = 0+2i
        XCTAssertEqual(cm.real[0], 0.0, accuracy: 1e-12)
        XCTAssertEqual(cm.imag[0], 2.0, accuracy: 1e-12)
        // (1+i)*(1+0i) = 1+i
        XCTAssertEqual(cm.real[3], 1.0, accuracy: 1e-12)
        XCTAssertEqual(cm.imag[3], 1.0, accuracy: 1e-12)
    }

    // MARK: - applyFunction: crossProduct

    func testCrossProduct_throws_unsupportedNode() {
        let result = run {
            try NumericDispatch.applyFunction("crossProduct", args: [self.makeMatrix(3, 1)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.unsupportedNode = err { return }
        XCTFail("Expected .unsupportedNode, got \(err)")
    }

    // MARK: - applyFunction: min / max

    func testMin1Arg() throws {
        let result = try NumericDispatch.applyFunction("min", args: [.scalar(7.0)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 7.0)
    }

    func testMin2Args() throws {
        let result = try NumericDispatch.applyFunction("min", args: [.scalar(3), .scalar(5)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 3.0)
    }

    func testMax2Args() throws {
        let result = try NumericDispatch.applyFunction("max", args: [.scalar(3), .scalar(5)])
        guard case .scalar(let v) = result else { return XCTFail("Expected scalar") }
        XCTAssertEqual(v, 5.0)
    }

    func testMinMatrixArg_throws() {
        let result = run {
            try NumericDispatch.applyFunction("min", args: [self.makeMatrix(2, 2)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.invalidArguments = err { return }
        XCTFail("Expected .invalidArguments, got \(err)")
    }

    // MARK: - applyFunction: unknown function

    func testUnknownFunction_throws() {
        let result = run {
            try NumericDispatch.applyFunction("squiggle", args: [.scalar(1)])
        }
        guard case .failure(let err) = result else { return XCTFail("Expected failure") }
        if case MathExprError.unknownFunction(let name) = err {
            XCTAssertEqual(name, "squiggle")
            return
        }
        XCTFail("Expected .unknownFunction, got \(err)")
    }

    // MARK: - Size cap

    func testSizeCap_firesForOversizedMatrixMul() {
        // Set a tiny soft cap, attempt a matrix multiplication that would exceed it,
        // then restore the default.
        let originalCap = LinAlg.maxEvaluatorMatrixElements
        do {
            try LinAlg.setMaxEvaluatorMatrixElements(4)
        } catch {
            XCTFail("Unexpected error setting cap: \(error)")
            return
        }
        defer {
            // Restore — ignore error since default is always valid
            try? LinAlg.setMaxEvaluatorMatrixElements(originalCap)
        }

        // 3×3 * 3×3 → 3×3 = 9 elements > cap of 4
        let result = run {
            try NumericDispatch.applyBinary(
                .mul, lhs: self.makeMatrix(3, 3), rhs: self.makeMatrix(3, 3))
        }
        guard case .failure(let err) = result else {
            return XCTFail("Expected failure due to size cap")
        }
        if case LinAlg.LinAlgError.invalidParameter(let msg) = err {
            XCTAssertTrue(msg.contains("soft cap"), "Expected 'soft cap' in message, got: \(msg)")
            return
        }
        XCTFail("Expected .invalidParameter from checkSoftCap, got \(err)")
    }

    // MARK: - 1×1 coercion via dotProduct function form

    func testDotProduct_2x3_3x1_producesMatrix_not1x1() throws {
        // (2×3) · (3×1) = (2×1) — not 1×1, must stay matrix
        let l = makeMatrix(2, 3, value: 1.0)
        let r = makeMatrix(3, 1, value: 1.0)
        let result = try NumericDispatch.applyFunction("dotProduct", args: [l, r])
        guard case .matrix(let m) = result else {
            return XCTFail("Expected matrix for 2×1 result, got \(result)")
        }
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 1)
    }
}
