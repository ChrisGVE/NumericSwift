//
//  AuditCoverageCRD8Tests.swift
//  NumericSwiftTests
//
//  Coverage for the gaps flagged by post-tag audit item CR-D8 (code_review.md):
//  the complex-matrix elementDiv error path, mixed-kind dotProduct, factorial of
//  a complex matrix, the log2 negative-real path (and its non-promotion under
//  complexMode), and NaN propagation through complex-matrix arithmetic.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class AuditCoverageCRD8Tests: XCTestCase {

    // MARK: - Helpers

    /// 1×1 complex matrix with the given real/imag parts.
    private func cm1(_ re: Double, _ im: Double) -> LinAlg.ComplexMatrix {
        LinAlg.ComplexMatrix(rows: 1, cols: 1, real: [re], imag: [im])
    }

    /// 1×1 real matrix.
    private func m1(_ x: Double) -> LinAlg.Matrix {
        LinAlg.Matrix(rows: 1, cols: 1, data: [x])
    }

    private func assertInvalidArguments(
        _ body: @autoclosure () throws -> NumericValue,
        _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(try body(), message, file: file, line: line) { error in
            guard case MathExprError.invalidArguments = error else {
                return XCTFail("expected .invalidArguments, got \(error)", file: file, line: line)
            }
        }
    }

    // MARK: - elementDiv(CM, CM): unsupported-kind error path

    func testElementDiv_complexMatrix_throwsInvalidArguments() {
        // elementDiv is real-matrix only; two complex matrices hit the default
        // arm and raise .invalidArguments rather than computing or trapping.
        assertInvalidArguments(
            try NumericDispatch.applyFunction(
                "elementDiv",
                args: [.complexMatrix(cm1(1, 0)), .complexMatrix(cm1(2, 0))]))
    }

    // MARK: - Mixed-kind dotProduct error paths

    func testDotProduct_matrixAndComplexMatrix_throwsInvalidArguments() {
        // (real matrix, complex matrix) is not a supported dot pairing.
        assertInvalidArguments(
            try NumericDispatch.applyFunction(
                "dotProduct",
                args: [.matrix(m1(1)), .complexMatrix(cm1(1, 0))]))
    }

    func testDotProduct_scalarAndMatrix_throwsInvalidArguments() {
        // dotProduct requires two matrix arguments; a scalar operand is rejected.
        assertInvalidArguments(
            try NumericDispatch.applyFunction(
                "dotProduct",
                args: [.scalar(2), .matrix(m1(3))]))
    }

    // MARK: - NumberTheory.factorial(complexMatrix): unsupported-kind error path

    func testFactorial_complexMatrix_throwsInvalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyUnary(.factorial, operand: .complexMatrix(cm1(3, 0)))
        ) { error in
            guard case MathExprError.invalidArguments(let msg) = error else {
                return XCTFail("expected .invalidArguments, got \(error)")
            }
            XCTAssertTrue(msg.contains("complex matrices"),
                          "message should name complex matrices; got: \(msg)")
        }
    }

    // MARK: - log2 negative-real path (and complexMode non-promotion)

    func testLog2_negativeReal_realPath_returnsNaN() throws {
        // log2(-1) on the real path is NaN, like the underlying Foundation.log2.
        let r = try NumericDispatch.applyFunction("log2", args: [.scalar(-1)])
        let x = try XCTUnwrap(r.asScalar)
        XCTAssertTrue(x.isNaN, "log2(-1) should be NaN on the real path, got \(x)")
    }

    func testLog2_negativeReal_complexMode_staysNaN_notPromoted() throws {
        // Issue-#1 promotion is intentionally narrow: only sqrt/log/ln (and the
        // ^ operator) promote negative-real scalars to the complex principal
        // value. log2 is NOT in that set, so complexMode leaves log2(-1) as NaN.
        let r = try NumericDispatch.applyFunction(
            "log2", args: [.scalar(-1)], complexMode: true)
        let x = try XCTUnwrap(r.asScalar)
        XCTAssertEqual(r.kind, .scalar, "log2 must not promote to complex under complexMode")
        XCTAssertTrue(x.isNaN, "log2(-1) must stay NaN under complexMode, got \(x)")
    }

    // MARK: - NaN propagation through complex-matrix arithmetic

    func testComplexMatrixAdd_propagatesNaN_inRealPart() throws {
        // A NaN real component flows through element-wise CM addition unchanged.
        let a = cm1(.nan, 1)
        let b = cm1(2, 3)
        let r = try NumericDispatch.applyBinary(.add, lhs: .complexMatrix(a), rhs: .complexMatrix(b))
        let out = try XCTUnwrap(r.asComplexMatrix)
        XCTAssertTrue(out.real[0].isNaN, "NaN real part must propagate through CM add")
        XCTAssertEqual(out.imag[0], 4, accuracy: 1e-15, "finite imag part still sums")
    }

    func testComplexMatrixHadamard_propagatesNaN() throws {
        // NaN in either operand poisons the complex product (NaN·anything = NaN).
        let a = cm1(.nan, 0)
        let b = cm1(5, 7)
        let r = try NumericDispatch.applyFunction(
            "hadamard", args: [.complexMatrix(a), .complexMatrix(b)])
        let out = try XCTUnwrap(r.asComplexMatrix)
        XCTAssertTrue(out.real[0].isNaN, "NaN must propagate through complex hadamard real part")
        XCTAssertTrue(out.imag[0].isNaN, "NaN must propagate through complex hadamard imag part")
    }
}
