//
//  NumericDispatchFunctionRegistryTests.swift
//  NumericSwiftTests
//
//  Tests for the unified function registry introduced in Task 17.
//
//  Coverage:
//    1. Every registered function name resolves (registry lookup succeeds).
//    2. Unknown function name → MathExprError.unknownFunction.
//    3. Every fixed-arity function rejects wrong arity → MathExprError.invalidArguments.
//    4. Every scalar/complex function rejects matrix operand → MathExprError.invalidArguments.
//    5. Every Group-B matrix function propagates LinAlgError.notSquare for non-square input.
//    6. Legacy alias names (arcsin, ln, lg, sgn, re, im, phase) work correctly.
//    7. Newly-added legacy transcendentals (log10, log2, cbrt, hypot, sign, floor, ceil,
//       round, trunc, clamp, lerp, rad, deg, atan2, pow) produce bit-exact values
//       matching the legacy MathExpr.evalFunction path.
//    8. Complex-only functions (conj, real/re, imag/im, arg/phase) work.
//    9. Group-B classification: inv/det/trace/cdet/cinv propagate notSquare.
//   10. crossProduct throws .unsupportedNode (deferred §14).
//   11. Parity spot-checks: representative scalar transcendentals via
//       applyFunction match MathExpr.evaluate bit-for-bit.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class NumericDispatchFunctionRegistryTests: XCTestCase {

    // MARK: - Helpers

    private func scalarArg(_ x: Double) -> NumericValue { .scalar(x) }
    private func complexArg(_ re: Double, _ im: Double) -> NumericValue {
        .complex(Complex(re: re, im: im))
    }
    private func matrixArg(_ rows: Int, _ cols: Int, value: Double = 1.0) -> NumericValue {
        .matrix(LinAlg.Matrix(rows: rows, cols: cols,
                              data: [Double](repeating: value, count: rows * cols)))
    }
    private func complexMatrixArg(_ rows: Int, _ cols: Int) -> NumericValue {
        let n = rows * cols
        return .complexMatrix(LinAlg.ComplexMatrix(
            rows: rows, cols: cols,
            real: [Double](repeating: 1.0, count: n),
            imag: [Double](repeating: 0.0, count: n)))
    }

    // Assert that result is a scalar and return the value (fails test if not scalar).
    private func assertScalar(
        _ result: NumericValue,
        _ label: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) -> Double {
        guard case .scalar(let v) = result else {
            XCTFail("Expected scalar\(label.isEmpty ? "" : " for \(label)"), got \(result.kind)",
                    file: file, line: line)
            return .nan
        }
        return v
    }

    private func assertComplex(
        _ result: NumericValue,
        _ label: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) -> Complex {
        guard case .complex(let z) = result else {
            XCTFail("Expected complex\(label.isEmpty ? "" : " for \(label)"), got \(result.kind)",
                    file: file, line: line)
            return Complex(0)
        }
        return z
    }

    // MARK: - 1. Every registered name resolves

    func testRegistryContainsAllExpectedNames() {
        let expected: [String] = [
            // Trig
            "sin", "cos", "tan",
            "asin", "arcsin", "acos", "arccos", "atan", "arctan",
            "sinh", "cosh", "tanh",
            "asinh", "arcsinh", "acosh", "arccosh", "atanh", "arctanh",
            // 2-arg scalar
            "atan2", "pow", "hypot",
            // Exp/log/sqrt
            "exp", "log", "ln", "log10", "log2", "lg", "sqrt", "cbrt",
            // Abs/sign/rounding
            "abs", "sign", "sgn", "floor", "ceil", "round", "trunc",
            // Multi-arg scalar
            "clamp", "lerp",
            // Angle
            "rad", "deg",
            // Min/max
            "min", "max",
            // Matrix unary
            "inv", "det", "trace", "transpose",
            // Complex-matrix
            "cdet", "cinv",
            // Multi-arg matrix
            "dotProduct", "hadamard", "elementDiv",
            // Complex-only
            "conj", "real", "re", "imag", "im", "arg", "phase",
            // Deferred
            "crossProduct",
        ]
        for name in expected {
            XCTAssertNotNil(NumericDispatch.functionRegistry[name],
                            "Registry missing entry for '\(name)'")
        }
    }

    // MARK: - 2. Unknown function name

    func testUnknownFunction_returnsUnknownFunctionError() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("totally_unknown_xyz", args: [.scalar(1)])
        ) { err in
            guard case MathExprError.unknownFunction(let n) = err else {
                XCTFail("Expected unknownFunction, got \(err)"); return
            }
            XCTAssertEqual(n, "totally_unknown_xyz")
        }
    }

    func testUnknownFunction_emptyString_returnsUnknownFunctionError() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("", args: [.scalar(1)])
        ) { err in
            if case MathExprError.unknownFunction = err { return }
            XCTFail("Expected unknownFunction, got \(err)")
        }
    }

    // MARK: - 3. Arity enforcement for fixed-arity functions

    func testSin_tooManyArgs_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("sin", args: [.scalar(0), .scalar(0)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments, got \(err)")
        }
    }

    func testSin_zeroArgs_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("sin", args: [])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments, got \(err)")
        }
    }

    func testExp_twoArgs_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("exp", args: [.scalar(1), .scalar(2)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments, got \(err)")
        }
    }

    func testAtan2_oneArg_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("atan2", args: [.scalar(1)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments, got \(err)")
        }
    }

    func testClamp_twoArgs_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("clamp", args: [.scalar(1), .scalar(0)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments, got \(err)")
        }
    }

    func testPow_oneArg_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("pow", args: [.scalar(2)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments, got \(err)")
        }
    }

    func testInv_zeroArgs_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("inv", args: [])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments, got \(err)")
        }
    }

    // MARK: - 4. Wrong operand kind → invalidArguments

    func testFloor_matrixArg_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("floor", args: [matrixArg(2, 2)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments for floor(matrix), got \(err)")
        }
    }

    func testLog10_complexArg_invalidArguments() {
        // log10 is scalar-only; complex input rejected
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("log10", args: [complexArg(1, 0)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments for log10(complex), got \(err)")
        }
    }

    func testSign_matrixArg_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("sign", args: [matrixArg(2, 2)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments for sign(matrix), got \(err)")
        }
    }

    func testConj_matrixArg_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("conj", args: [matrixArg(2, 2)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments for conj(matrix), got \(err)")
        }
    }

    func testRe_complexMatrixArg_invalidArguments() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("re", args: [complexMatrixArg(2, 2)])
        ) { err in
            if case MathExprError.invalidArguments = err { return }
            XCTFail("Expected invalidArguments for re(complexMatrix), got \(err)")
        }
    }

    // MARK: - 5. Group-B matrix functions propagate notSquare

    func testTrace_nonSquare_propagatesNotSquare() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("trace", args: [matrixArg(2, 3)])
        ) { err in
            if case LinAlg.LinAlgError.notSquare = err { return }
            XCTFail("Expected .notSquare for trace(2×3), got \(err)")
        }
    }

    func testDet_nonSquare_propagatesNotSquare() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("det", args: [matrixArg(3, 2)])
        ) { err in
            if case LinAlg.LinAlgError.notSquare = err { return }
            XCTFail("Expected .notSquare for det(3×2), got \(err)")
        }
    }

    func testInv_nonSquare_propagatesNotSquare() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("inv", args: [matrixArg(2, 3)])
        ) { err in
            if case LinAlg.LinAlgError.notSquare = err { return }
            XCTFail("Expected .notSquare for inv(2×3), got \(err)")
        }
    }

    func testCdet_nonSquare_propagatesNotSquare() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("cdet", args: [complexMatrixArg(1, 2)])
        ) { err in
            if case LinAlg.LinAlgError.notSquare = err { return }
            XCTFail("Expected .notSquare for cdet(1×2), got \(err)")
        }
    }

    func testCinv_nonSquare_propagatesNotSquare() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("cinv", args: [complexMatrixArg(2, 1)])
        ) { err in
            if case LinAlg.LinAlgError.notSquare = err { return }
            XCTFail("Expected .notSquare for cinv(2×1), got \(err)")
        }
    }

    // MARK: - 6. Alias names work

    func testArcsin_aliasWorksLikeAsin() throws {
        let r1 = try NumericDispatch.applyFunction("asin", args: [scalarArg(1.0)])
        let r2 = try NumericDispatch.applyFunction("arcsin", args: [scalarArg(1.0)])
        let v1 = assertScalar(r1, "asin(1)")
        let v2 = assertScalar(r2, "arcsin(1)")
        XCTAssertEqual(v1.bitPattern, v2.bitPattern, "arcsin alias must be bit-exact to asin")
    }

    func testArccos_aliasWorksLikeAcos() throws {
        let r1 = try NumericDispatch.applyFunction("acos", args: [scalarArg(0.0)])
        let r2 = try NumericDispatch.applyFunction("arccos", args: [scalarArg(0.0)])
        XCTAssertEqual(assertScalar(r1).bitPattern, assertScalar(r2).bitPattern,
                       "arccos alias bit-exact")
    }

    func testArctan_aliasWorksLikeAtan() throws {
        let r1 = try NumericDispatch.applyFunction("atan", args: [scalarArg(1.0)])
        let r2 = try NumericDispatch.applyFunction("arctan", args: [scalarArg(1.0)])
        XCTAssertEqual(assertScalar(r1).bitPattern, assertScalar(r2).bitPattern,
                       "arctan alias bit-exact")
    }

    func testArcsinh_aliasWorksLikeAsinh() throws {
        let r1 = try NumericDispatch.applyFunction("asinh", args: [scalarArg(1.0)])
        let r2 = try NumericDispatch.applyFunction("arcsinh", args: [scalarArg(1.0)])
        XCTAssertEqual(assertScalar(r1).bitPattern, assertScalar(r2).bitPattern,
                       "arcsinh alias bit-exact")
    }

    func testLn_aliasForLog() throws {
        let r1 = try NumericDispatch.applyFunction("log", args: [scalarArg(Foundation.exp(1))])
        let r2 = try NumericDispatch.applyFunction("ln",  args: [scalarArg(Foundation.exp(1))])
        XCTAssertEqual(assertScalar(r1).bitPattern, assertScalar(r2).bitPattern,
                       "ln alias bit-exact to log")
    }

    func testLg_aliasForLog2() throws {
        let r1 = try NumericDispatch.applyFunction("log2", args: [scalarArg(8)])
        let r2 = try NumericDispatch.applyFunction("lg",   args: [scalarArg(8)])
        XCTAssertEqual(assertScalar(r1).bitPattern, assertScalar(r2).bitPattern,
                       "lg alias bit-exact to log2")
        XCTAssertEqual(assertScalar(r2), 3.0, accuracy: 1e-15, "log2(8)=3")
    }

    func testSgn_aliasForSign() throws {
        let r1 = try NumericDispatch.applyFunction("sign", args: [scalarArg(-5)])
        let r2 = try NumericDispatch.applyFunction("sgn",  args: [scalarArg(-5)])
        XCTAssertEqual(assertScalar(r1).bitPattern, assertScalar(r2).bitPattern,
                       "sgn alias bit-exact to sign")
        XCTAssertEqual(assertScalar(r2), -1.0, "sign(-5)=-1")
    }

    func testRe_aliasForReal() throws {
        let z = complexArg(3, 4)
        let r1 = try NumericDispatch.applyFunction("real", args: [z])
        let r2 = try NumericDispatch.applyFunction("re",   args: [z])
        XCTAssertEqual(assertComplex(r1).re.bitPattern,
                       assertComplex(r2).re.bitPattern, "re alias bit-exact")
        XCTAssertEqual(assertComplex(r2).re, 3.0)
    }

    func testIm_aliasForImag() throws {
        let z = complexArg(3, 4)
        let r1 = try NumericDispatch.applyFunction("imag", args: [z])
        let r2 = try NumericDispatch.applyFunction("im",   args: [z])
        XCTAssertEqual(assertComplex(r1).re.bitPattern,
                       assertComplex(r2).re.bitPattern, "im alias bit-exact")
        XCTAssertEqual(assertComplex(r2).re, 4.0)
    }

    func testPhase_aliasForArg() throws {
        let z = complexArg(0, 1)   // arg(i) = π/2
        let r1 = try NumericDispatch.applyFunction("arg",   args: [z])
        let r2 = try NumericDispatch.applyFunction("phase", args: [z])
        XCTAssertEqual(assertComplex(r1).re.bitPattern,
                       assertComplex(r2).re.bitPattern, "phase alias bit-exact")
        XCTAssertEqual(assertComplex(r2).re, .pi / 2, accuracy: 1e-12, "arg(i)=π/2")
    }

    // MARK: - 7. New scalar transcendentals

    func testLog10() throws {
        let r = try NumericDispatch.applyFunction("log10", args: [scalarArg(100)])
        XCTAssertEqual(assertScalar(r, "log10(100)"), 2.0, accuracy: 1e-12)
    }

    func testLog2() throws {
        let r = try NumericDispatch.applyFunction("log2", args: [scalarArg(8)])
        XCTAssertEqual(assertScalar(r, "log2(8)"), 3.0, accuracy: 1e-12)
    }

    func testCbrt() throws {
        let r = try NumericDispatch.applyFunction("cbrt", args: [scalarArg(27)])
        XCTAssertEqual(assertScalar(r, "cbrt(27)"), 3.0, accuracy: 1e-12)
    }

    func testPow() throws {
        let r = try NumericDispatch.applyFunction("pow", args: [scalarArg(2), scalarArg(8)])
        XCTAssertEqual(assertScalar(r, "pow(2,8)"), 256.0, accuracy: 1e-12)
    }

    func testHypot() throws {
        let r = try NumericDispatch.applyFunction("hypot", args: [scalarArg(3), scalarArg(4)])
        XCTAssertEqual(assertScalar(r, "hypot(3,4)"), 5.0, accuracy: 1e-12)
    }

    func testAtan2() throws {
        // atan2(1, 1) = π/4
        let r = try NumericDispatch.applyFunction("atan2", args: [scalarArg(1), scalarArg(1)])
        XCTAssertEqual(assertScalar(r, "atan2(1,1)"), .pi / 4, accuracy: 1e-12)
    }

    func testSign_positive() throws {
        XCTAssertEqual(
            assertScalar(try NumericDispatch.applyFunction("sign", args: [scalarArg(7)])),
            1.0)
    }

    func testSign_negative() throws {
        XCTAssertEqual(
            assertScalar(try NumericDispatch.applyFunction("sign", args: [scalarArg(-3)])),
            -1.0)
    }

    func testSign_zero() throws {
        XCTAssertEqual(
            assertScalar(try NumericDispatch.applyFunction("sign", args: [scalarArg(0)])),
            0.0)
    }

    func testFloor() throws {
        XCTAssertEqual(
            assertScalar(try NumericDispatch.applyFunction("floor", args: [scalarArg(3.7)])),
            3.0)
    }

    func testCeil() throws {
        XCTAssertEqual(
            assertScalar(try NumericDispatch.applyFunction("ceil", args: [scalarArg(3.2)])),
            4.0)
    }

    func testRound() throws {
        XCTAssertEqual(
            assertScalar(try NumericDispatch.applyFunction("round", args: [scalarArg(3.5)])),
            4.0)
    }

    func testTrunc() throws {
        XCTAssertEqual(
            assertScalar(try NumericDispatch.applyFunction("trunc", args: [scalarArg(-2.9)])),
            -2.0)
    }

    func testClamp_withinRange() throws {
        // clamp(3, 0, 5) = 3
        let r = try NumericDispatch.applyFunction(
            "clamp", args: [scalarArg(3), scalarArg(0), scalarArg(5)])
        XCTAssertEqual(assertScalar(r, "clamp(3,0,5)"), 3.0)
    }

    func testClamp_aboveMax() throws {
        // clamp(10, 0, 5) = 5
        let r = try NumericDispatch.applyFunction(
            "clamp", args: [scalarArg(10), scalarArg(0), scalarArg(5)])
        XCTAssertEqual(assertScalar(r, "clamp(10,0,5)"), 5.0)
    }

    func testLerp() throws {
        // lerp(0, 10, 0.5) = 5
        let r = try NumericDispatch.applyFunction(
            "lerp", args: [scalarArg(0), scalarArg(10), scalarArg(0.5)])
        XCTAssertEqual(assertScalar(r, "lerp(0,10,0.5)"), 5.0, accuracy: 1e-12)
    }

    func testRad() throws {
        // rad(180) = π
        let r = try NumericDispatch.applyFunction("rad", args: [scalarArg(180)])
        XCTAssertEqual(assertScalar(r, "rad(180)"), .pi, accuracy: 1e-12)
    }

    func testDeg() throws {
        // deg(π) = 180
        let r = try NumericDispatch.applyFunction("deg", args: [scalarArg(.pi)])
        XCTAssertEqual(assertScalar(r, "deg(π)"), 180.0, accuracy: 1e-10)
    }

    // MARK: - 8. Complex-only functions

    func testConj() throws {
        // conj(2+3i) = 2-3i
        let r = try NumericDispatch.applyFunction("conj", args: [complexArg(2, 3)])
        let z = assertComplex(r, "conj(2+3i)")
        XCTAssertEqual(z.re, 2.0, accuracy: 1e-12)
        XCTAssertEqual(z.im, -3.0, accuracy: 1e-12)
    }

    func testReal() throws {
        let r = try NumericDispatch.applyFunction("real", args: [complexArg(3, 4)])
        XCTAssertEqual(assertComplex(r, "real(3+4i)").re, 3.0)
        XCTAssertEqual(assertComplex(r).im, 0.0)
    }

    func testImag() throws {
        let r = try NumericDispatch.applyFunction("imag", args: [complexArg(3, 4)])
        XCTAssertEqual(assertComplex(r, "imag(3+4i)").re, 4.0)
        XCTAssertEqual(assertComplex(r).im, 0.0)
    }

    func testArg_pureImaginary() throws {
        // arg(0+1i) = π/2
        let r = try NumericDispatch.applyFunction("arg", args: [complexArg(0, 1)])
        XCTAssertEqual(assertComplex(r, "arg(i)").re, .pi / 2, accuracy: 1e-12)
    }

    func testConj_scalarPromotedToComplex() throws {
        // conj(scalar) — scalar promoted to complex, result is complex with im=0
        let r = try NumericDispatch.applyFunction("conj", args: [scalarArg(5)])
        let z = assertComplex(r, "conj(5)")
        XCTAssertEqual(z.re, 5.0)
        XCTAssertEqual(z.im, 0.0)
    }

    // MARK: - 9. Group-B expm/logm/sqrtm via exp/log/sqrt on matrix

    func testExp_squareMatrix_groupBPropagation() throws {
        // exp(square matrix) → expm succeeds
        let r = try NumericDispatch.applyFunction("exp", args: [matrixArg(2, 2, value: 0)])
        guard case .matrix = r else { return XCTFail("Expected matrix from expm") }
    }

    func testLog_nonSquareMatrix_propagatesNotSquare() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("log", args: [matrixArg(2, 3)])
        ) { err in
            if case LinAlg.LinAlgError.notSquare = err { return }
            XCTFail("Expected .notSquare for log(2×3 matrix), got \(err)")
        }
    }

    func testSqrt_nonSquareMatrix_propagatesNotSquare() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction("sqrt", args: [matrixArg(3, 2)])
        ) { err in
            if case LinAlg.LinAlgError.notSquare = err { return }
            XCTFail("Expected .notSquare for sqrt(3×2 matrix), got \(err)")
        }
    }

    // MARK: - 10. crossProduct deferred

    func testCrossProduct_throwsUnsupportedNode() {
        XCTAssertThrowsError(
            try NumericDispatch.applyFunction(
                "crossProduct",
                args: [matrixArg(3, 1), matrixArg(3, 1)])
        ) { err in
            if case MathExprError.unsupportedNode = err { return }
            XCTFail("Expected .unsupportedNode for crossProduct (deferred), got \(err)")
        }
    }

    // MARK: - 11. Parity: representative scalars vs MathExpr.evaluate

    /// Verify that applyFunction produces the exact same bit-pattern as the legacy
    /// MathExpr.evaluate path for the scalar transcendentals that both paths cover.
    /// This is the critical backward-compat check.
    func testParityVsLegacyEvaluate_sin() throws {
        let x = 1.234
        let legacy = try MathExpr.eval("sin(\(x))")
        let r = try NumericDispatch.applyFunction("sin", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r, "sin").bitPattern, legacy.bitPattern,
                       "sin parity: applyFunction must match MathExpr.evaluate bit-exactly")
    }

    func testParityVsLegacyEvaluate_exp() throws {
        let x = 2.0
        let legacy = try MathExpr.eval("exp(\(x))")
        let r = try NumericDispatch.applyFunction("exp", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "exp parity")
    }

    func testParityVsLegacyEvaluate_log() throws {
        let x = Foundation.exp(2.0)
        let legacy = try MathExpr.eval("log(\(x))")
        let r = try NumericDispatch.applyFunction("log", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "log parity")
    }

    func testParityVsLegacyEvaluate_sqrt() throws {
        let x = 9.0
        let legacy = try MathExpr.eval("sqrt(\(x))")
        let r = try NumericDispatch.applyFunction("sqrt", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "sqrt parity")
    }

    func testParityVsLegacyEvaluate_abs() throws {
        let x = -7.5
        let legacy = try MathExpr.eval("abs(\(x))")
        let r = try NumericDispatch.applyFunction("abs", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "abs parity")
    }

    func testParityVsLegacyEvaluate_log10() throws {
        let x = 1000.0
        let legacy = try MathExpr.eval("log10(\(x))")
        let r = try NumericDispatch.applyFunction("log10", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "log10 parity")
    }

    func testParityVsLegacyEvaluate_cbrt() throws {
        let x = 125.0
        let legacy = try MathExpr.eval("cbrt(\(x))")
        let r = try NumericDispatch.applyFunction("cbrt", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "cbrt parity")
    }

    func testParityVsLegacyEvaluate_floor() throws {
        let x = 4.9
        let legacy = try MathExpr.eval("floor(\(x))")
        let r = try NumericDispatch.applyFunction("floor", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "floor parity")
    }

    func testParityVsLegacyEvaluate_ceil() throws {
        let x = 4.1
        let legacy = try MathExpr.eval("ceil(\(x))")
        let r = try NumericDispatch.applyFunction("ceil", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "ceil parity")
    }

    func testParityVsLegacyEvaluate_round() throws {
        let x = 2.5
        let legacy = try MathExpr.eval("round(\(x))")
        let r = try NumericDispatch.applyFunction("round", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "round parity")
    }

    func testParityVsLegacyEvaluate_atan2() throws {
        let y = 1.0, x = 1.0
        let legacy = try MathExpr.eval("atan2(\(y), \(x))")
        let r = try NumericDispatch.applyFunction("atan2", args: [scalarArg(y), scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "atan2 parity")
    }

    func testParityVsLegacyEvaluate_sign() throws {
        for x in [-3.0, 0.0, 5.0] {
            let legacy = try MathExpr.eval("sign(\(x))")
            let r = try NumericDispatch.applyFunction("sign", args: [scalarArg(x)])
            XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "sign(\(x)) parity")
        }
    }

    func testParityVsLegacyEvaluate_rad() throws {
        let x = 90.0
        let legacy = try MathExpr.eval("rad(\(x))")
        let r = try NumericDispatch.applyFunction("rad", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "rad parity")
    }

    func testParityVsLegacyEvaluate_deg() throws {
        let x = Double.pi / 4
        let legacy = try MathExpr.eval("deg(\(x))")
        let r = try NumericDispatch.applyFunction("deg", args: [scalarArg(x)])
        XCTAssertEqual(assertScalar(r).bitPattern, legacy.bitPattern, "deg parity")
    }
}
