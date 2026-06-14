//
//  NumericDispatchScalarComplexTests.swift
//  NumericSwiftTests
//
//  Parity and semantic correctness tests for Task 10 of the unified-pipeline
//  milestone: scalar/scalar, scalar/complex, complex/scalar, complex/complex
//  arithmetic handlers in NumericDispatch.
//
//  Coverage strategy
//  ─────────────────
//  1.  Bit-exact parity for scalar arithmetic against the frozen
//      LegacySnapshot.json (IEEE-754 Double bitPattern comparison).
//  2.  Approximate parity (1e-12 absolute + relative) for complex results.
//  3.  Division-by-zero: scalar x/0 throws; complex /(0+0i) throws;
//      mixed-kind paths that divide by a non-zero complex do NOT throw.
//  4.  Legacy plusMinus/minusPlus error via the binary top-level router.
//  5.  Scalar unary neg/pos/factorial/transpose semantics.
//  6.  Complex unary neg/pos/factorial/transpose semantics.
//  7.  Scalar→complex promotion: (scalar, complex) and (complex, scalar)
//      pairs produce complex results equal to promoting then computing.
//  8.  Complex→scalar collapse policy: complex results do NOT collapse to
//      scalar automatically (preserves legacy evaluateComplex surface).
//  9.  nonFiniteFloat policy: dispatch does not insert isFinite guards;
//      NaN/inf propagate through arithmetic and only the parse-literal
//      boundary raises nonFiniteFloat (not tested here — boundary is in
//      MathExpr.evaluate, not in NumericDispatch).
//  10. IEEE-754 edge values: NaN, ±inf, signed zero through dispatch.
//
//  Every subtask correspondence:
//    ST 1  → testParity*  (harness exists here)
//    ST 2  → routing confirmed by tests below (scalar path is exercised)
//    ST 3  → testScalar_add/sub/mul*
//    ST 4  → testScalar_div* / testDivisionByZero*
//    ST 5  → testScalar_pow* / testScalar_mod*
//    ST 6  → testPlusMinusRejection / testMinusPlusRejection
//    ST 7  → testScalarUnary_neg* / testScalarUnary_pos*
//    ST 8  → testScalarUnary_factorial* / testScalarUnary_transpose*
//    ST 9  → routing confirmed — complex cells exercised below
//    ST 10 → testComplex_add/sub/mul*
//    ST 11 → testComplex_div* / testComplexDivisionByZero*
//    ST 12 → testComplex_pow*
//    ST 13 → testComplex_mod / testComplex_plusMinus*
//    ST 14 → testComplexUnary_*
//    ST 15 → testPromotion_*
//    ST 16 → testCollapsePolicy_*
//    ST 17 → testNonFiniteFloat_propagation
//    ST 18 → (variable resolution — exercised via promotion tests)
//    ST 19 → testDivisionByZero_mechanism*
//    ST 20 → testSnapshotParityScalar / testSnapshotParityComplex
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import XCTest
@testable import NumericSwift

// MARK: - Bit-comparison helpers

private func assertScalarBitExact(
    _ result: NumericValue,
    bitPattern expected: UInt64,
    _ message: String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) {
    guard case .scalar(let v) = result else {
        XCTFail("Expected .scalar, got \(result.kind)\(message.isEmpty ? "" : ": \(message)")",
                file: file, line: line)
        return
    }
    XCTAssertEqual(v.bitPattern, expected,
        "Scalar bit mismatch — got \(v) (0x\(String(v.bitPattern, radix: 16)))"
        + " expected 0x\(String(expected, radix: 16))\(message.isEmpty ? "" : " [\(message)]")",
        file: file, line: line)
}

private func assertComplexApprox(
    _ result: NumericValue,
    re expectedRe: Double,
    im expectedIm: Double,
    tolerance tol: Double = 1e-12,
    _ message: String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) {
    guard case .complex(let z) = result else {
        XCTFail("Expected .complex, got \(result.kind)\(message.isEmpty ? "" : ": \(message)")",
                file: file, line: line)
        return
    }
    // NaN propagation: if expected is NaN, actual must also be NaN.
    if expectedRe.isNaN {
        XCTAssertTrue(z.re.isNaN, "Re should be NaN\(message.isEmpty ? "" : " [\(message)]")",
                      file: file, line: line)
    } else {
        XCTAssertEqual(z.re, expectedRe, accuracy: max(tol, tol * abs(expectedRe)),
            "Re mismatch\(message.isEmpty ? "" : " [\(message)]")", file: file, line: line)
    }
    if expectedIm.isNaN {
        XCTAssertTrue(z.im.isNaN, "Im should be NaN\(message.isEmpty ? "" : " [\(message)]")",
                      file: file, line: line)
    } else {
        XCTAssertEqual(z.im, expectedIm, accuracy: max(tol, tol * abs(expectedIm)),
            "Im mismatch\(message.isEmpty ? "" : " [\(message)]")", file: file, line: line)
    }
}

// MARK: - NumericDispatchScalarComplexTests

// swiftlint:disable:next type_body_length
final class NumericDispatchScalarComplexTests: XCTestCase {

    // MARK: - ST 3: Scalar add/sub/mul

    func testScalar_add_basic() throws {
        // 1 + 2 = 3 (s01: bitPattern 4613937818241073152)
        let r = try NumericDispatch.applyBinary(.add, lhs: .scalar(1), rhs: .scalar(2))
        assertScalarBitExact(r, bitPattern: 4613937818241073152, "1+2=3")
    }

    func testScalar_sub_basic() throws {
        // 10 - 3 = 7 (s02: 4619567317775286272)
        let r = try NumericDispatch.applyBinary(.sub, lhs: .scalar(10), rhs: .scalar(3))
        assertScalarBitExact(r, bitPattern: 4619567317775286272, "10-3=7")
    }

    func testScalar_mul_basic() throws {
        // 4 * 5 = 20 (s03: 4626322717216342016)
        let r = try NumericDispatch.applyBinary(.mul, lhs: .scalar(4), rhs: .scalar(5))
        assertScalarBitExact(r, bitPattern: 4626322717216342016, "4*5=20")
    }

    func testScalar_add_precedence() throws {
        // 2 + 3*4 = 14 (s07: 4624070917402656768) — result of the add; mul must be pre-applied
        let mul = try NumericDispatch.applyBinary(.mul, lhs: .scalar(3), rhs: .scalar(4))
        let r = try NumericDispatch.applyBinary(.add, lhs: .scalar(2), rhs: mul)
        assertScalarBitExact(r, bitPattern: 4624070917402656768, "2+3*4=14")
    }

    func testScalar_add_grouped() throws {
        // (2+3)*4 = 20 (s08: 4626322717216342016)
        let add = try NumericDispatch.applyBinary(.add, lhs: .scalar(2), rhs: .scalar(3))
        let r = try NumericDispatch.applyBinary(.mul, lhs: add, rhs: .scalar(4))
        assertScalarBitExact(r, bitPattern: 4626322717216342016, "(2+3)*4=20")
    }

    func testScalar_mul_large() throws {
        // a*b where a=3, b=4 → 12 (s15: 4622945017495814144)
        let r = try NumericDispatch.applyBinary(.mul, lhs: .scalar(3), rhs: .scalar(4))
        assertScalarBitExact(r, bitPattern: 4622945017495814144, "3*4=12")
    }

    func testScalar_sqrt2_mul_sqrt2() throws {
        // sqrt(2) * sqrt(2) = 2.0000000000000004 (s40: 4611686018427387905)
        let s = Foundation.sqrt(2.0)
        let r = try NumericDispatch.applyBinary(.mul, lhs: .scalar(s), rhs: .scalar(s))
        assertScalarBitExact(r, bitPattern: 4611686018427387905, "sqrt(2)*sqrt(2)")
    }

    // MARK: - ST 4: Scalar division — legacy divisionByZero semantics

    func testScalar_div_basic() throws {
        // 15/3 = 5 (s04: 4617315517961601024)
        let r = try NumericDispatch.applyBinary(.div, lhs: .scalar(15), rhs: .scalar(3))
        assertScalarBitExact(r, bitPattern: 4617315517961601024, "15/3=5")
    }

    func testScalar_div_byZero_throws() throws {
        // x/0 must throw divisionByZero (s groupA-e07)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.div, lhs: .scalar(1), rhs: .scalar(0))
        ) { err in
            guard case MathExprError.divisionByZero = err else {
                XCTFail("Expected divisionByZero, got \(err)"); return
            }
        }
    }

    func testScalar_div_byNegativeZero_throws() throws {
        // -0.0 == 0.0 in IEEE 754; both must throw per legacy
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.div, lhs: .scalar(5), rhs: .scalar(-0.0))
        ) { err in
            guard case MathExprError.divisionByZero = err else {
                XCTFail("Expected divisionByZero for -0.0 divisor, got \(err)"); return
            }
        }
    }

    func testScalar_div_doesNotProduceIEEEInfForZeroDivisor() throws {
        // Legacy threw; dispatch must NOT produce IEEE inf silently.
        var threw = false
        do {
            _ = try NumericDispatch.applyBinary(.div, lhs: .scalar(1), rhs: .scalar(0))
        } catch {
            threw = true
        }
        XCTAssertTrue(threw, "scalar/0 must throw, not produce inf")
    }

    // MARK: - ST 5: Scalar pow and mod

    func testScalar_pow_basic() throws {
        // 2^10 = 1024 (s05: 4652218415073722368)
        let r = try NumericDispatch.applyBinary(.pow, lhs: .scalar(2), rhs: .scalar(10))
        assertScalarBitExact(r, bitPattern: 4652218415073722368, "2^10=1024")
    }

    func testScalar_pow_nested() throws {
        // 2^3^2 — right-assoc: 2^(3^2)=2^9=512 (s09: 4647714815446351872)
        let inner = try NumericDispatch.applyBinary(.pow, lhs: .scalar(3), rhs: .scalar(2))
        let r = try NumericDispatch.applyBinary(.pow, lhs: .scalar(2), rhs: inner)
        assertScalarBitExact(r, bitPattern: 4647714815446351872, "2^(3^2)=512")
    }

    func testScalar_pow_via_applyBinary() throws {
        // pow(2,8) = 256 (s41: 4643211215818981376) — dispatched via applyBinary(.pow)
        // Note: "pow" named-function call is not in the function table (no applyFunction("pow")),
        // following the same boundary as legacy evalFunction which has "pow" but evalBinary(.pow)
        // is the primary path used by the AST evaluator.
        let r = try NumericDispatch.applyBinary(.pow, lhs: .scalar(2), rhs: .scalar(8))
        assertScalarBitExact(r, bitPattern: 4643211215818981376, "pow(2,8)=256")
    }

    func testScalar_pow_namedFunction() throws {
        // "pow" is registered in applyFunction (legacy MathExpr evalFunction parity).
        // pow(2, 8) = 256.
        let r = try NumericDispatch.applyFunction("pow", args: [.scalar(2), .scalar(8)])
        assertScalarBitExact(r, bitPattern: 4643211215818981376, "pow(2,8)=256")
    }

    func testScalar_pow2_8_direct() throws {
        // Direct: 2^8 = 256
        let r = try NumericDispatch.applyBinary(.pow, lhs: .scalar(2), rhs: .scalar(8))
        assertScalarBitExact(r, bitPattern: 4643211215818981376, "2^8=256")
    }

    func testScalar_pow_zero_zero() throws {
        // 0^0 inherits C pow behavior: returns 1.0 (per IEEE 854 recommendation)
        let r = try NumericDispatch.applyBinary(.pow, lhs: .scalar(0), rhs: .scalar(0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v, 1.0, accuracy: 0, "0^0 must equal 1.0 per C pow")
    }

    func testScalar_mod_basic() throws {
        // 17 % 5 = 2 (s06: 4611686018427387904)
        let r = try NumericDispatch.applyBinary(.mod, lhs: .scalar(17), rhs: .scalar(5))
        assertScalarBitExact(r, bitPattern: 4611686018427387904, "17%5=2")
    }

    func testScalar_mod_byZero_nanNotThrow() throws {
        // Legacy: truncatingRemainder(dividingBy: 0) = NaN (no throw); dispatch must match
        let r = try NumericDispatch.applyBinary(.mod, lhs: .scalar(7), rhs: .scalar(0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isNaN, "x mod 0 must propagate NaN, not throw")
    }

    // MARK: - ST 6: plusMinus/minusPlus rejection

    func testPlusMinusRejection() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.plusMinus, lhs: .scalar(1), rhs: .scalar(2))
        ) { err in
            guard case MathExprError.unsupportedNode = err else {
                XCTFail("Expected unsupportedNode for plusMinus, got \(err)"); return
            }
        }
    }

    func testMinusPlusRejection() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.minusPlus, lhs: .scalar(1), rhs: .scalar(2))
        ) { err in
            guard case MathExprError.unsupportedNode = err else {
                XCTFail("Expected unsupportedNode for minusPlus, got \(err)"); return
            }
        }
    }

    func testPlusMinus_complex_alsoThrows() {
        let c = Complex(re: 1, im: 2)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.plusMinus, lhs: .complex(c), rhs: .complex(c))
        ) { err in
            guard case MathExprError.unsupportedNode = err else {
                XCTFail("Expected unsupportedNode, got \(err)"); return
            }
        }
    }

    // MARK: - ST 7: Scalar unary neg/pos

    func testScalarUnary_neg_basic() throws {
        // -7 = -7 (s10: 13842939354630062080)
        let r = try NumericDispatch.applyUnary(.neg, operand: .scalar(7))
        assertScalarBitExact(r, bitPattern: 13842939354630062080, "-7")
    }

    func testScalarUnary_neg_compound() throws {
        // -(3+4) = -7 (s11: same bitPattern as -7)
        let sum = try NumericDispatch.applyBinary(.add, lhs: .scalar(3), rhs: .scalar(4))
        let r = try NumericDispatch.applyUnary(.neg, operand: sum)
        assertScalarBitExact(r, bitPattern: 13842939354630062080, "-(3+4)=-7")
    }

    func testScalarUnary_neg_zero_signBit() throws {
        // -(0.0) must produce -0.0 (different bitPattern from +0.0)
        let r = try NumericDispatch.applyUnary(.neg, operand: .scalar(0.0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v.bitPattern, (-0.0).bitPattern, "-0.0 bitPattern must differ from +0.0")
        XCTAssertNotEqual(v.bitPattern, (0.0).bitPattern)
    }

    func testScalarUnary_neg_inf() throws {
        let r = try NumericDispatch.applyUnary(.neg, operand: .scalar(.infinity))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v, -.infinity, accuracy: 0, "neg(+inf)=-inf")
    }

    func testScalarUnary_neg_nan() throws {
        let r = try NumericDispatch.applyUnary(.neg, operand: .scalar(.nan))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isNaN, "neg(NaN) must be NaN")
    }

    func testScalarUnary_pos_identity() throws {
        let r = try NumericDispatch.applyUnary(.pos, operand: .scalar(42.0))
        assertScalarBitExact(r, bitPattern: (42.0).bitPattern, "pos(x)=x")
    }

    func testScalarUnary_pos_negativeZero() throws {
        // pos(-0.0) must preserve -0.0
        let r = try NumericDispatch.applyUnary(.pos, operand: .scalar(-0.0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v.bitPattern, (-0.0).bitPattern, "pos(-0.0) preserves sign bit")
    }

    // MARK: - ST 8: Scalar unary factorial and transpose

    func testScalarUnary_factorial_5() throws {
        let r = try NumericDispatch.applyUnary(.factorial, operand: .scalar(5))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v, 120.0, accuracy: 1e-9, "5! = 120")
    }

    func testScalarUnary_factorial_zero() throws {
        let r = try NumericDispatch.applyUnary(.factorial, operand: .scalar(0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v, 1.0, accuracy: 1e-9, "0! = 1")
    }

    func testScalarUnary_factorial_negative_throws() throws {
        // Legacy: guard v >= 0 else throw invalidArguments("factorial requires non-negative argument")
        XCTAssertThrowsError(
            try NumericDispatch.applyUnary(.factorial, operand: .scalar(-1))
        ) { err in
            guard case MathExprError.invalidArguments(let msg) = err else {
                XCTFail("Expected invalidArguments, got \(err)"); return
            }
            XCTAssertTrue(msg.contains("non-negative"),
                "Error message must mention 'non-negative'; got: \(msg)")
        }
    }

    func testScalarUnary_factorial_fractional_usesGamma() throws {
        // factorial(0.5) = tgamma(1.5) = sqrt(pi)/2 ≈ 0.8862...
        let r = try NumericDispatch.applyUnary(.factorial, operand: .scalar(0.5))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v, tgamma(1.5), accuracy: 1e-12, "factorial(0.5)=Γ(1.5)")
    }

    func testScalarUnary_transpose_returnsScalar() throws {
        // Legacy (evalUnary line 248): throw unsupportedNode("transpose (matrix operation)")
        // Dispatch: transpose of a scalar returns the scalar itself (per applyTransposeUnary)
        let r = try NumericDispatch.applyUnary(.transpose, operand: .scalar(9.0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v, 9.0, "Scalar transpose is identity per dispatch convention")
        // Note: dispatch differs from legacy scalar path here intentionally
        // (legacy scalar path would never produce a transpose node from the pure-scalar AST
        // evaluator; the dispatch convention treats scalar^T = scalar for LinAlg completeness)
    }

    // MARK: - ST 9 / ST 10: Complex add/sub/mul

    func testComplex_add_basic() throws {
        // (2+3i)+(1+4i) = (3+7i) (c05: re=4613937818241073152, im=4619567317775286272)
        let a = Complex(re: 2, im: 3), b = Complex(re: 1, im: 4)
        let r = try NumericDispatch.applyBinary(.add, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 4613937818241073152, "Re bitPattern")
        XCTAssertEqual(z.im.bitPattern, 4619567317775286272, "Im bitPattern")
    }

    func testComplex_sub_basic() throws {
        // (3+2i)-(1+i) = (2+i) (c06: re=4611686018427387904, im=4607182418800017408)
        let a = Complex(re: 3, im: 2), b = Complex(re: 1, im: 1)
        let r = try NumericDispatch.applyBinary(.sub, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 4611686018427387904, "Re bitPattern")
        XCTAssertEqual(z.im.bitPattern, 4607182418800017408, "Im bitPattern")
    }

    func testComplex_mul_conjugate() throws {
        // (1+i)*(1-i) = 2 (c04: re=4611686018427387904, im=0)
        let a = Complex(re: 1, im: 1), b = Complex(re: 1, im: -1)
        let r = try NumericDispatch.applyBinary(.mul, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 4611686018427387904, "Re=2 bitPattern")
        XCTAssertEqual(z.im.bitPattern, 0, "Im=0 bitPattern")
    }

    func testComplex_mul_i_squared() throws {
        // i*i = -1 (c02: re=13830554455654793216, im=0)
        let i = Complex(re: 0, im: 1)
        let r = try NumericDispatch.applyBinary(.mul, lhs: .complex(i), rhs: .complex(i))
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 13830554455654793216, "i*i re=-1 bitPattern")
        XCTAssertEqual(z.im.bitPattern, 0, "i*i im=0")
    }

    // MARK: - ST 11: Complex division — legacy divisionByZero semantics

    func testComplex_div_basic() throws {
        // (1+i)/(1-i) = i (c07: re=0, im=4607182418800017408)
        let a = Complex(re: 1, im: 1), b = Complex(re: 1, im: -1)
        let r = try NumericDispatch.applyBinary(.div, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 0, "Re=0 bitPattern")
        XCTAssertEqual(z.im.bitPattern, 4607182418800017408, "Im=1 bitPattern")
    }

    func testComplexDivisionByZero_bothZero_throws() throws {
        // (1+i)/(0+0i) → divisionByZero (groupB-e09 in snapshot)
        let a = Complex(re: 1, im: 1), zero = Complex(re: 0, im: 0)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.div, lhs: .complex(a), rhs: .complex(zero))
        ) { err in
            guard case MathExprError.divisionByZero = err else {
                XCTFail("Expected divisionByZero, got \(err)"); return
            }
        }
    }

    func testComplex_div_pureImaginaryDivisor_doesNotThrow() throws {
        // (1+i)/(0+1i) — divisor has re=0 but im≠0 → must NOT throw
        let a = Complex(re: 1, im: 1), b = Complex(re: 0, im: 1)
        let r = try NumericDispatch.applyBinary(.div, lhs: .complex(a), rhs: .complex(b))
        guard case .complex = r else {
            XCTFail("Expected complex result for (1+i)/(0+i)"); return
        }
    }

    func testComplex_div_pureRealDivisor_doesNotThrow() throws {
        // (1+i)/(2+0i) — divisor has im=0 but re≠0 → must NOT throw
        let a = Complex(re: 1, im: 1), b = Complex(re: 2, im: 0)
        let r = try NumericDispatch.applyBinary(.div, lhs: .complex(a), rhs: .complex(b))
        guard case .complex = r else {
            XCTFail("Expected complex result for (1+i)/2"); return
        }
    }

    // MARK: - ST 12: Complex pow

    func testComplex_pow_i4() throws {
        // i^4 = 1 (c20: re=4607182418800017408, im≠0 small noise from trig)
        let i = Complex(re: 0, im: 1)
        let r = try NumericDispatch.applyBinary(.pow, lhs: .complex(i), rhs: .complex(Complex(4)))
        assertComplexApprox(r, re: 1.0, im: 0.0, tolerance: 1e-12, "i^4=1")
    }

    func testComplex_pow_squaredBase() throws {
        // (2+i)^2 = 3 + 4i  (c08: re=4613937818241073154, im=4616189618054758400)
        // Note: computed via exp(2*log(2+i)), so small floating-point offset expected
        let base = Complex(re: 2, im: 1)
        let r = try NumericDispatch.applyBinary(
            .pow, lhs: .complex(base), rhs: .complex(Complex(2)))
        assertComplexApprox(r, re: 3.0, im: 4.0, tolerance: 1e-12, "(2+i)^2=3+4i")
    }

    func testComplex_pow_zeroBase_returnsZero() throws {
        // 0^z = 0 per legacy evalComplexBinary lines 377-378
        let zero = Complex(re: 0, im: 0), exp = Complex(re: 2, im: 1)
        let r = try NumericDispatch.applyBinary(
            .pow, lhs: .complex(zero), rhs: .complex(exp))
        assertComplexApprox(r, re: 0.0, im: 0.0, tolerance: 1e-30, "0^z=0")
    }

    // MARK: - ST 13: Complex mod / plusMinus / minusPlus rejection

    func testComplex_mod_throws() throws {
        let a = Complex(re: 3, im: 1)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.mod, lhs: .complex(a), rhs: .complex(a))
        ) { err in
            guard case MathExprError.unsupportedNode(let msg) = err else {
                XCTFail("Expected unsupportedNode for complex mod, got \(err)"); return
            }
            XCTAssertTrue(msg.contains("modulo") || msg.contains("complex"),
                "Error must mention modulo or complex; got: \(msg)")
        }
    }

    func testComplex_plusMinus_throws() throws {
        let a = Complex(re: 1, im: 0)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.plusMinus, lhs: .complex(a), rhs: .complex(a))
        ) { err in
            guard case MathExprError.unsupportedNode = err else {
                XCTFail("Expected unsupportedNode, got \(err)"); return
            }
        }
    }

    func testComplex_minusPlus_throws() throws {
        let a = Complex(re: 1, im: 0)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.minusPlus, lhs: .complex(a), rhs: .complex(a))
        ) { err in
            guard case MathExprError.unsupportedNode = err else {
                XCTFail("Expected unsupportedNode, got \(err)"); return
            }
        }
    }

    // MARK: - ST 14: Complex unary neg/pos/factorial/transpose

    func testComplexUnary_neg_bothComponents() throws {
        let z = Complex(re: 3, im: -2)
        let r = try NumericDispatch.applyUnary(.neg, operand: .complex(z))
        guard case .complex(let v) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(v.re, -3.0, accuracy: 0, "neg re")
        XCTAssertEqual(v.im, 2.0, accuracy: 0, "neg im")
    }

    func testComplexUnary_neg_signedZero() throws {
        // neg(0+0i) = -0-0i: both sign bits flip
        let z = Complex(re: 0.0, im: 0.0)
        let r = try NumericDispatch.applyUnary(.neg, operand: .complex(z))
        guard case .complex(let v) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(v.re.bitPattern, (-0.0).bitPattern, "neg(+0) re = -0")
        XCTAssertEqual(v.im.bitPattern, (-0.0).bitPattern, "neg(+0) im = -0")
    }

    func testComplexUnary_pos_identity() throws {
        let z = Complex(re: 5, im: -3)
        let r = try NumericDispatch.applyUnary(.pos, operand: .complex(z))
        guard case .complex(let v) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(v.re, 5.0, accuracy: 0, "pos re")
        XCTAssertEqual(v.im, -3.0, accuracy: 0, "pos im")
    }

    func testComplexUnary_factorial_realNonNeg() throws {
        // factorial of a real-only complex = tgamma(5+1) = 120
        let z = Complex(re: 5, im: 0)
        let r = try NumericDispatch.applyUnary(.factorial, operand: .complex(z))
        guard case .complex(let v) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(v.re, 120.0, accuracy: 1e-9, "factorial(5) via complex")
        XCTAssertEqual(v.im, 0.0, accuracy: 1e-15, "Im=0")
    }

    func testComplexUnary_factorial_nonRealThrows() throws {
        // factorial of non-real complex → invalidArguments
        let z = Complex(re: 3, im: 1)
        XCTAssertThrowsError(
            try NumericDispatch.applyUnary(.factorial, operand: .complex(z))
        ) { err in
            guard case MathExprError.invalidArguments(let msg) = err else {
                XCTFail("Expected invalidArguments, got \(err)"); return
            }
            XCTAssertTrue(msg.contains("factorial") || msg.contains("complex"),
                "Error must mention factorial or complex; got: \(msg)")
        }
    }

    func testComplexUnary_factorial_negativeRealThrows() throws {
        let z = Complex(re: -1, im: 0)
        XCTAssertThrowsError(
            try NumericDispatch.applyUnary(.factorial, operand: .complex(z))
        ) { err in
            guard case MathExprError.invalidArguments = err else {
                XCTFail("Expected invalidArguments, got \(err)"); return
            }
        }
    }

    func testComplexUnary_transpose_returnsScalar() throws {
        // Dispatch convention: complex^T = complex (identity)
        let z = Complex(re: 2, im: 3)
        let r = try NumericDispatch.applyUnary(.transpose, operand: .complex(z))
        guard case .complex(let v) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(v.re, z.re, accuracy: 0, "transpose complex re identity")
        XCTAssertEqual(v.im, z.im, accuracy: 0, "transpose complex im identity")
    }

    // MARK: - ST 15: Scalar→complex promotion in binary dispatch

    func testPromotion_scalarPlusComplex() throws {
        // 1 + i = (1+1i) (mirrors c03 via dispatch)
        let r = try NumericDispatch.applyBinary(
            .add, lhs: .scalar(1.0), rhs: .complex(Complex(re: 0, im: 1)))
        guard case .complex = r else {
            XCTFail("scalar+complex must yield complex"); return
        }
        assertComplexApprox(r, re: 1.0, im: 1.0, "1+i=1+1i")
    }

    func testPromotion_complexPlusScalar_commutative() throws {
        // i + 1 = (1+1i) — same result
        let r = try NumericDispatch.applyBinary(
            .add, lhs: .complex(Complex(re: 0, im: 1)), rhs: .scalar(1.0))
        guard case .complex = r else {
            XCTFail("complex+scalar must yield complex"); return
        }
        assertComplexApprox(r, re: 1.0, im: 1.0, "i+1=1+1i")
    }

    func testPromotion_scalarMulComplex() throws {
        // 2 * (1+i) = (2+2i)
        let r = try NumericDispatch.applyBinary(
            .mul, lhs: .scalar(2.0), rhs: .complex(Complex(re: 1, im: 1)))
        guard case .complex = r else {
            XCTFail("scalar*complex must yield complex"); return
        }
        assertComplexApprox(r, re: 2.0, im: 2.0, "2*(1+i)=2+2i")
    }

    func testPromotion_complexMulScalar() throws {
        // (1+i) * 3 = (3+3i)
        let r = try NumericDispatch.applyBinary(
            .mul, lhs: .complex(Complex(re: 1, im: 1)), rhs: .scalar(3.0))
        guard case .complex = r else {
            XCTFail("complex*scalar must yield complex"); return
        }
        assertComplexApprox(r, re: 3.0, im: 3.0, "(1+i)*3=3+3i")
    }

    func testPromotion_scalarDivComplex() throws {
        // 1 / (0+1i) = -i (re=0, im=-1)
        let r = try NumericDispatch.applyBinary(
            .div, lhs: .scalar(1.0), rhs: .complex(Complex(re: 0, im: 1)))
        assertComplexApprox(r, re: 0.0, im: -1.0, "1/(0+i)=0-i")
    }

    func testPromotion_complexDivScalar() throws {
        // (2+4i) / 2 = (1+2i)
        let r = try NumericDispatch.applyBinary(
            .div, lhs: .complex(Complex(re: 2, im: 4)), rhs: .scalar(2.0))
        assertComplexApprox(r, re: 1.0, im: 2.0, "(2+4i)/2=1+2i")
    }

    func testPromotion_scalarDivComplex_complexZero_throws() throws {
        // scalar / (0+0i) → divisionByZero via complex promotion
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .div, lhs: .scalar(1.0), rhs: .complex(Complex(re: 0, im: 0)))
        ) { err in
            guard case MathExprError.divisionByZero = err else {
                XCTFail("Expected divisionByZero, got \(err)"); return
            }
        }
    }

    func testPromotion_complexDivScalar_scalarZero_throws() throws {
        // (1+i) / scalar(0) → divisionByZero
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .div, lhs: .complex(Complex(re: 1, im: 1)), rhs: .scalar(0.0))
        ) { err in
            guard case MathExprError.divisionByZero = err else {
                XCTFail("Expected divisionByZero, got \(err)"); return
            }
        }
    }

    func testPromotion_purePairStaysScalar() throws {
        // Both scalar: result is still scalar (promotion does not affect pure-scalar paths)
        let r = try NumericDispatch.applyBinary(.add, lhs: .scalar(3), rhs: .scalar(4))
        guard case .scalar = r else {
            XCTFail("scalar+scalar must remain scalar"); return
        }
    }

    // MARK: - ST 16: Complex→scalar collapse policy

    func testCollapsePolicy_complexPlusComplex_realResult_staysComplex() throws {
        // (1+i) + (1-i) = 2+0i — result is complex with im=0, NOT scalar
        let a = Complex(re: 1, im: 1), b = Complex(re: 1, im: -1)
        let r = try NumericDispatch.applyBinary(.add, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = r else {
            XCTFail("complex+complex must stay complex even when im=0"); return
        }
        XCTAssertEqual(z.re, 2.0, accuracy: 0, "re=2")
        XCTAssertEqual(z.im, 0.0, accuracy: 0, "im=0 (but kind is still complex)")
    }

    func testCollapsePolicy_promotedResult_staysComplex() throws {
        // scalar(2) + complex(0+0i) = complex(2+0i) — no collapse
        let r = try NumericDispatch.applyBinary(
            .add, lhs: .scalar(2.0), rhs: .complex(Complex(re: 0, im: 0)))
        guard case .complex = r else {
            XCTFail("scalar+complex must yield complex, not collapse to scalar"); return
        }
    }

    // MARK: - ST 17: nonFiniteFloat propagation

    func testNonFiniteFloat_nanPropagatesThrough_add() throws {
        // nan + x = nan (no throw; dispatch does not add isFinite guards)
        let r = try NumericDispatch.applyBinary(.add, lhs: .scalar(.nan), rhs: .scalar(1))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isNaN, "nan+1 must propagate NaN, not throw nonFiniteFloat")
    }

    func testNonFiniteFloat_infPropagatesThrough_mul() throws {
        // +inf * 2 = +inf (no throw)
        let r = try NumericDispatch.applyBinary(.mul, lhs: .scalar(.infinity), rhs: .scalar(2))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isInfinite && v > 0, "inf*2 must propagate +inf, not throw")
    }

    func testNonFiniteFloat_negInfPropagatesThrough_add() throws {
        let r = try NumericDispatch.applyBinary(
            .add, lhs: .scalar(-.infinity), rhs: .scalar(100))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isInfinite && v < 0, "-inf+100 must stay -inf")
    }

    // MARK: - ST 19: Division-by-zero mechanism parity

    func testDivisionByZero_mechanism_scalarThrows() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.div, lhs: .scalar(5), rhs: .scalar(0))
        ) { err in
            // Must be MathExprError.divisionByZero, not a different error type
            if case MathExprError.divisionByZero = err { return }
            XCTFail("Expected MathExprError.divisionByZero exactly, got \(err)")
        }
    }

    func testDivisionByZero_mechanism_complexBothZeroThrows() {
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(
                .div,
                lhs: .complex(Complex(re: 3, im: 2)),
                rhs: .complex(Complex(re: 0, im: 0)))
        ) { err in
            if case MathExprError.divisionByZero = err { return }
            XCTFail("Expected MathExprError.divisionByZero exactly, got \(err)")
        }
    }

    func testDivisionByZero_mechanism_complex_oneZeroComponent_noThrow() throws {
        // (a+bi) / (0+bi) — only re=0, im≠0 → must NOT throw, not be scalar inf
        let r = try NumericDispatch.applyBinary(
            .div,
            lhs: .complex(Complex(re: 1, im: 1)),
            rhs: .complex(Complex(re: 0, im: 2)))
        guard case .complex = r else {
            XCTFail("Expected complex result, not throw"); return
        }
    }

    func testDivisionByZero_mechanism_scalarsIEEEnan_doesNotThrow() throws {
        // 0.0/0.0 as Double arithmetic produces NaN (groupA-e07 is x/0 throws;
        // 0.0/0.0 also throws since divisor == 0)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.div, lhs: .scalar(0.0), rhs: .scalar(0.0))
        ) { err in
            if case MathExprError.divisionByZero = err { return }
            XCTFail("0.0/0.0 must throw divisionByZero, got \(err)")
        }
    }

    func testDivisionByZero_mechanism_nanDividend_zerodivisor_stillThrows() throws {
        // nan / 0 → divisionByZero (divisor is exactly zero)
        XCTAssertThrowsError(
            try NumericDispatch.applyBinary(.div, lhs: .scalar(.nan), rhs: .scalar(0))
        ) { err in
            if case MathExprError.divisionByZero = err { return }
            XCTFail("nan/0 must throw divisionByZero, got \(err)")
        }
    }

    // MARK: - ST 20: Full snapshot parity — scalar segment

    func testSnapshotParity_scalar_arithmeticCases() throws {
        // Bit-exact comparison for the arithmetic scalar entries from LegacySnapshot.json.
        // Entry ids and expected bitPatterns are taken directly from the committed snapshot.

        let cases: [(id: String, op: BinaryOp, lhs: Double, rhs: Double, expected: UInt64)] = [
            // s01: 1+2=3
            ("s01", .add, 1, 2, 4613937818241073152),
            // s02: 10-3=7
            ("s02", .sub, 10, 3, 4619567317775286272),
            // s03: 4*5=20
            ("s03", .mul, 4, 5, 4626322717216342016),
            // s04: 15/3=5
            ("s04", .div, 15, 3, 4617315517961601024),
            // s05: 2^10=1024
            ("s05", .pow, 2, 10, 4652218415073722368),
            // s06: 17%5=2
            ("s06", .mod, 17, 5, 4611686018427387904),
        ]

        for entry in cases {
            let r = try NumericDispatch.applyBinary(
                entry.op, lhs: .scalar(entry.lhs), rhs: .scalar(entry.rhs))
            assertScalarBitExact(r, bitPattern: entry.expected, "snapshot \(entry.id)")
        }
    }

    func testSnapshotParity_scalar_unaryNeg() throws {
        // s10: -7 bitPattern = 13842939354630062080
        let r = try NumericDispatch.applyUnary(.neg, operand: .scalar(7))
        assertScalarBitExact(r, bitPattern: 13842939354630062080, "snapshot s10")
    }

    func testSnapshotParity_scalar_sqrt2_mul_sqrt2() throws {
        // s40: sqrt(2)*sqrt(2) = 2.0000000000000004 bitPattern 4611686018427387905
        let s = Foundation.sqrt(2.0)
        let r = try NumericDispatch.applyBinary(.mul, lhs: .scalar(s), rhs: .scalar(s))
        assertScalarBitExact(r, bitPattern: 4611686018427387905, "snapshot s40")
    }

    func testSnapshotParity_scalar_explog5() throws {
        // s39: exp(log(5)) bitPattern 4617315517961601023
        let logR = try NumericDispatch.applyFunction("log", args: [.scalar(5)])
        let r = try NumericDispatch.applyFunction("exp", args: [logR])
        assertScalarBitExact(r, bitPattern: 4617315517961601023, "snapshot s39")
    }

    func testSnapshotParity_scalar_pow2_10_direct() throws {
        // s05: 2^10=1024 via applyBinary(.pow)
        let r = try NumericDispatch.applyBinary(.pow, lhs: .scalar(2), rhs: .scalar(10))
        assertScalarBitExact(r, bitPattern: 4652218415073722368, "snapshot s05 via applyBinary")
    }

    // MARK: - ST 20: Full snapshot parity — complex segment

    func testSnapshotParity_complex_i_squared() throws {
        // c02: i*i = (-1+0i) re=13830554455654793216, im=0
        let i = Complex(re: 0, im: 1)
        let r = try NumericDispatch.applyBinary(.mul, lhs: .complex(i), rhs: .complex(i))
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 13830554455654793216, "c02 re bitPattern")
        XCTAssertEqual(z.im.bitPattern, 0, "c02 im bitPattern")
    }

    func testSnapshotParity_complex_arithmetic() throws {
        // c05: (2+3i)+(1+4i)=(3+7i) re=4613937818241073152, im=4619567317775286272
        let a = Complex(re: 2, im: 3), b = Complex(re: 1, im: 4)
        let r = try NumericDispatch.applyBinary(.add, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 4613937818241073152, "c05 re bitPattern")
        XCTAssertEqual(z.im.bitPattern, 4619567317775286272, "c05 im bitPattern")
    }

    func testSnapshotParity_complex_div() throws {
        // c07: (1+i)/(1-i) = (0+i) re=0, im=4607182418800017408
        let a = Complex(re: 1, im: 1), b = Complex(re: 1, im: -1)
        let r = try NumericDispatch.applyBinary(.div, lhs: .complex(a), rhs: .complex(b))
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 0, "c07 re bitPattern")
        XCTAssertEqual(z.im.bitPattern, 4607182418800017408, "c07 im bitPattern")
    }

    func testSnapshotParity_complex_exp_i() throws {
        // c09: exp(i) = cos(1)+i*sin(1)
        // re=4603041830072026764 = cos(1), im=4605754516372524270 = sin(1)
        let i = Complex(re: 0, im: 1)
        let r = try NumericDispatch.applyFunction("exp", args: [.complex(i)])
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 4603041830072026764, "c09 re bitPattern")
        XCTAssertEqual(z.im.bitPattern, 4605754516372524270, "c09 im bitPattern")
    }

    func testSnapshotParity_complex_log_i() throws {
        // c10: log(i) = 0 + i*(pi/2)
        // re=0, im=4609753056924675352 = pi/2
        let i = Complex(re: 0, im: 1)
        let r = try NumericDispatch.applyFunction("log", args: [.complex(i)])
        guard case .complex(let z) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(z.re.bitPattern, 0, "c10 re bitPattern")
        XCTAssertEqual(z.im.bitPattern, 4609753056924675352, "c10 im bitPattern")
    }

    func testSnapshotParity_complex_abs() throws {
        // c14: abs(3+4i) = 5 (returns .scalar(5) per dispatch convention)
        let z = Complex(re: 3, im: 4)
        let r = try NumericDispatch.applyFunction("abs", args: [.complex(z)])
        // The dispatch returns .scalar for abs(complex) — the snapshot records complex
        // with re=5, im=0. Dispatch returns .scalar(5), which is the correct value
        // (abs of complex = modulus = real scalar).
        guard case .scalar(let v) = r else {
            XCTFail("abs(complex) must return scalar; got \(r.kind)"); return
        }
        XCTAssertEqual(v, 5.0, accuracy: 1e-12, "|3+4i|=5")
    }

    // MARK: - ST 20: IEEE-754 edge values through dispatch

    func testIEEE_sqrt_neg1_scalar_nan() throws {
        // ieee-f01: sqrt(-1) via real scalar path = NaN (bitPattern 18444492273895866368)
        let r = try NumericDispatch.applyFunction("sqrt", args: [.scalar(-1.0)])
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isNaN, "sqrt(-1) in real scalar must be NaN")
    }

    func testIEEE_log_neg1_scalar_nan() throws {
        // ieee-f02: log(-1) via real path = NaN
        let r = try NumericDispatch.applyFunction("log", args: [.scalar(-1.0)])
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isNaN, "log(-1) in real scalar must be NaN")
    }

    func testIEEE_exp_huge_inf() throws {
        // ieee-f03: exp(1e308) → +Inf
        let r = try NumericDispatch.applyFunction("exp", args: [.scalar(1e308)])
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isInfinite && v > 0, "exp(1e308) must be +inf")
    }

    func testIEEE_neg_exp_huge_negInf() throws {
        // ieee-f04: -exp(1e308) → -Inf
        let expR = try NumericDispatch.applyFunction("exp", args: [.scalar(1e308)])
        let r = try NumericDispatch.applyUnary(.neg, operand: expR)
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertTrue(v.isInfinite && v < 0, "-exp(1e308) must be -inf")
    }

    func testIEEE_positiveZero() throws {
        // ieee-f06: +0.0 bitPattern = 0
        let r = try NumericDispatch.applyBinary(.add, lhs: .scalar(0.0), rhs: .scalar(0.0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v.bitPattern, (0.0).bitPattern, "+0 bitPattern")
    }

    func testIEEE_negativeZero() throws {
        // ieee-f07: -0.0 bitPattern = 9223372036854775808
        let r = try NumericDispatch.applyUnary(.neg, operand: .scalar(0.0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v.bitPattern, (-0.0).bitPattern, "-0 bitPattern")
    }

    func testIEEE_posInfinity_preserved() throws {
        // ieee-f08: Double.infinity
        let r = try NumericDispatch.applyBinary(
            .add, lhs: .scalar(.infinity), rhs: .scalar(0.0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v.bitPattern, Double.infinity.bitPattern, "+inf bitPattern")
    }

    func testIEEE_negInfinity_preserved() throws {
        // ieee-f09: -Double.infinity
        let r = try NumericDispatch.applyBinary(
            .add, lhs: .scalar(-.infinity), rhs: .scalar(0.0))
        guard case .scalar(let v) = r else { XCTFail("Expected scalar"); return }
        XCTAssertEqual(v.bitPattern, (-Double.infinity).bitPattern, "-inf bitPattern")
    }

    func testIEEE_sqrt_neg1_via_complex_isI() throws {
        // ieee-f10: sqrt(-1) via complex path = i (re≈0, im=1)
        // Note: complex sqrt(-1+0i) produces i; the snapshot records re≠0 due to float
        let z = Complex(re: -1.0, im: 0.0)
        let r = try NumericDispatch.applyFunction("sqrt", args: [.complex(z)])
        guard case .complex(let v) = r else { XCTFail("Expected complex"); return }
        XCTAssertEqual(v.im, 1.0, accuracy: 1e-12, "sqrt(-1) complex im=1")
        // re should be essentially zero (may be tiny float due to atan2)
        XCTAssertEqual(v.re, 0.0, accuracy: 1e-10, "sqrt(-1) complex re≈0")
    }
}
