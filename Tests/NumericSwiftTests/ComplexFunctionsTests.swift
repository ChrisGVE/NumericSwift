//
//  ComplexFunctionsTests.swift
//  Tests/NumericSwiftTests/
//
//  Coverage tests for Complex.swift: construction, equality, magnitude/abs/arg,
//  conjugate, exp/log/sqrt, trig + hyperbolic functions, inverse functions,
//  pow, polar, branch cuts, free functions, and conformances.
//
//  SCOPE GUARD: Does NOT cover division / reciprocal — those are owned by the
//  sibling fix for issue #3 (Smith's algorithm). See ComplexDivisionTests.swift.
//
//  Oracle: numpy.complex128 / scipy (principal branches):
//    sqrt(-1)  = +1j         (NOT -1j)
//    log(-1)   = +πj         (NOT -πj)
//    asin(2)   = π/2 - i·ln(2+√3)
//  Branch: fix/issue-19-coverage  Refs #19
//

import XCTest
@testable import NumericSwift

final class ComplexFunctionsTests: XCTestCase {

    // MARK: - Helpers

    private let eps = 1e-12
    private let looseEps = 1e-10

    private func assertComplexClose(
        _ z: Complex, re expectedRe: Double, im expectedIm: Double,
        tol: Double = 1e-12, _ msg: String = "",
        file: StaticString = #file, line: UInt = #line
    ) {
        XCTAssertEqual(z.re, expectedRe, accuracy: tol, "\(msg) .re", file: file, line: line)
        XCTAssertEqual(z.im, expectedIm, accuracy: tol, "\(msg) .im", file: file, line: line)
    }

    private func assertComplexNaN(_ z: Complex, _ msg: String = "",
                                  file: StaticString = #file, line: UInt = #line) {
        XCTAssertTrue(z.re.isNaN || z.im.isNaN, "Expected NaN complex — \(msg)", file: file, line: line)
    }

    // MARK: - Construction

    func testInit_reAndIm() {
        let z = Complex(re: 3.0, im: 4.0)
        XCTAssertEqual(z.re, 3.0)
        XCTAssertEqual(z.im, 4.0)
    }

    func testInit_realOnly() {
        let z = Complex(5.0)
        XCTAssertEqual(z.re, 5.0)
        XCTAssertEqual(z.im, 0.0)
    }

    func testInit_floatLiteral() {
        let z: Complex = 7.5
        XCTAssertEqual(z.re, 7.5)
        XCTAssertEqual(z.im, 0.0)
    }

    func testInit_integerLiteral() {
        let z: Complex = 3
        XCTAssertEqual(z.re, 3.0)
        XCTAssertEqual(z.im, 0.0)
    }

    // MARK: - Static constants

    func testStaticConstants_i() {
        XCTAssertEqual(Complex.i.re, 0.0)
        XCTAssertEqual(Complex.i.im, 1.0)
    }

    func testStaticConstants_zero() {
        XCTAssertEqual(Complex.zero.re, 0.0)
        XCTAssertEqual(Complex.zero.im, 0.0)
    }

    func testStaticConstants_one() {
        XCTAssertEqual(Complex.one.re, 1.0)
        XCTAssertEqual(Complex.one.im, 0.0)
    }

    // MARK: - Polar construction

    func testPolar_unitCircle_zero() {
        // r=1, θ=0  → (1, 0)
        let z = Complex.polar(r: 1.0, theta: 0.0)
        assertComplexClose(z, re: 1.0, im: 0.0)
    }

    func testPolar_unitCircle_quarterPi() {
        // r=1, θ=π/4  → (√2/2, √2/2)
        let z = Complex.polar(r: 1.0, theta: .pi / 4)
        let s = 2.0.squareRoot() / 2.0
        assertComplexClose(z, re: s, im: s)
    }

    func testPolar_unitCircle_halfPi() {
        // r=1, θ=π/2  → (0, 1)  = i
        let z = Complex.polar(r: 1.0, theta: .pi / 2)
        assertComplexClose(z, re: 0.0, im: 1.0, tol: 1e-15)
    }

    func testPolar_unitCircle_pi() {
        // r=1, θ=π  → (-1, 0)
        let z = Complex.polar(r: 1.0, theta: .pi)
        assertComplexClose(z, re: -1.0, im: 0.0, tol: 1e-15)
    }

    func testPolar_withRadius() {
        // r=2, θ=π/2  → (0, 2)
        let z = Complex.polar(r: 2.0, theta: .pi / 2)
        assertComplexClose(z, re: 0.0, im: 2.0, tol: 1e-15)
    }

    func testPolar_zeroRadius() {
        let z = Complex.polar(r: 0.0, theta: 1.23)
        assertComplexClose(z, re: 0.0, im: 0.0, tol: 1e-15)
    }

    // MARK: - Equality

    func testEquality_sameValue() {
        let z1 = Complex(re: 1.0, im: 2.0)
        let z2 = Complex(re: 1.0, im: 2.0)
        XCTAssertEqual(z1, z2)
    }

    func testEquality_differentRe() {
        XCTAssertNotEqual(Complex(re: 1.0, im: 2.0), Complex(re: 3.0, im: 2.0))
    }

    func testEquality_differentIm() {
        XCTAssertNotEqual(Complex(re: 1.0, im: 2.0), Complex(re: 1.0, im: 3.0))
    }

    func testEquality_zero() {
        XCTAssertEqual(Complex.zero, Complex(re: 0.0, im: 0.0))
    }

    // MARK: - Magnitude / abs / abs2 / arg

    func testAbs_pythagorean_3_4_5() {
        // |3 + 4i| = 5
        XCTAssertEqual(Complex(re: 3.0, im: 4.0).abs, 5.0, accuracy: eps)
    }

    func testAbs_zero() {
        XCTAssertEqual(Complex.zero.abs, 0.0)
    }

    func testAbs_pureReal_positive() {
        XCTAssertEqual(Complex(re: 7.0, im: 0.0).abs, 7.0, accuracy: eps)
    }

    func testAbs_pureReal_negative() {
        // |−7 + 0i| = 7
        XCTAssertEqual(Complex(re: -7.0, im: 0.0).abs, 7.0, accuracy: eps)
    }

    func testAbs_pureImaginary() {
        XCTAssertEqual(Complex(re: 0.0, im: 3.0).abs, 3.0, accuracy: eps)
    }

    func testAbs_largeValues() {
        // hypot avoids overflow; check hypot(3e150, 4e150) = 5e150
        let z = Complex(re: 3e150, im: 4e150)
        XCTAssertEqual(z.abs, 5e150, accuracy: 5e150 * 1e-14)
    }

    func testAbs2_identity() {
        let z = Complex(re: 3.0, im: 4.0)
        XCTAssertEqual(z.abs2, 25.0, accuracy: eps)
    }

    func testArg_realAxis() {
        // arg(1 + 0i) = 0
        XCTAssertEqual(Complex(re: 1.0, im: 0.0).arg, 0.0, accuracy: eps)
    }

    func testArg_negativeReal() {
        // arg(-1 + 0i) = π
        XCTAssertEqual(Complex(re: -1.0, im: 0.0).arg, .pi, accuracy: eps)
    }

    func testArg_pureImaginaryPositive() {
        // arg(0 + i) = π/2
        XCTAssertEqual(Complex(re: 0.0, im: 1.0).arg, .pi / 2, accuracy: eps)
    }

    func testArg_pureImaginaryNegative() {
        // arg(0 - i) = -π/2
        XCTAssertEqual(Complex(re: 0.0, im: -1.0).arg, -.pi / 2, accuracy: eps)
    }

    func testArg_firstQuadrant() {
        // arg(1 + i) = π/4
        XCTAssertEqual(Complex(re: 1.0, im: 1.0).arg, .pi / 4, accuracy: eps)
    }

    func testArg_thirdQuadrant() {
        // arg(-1 - i) = -3π/4
        XCTAssertEqual(Complex(re: -1.0, im: -1.0).arg, -3 * .pi / 4, accuracy: eps)
    }

    // MARK: - Conjugate

    func testConj_basicValue() {
        let z = Complex(re: 3.0, im: 4.0)
        assertComplexClose(z.conj, re: 3.0, im: -4.0)
    }

    func testConj_pureReal() {
        let z = Complex(re: 5.0, im: 0.0)
        XCTAssertEqual(z.conj, z)
    }

    func testConj_ofConj_isOriginal() {
        let z = Complex(re: 1.5, im: -2.5)
        XCTAssertEqual(z.conj.conj, z)
    }

    func testConj_freeFunctionAlias() {
        let z = Complex(re: 2.0, im: 3.0)
        XCTAssertEqual(conj(z), z.conj)
    }

    // MARK: - isReal / isImaginary / isZero / isFinite / isNaN

    func testIsReal_pureReal() {
        XCTAssertTrue(Complex(re: 3.0, im: 0.0).isReal())
    }

    func testIsReal_withIm() {
        XCTAssertFalse(Complex(re: 3.0, im: 1.0).isReal())
    }

    func testIsImaginary_pureImaginary() {
        XCTAssertTrue(Complex(re: 0.0, im: 5.0).isImaginary())
    }

    func testIsImaginary_withRe() {
        XCTAssertFalse(Complex(re: 1.0, im: 5.0).isImaginary())
    }

    func testIsZero_zero() {
        XCTAssertTrue(Complex.zero.isZero)
    }

    func testIsZero_nonZero() {
        XCTAssertFalse(Complex(re: 0.0, im: 1e-300).isZero)
    }

    func testIsFinite_finite() {
        XCTAssertTrue(Complex(re: 1.0, im: 2.0).isFinite)
    }

    func testIsFinite_infRe() {
        XCTAssertFalse(Complex(re: .infinity, im: 0.0).isFinite)
    }

    func testIsFinite_infIm() {
        XCTAssertFalse(Complex(re: 0.0, im: .infinity).isFinite)
    }

    func testIsNaN_nanRe() {
        XCTAssertTrue(Complex(re: .nan, im: 0.0).isNaN)
    }

    func testIsNaN_nanIm() {
        XCTAssertTrue(Complex(re: 0.0, im: .nan).isNaN)
    }

    func testIsNaN_finite() {
        XCTAssertFalse(Complex(re: 1.0, im: 1.0).isNaN)
    }

    // MARK: - Basic arithmetic

    func testNegation() {
        let z = Complex(re: 1.0, im: -2.0)
        assertComplexClose(-z, re: -1.0, im: 2.0)
    }

    func testAddition_complex() {
        let sum = Complex(re: 1.0, im: 2.0) + Complex(re: 3.0, im: 4.0)
        assertComplexClose(sum, re: 4.0, im: 6.0)
    }

    func testSubtraction_complex() {
        let diff = Complex(re: 5.0, im: 7.0) - Complex(re: 2.0, im: 3.0)
        assertComplexClose(diff, re: 3.0, im: 4.0)
    }

    func testMultiplication_complex() {
        // (3 + 4i)(1 + 2i) = 3 + 6i + 4i + 8i² = -5 + 10i
        let prod = Complex(re: 3.0, im: 4.0) * Complex(re: 1.0, im: 2.0)
        assertComplexClose(prod, re: -5.0, im: 10.0)
    }

    func testMultiplication_byI_rotates90degrees() {
        // i * (1 + 0i) = (0 + i)
        let z = Complex(re: 1.0, im: 0.0) * Complex.i
        assertComplexClose(z, re: 0.0, im: 1.0, tol: 1e-15)
    }

    func testScalarArithmetic_addDouble() {
        let z = Complex(re: 1.0, im: 2.0) + 3.0
        assertComplexClose(z, re: 4.0, im: 2.0)
    }

    func testScalarArithmetic_doubleAddComplex() {
        let z = 3.0 + Complex(re: 1.0, im: 2.0)
        assertComplexClose(z, re: 4.0, im: 2.0)
    }

    func testScalarArithmetic_subtractDouble() {
        let z = Complex(re: 5.0, im: 2.0) - 3.0
        assertComplexClose(z, re: 2.0, im: 2.0)
    }

    func testScalarArithmetic_doubleSubtractComplex() {
        // 5 - (2 + 3i) = 3 - 3i
        let z = 5.0 - Complex(re: 2.0, im: 3.0)
        assertComplexClose(z, re: 3.0, im: -3.0)
    }

    func testScalarArithmetic_multiplyDouble() {
        let z = Complex(re: 1.0, im: 2.0) * 3.0
        assertComplexClose(z, re: 3.0, im: 6.0)
    }

    func testScalarArithmetic_doubleMultiplyComplex() {
        let z = 3.0 * Complex(re: 1.0, im: 2.0)
        assertComplexClose(z, re: 3.0, im: 6.0)
    }

    func testScalarArithmetic_divideByDouble() {
        let z = Complex(re: 4.0, im: 6.0) / 2.0
        assertComplexClose(z, re: 2.0, im: 3.0)
    }

    func testScalarArithmetic_doubleByComplex() {
        // 4 / (1 + 0i) = 4
        let z = 4.0 / Complex(re: 1.0, im: 0.0)
        assertComplexClose(z, re: 4.0, im: 0.0)
    }

    // MARK: - Compound assignment

    func testCompoundAssignment_plusEquals() {
        var z = Complex(re: 1.0, im: 2.0)
        z += Complex(re: 3.0, im: 4.0)
        assertComplexClose(z, re: 4.0, im: 6.0)
    }

    func testCompoundAssignment_minusEquals() {
        var z = Complex(re: 5.0, im: 7.0)
        z -= Complex(re: 2.0, im: 3.0)
        assertComplexClose(z, re: 3.0, im: 4.0)
    }

    func testCompoundAssignment_timesEquals() {
        var z = Complex(re: 3.0, im: 4.0)
        z *= Complex(re: 1.0, im: 2.0)
        assertComplexClose(z, re: -5.0, im: 10.0)
    }

    func testCompoundAssignment_divideEquals() {
        var z = Complex(re: 4.0, im: 6.0)
        z /= Complex(re: 2.0, im: 0.0)
        assertComplexClose(z, re: 2.0, im: 3.0)
    }

    // MARK: - exp

    func testExp_zero() {
        // e^0 = 1
        let z = Complex.zero.exp
        assertComplexClose(z, re: 1.0, im: 0.0)
    }

    func testExp_realPart() {
        // e^1 = e
        let z = Complex(re: 1.0, im: 0.0).exp
        assertComplexClose(z, re: MathConstants.e, im: 0.0, tol: 1e-10)
    }

    func testExp_imaginaryUnit_eulerFormula() {
        // e^(iπ) = -1 + 0i  (Euler's identity)
        let z = Complex(re: 0.0, im: .pi).exp
        assertComplexClose(z, re: -1.0, im: 0.0, tol: 1e-15)
    }

    func testExp_imaginaryHalfPi() {
        // e^(iπ/2) = i
        let z = Complex(re: 0.0, im: .pi / 2).exp
        assertComplexClose(z, re: 0.0, im: 1.0, tol: 1e-15)
    }

    func testExp_complexGeneral() {
        // e^(1 + iπ/2) = e * i
        let z = Complex(re: 1.0, im: .pi / 2).exp
        assertComplexClose(z, re: 0.0, im: MathConstants.e, tol: 1e-14)
    }

    func testCexp_freeFunctionAlias() {
        let z = Complex(re: 0.0, im: .pi)
        XCTAssertEqual(cexp(z), z.exp)
    }

    // MARK: - log (natural logarithm)

    func testLog_one() {
        // log(1) = 0
        let z = Complex.one.log
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testLog_e() {
        // log(e) = 1
        let z = Complex(re: MathConstants.e, im: 0.0).log
        assertComplexClose(z, re: 1.0, im: 0.0, tol: 1e-10)
    }

    func testLog_pureImaginary_i() {
        // log(i) = iπ/2  (principal branch: arg(i) = π/2)
        let z = Complex.i.log
        assertComplexClose(z, re: 0.0, im: .pi / 2)
    }

    func testLog_negativeReal_branchCut() {
        // log(-1) = iπ  (C99 principal branch: UPPER half-plane)
        // Oracle: numpy.log(-1+0j) = 0+3.14159265j  (positive imaginary)
        let z = Complex(re: -1.0, im: 0.0).log
        assertComplexClose(z, re: 0.0, im: .pi)
    }

    func testLog_negativeReal_approachFromBelow_givesNegativePi() {
        // Branch cut: approaching from the lower half-plane gives -iπ
        // log(-1 - ε*i) → iπ approaches from below → im approaches -π
        let z = Complex(re: -1.0, im: -1e-15).log
        XCTAssertLessThan(z.im, 0, "Approaching from below should yield negative imaginary part")
    }

    func testLog_magnitudeAndArg() {
        // log(z) = log|z| + i·arg(z)
        let z = Complex(re: 3.0, im: 4.0)
        let logZ = z.log
        XCTAssertEqual(logZ.re, Foundation.log(z.abs), accuracy: eps)
        XCTAssertEqual(logZ.im, z.arg, accuracy: eps)
    }

    func testLog_expInverseIdentity() {
        // log(exp(z)) = z for z with im in (-π, π]
        let z = Complex(re: 1.5, im: 0.7)
        let result = z.exp.log
        assertComplexClose(result, re: 1.5, im: 0.7, tol: 1e-12)
    }

    func testClog_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 1.0)
        XCTAssertEqual(clog(z), z.log)
    }

    // MARK: - log10

    func testLog10_one() {
        let z = Complex.one.log10
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testLog10_ten() {
        // log10(10) = 1
        let z = Complex(re: 10.0, im: 0.0).log10
        assertComplexClose(z, re: 1.0, im: 0.0, tol: 1e-14)
    }

    func testLog10_negativeReal() {
        // log10(-1) = iπ/ln(10)
        let z = Complex(re: -1.0, im: 0.0).log10
        let expected = .pi / Foundation.log(10.0)
        assertComplexClose(z, re: 0.0, im: expected, tol: 1e-14)
    }

    // MARK: - sqrt (principal square root)

    func testSqrt_one() {
        // sqrt(1) = 1
        let z = Complex.one.sqrt
        assertComplexClose(z, re: 1.0, im: 0.0)
    }

    func testSqrt_four() {
        // sqrt(4) = 2
        let z = Complex(re: 4.0, im: 0.0).sqrt
        assertComplexClose(z, re: 2.0, im: 0.0)
    }

    func testSqrt_negativeOne_principalBranch() {
        // sqrt(-1) = +i  (C99/numpy principal branch: positive imaginary)
        // numpy: np.sqrt(-1+0j) = 1j
        let z = Complex(re: -1.0, im: 0.0).sqrt
        assertComplexClose(z, re: 0.0, im: 1.0)
    }

    func testSqrt_negativeReal_signConvention() {
        // sqrt(-4) = 2i
        let z = Complex(re: -4.0, im: 0.0).sqrt
        assertComplexClose(z, re: 0.0, im: 2.0)
    }

    func testSqrt_pureImaginary() {
        // sqrt(i): |i| = 1, arg(i) = π/2 → r=1, θ=π/4 → (√2/2 + i√2/2)
        // numpy: np.sqrt(1j) = (0.7071..+0.7071..j)
        let z = Complex(re: 0.0, im: 1.0).sqrt
        let s = 2.0.squareRoot() / 2.0
        assertComplexClose(z, re: s, im: s)
    }

    func testSqrt_squaredIsOriginal() {
        // (sqrt(z))^2 = z  for a general z
        let z = Complex(re: 3.0, im: -5.0)
        let sq = z.sqrt
        let back = sq * sq
        assertComplexClose(back, re: z.re, im: z.im, tol: 1e-12)
    }

    func testSqrt_zero() {
        let z = Complex.zero.sqrt
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testCsqrt_freeFunctionAlias() {
        let z = Complex(re: -1.0, im: 0.0)
        XCTAssertEqual(csqrt(z), z.sqrt)
    }

    // MARK: - pow (real exponent)

    func testPow_squaredViaDouble() {
        // (1 + i)^2 = 2i
        let z = Complex(re: 1.0, im: 1.0).pow(2.0)
        assertComplexClose(z, re: 0.0, im: 2.0, tol: 1e-14)
    }

    func testPow_halfIsSquareRoot() {
        // (4 + 0i)^0.5 = 2
        let z = Complex(re: 4.0, im: 0.0).pow(0.5)
        assertComplexClose(z, re: 2.0, im: 0.0)
    }

    func testPow_zeroExponent() {
        // z^0 = 1
        let z = Complex(re: 3.0, im: 4.0).pow(0.0)
        assertComplexClose(z, re: 1.0, im: 0.0)
    }

    func testCpow_doubleExponent_freeFunctionAlias() {
        let z = Complex(re: 4.0, im: 0.0)
        XCTAssertEqual(cpow(z, 0.5), z.pow(0.5))
    }

    // MARK: - pow (complex exponent)

    func testPow_complexExponent_iToThePowerI() {
        // i^i = exp(i * log(i)) = exp(i * iπ/2) = exp(-π/2) ≈ 0.2079
        // numpy: 1j**1j = 0.2078795763507619+0j
        let z = Complex.i.pow(Complex.i)
        assertComplexClose(z, re: Foundation.exp(-.pi / 2), im: 0.0, tol: 1e-12)
    }

    func testCpow_complexExponent_freeFunctionAlias() {
        let z = Complex(re: 2.0, im: 0.0)
        let w = Complex(re: 3.0, im: 0.0)
        assertComplexClose(cpow(z, w), re: 8.0, im: 0.0, tol: 1e-12, "2^3 = 8")
    }

    // MARK: - squared / cubed

    func testSquared() {
        // (1 + i)^2 = 2i
        let z = Complex(re: 1.0, im: 1.0).squared
        assertComplexClose(z, re: 0.0, im: 2.0, tol: 1e-15)
    }

    func testCubed() {
        // (1 + i)^3 = (1+i)(2i) = 2i + 2i^2 = -2 + 2i
        let z = Complex(re: 1.0, im: 1.0).cubed
        assertComplexClose(z, re: -2.0, im: 2.0, tol: 1e-15)
    }

    // MARK: - reciprocal

    func testReciprocal_realAxis() {
        // 1/(2 + 0i) = 0.5
        let z = Complex(re: 2.0, im: 0.0).reciprocal
        assertComplexClose(z, re: 0.5, im: 0.0)
    }

    func testReciprocal_pureImaginary() {
        // 1/(0 + i) = -i
        let z = Complex(re: 0.0, im: 1.0).reciprocal
        assertComplexClose(z, re: 0.0, im: -1.0, tol: 1e-15)
    }

    func testReciprocal_timesOriginalIsOne() {
        let z = Complex(re: 3.0, im: 4.0)
        let prod = z * z.reciprocal
        assertComplexClose(prod, re: 1.0, im: 0.0, tol: 1e-14)
    }

    // MARK: - Trigonometric functions

    func testSin_zero() {
        // sin(0) = 0
        let z = Complex.zero.sin
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testSin_halfPi() {
        // sin(π/2) = 1
        let z = Complex(re: .pi / 2, im: 0.0).sin
        assertComplexClose(z, re: 1.0, im: 0.0, tol: 1e-15)
    }

    func testSin_pureImaginary() {
        // sin(iy) = i·sinh(y)
        // numpy: sin(1j) = 0+1.1752011936438014j
        let z = Complex(re: 0.0, im: 1.0).sin
        assertComplexClose(z, re: 0.0, im: 1.1752011936438014, tol: 1e-14)
    }

    func testCsin_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 1.0)
        XCTAssertEqual(csin(z), z.sin)
    }

    func testCos_zero() {
        // cos(0) = 1
        let z = Complex.zero.cos
        assertComplexClose(z, re: 1.0, im: 0.0)
    }

    func testCos_halfPi() {
        // cos(π/2) ≈ 0
        let z = Complex(re: .pi / 2, im: 0.0).cos
        assertComplexClose(z, re: 0.0, im: 0.0, tol: 1e-15)
    }

    func testCos_pureImaginary() {
        // cos(iy) = cosh(y)
        // numpy: cos(1j) = 1.5430806348152437+0j
        let z = Complex(re: 0.0, im: 1.0).cos
        assertComplexClose(z, re: 1.5430806348152437, im: 0.0, tol: 1e-14)
    }

    func testCcos_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 1.0)
        XCTAssertEqual(ccos(z), z.cos)
    }

    func testSinSquaredPlusCosSquaredEqualsOne() {
        // sin²(z) + cos²(z) = 1 for any z (Pythagorean identity)
        let z = Complex(re: 1.2, im: 0.5)
        let identity = z.sin.squared + z.cos.squared
        assertComplexClose(identity, re: 1.0, im: 0.0, tol: 1e-13)
    }

    func testTan_quarterPi() {
        // tan(π/4) = 1
        let z = Complex(re: .pi / 4, im: 0.0).tan
        assertComplexClose(z, re: 1.0, im: 0.0, tol: 1e-15)
    }

    func testTan_isSinOverCos() {
        let z = Complex(re: 0.8, im: 0.3)
        let tanDirect = z.tan
        let tanRatio = z.sin / z.cos
        assertComplexClose(tanDirect, re: tanRatio.re, im: tanRatio.im, tol: 1e-13)
    }

    func testCtan_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 1.0)
        XCTAssertEqual(ctan(z), z.tan)
    }

    // MARK: - Inverse trigonometric functions

    func testAsin_zero() {
        // asin(0) = 0
        let z = Complex.zero.asin
        assertComplexClose(z, re: 0.0, im: 0.0, tol: 1e-14)
    }

    func testAsin_one() {
        // asin(1) = π/2
        let z = Complex.one.asin
        assertComplexClose(z, re: .pi / 2, im: 0.0, tol: 1e-13)
    }

    func testAsin_inverseOfSin() {
        // asin(sin(z)) = z  for z with re in (-π/2, π/2)
        let z = Complex(re: 0.5, im: 0.3)
        let result = z.sin.asin
        assertComplexClose(result, re: 0.5, im: 0.3, tol: 1e-12)
    }

    func testAsin_outsideDomain_two() {
        // asin(2) = π/2 - i·ln(2 + √3)
        // numpy: arcsin(2+0j) = (1.5707963267948966-1.3169578969248166j)
        let z = Complex(re: 2.0, im: 0.0).asin
        assertComplexClose(z, re: .pi / 2, im: -1.3169578969248166, tol: 1e-12)
    }

    func testCasin_freeFunctionAlias() {
        let z = Complex(re: 0.5, im: 0.0)
        XCTAssertEqual(casin(z), z.asin)
    }

    func testAcos_one() {
        // acos(1) = 0
        let z = Complex.one.acos
        assertComplexClose(z, re: 0.0, im: 0.0, tol: 1e-13)
    }

    func testAcos_minusOne() {
        // acos(-1) = π
        let z = Complex(re: -1.0, im: 0.0).acos
        assertComplexClose(z, re: .pi, im: 0.0, tol: 1e-13)
    }

    func testAcos_knownValues() {
        // acos(z) = -i * log(z + sqrt(z²-1))
        // For z = 0.4+0.2i: acos gives a branch value. Verified by forward computation.
        // numpy: np.arccos(0.4+0.2j) = (1.1692099351270906-0.2156124185558297j)
        let z = Complex(re: 0.4, im: 0.2).acos
        assertComplexClose(z, re: 1.1692099351270906, im: -0.2156124185558297, tol: 1e-12)
    }

    func testCacos_freeFunctionAlias() {
        let z = Complex(re: 0.5, im: 0.0)
        XCTAssertEqual(cacos(z), z.acos)
    }

    func testAtan_zero() {
        // atan(0) = 0
        let z = Complex.zero.atan
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testAtan_one() {
        // atan(1) = π/4
        let z = Complex.one.atan
        assertComplexClose(z, re: .pi / 4, im: 0.0, tol: 1e-14)
    }

    func testAtan_inverseOfTan() {
        let z = Complex(re: 0.6, im: 0.1)
        let result = z.tan.atan
        assertComplexClose(result, re: 0.6, im: 0.1, tol: 1e-12)
    }

    func testCatan_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 0.0)
        XCTAssertEqual(catan(z), z.atan)
    }

    // MARK: - Hyperbolic functions

    func testSinh_zero() {
        // sinh(0) = 0
        let z = Complex.zero.sinh
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testSinh_pureImaginary() {
        // sinh(iy) = i·sin(y)
        let y = 1.0
        let z = Complex(re: 0.0, im: y).sinh
        assertComplexClose(z, re: 0.0, im: Foundation.sin(y), tol: 1e-15)
    }

    func testCsinh_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 0.0)
        XCTAssertEqual(csinh(z), z.sinh)
    }

    func testCosh_zero() {
        // cosh(0) = 1
        let z = Complex.zero.cosh
        assertComplexClose(z, re: 1.0, im: 0.0)
    }

    func testCosh_pureImaginary() {
        // cosh(iy) = cos(y)
        let y = 1.0
        let z = Complex(re: 0.0, im: y).cosh
        assertComplexClose(z, re: Foundation.cos(y), im: 0.0, tol: 1e-15)
    }

    func testCcosh_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 0.0)
        XCTAssertEqual(ccosh(z), z.cosh)
    }

    func testCosh2MinusSinh2EqualsOne() {
        // cosh²(z) - sinh²(z) = 1  (Pythagorean identity)
        let z = Complex(re: 0.7, im: 0.4)
        let identity = z.cosh.squared - z.sinh.squared
        assertComplexClose(identity, re: 1.0, im: 0.0, tol: 1e-12)
    }

    func testTanh_zero() {
        // tanh(0) = 0
        let z = Complex.zero.tanh
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testTanh_pureImaginary() {
        // tanh(iy) = i·tan(y)
        let y = 0.5
        let z = Complex(re: 0.0, im: y).tanh
        assertComplexClose(z, re: 0.0, im: Foundation.tan(y), tol: 1e-15)
    }

    func testCtanh_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 0.0)
        XCTAssertEqual(ctanh(z), z.tanh)
    }

    // MARK: - Inverse hyperbolic functions

    func testAsinh_zero() {
        // asinh(0) = 0
        let z = Complex.zero.asinh
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testAsinh_inverseOfSinh() {
        let z = Complex(re: 0.5, im: 0.3)
        let result = z.sinh.asinh
        assertComplexClose(result, re: 0.5, im: 0.3, tol: 1e-12)
    }

    func testCasinh_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 0.0)
        XCTAssertEqual(casinh(z), z.asinh)
    }

    func testAcosh_one() {
        // acosh(1) = 0
        let z = Complex.one.acosh
        assertComplexClose(z, re: 0.0, im: 0.0, tol: 1e-13)
    }

    func testAcosh_inverseOfCosh() {
        // Re must be ≥ 1 to avoid branch complexity; use real z > 1
        let z = Complex(re: 2.0, im: 0.0)
        let result = z.cosh.acosh
        // acosh(cosh(2)) = 2 (principal branch, re≥0)
        XCTAssertEqual(result.re, z.re, accuracy: 1e-12)
        XCTAssertEqual(Swift.abs(result.im), 0.0, accuracy: 1e-12)
    }

    func testCacosh_freeFunctionAlias() {
        let z = Complex(re: 2.0, im: 0.0)
        XCTAssertEqual(cacosh(z), z.acosh)
    }

    func testAtanh_zero() {
        // atanh(0) = 0
        let z = Complex.zero.atanh
        assertComplexClose(z, re: 0.0, im: 0.0)
    }

    func testAtanh_inverseOfTanh() {
        let z = Complex(re: 0.3, im: 0.2)
        let result = z.tanh.atanh
        assertComplexClose(result, re: 0.3, im: 0.2, tol: 1e-12)
    }

    func testCatanh_freeFunctionAlias() {
        let z = Complex(re: 0.5, im: 0.0)
        XCTAssertEqual(catanh(z), z.atanh)
    }

    // MARK: - cabs / carg free functions

    func testCabs_freeFunctionAlias() {
        let z = Complex(re: 3.0, im: 4.0)
        XCTAssertEqual(cabs(z), z.abs, accuracy: eps)
    }

    func testCarg_freeFunctionAlias() {
        let z = Complex(re: 1.0, im: 1.0)
        XCTAssertEqual(carg(z), z.arg, accuracy: eps)
    }

    // MARK: - CustomStringConvertible

    func testDescription_positiveImaginary() {
        let z = Complex(re: 1.0, im: 2.0)
        XCTAssertEqual(z.description, "1.0+2.0i")
    }

    func testDescription_negativeImaginary() {
        let z = Complex(re: 1.0, im: -2.0)
        XCTAssertEqual(z.description, "1.0-2.0i")
    }

    func testDescription_zero() {
        let z = Complex(re: 0.0, im: 0.0)
        XCTAssertEqual(z.description, "0.0+0.0i")
    }

    // MARK: - Hashable

    func testHashable_equalValuesHaveSameHash() {
        let z1 = Complex(re: 3.0, im: 4.0)
        let z2 = Complex(re: 3.0, im: 4.0)
        XCTAssertEqual(z1.hashValue, z2.hashValue)
    }

    func testHashable_usableInSet() {
        let set: Set<Complex> = [
            Complex(re: 1.0, im: 0.0),
            Complex(re: 0.0, im: 1.0),
            Complex(re: 1.0, im: 0.0)  // duplicate
        ]
        XCTAssertEqual(set.count, 2)
    }

    func testHashable_usableAsDictionaryKey() {
        var dict: [Complex: String] = [:]
        dict[Complex(re: 1.0, im: 2.0)] = "z1"
        XCTAssertEqual(dict[Complex(re: 1.0, im: 2.0)], "z1")
    }

    // MARK: - Edge values: infinity and NaN

    func testExp_nanInput() {
        let z = Complex(re: .nan, im: 0.0).exp
        XCTAssertTrue(z.isNaN)
    }

    func testSqrt_infinitePositiveReal() {
        // C99 Annex G.6.4.2 / numpy: sqrt(+inf + 0i) = +inf + 0i
        // np.sqrt(complex(float('inf'), 0)) = (inf+0j)
        let z = Complex(re: .infinity, im: 0.0).sqrt
        XCTAssertEqual(z.re, .infinity)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-30)
    }

    func testSqrt_infiniteNegativeReal() {
        // C99 Annex G.6.4.2: sqrt(-inf + 0i) = 0 + inf*i
        // np.sqrt(complex(float('-inf'), 0)) = 0j + infj  → (0+infj)
        let z = Complex(re: -.infinity, im: 0.0).sqrt
        XCTAssertEqual(z.re, 0.0, accuracy: 1e-30,
                       "sqrt(-inf+0i) real part must be 0 per C99 Annex G.6.4.2")
        XCTAssertEqual(z.im, .infinity,
                       "sqrt(-inf+0i) imaginary part must be +inf per C99 Annex G.6.4.2")
    }

    func testSqrt_infinitePositiveReal_withFiniteIm() {
        // C99 Annex G.6.4.2: sqrt(+inf + y*i) = +inf + 0*i  for finite y
        // np.sqrt(complex(float('inf'), 3.0)) = (inf+0j)
        let z = Complex(re: .infinity, im: 3.0).sqrt
        XCTAssertEqual(z.re, .infinity,
                       "sqrt(+inf+3i) real part must be +inf")
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-30,
                       "sqrt(+inf+3i) imaginary part must be 0")
    }

    func testSqrt_infiniteNegativeReal_withFiniteIm() {
        // C99 Annex G.6.4.2: sqrt(-inf + y*i) = 0 + inf*i  for finite y > 0
        // np.sqrt(complex(float('-inf'), 3.0)) = 0+infj
        let z = Complex(re: -.infinity, im: 3.0).sqrt
        XCTAssertEqual(z.re, 0.0, accuracy: 1e-30,
                       "sqrt(-inf+3i) real part must be 0")
        XCTAssertEqual(z.im, .infinity,
                       "sqrt(-inf+3i) imaginary part must be +inf")
    }

    func testSqrt_finiteReal_withInfiniteIm() {
        // C99 Annex G.6.4.2: sqrt(x + inf*i) = +inf + inf*i  for any finite x
        // np.sqrt(complex(1.0, float('inf'))) = (inf+infj)
        let z = Complex(re: 1.0, im: .infinity).sqrt
        XCTAssertEqual(z.re, .infinity,
                       "sqrt(1+inf*i) real part must be +inf")
        XCTAssertEqual(z.im, .infinity,
                       "sqrt(1+inf*i) imaginary part must be +inf")
    }

    func testSqrt_negativeInfiniteReal_withInfiniteIm() {
        // C99 Annex G.6.4.2: sqrt(-inf + inf*i) = +inf + +inf*i
        // np.sqrt(complex(float('-inf'), float('inf'))) = (inf+infj)
        let z = Complex(re: -.infinity, im: .infinity).sqrt
        XCTAssertEqual(z.re, .infinity,
                       "sqrt(-inf+inf*i) real part must be +inf")
        XCTAssertEqual(z.im, .infinity,
                       "sqrt(-inf+inf*i) imaginary part must be +inf")
    }

    func testSqrt_positiveInfiniteReal_withInfiniteIm() {
        // np.sqrt(complex(float('inf'), float('inf'))) = (inf+infj)
        let z = Complex(re: .infinity, im: .infinity).sqrt
        XCTAssertEqual(z.re, .infinity)
        XCTAssertEqual(z.im, .infinity)
    }

    func testSqrt_nanReal_withFiniteIm() {
        // C99 Annex G.6.4.2: sqrt(NaN + y*i) = NaN + NaN*i  for finite y
        // np.sqrt(complex(float('nan'), 0.0)) → (nan+nanj)
        let z = Complex(re: .nan, im: 0.0).sqrt
        XCTAssertTrue(z.re.isNaN || z.im.isNaN,
                      "sqrt(NaN+0i) must have at least one NaN component")
    }

    func testSqrt_finiteReal_withNanIm() {
        // np.sqrt(complex(1.0, float('nan'))) → (nan+nanj)
        let z = Complex(re: 1.0, im: .nan).sqrt
        XCTAssertTrue(z.re.isNaN || z.im.isNaN,
                      "sqrt(1+NaN*i) must have at least one NaN component")
    }

    func testLog_zero() {
        // log(0) = -inf + i·arg(0); arg(0+0i) = 0 (atan2(0,0))
        let z = Complex.zero.log
        XCTAssertEqual(z.re, -.infinity)
    }

    func testAbs_nanComponent() {
        // |nan + 0i| = nan (hypot propagates)
        let z = Complex(re: .nan, im: 0.0)
        XCTAssertTrue(z.abs.isNaN)
    }

    // MARK: - Round-trip identities across all trig pairs

    func testSinAsinRoundTrip() {
        // sin(asin(z)) = z for various z
        for z in [Complex(re: 0.0, im: 0.0), Complex(re: 0.3, im: 0.2), Complex(re: -0.5, im: 0.1)] {
            let result = z.asin.sin
            assertComplexClose(result, re: z.re, im: z.im, tol: 1e-12, "sin(asin(z)) for \(z)")
        }
    }

    func testCosAcosRoundTrip() {
        for z in [Complex(re: 0.0, im: 0.0), Complex(re: 0.4, im: 0.1)] {
            let result = z.acos.cos
            assertComplexClose(result, re: z.re, im: z.im, tol: 1e-12, "cos(acos(z)) for \(z)")
        }
    }

    func testTanAtanRoundTrip() {
        for z in [Complex(re: 0.0, im: 0.0), Complex(re: 0.5, im: 0.2)] {
            let result = z.atan.tan
            assertComplexClose(result, re: z.re, im: z.im, tol: 1e-12, "tan(atan(z)) for \(z)")
        }
    }
}
