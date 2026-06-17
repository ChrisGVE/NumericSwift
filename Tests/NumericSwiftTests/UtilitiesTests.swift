//
//  UtilitiesTests.swift
//  Tests/NumericSwiftTests/
//
//  Coverage tests for Utilities.swift: scalar functions and all vDSP-backed array
//  operations. Reference values come from numpy/Python for non-trivial cases;
//  analytic values are used for basic identities.
//
//  Oracle: numpy 1.x  (numpy.round, numpy.sign, numpy.clip, numpy.sqrt, etc.)
//  Branch: fix/issue-19-coverage  Refs #19
//

import XCTest
@testable import NumericSwift

final class UtilitiesTests: XCTestCase {

    // MARK: - Tolerance helpers

    private let eps = 1e-12

    private func assertClose(_ a: Double, _ b: Double, tol: Double = 1e-12,
                             _ msg: String = "", file: StaticString = #file, line: UInt = #line) {
        XCTAssertEqual(a, b, accuracy: tol, msg, file: file, line: line)
    }

    private func assertNaN(_ x: Double, _ msg: String = "", file: StaticString = #file, line: UInt = #line) {
        XCTAssertTrue(x.isNaN, "expected NaN but got \(x) — \(msg)", file: file, line: line)
    }

    // MARK: - roundValue

    func testRoundValue_positiveHalfUp() {
        // 0.5 rounds to nearest-even in some modes; Darwin.round always rounds away from zero
        XCTAssertEqual(roundValue(0.5), 1.0)
        XCTAssertEqual(roundValue(1.5), 2.0)
        XCTAssertEqual(roundValue(2.5), 3.0)
    }

    func testRoundValue_negative() {
        XCTAssertEqual(roundValue(-0.5), -1.0)
        XCTAssertEqual(roundValue(-1.4), -1.0)
        XCTAssertEqual(roundValue(-1.6), -2.0)
    }

    func testRoundValue_zero() {
        XCTAssertEqual(roundValue(0.0), 0.0)
    }

    func testRoundValue_infinity() {
        // round(±inf) == ±inf  (IEEE-754)
        XCTAssertEqual(roundValue(.infinity), .infinity)
        XCTAssertEqual(roundValue(-.infinity), -.infinity)
    }

    func testRoundValue_nan() {
        assertNaN(roundValue(.nan), "round(nan)")
    }

    // MARK: - truncValue

    func testTruncValue_positive() {
        XCTAssertEqual(truncValue(3.9), 3.0)
        XCTAssertEqual(truncValue(3.1), 3.0)
    }

    func testTruncValue_negative() {
        XCTAssertEqual(truncValue(-3.9), -3.0)
        XCTAssertEqual(truncValue(-3.1), -3.0)
    }

    func testTruncValue_zero() {
        XCTAssertEqual(truncValue(0.0), 0.0)
    }

    func testTruncValue_infinity() {
        XCTAssertEqual(truncValue(.infinity), .infinity)
        XCTAssertEqual(truncValue(-.infinity), -.infinity)
    }

    func testTruncValue_nan() {
        assertNaN(truncValue(.nan))
    }

    // MARK: - signValue

    func testSignValue_positive() {
        XCTAssertEqual(signValue(3.7), 1.0)
        XCTAssertEqual(signValue(.infinity), 1.0)
    }

    func testSignValue_negative() {
        XCTAssertEqual(signValue(-3.7), -1.0)
        XCTAssertEqual(signValue(-.infinity), -1.0)
    }

    func testSignValue_zero() {
        XCTAssertEqual(signValue(0.0), 0.0)
        XCTAssertEqual(signValue(-0.0), 0.0)  // -0.0 is not < 0
    }

    func testSignValue_nan() {
        // nan is not > 0 and not < 0, so returns 0.0
        XCTAssertEqual(signValue(.nan), 0.0)
    }

    // MARK: - clipValue

    func testClipValue_insideRange() {
        XCTAssertEqual(clipValue(3.0, lo: 0.0, hi: 5.0), 3.0)
    }

    func testClipValue_belowLo() {
        XCTAssertEqual(clipValue(-1.0, lo: 0.0, hi: 5.0), 0.0)
    }

    func testClipValue_aboveHi() {
        XCTAssertEqual(clipValue(10.0, lo: 0.0, hi: 5.0), 5.0)
    }

    func testClipValue_atBoundary() {
        XCTAssertEqual(clipValue(0.0, lo: 0.0, hi: 5.0), 0.0)
        XCTAssertEqual(clipValue(5.0, lo: 0.0, hi: 5.0), 5.0)
    }

    func testClipValue_negativeRange() {
        XCTAssertEqual(clipValue(-3.0, lo: -5.0, hi: -1.0), -3.0)
        XCTAssertEqual(clipValue(0.0, lo: -5.0, hi: -1.0), -1.0)
        XCTAssertEqual(clipValue(-10.0, lo: -5.0, hi: -1.0), -5.0)
    }

    // MARK: - roundArray

    func testRoundArray_empty() {
        XCTAssertEqual(roundArray([]), [])
    }

    func testRoundArray_singleElement() {
        XCTAssertEqual(roundArray([2.7]), [3.0])
    }

    func testRoundArray_typicalValues() {
        // Darwin.round uses round-half-away-from-zero (C99 §7.12.9.6):
        //   1.1 → 1.0  (below midpoint)
        //   1.5 → 2.0  (half → away from zero, i.e. up)
        //  -1.5 → -2.0 (half → away from zero, i.e. down)
        //   2.9 → 3.0  (above midpoint)
        // Input: [1.1, 1.5, -1.5, 2.9]
        let result = roundArray([1.1, 1.5, -1.5, 2.9])
        XCTAssertEqual(result, [1.0, 2.0, -2.0, 3.0])
    }

    func testRoundArray_infinity() {
        let result = roundArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], .infinity)
        XCTAssertEqual(result[1], -.infinity)
    }

    func testRoundArray_nan() {
        let result = roundArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - truncArray

    func testTruncArray_empty() {
        XCTAssertEqual(truncArray([]), [])
    }

    func testTruncArray_singleElement() {
        XCTAssertEqual(truncArray([3.9]), [3.0])
    }

    func testTruncArray_mixedSigns() {
        let result = truncArray([1.9, -1.9, 0.1, -0.1])
        XCTAssertEqual(result, [1.0, -1.0, 0.0, 0.0])
    }

    // MARK: - signArray

    func testSignArray_empty() {
        XCTAssertEqual(signArray([]), [])
    }

    func testSignArray_singleElement() {
        XCTAssertEqual(signArray([5.0]), [1.0])
        XCTAssertEqual(signArray([-5.0]), [-1.0])
        XCTAssertEqual(signArray([0.0]), [0.0])
    }

    func testSignArray_mixedValues() {
        let result = signArray([3.0, -2.0, 0.0, 1e-300, -1e-300])
        XCTAssertEqual(result, [1.0, -1.0, 0.0, 1.0, -1.0])
    }

    func testSignArray_infinity() {
        let result = signArray([.infinity, -.infinity])
        XCTAssertEqual(result, [1.0, -1.0])
    }

    // MARK: - clipArray

    func testClipArray_empty() {
        XCTAssertEqual(clipArray([], lo: 0, hi: 1), [])
    }

    func testClipArray_singleElement() {
        XCTAssertEqual(clipArray([3.0], lo: 0, hi: 2), [2.0])
    }

    func testClipArray_allBelow() {
        let result = clipArray([-5.0, -3.0, -1.0], lo: 0, hi: 10)
        XCTAssertEqual(result, [0.0, 0.0, 0.0])
    }

    func testClipArray_allAbove() {
        let result = clipArray([11.0, 20.0, 100.0], lo: 0, hi: 10)
        XCTAssertEqual(result, [10.0, 10.0, 10.0])
    }

    func testClipArray_mixedInOut() {
        // numpy: np.clip([-1,0,3,5,6], 0, 5) = [0,0,3,5,5]
        let result = clipArray([-1.0, 0.0, 3.0, 5.0, 6.0], lo: 0, hi: 5)
        XCTAssertEqual(result, [0.0, 0.0, 3.0, 5.0, 5.0])
    }

    // MARK: - floorArray

    func testFloorArray_empty() {
        XCTAssertEqual(floorArray([]), [])
    }

    func testFloorArray_singleElement() {
        XCTAssertEqual(floorArray([2.7]), [2.0])
    }

    func testFloorArray_negativeValues() {
        // numpy: np.floor([-1.1, -1.9]) = [-2., -2.]
        let result = floorArray([-1.1, -1.9, 1.1, 1.9])
        XCTAssertEqual(result, [-2.0, -2.0, 1.0, 1.0])
    }

    func testFloorArray_infinity() {
        let result = floorArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], .infinity)
        XCTAssertEqual(result[1], -.infinity)
    }

    func testFloorArray_nan() {
        let result = floorArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - ceilArray

    func testCeilArray_empty() {
        XCTAssertEqual(ceilArray([]), [])
    }

    func testCeilArray_singleElement() {
        XCTAssertEqual(ceilArray([2.1]), [3.0])
    }

    func testCeilArray_negativeValues() {
        // numpy: np.ceil([-1.1, -1.9]) = [-1., -1.]
        let result = ceilArray([-1.1, -1.9, 1.1, 1.9])
        XCTAssertEqual(result, [-1.0, -1.0, 2.0, 2.0])
    }

    func testCeilArray_infinity() {
        let result = ceilArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], .infinity)
        XCTAssertEqual(result[1], -.infinity)
    }

    // MARK: - absArray

    func testAbsArray_empty() {
        XCTAssertEqual(absArray([]), [])
    }

    func testAbsArray_singleElement() {
        XCTAssertEqual(absArray([-3.0]), [3.0])
    }

    func testAbsArray_mixedSigns() {
        let result = absArray([-1.0, 0.0, 1.0, -100.0, 100.0])
        XCTAssertEqual(result, [1.0, 0.0, 1.0, 100.0, 100.0])
    }

    func testAbsArray_infinity() {
        let result = absArray([.infinity, -.infinity])
        XCTAssertEqual(result, [.infinity, .infinity])
    }

    func testAbsArray_nan() {
        let result = absArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - negArray

    func testNegArray_empty() {
        XCTAssertEqual(negArray([]), [])
    }

    func testNegArray_singleElement() {
        XCTAssertEqual(negArray([3.0]), [-3.0])
    }

    func testNegArray_mixedSigns() {
        let result = negArray([1.0, -2.0, 0.0])
        XCTAssertEqual(result, [-1.0, 2.0, 0.0])
    }

    func testNegArray_infinity() {
        let result = negArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], -.infinity)
        XCTAssertEqual(result[1], .infinity)
    }

    func testNegArray_nan() {
        let result = negArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - sqrtArray

    func testSqrtArray_empty() {
        XCTAssertEqual(sqrtArray([]), [])
    }

    func testSqrtArray_singleElement() {
        assertClose(sqrtArray([4.0])[0], 2.0)
    }

    func testSqrtArray_typicalValues() {
        // numpy: np.sqrt([0, 1, 4, 9, 2]) = [0, 1, 2, 3, 1.41421356...]
        let result = sqrtArray([0.0, 1.0, 4.0, 9.0, 2.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.0)
        assertClose(result[2], 2.0)
        assertClose(result[3], 3.0)
        assertClose(result[4], 1.4142135623730951)
    }

    func testSqrtArray_negativeReturnsNaN() {
        // IEEE-754: sqrt of negative real → NaN
        let result = sqrtArray([-1.0])
        XCTAssertTrue(result[0].isNaN, "sqrt(-1) should be NaN for real array")
    }

    func testSqrtArray_infinity() {
        let result = sqrtArray([.infinity])
        XCTAssertEqual(result[0], .infinity)
    }

    func testSqrtArray_nan() {
        let result = sqrtArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - squareArray

    func testSquareArray_empty() {
        XCTAssertEqual(squareArray([]), [])
    }

    func testSquareArray_singleElement() {
        assertClose(squareArray([3.0])[0], 9.0)
    }

    func testSquareArray_negativesSquarePositive() {
        let result = squareArray([-3.0, 0.0, 3.0])
        assertClose(result[0], 9.0)
        assertClose(result[1], 0.0)
        assertClose(result[2], 9.0)
    }

    func testSquareArray_infinity() {
        let result = squareArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], .infinity)
        XCTAssertEqual(result[1], .infinity)
    }

    // MARK: - logArray

    func testLogArray_empty() {
        XCTAssertEqual(logArray([]), [])
    }

    func testLogArray_singleElement() {
        // ln(e) = 1
        assertClose(logArray([MathConstants.e])[0], 1.0)
    }

    func testLogArray_typicalValues() {
        // numpy: np.log([1, e, e^2]) = [0, 1, 2]
        let result = logArray([1.0, MathConstants.e, MathConstants.e * MathConstants.e])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.0)
        assertClose(result[2], 2.0, tol: 1e-10)
    }

    func testLogArray_zeroReturnsNegInf() {
        // ln(0) = -inf
        let result = logArray([0.0])
        XCTAssertEqual(result[0], -.infinity)
    }

    func testLogArray_negativeReturnsNaN() {
        // ln(-1) = NaN for real
        let result = logArray([-1.0])
        XCTAssertTrue(result[0].isNaN)
    }

    func testLogArray_infinity() {
        let result = logArray([.infinity])
        XCTAssertEqual(result[0], .infinity)
    }

    // MARK: - log10Array

    func testLog10Array_empty() {
        XCTAssertEqual(log10Array([]), [])
    }

    func testLog10Array_singleElement() {
        assertClose(log10Array([100.0])[0], 2.0)
    }

    func testLog10Array_typicalValues() {
        // log10([1, 10, 1000]) = [0, 1, 3]
        let result = log10Array([1.0, 10.0, 1000.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.0)
        assertClose(result[2], 3.0)
    }

    func testLog10Array_zeroReturnsNegInf() {
        let result = log10Array([0.0])
        XCTAssertEqual(result[0], -.infinity)
    }

    func testLog10Array_negativeReturnsNaN() {
        let result = log10Array([-1.0])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - expArray

    func testExpArray_empty() {
        XCTAssertEqual(expArray([]), [])
    }

    func testExpArray_singleElement() {
        // exp(1) = e
        assertClose(expArray([1.0])[0], MathConstants.e, tol: 1e-10)
    }

    func testExpArray_typicalValues() {
        // numpy: np.exp([0, 1, -1]) = [1, e, 1/e]
        let result = expArray([0.0, 1.0, -1.0])
        assertClose(result[0], 1.0)
        assertClose(result[1], MathConstants.e, tol: 1e-10)
        assertClose(result[2], 1.0 / MathConstants.e, tol: 1e-10)
    }

    func testExpArray_negativeInfinity() {
        // exp(-inf) = 0
        let result = expArray([-.infinity])
        XCTAssertEqual(result[0], 0.0)
    }

    func testExpArray_positiveInfinity() {
        let result = expArray([.infinity])
        XCTAssertEqual(result[0], .infinity)
    }

    func testExpArray_nan() {
        let result = expArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - powArray

    func testPowArray_emptyBases() {
        XCTAssertEqual(powArray([], []), [])
    }

    func testPowArray_mismatchedLengths() {
        // Guard: mismatched → empty
        XCTAssertEqual(powArray([1.0, 2.0], [1.0]), [])
    }

    func testPowArray_singleElement() {
        // 2^3 = 8
        assertClose(powArray([2.0], [3.0])[0], 8.0)
    }

    func testPowArray_typicalValues() {
        // numpy: np.power([2,3,4], [2,2,0.5]) = [4, 9, 2]
        let result = powArray([2.0, 3.0, 4.0], [2.0, 2.0, 0.5])
        assertClose(result[0], 4.0)
        assertClose(result[1], 9.0)
        assertClose(result[2], 2.0)
    }

    func testPowArray_zeroExponent() {
        // x^0 = 1 for x > 0
        let result = powArray([5.0, 100.0], [0.0, 0.0])
        assertClose(result[0], 1.0)
        assertClose(result[1], 1.0)
    }

    // MARK: - sinArray

    func testSinArray_empty() {
        XCTAssertEqual(sinArray([]), [])
    }

    func testSinArray_singleElement() {
        // sin(π/2) = 1
        assertClose(sinArray([.pi / 2])[0], 1.0)
    }

    func testSinArray_typicalValues() {
        // numpy: np.sin([0, π/6, π/2, π]) ≈ [0, 0.5, 1, 0]
        let result = sinArray([0.0, .pi / 6, .pi / 2, .pi])
        assertClose(result[0], 0.0)
        assertClose(result[1], 0.5)
        assertClose(result[2], 1.0)
        assertClose(result[3], 0.0, tol: 1e-15)
    }

    func testSinArray_infinity() {
        // sin(inf) = NaN per IEEE-754
        let result = sinArray([.infinity])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - cosArray

    func testCosArray_empty() {
        XCTAssertEqual(cosArray([]), [])
    }

    func testCosArray_singleElement() {
        // cos(0) = 1
        assertClose(cosArray([0.0])[0], 1.0)
    }

    func testCosArray_typicalValues() {
        // numpy: np.cos([0, π/3, π/2, π]) ≈ [1, 0.5, 0, -1]
        let result = cosArray([0.0, .pi / 3, .pi / 2, .pi])
        assertClose(result[0], 1.0)
        assertClose(result[1], 0.5)
        assertClose(result[2], 0.0, tol: 1e-15)
        assertClose(result[3], -1.0)
    }

    func testCosArray_infinity() {
        let result = cosArray([.infinity])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - tanArray

    func testTanArray_empty() {
        XCTAssertEqual(tanArray([]), [])
    }

    func testTanArray_singleElement() {
        // tan(π/4) = 1
        assertClose(tanArray([.pi / 4])[0], 1.0)
    }

    func testTanArray_typicalValues() {
        // numpy: np.tan([0, π/4]) = [0, 1]
        let result = tanArray([0.0, .pi / 4])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.0)
    }

    // MARK: - asinArray

    func testAsinArray_empty() {
        XCTAssertEqual(asinArray([]), [])
    }

    func testAsinArray_singleElement() {
        // asin(1) = π/2
        assertClose(asinArray([1.0])[0], .pi / 2)
    }

    func testAsinArray_typicalValues() {
        // numpy: np.arcsin([0, 0.5, 1]) = [0, π/6, π/2]
        let result = asinArray([0.0, 0.5, 1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], .pi / 6)
        assertClose(result[2], .pi / 2)
    }

    func testAsinArray_outsideDomainReturnsNaN() {
        // asin(2) = NaN (out of domain)
        let result = asinArray([2.0])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - acosArray

    func testAcosArray_empty() {
        XCTAssertEqual(acosArray([]), [])
    }

    func testAcosArray_singleElement() {
        // acos(1) = 0
        assertClose(acosArray([1.0])[0], 0.0)
    }

    func testAcosArray_typicalValues() {
        // numpy: np.arccos([1, 0.5, 0, -1]) = [0, π/3, π/2, π]
        let result = acosArray([1.0, 0.5, 0.0, -1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], .pi / 3)
        assertClose(result[2], .pi / 2)
        assertClose(result[3], .pi)
    }

    func testAcosArray_outsideDomainReturnsNaN() {
        let result = acosArray([2.0])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - atanArray

    func testAtanArray_empty() {
        XCTAssertEqual(atanArray([]), [])
    }

    func testAtanArray_singleElement() {
        // atan(1) = π/4
        assertClose(atanArray([1.0])[0], .pi / 4)
    }

    func testAtanArray_typicalValues() {
        // numpy: np.arctan([0, 1, -1]) = [0, π/4, -π/4]
        let result = atanArray([0.0, 1.0, -1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], .pi / 4)
        assertClose(result[2], -.pi / 4)
    }

    func testAtanArray_infinity() {
        // atan(inf) = π/2
        assertClose(atanArray([.infinity])[0], .pi / 2)
        assertClose(atanArray([-.infinity])[0], -.pi / 2)
    }

    // MARK: - sinhArray

    func testSinhArray_empty() {
        XCTAssertEqual(sinhArray([]), [])
    }

    func testSinhArray_singleElement() {
        // sinh(0) = 0
        assertClose(sinhArray([0.0])[0], 0.0)
    }

    func testSinhArray_typicalValues() {
        // numpy: np.sinh([0, 1, -1]) = [0, 1.1752011936..., -1.1752011936...]
        let result = sinhArray([0.0, 1.0, -1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.1752011936438014)
        assertClose(result[2], -1.1752011936438014)
    }

    // MARK: - coshArray

    func testCoshArray_empty() {
        XCTAssertEqual(coshArray([]), [])
    }

    func testCoshArray_singleElement() {
        // cosh(0) = 1
        assertClose(coshArray([0.0])[0], 1.0)
    }

    func testCoshArray_typicalValues() {
        // numpy: np.cosh([0, 1]) = [1, 1.5430806348...]
        let result = coshArray([0.0, 1.0])
        assertClose(result[0], 1.0)
        assertClose(result[1], 1.5430806348152437)
    }

    // MARK: - tanhArray

    func testTanhArray_empty() {
        XCTAssertEqual(tanhArray([]), [])
    }

    func testTanhArray_singleElement() {
        // tanh(0) = 0
        assertClose(tanhArray([0.0])[0], 0.0)
    }

    func testTanhArray_typicalValues() {
        // numpy: np.tanh([0, 1, -1]) = [0, 0.76159415..., -0.76159415...]
        let result = tanhArray([0.0, 1.0, -1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], 0.7615941559557649)
        assertClose(result[2], -0.7615941559557649)
    }

    func testTanhArray_infinity() {
        // tanh(±inf) = ±1
        assertClose(tanhArray([.infinity])[0], 1.0)
        assertClose(tanhArray([-.infinity])[0], -1.0)
    }
}
