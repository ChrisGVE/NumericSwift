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
        XCTAssertEqual(ArrayOps.roundValue(0.5), 1.0)
        XCTAssertEqual(ArrayOps.roundValue(1.5), 2.0)
        XCTAssertEqual(ArrayOps.roundValue(2.5), 3.0)
    }

    func testRoundValue_negative() {
        XCTAssertEqual(ArrayOps.roundValue(-0.5), -1.0)
        XCTAssertEqual(ArrayOps.roundValue(-1.4), -1.0)
        XCTAssertEqual(ArrayOps.roundValue(-1.6), -2.0)
    }

    func testRoundValue_zero() {
        XCTAssertEqual(ArrayOps.roundValue(0.0), 0.0)
    }

    func testRoundValue_infinity() {
        // round(±inf) == ±inf  (IEEE-754)
        XCTAssertEqual(ArrayOps.roundValue(.infinity), .infinity)
        XCTAssertEqual(ArrayOps.roundValue(-.infinity), -.infinity)
    }

    func testRoundValue_nan() {
        assertNaN(ArrayOps.roundValue(.nan), "round(nan)")
    }

    // MARK: - truncValue

    func testTruncValue_positive() {
        XCTAssertEqual(ArrayOps.truncValue(3.9), 3.0)
        XCTAssertEqual(ArrayOps.truncValue(3.1), 3.0)
    }

    func testTruncValue_negative() {
        XCTAssertEqual(ArrayOps.truncValue(-3.9), -3.0)
        XCTAssertEqual(ArrayOps.truncValue(-3.1), -3.0)
    }

    func testTruncValue_zero() {
        XCTAssertEqual(ArrayOps.truncValue(0.0), 0.0)
    }

    func testTruncValue_infinity() {
        XCTAssertEqual(ArrayOps.truncValue(.infinity), .infinity)
        XCTAssertEqual(ArrayOps.truncValue(-.infinity), -.infinity)
    }

    func testTruncValue_nan() {
        assertNaN(ArrayOps.truncValue(.nan))
    }

    // MARK: - signValue

    func testSignValue_positive() {
        XCTAssertEqual(ArrayOps.signValue(3.7), 1.0)
        XCTAssertEqual(ArrayOps.signValue(.infinity), 1.0)
    }

    func testSignValue_negative() {
        XCTAssertEqual(ArrayOps.signValue(-3.7), -1.0)
        XCTAssertEqual(ArrayOps.signValue(-.infinity), -1.0)
    }

    func testSignValue_zero() {
        XCTAssertEqual(ArrayOps.signValue(0.0), 0.0)
        XCTAssertEqual(ArrayOps.signValue(-0.0), 0.0)  // -0.0 is not < 0
    }

    func testSignValue_nan() {
        // nan is not > 0 and not < 0, so returns 0.0
        XCTAssertEqual(ArrayOps.signValue(.nan), 0.0)
    }

    // MARK: - clipValue

    func testClipValue_insideRange() {
        XCTAssertEqual(ArrayOps.clipValue(3.0, lo: 0.0, hi: 5.0), 3.0)
    }

    func testClipValue_belowLo() {
        XCTAssertEqual(ArrayOps.clipValue(-1.0, lo: 0.0, hi: 5.0), 0.0)
    }

    func testClipValue_aboveHi() {
        XCTAssertEqual(ArrayOps.clipValue(10.0, lo: 0.0, hi: 5.0), 5.0)
    }

    func testClipValue_atBoundary() {
        XCTAssertEqual(ArrayOps.clipValue(0.0, lo: 0.0, hi: 5.0), 0.0)
        XCTAssertEqual(ArrayOps.clipValue(5.0, lo: 0.0, hi: 5.0), 5.0)
    }

    func testClipValue_negativeRange() {
        XCTAssertEqual(ArrayOps.clipValue(-3.0, lo: -5.0, hi: -1.0), -3.0)
        XCTAssertEqual(ArrayOps.clipValue(0.0, lo: -5.0, hi: -1.0), -1.0)
        XCTAssertEqual(ArrayOps.clipValue(-10.0, lo: -5.0, hi: -1.0), -5.0)
    }

    // MARK: - roundArray

    func testRoundArray_empty() {
        XCTAssertEqual(ArrayOps.roundArray([]), [])
    }

    func testRoundArray_singleElement() {
        XCTAssertEqual(ArrayOps.roundArray([2.7]), [3.0])
    }

    func testRoundArray_typicalValues() {
        // Darwin.round uses round-half-away-from-zero (C99 §7.12.9.6):
        //   1.1 → 1.0  (below midpoint)
        //   1.5 → 2.0  (half → away from zero, i.e. up)
        //  -1.5 → -2.0 (half → away from zero, i.e. down)
        //   2.9 → 3.0  (above midpoint)
        // Input: [1.1, 1.5, -1.5, 2.9]
        let result = ArrayOps.roundArray([1.1, 1.5, -1.5, 2.9])
        XCTAssertEqual(result, [1.0, 2.0, -2.0, 3.0])
    }

    func testRoundArray_infinity() {
        let result = ArrayOps.roundArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], .infinity)
        XCTAssertEqual(result[1], -.infinity)
    }

    func testRoundArray_nan() {
        let result = ArrayOps.roundArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - truncArray

    func testTruncArray_empty() {
        XCTAssertEqual(ArrayOps.truncArray([]), [])
    }

    func testTruncArray_singleElement() {
        XCTAssertEqual(ArrayOps.truncArray([3.9]), [3.0])
    }

    func testTruncArray_mixedSigns() {
        let result = ArrayOps.truncArray([1.9, -1.9, 0.1, -0.1])
        XCTAssertEqual(result, [1.0, -1.0, 0.0, 0.0])
    }

    // MARK: - signArray

    func testSignArray_empty() {
        XCTAssertEqual(ArrayOps.signArray([]), [])
    }

    func testSignArray_singleElement() {
        XCTAssertEqual(ArrayOps.signArray([5.0]), [1.0])
        XCTAssertEqual(ArrayOps.signArray([-5.0]), [-1.0])
        XCTAssertEqual(ArrayOps.signArray([0.0]), [0.0])
    }

    func testSignArray_mixedValues() {
        let result = ArrayOps.signArray([3.0, -2.0, 0.0, 1e-300, -1e-300])
        XCTAssertEqual(result, [1.0, -1.0, 0.0, 1.0, -1.0])
    }

    func testSignArray_infinity() {
        let result = ArrayOps.signArray([.infinity, -.infinity])
        XCTAssertEqual(result, [1.0, -1.0])
    }

    // MARK: - clipArray

    func testClipArray_empty() {
        XCTAssertEqual(ArrayOps.clipArray([], lo: 0, hi: 1), [])
    }

    func testClipArray_singleElement() {
        XCTAssertEqual(ArrayOps.clipArray([3.0], lo: 0, hi: 2), [2.0])
    }

    func testClipArray_allBelow() {
        let result = ArrayOps.clipArray([-5.0, -3.0, -1.0], lo: 0, hi: 10)
        XCTAssertEqual(result, [0.0, 0.0, 0.0])
    }

    func testClipArray_allAbove() {
        let result = ArrayOps.clipArray([11.0, 20.0, 100.0], lo: 0, hi: 10)
        XCTAssertEqual(result, [10.0, 10.0, 10.0])
    }

    func testClipArray_mixedInOut() {
        // numpy: np.clip([-1,0,3,5,6], 0, 5) = [0,0,3,5,5]
        let result = ArrayOps.clipArray([-1.0, 0.0, 3.0, 5.0, 6.0], lo: 0, hi: 5)
        XCTAssertEqual(result, [0.0, 0.0, 3.0, 5.0, 5.0])
    }

    // MARK: - floorArray

    func testFloorArray_empty() {
        XCTAssertEqual(ArrayOps.floorArray([]), [])
    }

    func testFloorArray_singleElement() {
        XCTAssertEqual(ArrayOps.floorArray([2.7]), [2.0])
    }

    func testFloorArray_negativeValues() {
        // numpy: np.floor([-1.1, -1.9]) = [-2., -2.]
        let result = ArrayOps.floorArray([-1.1, -1.9, 1.1, 1.9])
        XCTAssertEqual(result, [-2.0, -2.0, 1.0, 1.0])
    }

    func testFloorArray_infinity() {
        let result = ArrayOps.floorArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], .infinity)
        XCTAssertEqual(result[1], -.infinity)
    }

    func testFloorArray_nan() {
        let result = ArrayOps.floorArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - ceilArray

    func testCeilArray_empty() {
        XCTAssertEqual(ArrayOps.ceilArray([]), [])
    }

    func testCeilArray_singleElement() {
        XCTAssertEqual(ArrayOps.ceilArray([2.1]), [3.0])
    }

    func testCeilArray_negativeValues() {
        // numpy: np.ceil([-1.1, -1.9]) = [-1., -1.]
        let result = ArrayOps.ceilArray([-1.1, -1.9, 1.1, 1.9])
        XCTAssertEqual(result, [-1.0, -1.0, 2.0, 2.0])
    }

    func testCeilArray_infinity() {
        let result = ArrayOps.ceilArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], .infinity)
        XCTAssertEqual(result[1], -.infinity)
    }

    // MARK: - absArray

    func testAbsArray_empty() {
        XCTAssertEqual(ArrayOps.absArray([]), [])
    }

    func testAbsArray_singleElement() {
        XCTAssertEqual(ArrayOps.absArray([-3.0]), [3.0])
    }

    func testAbsArray_mixedSigns() {
        let result = ArrayOps.absArray([-1.0, 0.0, 1.0, -100.0, 100.0])
        XCTAssertEqual(result, [1.0, 0.0, 1.0, 100.0, 100.0])
    }

    func testAbsArray_infinity() {
        let result = ArrayOps.absArray([.infinity, -.infinity])
        XCTAssertEqual(result, [.infinity, .infinity])
    }

    func testAbsArray_nan() {
        let result = ArrayOps.absArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - negArray

    func testNegArray_empty() {
        XCTAssertEqual(ArrayOps.negArray([]), [])
    }

    func testNegArray_singleElement() {
        XCTAssertEqual(ArrayOps.negArray([3.0]), [-3.0])
    }

    func testNegArray_mixedSigns() {
        let result = ArrayOps.negArray([1.0, -2.0, 0.0])
        XCTAssertEqual(result, [-1.0, 2.0, 0.0])
    }

    func testNegArray_infinity() {
        let result = ArrayOps.negArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], -.infinity)
        XCTAssertEqual(result[1], .infinity)
    }

    func testNegArray_nan() {
        let result = ArrayOps.negArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - sqrtArray

    func testSqrtArray_empty() {
        XCTAssertEqual(ArrayOps.sqrtArray([]), [])
    }

    func testSqrtArray_singleElement() {
        assertClose(ArrayOps.sqrtArray([4.0])[0], 2.0)
    }

    func testSqrtArray_typicalValues() {
        // numpy: np.sqrt([0, 1, 4, 9, 2]) = [0, 1, 2, 3, 1.41421356...]
        let result = ArrayOps.sqrtArray([0.0, 1.0, 4.0, 9.0, 2.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.0)
        assertClose(result[2], 2.0)
        assertClose(result[3], 3.0)
        assertClose(result[4], 1.4142135623730951)
    }

    func testSqrtArray_negativeReturnsNaN() {
        // IEEE-754: sqrt of negative real → NaN
        let result = ArrayOps.sqrtArray([-1.0])
        XCTAssertTrue(result[0].isNaN, "sqrt(-1) should be NaN for real array")
    }

    func testSqrtArray_infinity() {
        let result = ArrayOps.sqrtArray([.infinity])
        XCTAssertEqual(result[0], .infinity)
    }

    func testSqrtArray_nan() {
        let result = ArrayOps.sqrtArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - squareArray

    func testSquareArray_empty() {
        XCTAssertEqual(ArrayOps.squareArray([]), [])
    }

    func testSquareArray_singleElement() {
        assertClose(ArrayOps.squareArray([3.0])[0], 9.0)
    }

    func testSquareArray_negativesSquarePositive() {
        let result = ArrayOps.squareArray([-3.0, 0.0, 3.0])
        assertClose(result[0], 9.0)
        assertClose(result[1], 0.0)
        assertClose(result[2], 9.0)
    }

    func testSquareArray_infinity() {
        let result = ArrayOps.squareArray([.infinity, -.infinity])
        XCTAssertEqual(result[0], .infinity)
        XCTAssertEqual(result[1], .infinity)
    }

    // MARK: - logArray

    func testLogArray_empty() {
        XCTAssertEqual(ArrayOps.logArray([]), [])
    }

    func testLogArray_singleElement() {
        // ln(e) = 1
        assertClose(ArrayOps.logArray([MathConstants.e])[0], 1.0)
    }

    func testLogArray_typicalValues() {
        // numpy: np.log([1, e, e^2]) = [0, 1, 2]
        let result = ArrayOps.logArray([1.0, MathConstants.e, MathConstants.e * MathConstants.e])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.0)
        assertClose(result[2], 2.0, tol: 1e-10)
    }

    func testLogArray_zeroReturnsNegInf() {
        // ln(0) = -inf
        let result = ArrayOps.logArray([0.0])
        XCTAssertEqual(result[0], -.infinity)
    }

    func testLogArray_negativeReturnsNaN() {
        // ln(-1) = NaN for real
        let result = ArrayOps.logArray([-1.0])
        XCTAssertTrue(result[0].isNaN)
    }

    func testLogArray_infinity() {
        let result = ArrayOps.logArray([.infinity])
        XCTAssertEqual(result[0], .infinity)
    }

    // MARK: - log10Array

    func testLog10Array_empty() {
        XCTAssertEqual(ArrayOps.log10Array([]), [])
    }

    func testLog10Array_singleElement() {
        assertClose(ArrayOps.log10Array([100.0])[0], 2.0)
    }

    func testLog10Array_typicalValues() {
        // log10([1, 10, 1000]) = [0, 1, 3]
        let result = ArrayOps.log10Array([1.0, 10.0, 1000.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.0)
        assertClose(result[2], 3.0)
    }

    func testLog10Array_zeroReturnsNegInf() {
        let result = ArrayOps.log10Array([0.0])
        XCTAssertEqual(result[0], -.infinity)
    }

    func testLog10Array_negativeReturnsNaN() {
        let result = ArrayOps.log10Array([-1.0])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - expArray

    func testExpArray_empty() {
        XCTAssertEqual(ArrayOps.expArray([]), [])
    }

    func testExpArray_singleElement() {
        // exp(1) = e
        assertClose(ArrayOps.expArray([1.0])[0], MathConstants.e, tol: 1e-10)
    }

    func testExpArray_typicalValues() {
        // numpy: np.exp([0, 1, -1]) = [1, e, 1/e]
        let result = ArrayOps.expArray([0.0, 1.0, -1.0])
        assertClose(result[0], 1.0)
        assertClose(result[1], MathConstants.e, tol: 1e-10)
        assertClose(result[2], 1.0 / MathConstants.e, tol: 1e-10)
    }

    func testExpArray_negativeInfinity() {
        // exp(-inf) = 0
        let result = ArrayOps.expArray([-.infinity])
        XCTAssertEqual(result[0], 0.0)
    }

    func testExpArray_positiveInfinity() {
        let result = ArrayOps.expArray([.infinity])
        XCTAssertEqual(result[0], .infinity)
    }

    func testExpArray_nan() {
        let result = ArrayOps.expArray([.nan])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - powArray

    func testPowArray_emptyBases() {
        XCTAssertEqual(ArrayOps.powArray([], []), [])
    }

    func testPowArray_mismatchedLengths() {
        // Guard: mismatched → empty
        XCTAssertEqual(ArrayOps.powArray([1.0, 2.0], [1.0]), [])
    }

    func testPowArray_singleElement() {
        // 2^3 = 8
        assertClose(ArrayOps.powArray([2.0], [3.0])[0], 8.0)
    }

    func testPowArray_typicalValues() {
        // numpy: np.power([2,3,4], [2,2,0.5]) = [4, 9, 2]
        let result = ArrayOps.powArray([2.0, 3.0, 4.0], [2.0, 2.0, 0.5])
        assertClose(result[0], 4.0)
        assertClose(result[1], 9.0)
        assertClose(result[2], 2.0)
    }

    func testPowArray_zeroExponent() {
        // x^0 = 1 for x > 0
        let result = ArrayOps.powArray([5.0, 100.0], [0.0, 0.0])
        assertClose(result[0], 1.0)
        assertClose(result[1], 1.0)
    }

    // MARK: - sinArray

    func testSinArray_empty() {
        XCTAssertEqual(ArrayOps.sinArray([]), [])
    }

    func testSinArray_singleElement() {
        // sin(π/2) = 1
        assertClose(ArrayOps.sinArray([.pi / 2])[0], 1.0)
    }

    func testSinArray_typicalValues() {
        // numpy: np.sin([0, π/6, π/2, π]) ≈ [0, 0.5, 1, 0]
        let result = ArrayOps.sinArray([0.0, .pi / 6, .pi / 2, .pi])
        assertClose(result[0], 0.0)
        assertClose(result[1], 0.5)
        assertClose(result[2], 1.0)
        assertClose(result[3], 0.0, tol: 1e-15)
    }

    func testSinArray_infinity() {
        // sin(inf) = NaN per IEEE-754
        let result = ArrayOps.sinArray([.infinity])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - cosArray

    func testCosArray_empty() {
        XCTAssertEqual(ArrayOps.cosArray([]), [])
    }

    func testCosArray_singleElement() {
        // cos(0) = 1
        assertClose(ArrayOps.cosArray([0.0])[0], 1.0)
    }

    func testCosArray_typicalValues() {
        // numpy: np.cos([0, π/3, π/2, π]) ≈ [1, 0.5, 0, -1]
        let result = ArrayOps.cosArray([0.0, .pi / 3, .pi / 2, .pi])
        assertClose(result[0], 1.0)
        assertClose(result[1], 0.5)
        assertClose(result[2], 0.0, tol: 1e-15)
        assertClose(result[3], -1.0)
    }

    func testCosArray_infinity() {
        let result = ArrayOps.cosArray([.infinity])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - tanArray

    func testTanArray_empty() {
        XCTAssertEqual(ArrayOps.tanArray([]), [])
    }

    func testTanArray_singleElement() {
        // tan(π/4) = 1
        assertClose(ArrayOps.tanArray([.pi / 4])[0], 1.0)
    }

    func testTanArray_typicalValues() {
        // numpy: np.tan([0, π/4]) = [0, 1]
        let result = ArrayOps.tanArray([0.0, .pi / 4])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.0)
    }

    // MARK: - asinArray

    func testAsinArray_empty() {
        XCTAssertEqual(ArrayOps.asinArray([]), [])
    }

    func testAsinArray_singleElement() {
        // asin(1) = π/2
        assertClose(ArrayOps.asinArray([1.0])[0], .pi / 2)
    }

    func testAsinArray_typicalValues() {
        // numpy: np.arcsin([0, 0.5, 1]) = [0, π/6, π/2]
        let result = ArrayOps.asinArray([0.0, 0.5, 1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], .pi / 6)
        assertClose(result[2], .pi / 2)
    }

    func testAsinArray_outsideDomainReturnsNaN() {
        // asin(2) = NaN (out of domain)
        let result = ArrayOps.asinArray([2.0])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - acosArray

    func testAcosArray_empty() {
        XCTAssertEqual(ArrayOps.acosArray([]), [])
    }

    func testAcosArray_singleElement() {
        // acos(1) = 0
        assertClose(ArrayOps.acosArray([1.0])[0], 0.0)
    }

    func testAcosArray_typicalValues() {
        // numpy: np.arccos([1, 0.5, 0, -1]) = [0, π/3, π/2, π]
        let result = ArrayOps.acosArray([1.0, 0.5, 0.0, -1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], .pi / 3)
        assertClose(result[2], .pi / 2)
        assertClose(result[3], .pi)
    }

    func testAcosArray_outsideDomainReturnsNaN() {
        let result = ArrayOps.acosArray([2.0])
        XCTAssertTrue(result[0].isNaN)
    }

    // MARK: - atanArray

    func testAtanArray_empty() {
        XCTAssertEqual(ArrayOps.atanArray([]), [])
    }

    func testAtanArray_singleElement() {
        // atan(1) = π/4
        assertClose(ArrayOps.atanArray([1.0])[0], .pi / 4)
    }

    func testAtanArray_typicalValues() {
        // numpy: np.arctan([0, 1, -1]) = [0, π/4, -π/4]
        let result = ArrayOps.atanArray([0.0, 1.0, -1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], .pi / 4)
        assertClose(result[2], -.pi / 4)
    }

    func testAtanArray_infinity() {
        // atan(inf) = π/2
        assertClose(ArrayOps.atanArray([.infinity])[0], .pi / 2)
        assertClose(ArrayOps.atanArray([-.infinity])[0], -.pi / 2)
    }

    // MARK: - sinhArray

    func testSinhArray_empty() {
        XCTAssertEqual(ArrayOps.sinhArray([]), [])
    }

    func testSinhArray_singleElement() {
        // sinh(0) = 0
        assertClose(ArrayOps.sinhArray([0.0])[0], 0.0)
    }

    func testSinhArray_typicalValues() {
        // numpy: np.sinh([0, 1, -1]) = [0, 1.1752011936..., -1.1752011936...]
        let result = ArrayOps.sinhArray([0.0, 1.0, -1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], 1.1752011936438014)
        assertClose(result[2], -1.1752011936438014)
    }

    // MARK: - coshArray

    func testCoshArray_empty() {
        XCTAssertEqual(ArrayOps.coshArray([]), [])
    }

    func testCoshArray_singleElement() {
        // cosh(0) = 1
        assertClose(ArrayOps.coshArray([0.0])[0], 1.0)
    }

    func testCoshArray_typicalValues() {
        // numpy: np.cosh([0, 1]) = [1, 1.5430806348...]
        let result = ArrayOps.coshArray([0.0, 1.0])
        assertClose(result[0], 1.0)
        assertClose(result[1], 1.5430806348152437)
    }

    // MARK: - tanhArray

    func testTanhArray_empty() {
        XCTAssertEqual(ArrayOps.tanhArray([]), [])
    }

    func testTanhArray_singleElement() {
        // tanh(0) = 0
        assertClose(ArrayOps.tanhArray([0.0])[0], 0.0)
    }

    func testTanhArray_typicalValues() {
        // numpy: np.tanh([0, 1, -1]) = [0, 0.76159415..., -0.76159415...]
        let result = ArrayOps.tanhArray([0.0, 1.0, -1.0])
        assertClose(result[0], 0.0)
        assertClose(result[1], 0.7615941559557649)
        assertClose(result[2], -0.7615941559557649)
    }

    func testTanhArray_infinity() {
        // tanh(±inf) = ±1
        assertClose(ArrayOps.tanhArray([.infinity])[0], 1.0)
        assertClose(ArrayOps.tanhArray([-.infinity])[0], -1.0)
    }
}
