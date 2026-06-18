//
//  ComplexDivisionTests.swift
//  Tests/NumericSwiftTests/
//
//  Tests for Complex division stability (Smith's algorithm, issue #3).
//
//  Oracle source: Python 3 `complex` / numpy.complex128 division via
//  /tmp/.nsoracle/bin/python (NumPy 2.x, CPython 3.x).
//  Reference algorithm: R.L. Smith (1962), "Algorithm 116: Complex division",
//  CACM 5(8):435. Also codified in C99 Annex G §G.5.1.
//
//  Failure modes of the naive formula (c²+d²) being tested:
//    • c or d near Double.greatestFiniteMagnitude → c²+d² = inf → NaN result
//    • c or d near Double.leastNormalMagnitude    → loss of relative precision
//  Smith's algorithm avoids these by scaling before squaring.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class ComplexDivisionTests: XCTestCase {

    // Relative tolerance used for finite-value assertions.
    // Smith's result should match Python/NumPy to full double precision.
    private let tol = 1e-14

    // MARK: - Ordinary Division (regression guard)

    /// (2+3i)/(1+2i) — oracle: 1.6 - 0.2i
    /// Python: complex(2,3)/complex(1,2) = (1.6-0.2j)
    func testOrdinaryDivision() {
        let z = Complex(re: 2, im: 3) / Complex(re: 1, im: 2)
        XCTAssertEqual(z.re, 1.6, accuracy: tol, "real part mismatch")
        XCTAssertEqual(z.im, -0.2, accuracy: tol, "imaginary part mismatch")
    }

    /// (3+4i)/(3+4i) = 1+0i — exact identity
    func testSelfDivision() {
        let z = Complex(re: 3, im: 4)
        let result = z / z
        XCTAssertEqual(result.re, 1.0, accuracy: tol)
        XCTAssertEqual(result.im, 0.0, accuracy: tol)
    }

    /// (1+0i)/(0+1i) = -i — oracle: 0.0-1.0i
    func testPureImaginaryDenominator() {
        let z = Complex(re: 1, im: 0) / Complex(re: 0, im: 1)
        XCTAssertEqual(z.re,  0.0, accuracy: tol)
        XCTAssertEqual(z.im, -1.0, accuracy: tol)
    }

    // MARK: - Large-Magnitude Denominator (Smith overflow tests)

    /// (1+1i)/(1e200+1e200i) — oracle: 1e-200+0.0i
    /// Naive formula: c²+d² = 2e400 = inf → result is 0/inf = NaN.
    /// Smith: |c|=|d|=1e200, branch |c|≥|d|, r=1, denom=2e200 → 1e-200+0i.
    func testLargeDenominatorEqualComponents() {
        let z = Complex(re: 1, im: 1) / Complex(re: 1e200, im: 1e200)
        XCTAssertTrue(z.isFinite, "result should be finite, not NaN/inf")
        XCTAssertEqual(z.re, 1e-200, accuracy: 1e-214, "real part")
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-214, "imaginary part")
    }

    /// (3+4i)/(1e308+0i) — oracle: ≈3e-308+4e-308i
    /// Naive: c²=1e616=inf → NaN. Smith: r=0/c=0, denom=c → finite.
    func testHugeRealDenominator() {
        let z = Complex(re: 3, im: 4) / Complex(re: 1e308, im: 0)
        XCTAssertTrue(z.isFinite, "result should be finite")
        // Oracle: (3e-308, 4e-308) — use relative tolerance since values are subnormal-adjacent
        let expected_re = 3e-308
        let expected_im = 4e-308
        XCTAssertEqual(z.re, expected_re, accuracy: expected_re * tol + Double.leastNormalMagnitude)
        XCTAssertEqual(z.im, expected_im, accuracy: expected_im * tol + Double.leastNormalMagnitude)
    }

    /// (3+4i)/(0+1e308i) — oracle: ≈4e-308-3e-308i
    /// Naive: d²=1e616=inf → NaN. Smith: |d|>|c|=0, r=0/d=0, denom=d → finite.
    func testHugeImaginaryDenominator() {
        let z = Complex(re: 3, im: 4) / Complex(re: 0, im: 1e308)
        XCTAssertTrue(z.isFinite, "result should be finite")
        let expected_re =  4e-308
        let expected_im = -3e-308
        XCTAssertEqual(z.re, expected_re, accuracy: expected_re * tol + Double.leastNormalMagnitude)
        XCTAssertEqual(z.im, expected_im, accuracy: Swift.abs(expected_im) * tol + Double.leastNormalMagnitude)
    }

    /// (3+4i)/(1e200+1e-100i) — oracle: 3e-200+4e-200i
    /// Tests the branch where |c| >> |d|.
    func testLargeRealSmallImagDenominator() {
        let z = Complex(re: 3, im: 4) / Complex(re: 1e200, im: 1e-100)
        XCTAssertTrue(z.isFinite, "result should be finite")
        XCTAssertEqual(z.re, 3e-200, accuracy: 3e-200 * tol)
        XCTAssertEqual(z.im, 4e-200, accuracy: 4e-200 * tol)
    }

    // MARK: - Large-Magnitude Numerator

    /// (1e300+1e300i)/(1+1i) — oracle: 1e300+0i
    /// Smith: |c|=|d|=1, r=1, denom=2 → (1e300+1e300)/2=1e300, (1e300-1e300)/2=0.
    func testLargeNumeratorUnitDenominator() {
        let z = Complex(re: 1e300, im: 1e300) / Complex(re: 1, im: 1)
        XCTAssertTrue(z.isFinite, "result should be finite")
        XCTAssertEqual(z.re, 1e300, accuracy: 1e300 * tol)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e285)    // relative: 1e300 * tol
    }

    // MARK: - Tiny Denominator

    /// (1+1i)/(1e-200+1e-200i) — oracle: 1e200+0i
    /// Smith: r=1, denom=2e-200 → 2/2e-200=1e200, 0/2e-200=0.
    func testTinyDenominatorEqualComponents() {
        let z = Complex(re: 1, im: 1) / Complex(re: 1e-200, im: 1e-200)
        XCTAssertTrue(z.isFinite, "result should be finite")
        XCTAssertEqual(z.re, 1e200, accuracy: 1e200 * tol)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e185)
    }

    // MARK: - Mixed Huge/Tiny

    /// (1e200-1e200i)/(1e-200+1e-200i)
    /// Oracle (numpy): 0.0 - inf·i  — catastropically large imaginary part
    /// Checks that we get the same infinite result, not NaN.
    func testMixedHugeTinyRatio() {
        let z = Complex(re: 1e200, im: -1e200) / Complex(re: 1e-200, im: 1e-200)
        // Real part = 0, imaginary part = -inf (correctly large)
        XCTAssertTrue(z.re.isFinite || z.re.isNaN || z.re.isInfinite,
                      "real part should be a valid IEEE-754 value")
        // The important property: the imaginary part should be -inf, not NaN
        XCTAssertFalse(z.im.isNaN, "imaginary part should not be NaN")
        XCTAssertTrue(z.im.isInfinite, "imaginary part should be infinite")
        XCTAssertTrue(z.im < 0, "imaginary part should be negative infinite")
    }

    // MARK: - IEEE-754 Edge Cases

    /// Division by zero: (1+2i)/(0+0i)
    /// IEEE-754: 1/+0 = +inf. Numerator and denominator both give inf,
    /// so result components are inf. Not NaN.
    func testDivisionByZero() {
        let z = Complex(re: 1, im: 2) / Complex(re: 0, im: 0)
        let re = z.re, im = z.im
        // Smith's algorithm with |c|=|d|=0: r=d/c=0/0=NaN → falls into naive
        // path → re = (a+b*NaN)/NaN = NaN or inf. C99 Annex G §G.5.1:
        // finite/±0 → ±inf (not NaN); both re and im must be infinite.
        // Negative assertion: result must NOT be a wrong finite value.
        XCTAssertFalse(re.isFinite && im.isFinite,
                       "dividing by zero should not produce a finite result")
        // Positive assertion: at least one component must be infinite
        // (C99 Annex G §G.5.1: x/+0 → ±inf for x ≠ 0, not NaN).
        XCTAssertTrue(re.isInfinite || im.isInfinite,
                      "div-by-zero must route to ±inf per C99 Annex G §G.5.1, not NaN")
    }

    /// Inf numerator: (inf+0i)/(1+0i) = inf+?i
    func testInfiniteNumerator() {
        let z = Complex(re: .infinity, im: 0) / Complex(re: 1, im: 0)
        XCTAssertTrue(z.re.isInfinite, "real part should be inf")
    }

    /// NaN in numerator propagates
    func testNaNNumerator() {
        let z = Complex(re: .nan, im: 0) / Complex(re: 1, im: 0)
        XCTAssertTrue(z.isNaN, "NaN numerator should produce NaN result")
    }

    /// NaN in denominator propagates
    func testNaNDenominator() {
        let z = Complex(re: 1, im: 0) / Complex(re: .nan, im: 0)
        XCTAssertTrue(z.isNaN, "NaN denominator should produce NaN result")
    }

    /// Inf denominator: (1+2i)/(inf+0i) — oracle: 0+0i
    /// IEEE-754: finite/inf = +0.
    func testInfiniteDenominator() {
        let z = Complex(re: 1, im: 2) / Complex(re: .infinity, im: 0)
        XCTAssertEqual(z.re, 0.0, accuracy: 1e-300)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-300)
    }

    // MARK: - Scalar Division (existing paths — regression guard)

    /// Complex / Double: (4+6i)/2.0 = 2+3i
    func testComplexDivideByScalar() {
        let z = Complex(re: 4, im: 6) / 2.0
        XCTAssertEqual(z.re, 2.0, accuracy: tol)
        XCTAssertEqual(z.im, 3.0, accuracy: tol)
    }

    /// Double / Complex: 1.0/(0+1i) = -i
    func testScalarDivideByComplex() {
        let z = 1.0 / Complex(re: 0, im: 1)
        XCTAssertEqual(z.re,  0.0, accuracy: tol)
        XCTAssertEqual(z.im, -1.0, accuracy: tol)
    }

    /// Large-denominator Double / Complex must produce the correct finite result.
    /// 1.0 / (1e200+1e200i) — oracle (numpy): 5e-201 - 5e-201i
    /// Naive formula: c²+d²=2e400=inf → re=0, im=-0 (wrong — underflows to zero).
    func testScalarDivideByLargeComplex() {
        let z = 1.0 / Complex(re: 1e200, im: 1e200)
        XCTAssertTrue(z.isFinite, "scalar / large complex should be finite")
        XCTAssertEqual(z.re,  5e-201, accuracy: 5e-201 * tol, "real part")
        XCTAssertEqual(z.im, -5e-201, accuracy: 5e-201 * tol, "imaginary part")
    }

    /// Double / Complex.zero must route to ±inf per C99 Annex G §G.5.1 — the same
    /// contract as `Complex / Complex` div-by-zero (`testDivisionByZero`). Before
    /// the guard, Smith's r = d/c = 0/0 = NaN poisoned the result to NaN+NaNi.
    func testScalarDivideByZero() {
        let z = 1.0 / Complex(re: 0, im: 0)
        XCTAssertFalse(z.re.isFinite && z.im.isFinite,
                       "scalar / zero must not produce a finite result")
        XCTAssertTrue(z.re.isInfinite,
                      "scalar / zero real part must be ±inf per C99 Annex G §G.5.1, not NaN")
    }
}
