//
//  NumericSwiftTests.swift
//  NumericSwift
//
//  Licensed under the MIT License.
//

import XCTest
@testable import NumericSwift

final class NumericSwiftTests: XCTestCase {

    // MARK: - Complex Number Tests

    func testComplexCreation() {
        let z = Complex(re: 3, im: 4)
        XCTAssertEqual(z.re, 3)
        XCTAssertEqual(z.im, 4)
    }

    func testComplexMagnitude() {
        let z = Complex(re: 3, im: 4)
        XCTAssertEqual(z.abs, 5, accuracy: 1e-10)
    }

    func testComplexAddition() {
        let z1 = Complex(re: 1, im: 2)
        let z2 = Complex(re: 3, im: 4)
        let sum = z1 + z2
        XCTAssertEqual(sum.re, 4, accuracy: 1e-10)
        XCTAssertEqual(sum.im, 6, accuracy: 1e-10)
    }

    func testComplexMultiplication() {
        let z1 = Complex(re: 3, im: 4)
        let z2 = Complex(re: 1, im: 2)
        let product = z1 * z2
        // (3+4i)(1+2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
        XCTAssertEqual(product.re, -5, accuracy: 1e-10)
        XCTAssertEqual(product.im, 10, accuracy: 1e-10)
    }

    func testComplexConjugate() {
        let z = Complex(re: 3, im: 4)
        let conj = z.conj
        XCTAssertEqual(conj.re, 3, accuracy: 1e-10)
        XCTAssertEqual(conj.im, -4, accuracy: 1e-10)
    }

    func testComplexExp() {
        // e^(i*pi) = -1
        let z = Complex(re: 0, im: .pi)
        let result = z.exp
        XCTAssertEqual(result.re, -1, accuracy: 1e-10)
        XCTAssertEqual(result.im, 0, accuracy: 1e-10)
    }

    // MARK: - Constants Tests

    func testMathConstants() {
        XCTAssertEqual(MathConstants.pi, .pi, accuracy: 1e-15)
        XCTAssertEqual(MathConstants.e, 2.718281828459045, accuracy: 1e-10)
        XCTAssertEqual(MathConstants.sqrt2, sqrt(2), accuracy: 1e-10)
    }

    func testPhysicalConstants() {
        XCTAssertEqual(PhysicalConstants.c, 299792458)
        XCTAssertEqual(PhysicalConstants.h, 6.62607015e-34, accuracy: 1e-44)
    }

    // MARK: - Statistics Tests

    func testMean() {
        XCTAssertEqual(mean([1, 2, 3, 4, 5]), 3, accuracy: 1e-10)
        XCTAssertEqual(mean([10]), 10, accuracy: 1e-10)
        XCTAssert(mean([]).isNaN)
    }

    func testMedian() {
        XCTAssertEqual(median([1, 2, 3, 4, 5]), 3, accuracy: 1e-10)
        XCTAssertEqual(median([1, 2, 3, 4]), 2.5, accuracy: 1e-10)
        XCTAssert(median([]).isNaN)
    }

    func testVariance() {
        // Population variance
        XCTAssertEqual(variance([2, 4, 4, 4, 5, 5, 7, 9], ddof: 0), 4, accuracy: 1e-10)
        // Sample variance
        XCTAssertEqual(variance([2, 4, 4, 4, 5, 5, 7, 9], ddof: 1), 4.571428571428571, accuracy: 1e-10)
    }

    func testStddev() {
        XCTAssertEqual(stddev([2, 4, 4, 4, 5, 5, 7, 9], ddof: 0), 2, accuracy: 1e-10)
    }

    func testPercentile() {
        XCTAssertEqual(percentile([1, 2, 3, 4, 5], 50), 3, accuracy: 1e-10)
        XCTAssertEqual(percentile([1, 2, 3, 4, 5], 0), 1, accuracy: 1e-10)
        XCTAssertEqual(percentile([1, 2, 3, 4, 5], 100), 5, accuracy: 1e-10)
    }

    func testGeometricMean() {
        XCTAssertEqual(gmean([1, 2, 4, 8]), 2.8284271247461903, accuracy: 1e-10)
        XCTAssert(gmean([1, 0, 2]).isNaN) // Non-positive values
    }

    func testHarmonicMean() {
        XCTAssertEqual(hmean([1, 2, 4]), 1.7142857142857142, accuracy: 1e-10)
    }

    func testMode() {
        XCTAssertEqual(mode([1, 2, 2, 3, 3, 3]), 3, accuracy: 1e-10)
        XCTAssertEqual(mode([1, 1, 2, 2]), 1, accuracy: 1e-10) // Returns smallest on ties
    }

    // MARK: - Cumulative Functions Tests

    func testCumsum() {
        XCTAssertEqual(cumsum([1, 2, 3, 4]), [1, 3, 6, 10])
    }

    func testCumprod() {
        XCTAssertEqual(cumprod([1, 2, 3, 4]), [1, 2, 6, 24])
    }

    func testDiff() {
        XCTAssertEqual(diff([1, 3, 6, 10]), [2, 3, 4])
    }

    // MARK: - Combinatorics Tests

    func testFactorial() {
        XCTAssertEqual(factorial(0), 1, accuracy: 1e-10)
        XCTAssertEqual(factorial(5), 120, accuracy: 1e-10)
        XCTAssertEqual(factorial(10), 3628800, accuracy: 1e-10)
    }

    func testPermutations() {
        XCTAssertEqual(perm(5, 2), 20, accuracy: 1e-10)
        XCTAssertEqual(perm(5, 0), 1, accuracy: 1e-10)
    }

    func testCombinations() {
        XCTAssertEqual(comb(5, 2), 10, accuracy: 1e-10)
        XCTAssertEqual(comb(10, 3), 120, accuracy: 1e-10)
        XCTAssertEqual(binomial(5, 2), 10, accuracy: 1e-10) // Alias
    }

    // MARK: - Coordinate Conversion Tests

    func testPolarToCartesian() {
        let (x, y) = polarToCart(r: 1, theta: .pi / 2)
        XCTAssertEqual(x, 0, accuracy: 1e-10)
        XCTAssertEqual(y, 1, accuracy: 1e-10)
    }

    func testCartesianToPolar() {
        let (r, theta) = cartToPolar(x: 1, y: 1)
        XCTAssertEqual(r, sqrt(2), accuracy: 1e-10)
        XCTAssertEqual(theta, .pi / 4, accuracy: 1e-10)
    }

    func testDegRadConversions() {
        XCTAssertEqual(deg2rad(180), .pi, accuracy: 1e-10)
        XCTAssertEqual(rad2deg(.pi), 180, accuracy: 1e-10)
    }

    func testClip() {
        XCTAssertEqual(clip(5, min: 0, max: 10), 5)
        XCTAssertEqual(clip(-5, min: 0, max: 10), 0)
        XCTAssertEqual(clip(15, min: 0, max: 10), 10)
        XCTAssertEqual(clip([1, 5, 10, 15], min: 3, max: 12), [3, 5, 10, 12])
    }
}
