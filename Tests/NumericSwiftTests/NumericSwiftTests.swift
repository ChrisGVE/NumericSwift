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
}
