//
//  EdgeInputHardeningTests.swift
//  NumericSwiftTests
//
//  Regression tests for the Codex convergence-audit round-5 hardening: public
//  functions in Geometry, Series, Integration, NumberTheory, and SpecialFunctions
//  must return a documented sentinel on degenerate/malformed input rather than
//  trap (force-unwrap, invalid Range, negative array count, divide-by-zero,
//  Int.min abs, or Int→Int32 narrowing).
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest

@testable import NumericSwift

final class EdgeInputHardeningTests: XCTestCase {

  // MARK: - NumberTheory

  /// gcd/lcm must not trap on Int.min (abs(Int.min) is not representable as Int).
  func testGcdLcmIntMinDoesNotTrap() {
    XCTAssertEqual(NumberTheory.gcd(Int.min, 0), Int.max)  // |Int.min| clamps to Int.max
    XCTAssertEqual(NumberTheory.gcd(Int.min, 2), 2)        // 2^63 and 2 → 2
    XCTAssertEqual(NumberTheory.lcm(Int.min, 0), 0)
    // Large-product overflow clamps to Int.max rather than trapping.
    XCTAssertEqual(NumberTheory.lcm(Int.max, Int.max - 1), Int.max)
  }

  /// modInverse with a non-positive modulus returns nil (x % 0 would trap).
  func testModInverseNonPositiveModulusReturnsNil() {
    XCTAssertNil(NumberTheory.modInverse(1, 0))
    XCTAssertNil(NumberTheory.modInverse(3, -7))
  }

  /// primePi / chebyshev must not trap converting a non-finite or out-of-range
  /// Double to Int.
  func testPrimeCountingNonFiniteReturnsZero() {
    XCTAssertEqual(NumberTheory.primePi(.infinity), 0)
    XCTAssertEqual(NumberTheory.primePi(.nan), 0)
    XCTAssertEqual(NumberTheory.chebyshevTheta(.infinity), 0)
    XCTAssertEqual(NumberTheory.chebyshevPsi(.infinity), 0)
  }

  // MARK: - Series

  /// Negative term/count must yield an empty array, not an invalid Range trap.
  func testSeriesNegativeCountsReturnEmpty() {
    XCTAssertEqual(Series.taylorCoefficients(for: "sin", terms: -1), [])
    XCTAssertTrue(Series.partialSums(from: 0, count: -3) { Double($0) }.isEmpty)
  }

  // MARK: - Integration

  /// romberg must not trap on divmax < 1 (clamped to the documented minimum).
  func testRombergNonPositiveDivmaxDoesNotTrap() {
    let r = romberg({ $0 }, 0.0, 1.0, divmax: 0)
    XCTAssertTrue(r.value.isFinite)
    XCTAssertEqual(r.value, 0.5, accuracy: 1e-9)  // ∫₀¹ x dx = 0.5
  }

  // MARK: - Geometry (B-spline)

  /// bsplineBasis returns 0 for out-of-range indices/degree or too-short knots.
  func testBsplineBasisOutOfRangeReturnsZero() {
    XCTAssertEqual(Geometry.bsplineBasis(i: -1, degree: 0, t: 0.5, knots: [0, 1]), 0.0)
    XCTAssertEqual(Geometry.bsplineBasis(i: 5, degree: 2, t: 0.5, knots: [0, 1]), 0.0)
    XCTAssertEqual(Geometry.bsplineBasis(i: 0, degree: -1, t: 0.5, knots: [0, 1]), 0.0)
  }

  /// bsplineUniformKnots returns [] for negative n/degree (no negative-count alloc).
  func testBsplineUniformKnotsNegativeReturnsEmpty() {
    XCTAssertTrue(Geometry.bsplineUniformKnots(n: -1, degree: 2).isEmpty)
    XCTAssertTrue(Geometry.bsplineUniformKnots(n: 3, degree: -1).isEmpty)
  }

  // MARK: - SpecialFunctions (integer-order Bessel)

  /// Integer-order Bessel APIs return NaN for an order outside Int32 range instead
  /// of trapping on abs(Int.min) or the Int→Int32 narrowing.
  func testBesselIntMinOrderReturnsNaN() {
    XCTAssertTrue(besseli(Int.min, 1.0).isNaN)
    XCTAssertTrue(besselk(Int.min, 1.0).isNaN)
    XCTAssertTrue(NumericSwift.jn(Int.min, 1.0).isNaN)
    XCTAssertTrue(NumericSwift.yn(Int.min, 1.0).isNaN)
  }
}
