//
//  DistributionPpfGuardTests.swift
//  NumericSwift
//
//  Verifies the ppf p-domain contract (audit "ppf p-range guards"): every
//  continuous distribution returns `NaN` for p strictly outside [0, 1], and the
//  support endpoints at p == 0 and p == 1 — matching scipy.stats.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

final class DistributionPpfGuardTests: XCTestCase {

  // MARK: - Out-of-range p → NaN (every distribution)

  func testOutOfRangeReturnsNaN() {
    XCTAssertTrue(NormalDistribution().ppf(-0.1).isNaN)
    XCTAssertTrue(NormalDistribution().ppf(1.1).isNaN)
    XCTAssertTrue(UniformDistribution(loc: 2, scale: 3).ppf(-1e-9).isNaN)
    XCTAssertTrue(ExponentialDistribution().ppf(2).isNaN)
    XCTAssertTrue(TDistribution(df: 5).ppf(-0.5).isNaN)
    XCTAssertTrue(ChiSquaredDistribution(df: 3).ppf(1.0001).isNaN)
    XCTAssertTrue(FDistribution(dfn: 3, dfd: 7).ppf(-0.2).isNaN)
    XCTAssertTrue(GammaDistribution(shape: 2).ppf(5).isNaN)
    XCTAssertTrue(BetaDistribution(a: 2, b: 3).ppf(1.5).isNaN)
    XCTAssertTrue(NormalDistribution().ppf(.nan).isNaN)
  }

  // MARK: - Unbounded-below distributions: p==0 → -inf, p==1 → +inf

  func testNormalBoundaries() {
    XCTAssertEqual(NormalDistribution().ppf(0), -.infinity)
    XCTAssertEqual(NormalDistribution().ppf(1), .infinity)
  }

  func testTBoundaries() {
    XCTAssertEqual(TDistribution(df: 4).ppf(0), -.infinity)
    XCTAssertEqual(TDistribution(df: 4).ppf(1), .infinity)
  }

  // MARK: - Half-line distributions: p==0 → loc, p==1 → +inf

  func testExponentialBoundaries() {
    let d = ExponentialDistribution(loc: 1.5, scale: 2)
    XCTAssertEqual(d.ppf(0), 1.5)
    XCTAssertEqual(d.ppf(1), .infinity)
  }

  func testChiSquaredBoundaries() {
    XCTAssertEqual(ChiSquaredDistribution(df: 3).ppf(0), 0)
    XCTAssertEqual(ChiSquaredDistribution(df: 3).ppf(1), .infinity)
  }

  func testFBoundaries() {
    XCTAssertEqual(FDistribution(dfn: 3, dfd: 7).ppf(0), 0)
    XCTAssertEqual(FDistribution(dfn: 3, dfd: 7).ppf(1), .infinity)
  }

  func testGammaBoundaries() {
    XCTAssertEqual(GammaDistribution(shape: 2).ppf(0), 0)
    XCTAssertEqual(GammaDistribution(shape: 2).ppf(1), .infinity)
  }

  // MARK: - Bounded distributions: p==0 → lower, p==1 → upper

  func testUniformBoundaries() {
    let d = UniformDistribution(loc: 2, scale: 3)   // support [2, 5]
    XCTAssertEqual(d.ppf(0), 2)
    XCTAssertEqual(d.ppf(1), 5)
  }

  func testBetaBoundaries() {
    let d = BetaDistribution(a: 2, b: 3)            // support [0, 1]
    XCTAssertEqual(d.ppf(0), 0)
    XCTAssertEqual(d.ppf(1), 1)
  }

  // MARK: - Interior p still computes normally (guard does not interfere)

  func testInteriorUnaffected() {
    XCTAssertEqual(NormalDistribution().ppf(0.5), 0, accuracy: 1e-12)
    XCTAssertEqual(UniformDistribution(loc: 2, scale: 3).ppf(0.5), 3.5, accuracy: 1e-12)
    XCTAssertEqual(ExponentialDistribution().ppf(0.5), Foundation.log(2.0), accuracy: 1e-10)
  }
}
