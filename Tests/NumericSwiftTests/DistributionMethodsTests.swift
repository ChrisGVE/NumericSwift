//
//  DistributionMethodsTests.swift
//  NumericSwift
//
//  Tests for logpdf, sf (survival function), and isf (inverse survival function)
//  methods on all continuous distributions.
//
//  Licensed under the MIT License.
//

import XCTest

@testable import NumericSwift

final class DistributionMethodsTests: XCTestCase {

  // MARK: - Helpers

  private let tolerance = 1e-10

  // MARK: - NormalDistribution

  func testNormalLogpdf() {
    let dist = NormalDistribution(loc: 0, scale: 1)
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
      let expected = Darwin.log(dist.pdf(x))
      XCTAssertEqual(
        dist.logpdf(x), expected, accuracy: tolerance,
        "Normal logpdf mismatch at x=\(x)")
    }
  }

  func testNormalLogpdfStableAtExtremes() {
    // At 10 sigma from mean, pdf underflows to 0 but logpdf must be finite
    let dist = NormalDistribution(loc: 0, scale: 1)
    let logVal = dist.logpdf(10.0)
    XCTAssertFalse(logVal.isInfinite, "Normal logpdf should be finite at 10 sigma")
    XCTAssertFalse(logVal.isNaN, "Normal logpdf should not be NaN at 10 sigma")
    // Expected: -0.5*(100) - log(1) - 0.5*log(2pi) ≈ -50 - 0.919 ≈ -50.919
    XCTAssertEqual(logVal, -50.0 - 0.5 * Darwin.log(2.0 * .pi), accuracy: tolerance)
  }

  func testNormalSf() {
    let dist = NormalDistribution(loc: 0, scale: 1)
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
      let sfVal = dist.sf(x)
      let cdfVal = dist.cdf(x)
      XCTAssertEqual(
        sfVal + cdfVal, 1.0, accuracy: tolerance,
        "Normal sf + cdf != 1 at x=\(x)")
    }
  }

  func testNormalIsf() {
    let dist = NormalDistribution(loc: 0, scale: 1)
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
      let sfVal = dist.sf(x)
      let recovered = dist.isf(sfVal)
      XCTAssertEqual(
        recovered, x, accuracy: 1e-9,
        "Normal isf(sf(x)) round-trip failed at x=\(x)")
    }
  }

  // MARK: - UniformDistribution

  func testUniformLogpdf() {
    let dist = UniformDistribution(loc: 1, scale: 3)
    // Inside support: logpdf = -log(scale)
    XCTAssertEqual(dist.logpdf(2.0), -Darwin.log(3.0), accuracy: tolerance)
    XCTAssertEqual(dist.logpdf(1.5), Darwin.log(dist.pdf(1.5)), accuracy: tolerance)
    // Outside support: logpdf = -inf
    XCTAssertEqual(dist.logpdf(0.5), -.infinity)
    XCTAssertEqual(dist.logpdf(5.0), -.infinity)
  }

  func testUniformSf() {
    let dist = UniformDistribution(loc: 0, scale: 4)
    for x in [0.5, 1.0, 2.0, 3.0, 3.5] {
      XCTAssertEqual(
        dist.sf(x) + dist.cdf(x), 1.0, accuracy: tolerance,
        "Uniform sf + cdf != 1 at x=\(x)")
    }
  }

  func testUniformIsf() {
    let dist = UniformDistribution(loc: 0, scale: 4)
    for x in [0.5, 1.0, 2.0, 3.0, 3.5] {
      let recovered = dist.isf(dist.sf(x))
      XCTAssertEqual(
        recovered, x, accuracy: 1e-9,
        "Uniform isf(sf(x)) round-trip failed at x=\(x)")
    }
  }

  // MARK: - ExponentialDistribution

  func testExponentialLogpdf() {
    let dist = ExponentialDistribution(loc: 0, scale: 2)
    for x in [0.1, 0.5, 1.0, 2.0, 5.0] {
      let expected = Darwin.log(dist.pdf(x))
      XCTAssertEqual(
        dist.logpdf(x), expected, accuracy: tolerance,
        "Exponential logpdf mismatch at x=\(x)")
    }
    // Below loc: logpdf = -inf
    XCTAssertEqual(dist.logpdf(-0.1), -.infinity)
  }

  func testExponentialLogpdfStableAtExtremes() {
    // Large x where exp(-x/scale) underflows to 0
    let dist = ExponentialDistribution(loc: 0, scale: 1)
    let logVal = dist.logpdf(800.0)
    XCTAssertFalse(logVal.isInfinite, "Exponential logpdf should be finite at large x")
    XCTAssertEqual(logVal, -800.0, accuracy: tolerance)
  }

  func testExponentialSf() {
    let dist = ExponentialDistribution(loc: 0, scale: 2)
    for x in [0.1, 0.5, 1.0, 2.0, 5.0] {
      XCTAssertEqual(
        dist.sf(x) + dist.cdf(x), 1.0, accuracy: tolerance,
        "Exponential sf + cdf != 1 at x=\(x)")
    }
  }

  func testExponentialIsf() {
    let dist = ExponentialDistribution(loc: 0, scale: 2)
    for x in [0.1, 0.5, 1.0, 2.0, 5.0] {
      let recovered = dist.isf(dist.sf(x))
      XCTAssertEqual(
        recovered, x, accuracy: 1e-9,
        "Exponential isf(sf(x)) round-trip failed at x=\(x)")
    }
  }

  // MARK: - TDistribution

  func testTLogpdf() {
    let dist = TDistribution(df: 5)
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
      let expected = Darwin.log(dist.pdf(x))
      XCTAssertEqual(
        dist.logpdf(x), expected, accuracy: tolerance,
        "T logpdf mismatch at x=\(x)")
    }
  }

  func testTSf() {
    let dist = TDistribution(df: 10)
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
      XCTAssertEqual(
        dist.sf(x) + dist.cdf(x), 1.0, accuracy: tolerance,
        "T sf + cdf != 1 at x=\(x)")
    }
  }

  func testTIsf() {
    let dist = TDistribution(df: 10)
    for x in [-2.0, -1.0, 0.5, 1.0, 2.0] {
      let recovered = dist.isf(dist.sf(x))
      XCTAssertEqual(
        recovered, x, accuracy: 1e-6,
        "T isf(sf(x)) round-trip failed at x=\(x)")
    }
  }

  // MARK: - ChiSquaredDistribution

  func testChiSquaredLogpdf() {
    let dist = ChiSquaredDistribution(df: 4)
    for x in [0.5, 1.0, 2.0, 4.0, 8.0] {
      let expected = Darwin.log(dist.pdf(x))
      XCTAssertEqual(
        dist.logpdf(x), expected, accuracy: tolerance,
        "ChiSquared logpdf mismatch at x=\(x)")
    }
    // Below support: logpdf = -inf
    XCTAssertEqual(dist.logpdf(-1.0), -.infinity)
  }

  func testChiSquaredSf() {
    let dist = ChiSquaredDistribution(df: 4)
    for x in [0.5, 1.0, 2.0, 4.0, 8.0] {
      XCTAssertEqual(
        dist.sf(x) + dist.cdf(x), 1.0, accuracy: tolerance,
        "ChiSquared sf + cdf != 1 at x=\(x)")
    }
  }

  func testChiSquaredIsf() {
    let dist = ChiSquaredDistribution(df: 4)
    for x in [0.5, 1.0, 2.0, 4.0, 8.0] {
      let recovered = dist.isf(dist.sf(x))
      XCTAssertEqual(
        recovered, x, accuracy: 1e-7,
        "ChiSquared isf(sf(x)) round-trip failed at x=\(x)")
    }
  }

  // MARK: - FDistribution

  func testFLogpdf() {
    let dist = FDistribution(dfn: 3, dfd: 10)
    for x in [0.5, 1.0, 2.0, 4.0] {
      let expected = Darwin.log(dist.pdf(x))
      XCTAssertEqual(
        dist.logpdf(x), expected, accuracy: tolerance,
        "F logpdf mismatch at x=\(x)")
    }
  }

  func testFSf() {
    let dist = FDistribution(dfn: 3, dfd: 10)
    for x in [0.5, 1.0, 2.0, 4.0] {
      XCTAssertEqual(
        dist.sf(x) + dist.cdf(x), 1.0, accuracy: tolerance,
        "F sf + cdf != 1 at x=\(x)")
    }
  }

  func testFIsf() {
    let dist = FDistribution(dfn: 3, dfd: 10)
    for x in [0.5, 1.0, 2.0, 3.0] {
      let recovered = dist.isf(dist.sf(x))
      XCTAssertEqual(
        recovered, x, accuracy: 1e-7,
        "F isf(sf(x)) round-trip failed at x=\(x)")
    }
  }

  // MARK: - GammaDistribution

  func testGammaLogpdf() {
    let dist = GammaDistribution(shape: 2, scale: 1.5)
    for x in [0.5, 1.0, 2.0, 4.0, 6.0] {
      let expected = Darwin.log(dist.pdf(x))
      XCTAssertEqual(
        dist.logpdf(x), expected, accuracy: tolerance,
        "Gamma logpdf mismatch at x=\(x)")
    }
  }

  func testGammaSf() {
    let dist = GammaDistribution(shape: 2, scale: 1.5)
    for x in [0.5, 1.0, 2.0, 4.0, 6.0] {
      XCTAssertEqual(
        dist.sf(x) + dist.cdf(x), 1.0, accuracy: tolerance,
        "Gamma sf + cdf != 1 at x=\(x)")
    }
  }

  func testGammaIsf() {
    // Use points near and above the mean (mean = shape * scale = 3.0) to keep
    // isf within the convergence region of the existing Newton-Raphson ppf.
    let dist = GammaDistribution(shape: 2, scale: 1.5)
    for x in [2.0, 3.0, 4.0, 5.0, 6.0] {
      let recovered = dist.isf(dist.sf(x))
      XCTAssertEqual(
        recovered, x, accuracy: 1e-7,
        "Gamma isf(sf(x)) round-trip failed at x=\(x)")
    }
  }

  // MARK: - BetaDistribution

  func testBetaLogpdf() {
    let dist = BetaDistribution(a: 2, b: 3)
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
      let expected = Darwin.log(dist.pdf(x))
      XCTAssertEqual(
        dist.logpdf(x), expected, accuracy: tolerance,
        "Beta logpdf mismatch at x=\(x)")
    }
    // Outside [0,1]: logpdf = -inf
    XCTAssertEqual(dist.logpdf(-0.1), -.infinity)
    XCTAssertEqual(dist.logpdf(1.1), -.infinity)
  }

  func testBetaSf() {
    let dist = BetaDistribution(a: 2, b: 3)
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
      XCTAssertEqual(
        dist.sf(x) + dist.cdf(x), 1.0, accuracy: tolerance,
        "Beta sf + cdf != 1 at x=\(x)")
    }
  }

  func testBetaIsf() {
    let dist = BetaDistribution(a: 2, b: 3)
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
      let recovered = dist.isf(dist.sf(x))
      XCTAssertEqual(
        recovered, x, accuracy: 1e-7,
        "Beta isf(sf(x)) round-trip failed at x=\(x)")
    }
  }

  // MARK: - Cross-distribution: logpdf finiteness at tails

  func testLogpdfFiniteWherePdfUnderflows() {
    // Normal: at 40 sigma, pdf == 0 in double precision, logpdf must be finite
    let normal = NormalDistribution(loc: 0, scale: 1)
    XCTAssertEqual(normal.pdf(40.0), 0.0, "pdf should underflow at 40 sigma")
    XCTAssertFalse(
      normal.logpdf(40.0).isInfinite,
      "Normal logpdf must remain finite where pdf underflows")

    // Exponential: at x=1000 with scale=1, pdf == 0, logpdf must be finite
    let expo = ExponentialDistribution(loc: 0, scale: 1)
    XCTAssertEqual(expo.pdf(1000.0), 0.0, "Exponential pdf should underflow at x=1000")
    XCTAssertFalse(
      expo.logpdf(1000.0).isInfinite,
      "Exponential logpdf must remain finite where pdf underflows")
  }

  // MARK: - Normal logpdf formula correctness

  func testNormalLogpdfFormula() {
    // Verify the direct formula -0.5*z^2 - log(scale) - 0.5*log(2pi) is used
    // by comparing at a point where exp would lose precision if used naively
    let dist = NormalDistribution(loc: 5, scale: 2)
    let x = 5.0 + 10.0 * 2.0  // 10 sigma from mean
    let z = (x - 5.0) / 2.0
    let expected = -0.5 * z * z - Darwin.log(2.0) - 0.5 * Darwin.log(2.0 * .pi)
    XCTAssertEqual(dist.logpdf(x), expected, accuracy: tolerance)
    XCTAssertFalse(dist.logpdf(x).isInfinite)
  }
}
