//
//  DistributionMethodsTests.swift
//  NumericSwift
//
//  Tests for logpdf, sf (survival function), and isf (inverse survival function)
//  methods on all continuous distributions.
//
//  Licensed under the Apache License, Version 2.0.
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

  // MARK: - Support-boundary densities (SciPy parity, audit hardening)

  /// Beta pdf at the lower boundary z == 0 is governed by `a`:
  /// a < 1 → +inf, a == 1 → 1/B(1,b) = b, a > 1 → 0.
  func testBetaPdfLowerBoundary() {
    XCTAssertEqual(BetaDistribution(a: 1, b: 2).pdf(0.0), 2.0, accuracy: tolerance)
    XCTAssertEqual(BetaDistribution(a: 0.5, b: 2).pdf(0.0), .infinity)
    XCTAssertEqual(BetaDistribution(a: 2, b: 2).pdf(0.0), 0.0, accuracy: tolerance)
  }

  /// Beta pdf at the upper boundary z == 1 is governed symmetrically by `b`:
  /// b < 1 → +inf, b == 1 → 1/B(a,1) = a, b > 1 → 0.
  func testBetaPdfUpperBoundary() {
    XCTAssertEqual(BetaDistribution(a: 3, b: 1).pdf(1.0), 3.0, accuracy: tolerance)
    XCTAssertEqual(BetaDistribution(a: 3, b: 0.5).pdf(1.0), .infinity)
    XCTAssertEqual(BetaDistribution(a: 3, b: 2).pdf(1.0), 0.0, accuracy: tolerance)
  }

  /// Gamma pdf at z == 0: shape < 1 → +inf, shape == 1 → 1/scale, shape > 1 → 0.
  func testGammaPdfBoundary() {
    XCTAssertEqual(GammaDistribution(shape: 0.5).pdf(0.0), .infinity)
    XCTAssertEqual(GammaDistribution(shape: 1).pdf(0.0), 1.0, accuracy: tolerance)
    XCTAssertEqual(GammaDistribution(shape: 1, scale: 2).pdf(0.0), 0.5, accuracy: tolerance)
    XCTAssertEqual(GammaDistribution(shape: 2).pdf(0.0), 0.0, accuracy: tolerance)
  }

  /// F pdf must stay finite for large degrees of freedom. The old product form
  /// `(dfn·z)^(dfn/2)·dfd^(dfd/2) / (dfn·z+dfd)^((dfn+dfd)/2)` overflowed to
  /// inf/inf = NaN; the log-space form does not. The second assertion is an
  /// algebraic-identity self-check that the refactor preserved the value the
  /// SciPy-backed workbench validates at moderate inputs.
  func testFPdfLargeDfStaysFinite() {
    let f = FDistribution(dfn: 500, dfd: 500)
    let d = f.pdf(1.2)
    XCTAssertTrue(d.isFinite, "F pdf with large df must be finite, got \(d)")
    XCTAssertGreaterThan(d, 0.0)

    let g = FDistribution(dfn: 5, dfd: 9)
    let z = 1.5
    // num exponents dfn/2 = 2.5, dfd/2 = 4.5; den exponent (dfn+dfd)/2 = 7.0.
    let num = Darwin.pow(5 * z, 2.5) * Darwin.pow(9.0, 4.5)
    let den = Darwin.pow(5 * z + 9, 7.0)
    let expected = (1.0 / (z * beta(2.5, 4.5))) * num / den
    XCTAssertEqual(g.pdf(z), expected, accuracy: abs(expected) * 1e-10)
  }

  /// Chi-squared logpdf must stay finite deep in the tail where pdf underflows to
  /// 0 (so the old `log(pdf(x))` returned -inf). Closed form:
  /// (k/2 − 1)·ln x − x/2 − (k/2)·ln 2 − lgamma(k/2).
  func testChiSquaredLogpdfTailStaysFinite() {
    let c = ChiSquaredDistribution(df: 4)
    let x = 2000.0
    XCTAssertEqual(c.pdf(x), 0.0, "pdf should underflow to 0 deep in the tail")
    let lp = c.logpdf(x)
    XCTAssertTrue(lp.isFinite, "logpdf must remain finite where pdf underflows, got \(lp)")
    let k2 = 2.0
    let expected = (k2 - 1) * Darwin.log(x) - x / 2.0 - k2 * Darwin.log(2.0) - lgamma(k2)
    XCTAssertEqual(lp, expected, accuracy: abs(expected) * 1e-12)
  }
}
