//
//  AuditFixesTests.swift
//  NumericSwift
//
//  Regression tests for correctness bugs found in the 2026-06-10 audit
//  (code_audit_2026-06-10.md): besseli sign (C1), the tgamma→lgamma overflow
//  sweep (M1, M3, M4, M5), and the NumberTheory Int-overflow fixes (M8, M9).
//
//  Licensed under the MIT License.
//

import XCTest

@testable import NumericSwift

final class AuditFixesTests: XCTestCase {

  // MARK: - C1: besseli sign for negative x

  /// Iₙ(−x) = (−1)ⁿ Iₙ(x): odd orders flip sign, even orders do not.
  /// Before the fix, besseli took abs(x) and always returned the positive branch,
  /// so besseli(1, −1) wrongly returned +0.565… instead of −0.565….
  func testBesseliNegativeArgumentSign() {
    let tol = 1e-12
    // Reference values from scipy.special.iv.
    XCTAssertEqual(besseli(1, 1.0), 0.5651591039924850, accuracy: tol)
    XCTAssertEqual(besseli(1, -1.0), -0.5651591039924850, accuracy: tol)
    XCTAssertEqual(besseli(2, 1.0), 0.13574766976703831, accuracy: tol)
    XCTAssertEqual(besseli(2, -1.0), 0.13574766976703831, accuracy: tol)  // even: unchanged
    XCTAssertEqual(besseli(0, -2.0), 2.2795853023360673, accuracy: tol)  // even: unchanged
    XCTAssertEqual(besseli(3, 2.0), 0.21273995923985266, accuracy: tol)
    XCTAssertEqual(besseli(3, -2.0), -0.21273995923985266, accuracy: tol)  // odd: flipped
  }

  // MARK: - M1: besseliSeries initial term overflow for order ≥ 171

  /// For n ≥ 171, `pow(x/2, n)/tgamma(n+1)` is 0/Inf = 0, which zeroed the whole
  /// series. The lgamma form keeps the term finite, so besseli stays positive and
  /// strictly increasing in x.
  func testBesseliHighOrderNoOverflow() {
    let a = besseli(180, 50.0)
    let b = besseli(180, 60.0)
    XCTAssertTrue(a.isFinite && a > 0, "besseli(180,50) must be finite and positive, got \(a)")
    XCTAssertTrue(b.isFinite && b > 0, "besseli(180,60) must be finite and positive, got \(b)")
    XCTAssertLessThan(a, b, "Iₙ(x) is increasing in x for x > 0")
  }

  // MARK: - M3: ChiSquared pdf overflow for df > 342

  func testChiSquaredPdfHighDFNoOverflow() {
    let dist = ChiSquaredDistribution(df: 700)
    let atMode = dist.pdf(698)  // mode = df − 2
    XCTAssertTrue(
      atMode.isFinite && atMode > 0,
      "ChiSquared(df:700).pdf(698) must be finite and positive, got \(atMode)")
    // ppf must also stay finite (uses the same density in its Newton step).
    let median = dist.ppf(0.5)
    XCTAssertTrue(median.isFinite && median > 0)
  }

  // MARK: - M4: Gamma pdf overflow for shape > 171

  func testGammaPdfHighShapeNoOverflow() {
    let dist = GammaDistribution(shape: 300)
    let atMean = dist.pdf(300)
    XCTAssertTrue(
      atMean.isFinite && atMean > 0,
      "Gamma(shape:300).pdf(300) must be finite and positive, got \(atMean)")
    let median = dist.ppf(0.5)
    XCTAssertTrue(median.isFinite && median > 0)
  }

  // MARK: - M5: TDistribution.logpdf direct log-space

  /// In the non-underflowing regime the closed form must agree with log(pdf(x)).
  func testTLogpdfMatchesLogPdf() {
    let dist = TDistribution(df: 5)
    for x in [-3.0, -1.5, 0.0, 0.7, 1.5, 4.0] {
      XCTAssertEqual(dist.logpdf(x), Foundation.log(dist.pdf(x)), accuracy: 1e-12)
    }
  }

  /// Far in the tail pdf(x) underflows to 0, so log(pdf(x)) = −Inf; the closed
  /// form stays finite.
  func testTLogpdfFiniteInFarTail() {
    let dist = TDistribution(df: 3)
    let lp = dist.logpdf(1e160)
    XCTAssertTrue(lp.isFinite && lp < 0, "logpdf far tail must be finite negative, got \(lp)")
    XCTAssertEqual(Foundation.log(dist.pdf(1e160)), -.infinity)  // documents the underflow it avoids
  }

  // MARK: - M9: modPow overflow for modulus > ~3e9

  func testModPowLargeModulusNoOverflow() {
    let m = 9_000_000_000_000_000_000  // 9e18, < Int.max
    // 2^63 mod 9e18 = 9223372036854775808 − 9e18
    XCTAssertEqual(modPow(2, 63, m), 223_372_036_854_775_808)
    // 10^19 mod 9e18 = 1e18 (10^19 is not even representable as Int)
    XCTAssertEqual(modPow(10, 19, m), 1_000_000_000_000_000_000)
    // 7^20 < 9e18, so the modulus leaves it unchanged
    XCTAssertEqual(modPow(7, 20, m), 79_792_266_297_612_001)
    // Small cases unchanged by the refactor
    XCTAssertEqual(modPow(2, 10, 1000), 24)
    XCTAssertEqual(modPow(3, 5, 7), 5)
  }

  // MARK: - M8: isPrime / primeFactors Int overflow near Int.max

  /// `i * i <= n` overflowed (and trapped in debug builds) once i passed ~3.04e9.
  /// `i <= n / i` is overflow-free; the largest signed-64-bit prime forces the
  /// trial-division loop to run to completion (~15 s — the full sqrt(n) sweep that
  /// triggers the original overflow). Gated behind NUMERICSWIFT_SLOW_TESTS so the
  /// default `swift test` loop stays fast; CI sets the flag.
  func testIsPrimeNearIntMaxNoOverflow() throws {
    try XCTSkipUnless(
      ProcessInfo.processInfo.environment["NUMERICSWIFT_SLOW_TESTS"] != nil,
      "set NUMERICSWIFT_SLOW_TESTS=1 to run the full near-Int.max trial-division sweep")
    XCTAssertTrue(isPrime(9_223_372_036_854_775_783))  // largest prime < 2^63
  }

  /// primeFactors uses the same `factor <= remaining / factor` guard.
  func testPrimeFactorsLargeSemiprime() {
    // 1_000_003 (prime) × 1_299_709 (the 100,000th prime).
    let n = 1_000_003 * 1_299_709
    let factors = primeFactors(n)
    XCTAssertEqual(factors.map { $0.prime }, [1_000_003, 1_299_709])
    XCTAssertEqual(factors.map { $0.exponent }, [1, 1])
  }
}
