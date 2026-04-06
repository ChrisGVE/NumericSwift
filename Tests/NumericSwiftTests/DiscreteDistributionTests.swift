//
//  DiscreteDistributionTests.swift
//  NumericSwiftTests
//
//  Tests for BernoulliDistribution, BinomialDistribution, and PoissonDistribution.
//

import XCTest

@testable import NumericSwift

final class DiscreteDistributionTests: XCTestCase {

  // MARK: - BernoulliDistribution

  func testBernoulliPMF() {
    let dist = BernoulliDistribution(p: 0.3)
    XCTAssertEqual(dist.pmf(0), 0.7, accuracy: 1e-14)
    XCTAssertEqual(dist.pmf(1), 0.3, accuracy: 1e-14)
    XCTAssertEqual(dist.pmf(2), 0.0)
    XCTAssertEqual(dist.pmf(-1), 0.0)
  }

  func testBernoulliPMFSumsToOne() {
    let dist = BernoulliDistribution(p: 0.6)
    let total = dist.pmf(0) + dist.pmf(1)
    XCTAssertEqual(total, 1.0, accuracy: 1e-14)
  }

  func testBernoulliCDF() {
    let dist = BernoulliDistribution(p: 0.4)
    XCTAssertEqual(dist.cdf(-1), 0.0)
    XCTAssertEqual(dist.cdf(0), 0.6, accuracy: 1e-14)
    XCTAssertEqual(dist.cdf(1), 1.0, accuracy: 1e-14)
    XCTAssertEqual(dist.cdf(10), 1.0)
  }

  func testBernoulliCDFMonotone() {
    let dist = BernoulliDistribution(p: 0.5)
    var prev = dist.cdf(-1)
    for k in 0...5 {
      let cur = dist.cdf(k)
      XCTAssertGreaterThanOrEqual(cur, prev)
      prev = cur
    }
  }

  func testBernoulliPPF() {
    let dist = BernoulliDistribution(p: 0.3)
    // cdf(0) = 0.7, so ppf(q) = 0 when q <= 0.7
    XCTAssertEqual(dist.ppf(0.0), 0)
    XCTAssertEqual(dist.ppf(0.5), 0)
    XCTAssertEqual(dist.ppf(0.7), 0)
    XCTAssertEqual(dist.ppf(0.71), 1)
    XCTAssertEqual(dist.ppf(1.0), 1)
  }

  func testBernoulliPPFRoundTrip() {
    let dist = BernoulliDistribution(p: 0.4)
    for q in [0.1, 0.3, 0.5, 0.7, 0.9] {
      let k = dist.ppf(q)
      XCTAssertGreaterThanOrEqual(dist.cdf(k), q)
    }
  }

  func testBernoulliMeanVariance() {
    let dist = BernoulliDistribution(p: 0.4)
    XCTAssertEqual(dist.mean, 0.4, accuracy: 1e-14)
    XCTAssertEqual(dist.variance, 0.24, accuracy: 1e-14)
  }

  func testBernoulliEdgeCaseP0() {
    let dist = BernoulliDistribution(p: 0.0)
    XCTAssertEqual(dist.pmf(0), 1.0)
    XCTAssertEqual(dist.pmf(1), 0.0)
    XCTAssertEqual(dist.cdf(0), 1.0)
    XCTAssertEqual(dist.ppf(0.5), 0)
  }

  func testBernoulliEdgeCaseP1() {
    let dist = BernoulliDistribution(p: 1.0)
    XCTAssertEqual(dist.pmf(0), 0.0)
    XCTAssertEqual(dist.pmf(1), 1.0)
    XCTAssertEqual(dist.cdf(0), 0.0)
    XCTAssertEqual(dist.ppf(0.5), 1)
  }

  func testBernoulliRVS() {
    let dist = BernoulliDistribution(p: 0.5)
    let samples = dist.rvs(1000)
    XCTAssertEqual(samples.count, 1000)
    XCTAssertTrue(samples.allSatisfy { $0 == 0 || $0 == 1 })
    // Empirical mean should be close to 0.5
    let empiricalMean = Double(samples.reduce(0, +)) / Double(samples.count)
    XCTAssertEqual(empiricalMean, 0.5, accuracy: 0.08)
  }

  // MARK: - BinomialDistribution

  func testBinomialPMF() {
    // scipy: binom.pmf(3, 10, 0.5) ≈ 0.1171875
    let dist = BinomialDistribution(n: 10, p: 0.5)
    XCTAssertEqual(dist.pmf(3), 0.1171875, accuracy: 1e-10)
    XCTAssertEqual(dist.pmf(5), 0.24609375, accuracy: 1e-10)
    XCTAssertEqual(dist.pmf(-1), 0.0)
    XCTAssertEqual(dist.pmf(11), 0.0)
  }

  func testBinomialPMFSumsToOne() {
    let dist = BinomialDistribution(n: 20, p: 0.3)
    let total = (0...20).reduce(0.0) { $0 + dist.pmf($1) }
    XCTAssertEqual(total, 1.0, accuracy: 1e-12)
  }

  func testBinomialCDF() {
    let dist = BinomialDistribution(n: 10, p: 0.5)
    XCTAssertEqual(dist.cdf(-1), 0.0)
    XCTAssertEqual(dist.cdf(10), 1.0, accuracy: 1e-12)
    // scipy: binom.cdf(5, 10, 0.5) ≈ 0.623046875
    XCTAssertEqual(dist.cdf(5), 0.623046875, accuracy: 1e-10)
  }

  func testBinomialCDFMonotone() {
    let dist = BinomialDistribution(n: 15, p: 0.4)
    var prev = dist.cdf(-1)
    for k in 0...15 {
      let cur = dist.cdf(k)
      XCTAssertGreaterThanOrEqual(cur, prev - 1e-15)
      prev = cur
    }
  }

  func testBinomialCDFAtMaxIsOne() {
    let dist = BinomialDistribution(n: 8, p: 0.7)
    XCTAssertEqual(dist.cdf(8), 1.0, accuracy: 1e-12)
    XCTAssertEqual(dist.cdf(100), 1.0, accuracy: 1e-12)
  }

  func testBinomialPPF() {
    let dist = BinomialDistribution(n: 10, p: 0.5)
    // Median of Binomial(10, 0.5) is 5
    XCTAssertEqual(dist.ppf(0.5), 5)
    XCTAssertEqual(dist.ppf(0.0), 0)
    XCTAssertEqual(dist.ppf(1.0), 10)
  }

  func testBinomialPPFRoundTrip() {
    let dist = BinomialDistribution(n: 20, p: 0.3)
    for q in [0.05, 0.25, 0.5, 0.75, 0.95] {
      let k = dist.ppf(q)
      XCTAssertGreaterThanOrEqual(dist.cdf(k), q)
      // Also verify k is the smallest such integer
      if k > 0 {
        XCTAssertLessThan(dist.cdf(k - 1), q)
      }
    }
  }

  func testBinomialMeanVariance() {
    let dist = BinomialDistribution(n: 20, p: 0.3)
    XCTAssertEqual(dist.mean, 6.0, accuracy: 1e-14)
    XCTAssertEqual(dist.variance, 4.2, accuracy: 1e-14)
  }

  func testBinomialEdgeCaseN0() {
    let dist = BinomialDistribution(n: 0, p: 0.5)
    XCTAssertEqual(dist.pmf(0), 1.0)
    XCTAssertEqual(dist.pmf(1), 0.0)
    XCTAssertEqual(dist.cdf(0), 1.0)
    XCTAssertEqual(dist.mean, 0.0)
    XCTAssertEqual(dist.variance, 0.0)
  }

  func testBinomialEdgeCaseP0() {
    let dist = BinomialDistribution(n: 5, p: 0.0)
    XCTAssertEqual(dist.pmf(0), 1.0)
    XCTAssertEqual(dist.pmf(1), 0.0)
    XCTAssertEqual(dist.cdf(0), 1.0)
  }

  func testBinomialEdgeCaseP1() {
    let dist = BinomialDistribution(n: 5, p: 1.0)
    XCTAssertEqual(dist.pmf(5), 1.0)
    XCTAssertEqual(dist.pmf(4), 0.0)
    XCTAssertEqual(dist.cdf(4), 0.0)
    XCTAssertEqual(dist.cdf(5), 1.0)
  }

  func testBinomialNegativeKReturnsZero() {
    let dist = BinomialDistribution(n: 10, p: 0.5)
    XCTAssertEqual(dist.pmf(-1), 0.0)
    XCTAssertEqual(dist.cdf(-1), 0.0)
  }

  func testBinomialKGreaterThanNReturnsZero() {
    let dist = BinomialDistribution(n: 10, p: 0.5)
    XCTAssertEqual(dist.pmf(11), 0.0)
  }

  func testBinomialRVS() {
    let dist = BinomialDistribution(n: 20, p: 0.5)
    let samples = dist.rvs(500)
    XCTAssertEqual(samples.count, 500)
    XCTAssertTrue(samples.allSatisfy { $0 >= 0 && $0 <= 20 })
    let empiricalMean = Double(samples.reduce(0, +)) / Double(samples.count)
    XCTAssertEqual(empiricalMean, dist.mean, accuracy: 1.0)
  }

  // MARK: - PoissonDistribution

  func testPoissonPMF() {
    // scipy: poisson.pmf(3, 2.5) ≈ 0.21376...
    let dist = PoissonDistribution(mu: 2.5)
    let expected = Darwin.exp(-2.5) * Darwin.pow(2.5, 3) / 6.0
    XCTAssertEqual(dist.pmf(3), expected, accuracy: 1e-12)
    XCTAssertEqual(dist.pmf(-1), 0.0)
  }

  func testPoissonPMFSumsToOne() {
    let dist = PoissonDistribution(mu: 3.0)
    // Sum first 50 terms - should be essentially 1
    let total = (0...50).reduce(0.0) { $0 + dist.pmf($1) }
    XCTAssertEqual(total, 1.0, accuracy: 1e-10)
  }

  func testPoissonPMFSumsToOneHighMu() {
    let dist = PoissonDistribution(mu: 20.0)
    let total = (0...100).reduce(0.0) { $0 + dist.pmf($1) }
    XCTAssertEqual(total, 1.0, accuracy: 1e-8)
  }

  func testPoissonCDF() {
    let dist = PoissonDistribution(mu: 3.0)
    XCTAssertEqual(dist.cdf(-1), 0.0)
    // For large k, CDF should be very close to 1
    XCTAssertEqual(dist.cdf(30), 1.0, accuracy: 1e-10)
    // scipy: poisson.cdf(3, 3) ≈ 0.64723...
    let cdf3 = (0...3).reduce(0.0) { $0 + dist.pmf($1) }
    XCTAssertEqual(dist.cdf(3), cdf3, accuracy: 1e-12)
  }

  func testPoissonCDFMonotone() {
    let dist = PoissonDistribution(mu: 5.0)
    var prev = dist.cdf(-1)
    for k in 0...20 {
      let cur = dist.cdf(k)
      XCTAssertGreaterThanOrEqual(cur, prev - 1e-15)
      prev = cur
    }
  }

  func testPoissonPPF() {
    let dist = PoissonDistribution(mu: 3.0)
    // ppf(0) should be 0
    XCTAssertEqual(dist.ppf(0.0), 0)
    // ppf(1) should return some finite k
    XCTAssertGreaterThanOrEqual(dist.ppf(1.0), 0)
    // Median of Poisson(3) is 3
    XCTAssertEqual(dist.ppf(0.5), 3)
  }

  func testPoissonPPFRoundTrip() {
    let dist = PoissonDistribution(mu: 4.0)
    for q in [0.05, 0.25, 0.5, 0.75, 0.95] {
      let k = dist.ppf(q)
      XCTAssertGreaterThanOrEqual(dist.cdf(k), q)
      if k > 0 {
        XCTAssertLessThan(dist.cdf(k - 1), q)
      }
    }
  }

  func testPoissonMeanVariance() {
    let dist = PoissonDistribution(mu: 7.5)
    XCTAssertEqual(dist.mean, 7.5, accuracy: 1e-14)
    XCTAssertEqual(dist.variance, 7.5, accuracy: 1e-14)
  }

  func testPoissonNegativeKReturnsZero() {
    let dist = PoissonDistribution(mu: 2.0)
    XCTAssertEqual(dist.pmf(-1), 0.0)
    XCTAssertEqual(dist.pmf(-10), 0.0)
    XCTAssertEqual(dist.cdf(-1), 0.0)
  }

  func testPoissonK0() {
    let dist = PoissonDistribution(mu: 1.5)
    XCTAssertEqual(dist.pmf(0), Darwin.exp(-1.5), accuracy: 1e-14)
  }

  func testPoissonRVS() {
    let dist = PoissonDistribution(mu: 5.0)
    let samples = dist.rvs(500)
    XCTAssertEqual(samples.count, 500)
    XCTAssertTrue(samples.allSatisfy { $0 >= 0 })
    let empiricalMean = Double(samples.reduce(0, +)) / Double(samples.count)
    XCTAssertEqual(empiricalMean, dist.mean, accuracy: 1.0)
  }

  func testPoissonLargeMuPMFStability() {
    // Verify log-space computation doesn't overflow for large mu
    let dist = PoissonDistribution(mu: 100.0)
    let modeValue = dist.pmf(100)
    XCTAssertFalse(modeValue.isNaN)
    XCTAssertFalse(modeValue.isInfinite)
    XCTAssertGreaterThan(modeValue, 0.0)
  }
}
