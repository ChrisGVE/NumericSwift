//
//  DiscreteDistributions.swift
//  NumericSwift
//
//  Discrete probability distributions following scipy.stats patterns.
//  Each distribution provides pmf, cdf, ppf (quantile), rvs (random sampling),
//  and moment functions (mean, variance).
//
//  Licensed under the MIT License.
//

import Darwin

// MARK: - Bernoulli Distribution

/// Bernoulli distribution: single trial with success probability p.
public struct BernoulliDistribution {
  /// Probability of success (k=1).
  public let p: Double

  /// - Parameter p: Success probability, must satisfy 0 ≤ p ≤ 1.
  public init(p: Double) {
    precondition(p >= 0 && p <= 1, "BernoulliDistribution: p must be in [0, 1], got \(p)")
    self.p = p
  }

  /// Probability mass function.
  ///
  /// - Returns: p if k == 1, 1-p if k == 0, 0 otherwise.
  public func pmf(_ k: Int) -> Double {
    switch k {
    case 0: return 1.0 - p
    case 1: return p
    default: return 0.0
    }
  }

  /// Cumulative distribution function.
  public func cdf(_ k: Int) -> Double {
    if k < 0 { return 0.0 }
    if k == 0 { return 1.0 - p }
    return 1.0
  }

  /// Percent point function (inverse CDF). Returns smallest k such that cdf(k) >= q.
  public func ppf(_ q: Double) -> Int {
    precondition(q >= 0 && q <= 1, "BernoulliDistribution.ppf: q must be in [0, 1]")
    return (q <= 1.0 - p) ? 0 : 1
  }

  /// Single random variate.
  public func rvs() -> Int {
    Double.random(in: 0..<1) < p ? 1 : 0
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Int] {
    (0..<n).map { _ in rvs() }
  }

  /// Distribution mean.
  public var mean: Double { p }

  /// Distribution variance.
  public var variance: Double { p * (1.0 - p) }
}

// MARK: - Binomial Distribution

/// Binomial distribution: number of successes in n independent Bernoulli trials.
public struct BinomialDistribution {
  /// Number of trials.
  public let n: Int
  /// Probability of success per trial.
  public let p: Double

  /// - Parameters:
  ///   - n: Number of trials, must be ≥ 0.
  ///   - p: Success probability per trial, must satisfy 0 ≤ p ≤ 1.
  public init(n: Int, p: Double) {
    precondition(n >= 0, "BinomialDistribution: n must be non-negative, got \(n)")
    precondition(p >= 0 && p <= 1, "BinomialDistribution: p must be in [0, 1], got \(p)")
    self.n = n
    self.p = p
  }

  /// Probability mass function using log-space arithmetic for numerical stability.
  public func pmf(_ k: Int) -> Double {
    guard k >= 0 && k <= n else { return 0.0 }
    if p == 0.0 { return k == 0 ? 1.0 : 0.0 }
    if p == 1.0 { return k == n ? 1.0 : 0.0 }
    let logPmf =
      Darwin.log(comb(n, k))
      + Double(k) * Darwin.log(p)
      + Double(n - k) * Darwin.log(1.0 - p)
    return Darwin.exp(logPmf)
  }

  /// Cumulative distribution function (sum of pmf from 0 to k).
  public func cdf(_ k: Int) -> Double {
    if k < 0 { return 0.0 }
    if k >= n { return 1.0 }
    return (0...k).reduce(0.0) { $0 + pmf($1) }
  }

  /// Percent point function: smallest k such that cdf(k) >= q.
  public func ppf(_ q: Double) -> Int {
    precondition(q >= 0 && q <= 1, "BinomialDistribution.ppf: q must be in [0, 1]")
    if q <= 0 { return 0 }
    if q >= 1 { return n }
    var cumulative = 0.0
    for k in 0...n {
      cumulative += pmf(k)
      if cumulative >= q { return k }
    }
    return n
  }

  /// Single random variate (sum of n Bernoulli trials).
  public func rvs() -> Int {
    var successes = 0
    for _ in 0..<n {
      if Double.random(in: 0..<1) < p { successes += 1 }
    }
    return successes
  }

  /// Generate count random variates.
  public func rvs(_ count: Int) -> [Int] {
    (0..<count).map { _ in rvs() }
  }

  /// Distribution mean.
  public var mean: Double { Double(n) * p }

  /// Distribution variance.
  public var variance: Double { Double(n) * p * (1.0 - p) }
}

// MARK: - Poisson Distribution

/// Poisson distribution: number of events in a fixed interval with average rate mu.
public struct PoissonDistribution {
  /// Expected number of events (rate parameter).
  public let mu: Double

  /// - Parameter mu: Mean rate, must be > 0.
  public init(mu: Double) {
    precondition(mu > 0, "PoissonDistribution: mu must be positive, got \(mu)")
    self.mu = mu
  }

  /// Probability mass function using log-space arithmetic for numerical stability.
  ///
  /// P(k) = exp(-mu) * mu^k / k!
  public func pmf(_ k: Int) -> Double {
    guard k >= 0 else { return 0.0 }
    // Use log-space: log P(k) = -mu + k*log(mu) - lgamma(k+1)
    let logPmf = -mu + Double(k) * Darwin.log(mu) - Darwin.lgamma(Double(k) + 1.0)
    return Darwin.exp(logPmf)
  }

  /// Cumulative distribution function (sum of pmf from 0 to k).
  public func cdf(_ k: Int) -> Double {
    if k < 0 { return 0.0 }
    // Use regularized upper incomplete gamma: CDF = 1 - gammainc(k+1, mu) / Gamma(k+1)
    // Equivalently, sum pmf values for stability at moderate k
    return (0...k).reduce(0.0) { $0 + pmf($1) }
  }

  /// Percent point function: smallest k such that cdf(k) >= q.
  public func ppf(_ q: Double) -> Int {
    precondition(q >= 0 && q <= 1, "PoissonDistribution.ppf: q must be in [0, 1]")
    if q <= 0 { return 0 }
    // Start search from floor(mu) as initial approximation
    var k = max(0, Int(mu) - 1)
    // Walk backward if needed
    while k > 0 && cdf(k - 1) >= q { k -= 1 }
    // Walk forward until cdf meets or exceeds q
    while cdf(k) < q { k += 1 }
    return k
  }

  /// Single random variate using Knuth's algorithm (efficient for small mu).
  public func rvs() -> Int {
    // For large mu, normal approximation seeded into integer search would be
    // preferable, but Knuth's method is correct for all mu.
    let threshold = Darwin.exp(-mu)
    var k = 0
    var product = Double.random(in: 0..<1)
    while product > threshold {
      k += 1
      product *= Double.random(in: 0..<1)
    }
    return k
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Int] {
    (0..<n).map { _ in rvs() }
  }

  /// Distribution mean.
  public var mean: Double { mu }

  /// Distribution variance.
  public var variance: Double { mu }
}
