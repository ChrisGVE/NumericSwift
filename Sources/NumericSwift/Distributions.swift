//
//  Distributions.swift
//  NumericSwift
//
//  Probability distributions following scipy.stats patterns.
//  Each distribution provides pdf, cdf, ppf (quantile), rvs (random sampling),
//  and moment functions (mean, variance, std).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - PPF Domain Guard

/// Shared p-domain handling for every continuous `ppf` (inverse CDF), matching
/// `scipy.stats`:
/// - `p` strictly outside `[0, 1]` (or `NaN`) → `NaN`;
/// - `p == 0` → the lower bound of the support;
/// - `p == 1` → the upper bound of the support.
///
/// Returns a value to short-circuit on, or `nil` to proceed with the
/// distribution's closed-form / iterative solve. The bounds are autoclosures so
/// a distribution only evaluates the one it needs.
///
/// - Parameters:
///   - p: Requested probability.
///   - lower: Lower support endpoint (e.g. `-.infinity`, or `loc`).
///   - upper: Upper support endpoint (e.g. `.infinity`, or `loc + scale`).
@inline(__always)
internal func ppfDomainGuard(
  _ p: Double,
  lower: @autoclosure () -> Double,
  upper: @autoclosure () -> Double
) -> Double? {
  if p.isNaN || p < 0 || p > 1 { return .nan }
  if p == 0 { return lower() }
  if p == 1 { return upper() }
  return nil
}

// MARK: - Random Number Generators

/// Box-Muller transform for standard normal random variates.
public func randomNormal() -> Double {
  let u1 = Double.random(in: Double.ulpOfOne..<1.0)
  let u2 = Double.random(in: 0..<1.0)
  return Darwin.sqrt(-2.0 * Darwin.log(u1)) * Darwin.cos(2.0 * .pi * u2)
}

/// Generate n standard normal random variates.
public func randomNormal(_ n: Int) -> [Double] {
  guard n > 0 else { return [] }
  return (0..<n).map { _ in randomNormal() }
}

/// Gamma random variate using Marsaglia and Tsang's method.
public func randomGamma(_ shape: Double) -> Double {
  if shape < 1 {
    // For shape < 1, use shape + 1 and transform
    return randomGamma(shape + 1) * Darwin.pow(Double.random(in: 0..<1), 1.0 / shape)
  }

  let d = shape - 1.0 / 3.0
  let c = 1.0 / Darwin.sqrt(9.0 * d)

  while true {
    var x: Double
    var v: Double

    repeat {
      x = randomNormal()
      v = 1.0 + c * x
    } while v <= 0

    v = v * v * v
    let u = Double.random(in: 0..<1)

    if u < 1.0 - 0.0331 * (x * x) * (x * x) {
      return d * v
    }

    if Darwin.log(u) < 0.5 * x * x + d * (1.0 - v + Darwin.log(v)) {
      return d * v
    }
  }
}

/// Generate n gamma random variates.
public func randomGamma(_ shape: Double, n: Int) -> [Double] {
  guard n > 0 else { return [] }
  return (0..<n).map { _ in randomGamma(shape) }
}

// MARK: - Normal Distribution

/// Normal (Gaussian) distribution.
public struct NormalDistribution {
  /// Location parameter (mean).
  public let loc: Double
  /// Scale parameter (standard deviation).
  public let scale: Double

  public init(loc: Double = 0, scale: Double = 1) {
    precondition(scale > 0, "NormalDistribution: scale must be positive, got \(scale)")
    self.loc = loc
    self.scale = scale
  }

  /// Probability density function.
  public func pdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    return Darwin.exp(-0.5 * z * z) / (scale * Darwin.sqrt(2.0 * .pi))
  }

  /// Cumulative distribution function.
  public func cdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    return 0.5 * (1.0 + erf(z / Darwin.sqrt(2.0)))
  }

  /// Percent point function (quantile function, inverse CDF).
  public func ppf(_ p: Double) -> Double {
    if let g = ppfDomainGuard(p, lower: -.infinity, upper: .infinity) { return g }
    return loc + scale * Darwin.sqrt(2.0) * erfinv(2.0 * p - 1.0)
  }

  /// Random variate sampling.
  public func rvs() -> Double {
    loc + scale * randomNormal()
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Double] {
    guard n > 0 else { return [] }
    return (0..<n).map { _ in rvs() }
  }

  /// Log of the probability density function.
  ///
  /// Uses the numerically stable formula `-0.5*z^2 - log(scale) - 0.5*log(2π)`
  /// to avoid the exp/log round-trip of `log(pdf(x))`.
  public func logpdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    return -0.5 * z * z - Darwin.log(scale) - 0.5 * Darwin.log(2.0 * .pi)
  }

  /// Survival function (complementary CDF): `1 - cdf(x)`.
  ///
  /// Uses `erfc` for better precision in the tails.
  public func sf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    return 0.5 * erfc(z / Darwin.sqrt(2.0))
  }

  /// Inverse survival function: `ppf(1 - p)`.
  public func isf(_ p: Double) -> Double {
    ppf(1.0 - p)
  }

  /// Distribution mean.
  public var mean: Double { loc }

  /// Distribution variance.
  public var variance: Double { scale * scale }

  /// Distribution standard deviation.
  public var std: Double { scale }
}

/// Convenience function for normal PDF.
public func normPdf(_ x: Double, loc: Double = 0, scale: Double = 1) -> Double {
  NormalDistribution(loc: loc, scale: scale).pdf(x)
}

/// Convenience function for normal CDF.
public func normCdf(_ x: Double, loc: Double = 0, scale: Double = 1) -> Double {
  NormalDistribution(loc: loc, scale: scale).cdf(x)
}

/// Convenience function for normal PPF (quantile).
public func normPpf(_ p: Double, loc: Double = 0, scale: Double = 1) -> Double {
  NormalDistribution(loc: loc, scale: scale).ppf(p)
}

// MARK: - Uniform Distribution

/// Uniform distribution on [loc, loc + scale].
public struct UniformDistribution {
  /// Location parameter (lower bound).
  public let loc: Double
  /// Scale parameter (width of interval).
  public let scale: Double

  public init(loc: Double = 0, scale: Double = 1) {
    precondition(scale > 0, "UniformDistribution: scale must be positive, got \(scale)")
    self.loc = loc
    self.scale = scale
  }

  /// Probability density function.
  public func pdf(_ x: Double) -> Double {
    if x >= loc && x <= loc + scale {
      return 1.0 / scale
    }
    return 0.0
  }

  /// Cumulative distribution function.
  public func cdf(_ x: Double) -> Double {
    if x < loc { return 0.0 }
    if x > loc + scale { return 1.0 }
    return (x - loc) / scale
  }

  /// Percent point function (quantile function, inverse CDF).
  public func ppf(_ p: Double) -> Double {
    if let g = ppfDomainGuard(p, lower: loc, upper: loc + scale) { return g }
    return loc + scale * p
  }

  /// Random variate sampling.
  public func rvs() -> Double {
    loc + scale * Double.random(in: 0..<1)
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Double] {
    guard n > 0 else { return [] }
    return (0..<n).map { _ in rvs() }
  }

  /// Log of the probability density function.
  public func logpdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z >= 0 && z <= 1 {
      return -Darwin.log(scale)
    }
    return -.infinity
  }

  /// Survival function (complementary CDF): `1 - cdf(x)`.
  public func sf(_ x: Double) -> Double {
    1.0 - cdf(x)
  }

  /// Inverse survival function: `ppf(1 - p)`.
  public func isf(_ p: Double) -> Double {
    ppf(1.0 - p)
  }

  /// Distribution mean.
  public var mean: Double { loc + scale / 2.0 }

  /// Distribution variance.
  public var variance: Double { scale * scale / 12.0 }

  /// Distribution standard deviation.
  public var std: Double { scale / Darwin.sqrt(12.0) }
}

// MARK: - Exponential Distribution

/// Exponential distribution.
public struct ExponentialDistribution {
  /// Location parameter.
  public let loc: Double
  /// Scale parameter (1/lambda).
  public let scale: Double

  public init(loc: Double = 0, scale: Double = 1) {
    precondition(scale > 0, "ExponentialDistribution: scale must be positive, got \(scale)")
    self.loc = loc
    self.scale = scale
  }

  /// Probability density function.
  public func pdf(_ x: Double) -> Double {
    let z = x - loc
    if z < 0 { return 0.0 }
    return Darwin.exp(-z / scale) / scale
  }

  /// Cumulative distribution function.
  public func cdf(_ x: Double) -> Double {
    let z = x - loc
    if z < 0 { return 0.0 }
    return 1.0 - Darwin.exp(-z / scale)
  }

  /// Percent point function (quantile function, inverse CDF).
  public func ppf(_ p: Double) -> Double {
    if let g = ppfDomainGuard(p, lower: loc, upper: .infinity) { return g }
    return loc - scale * Darwin.log(1.0 - p)
  }

  /// Random variate sampling.
  public func rvs() -> Double {
    loc - scale * Darwin.log(Double.random(in: Double.ulpOfOne..<1.0))
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Double] {
    guard n > 0 else { return [] }
    return (0..<n).map { _ in rvs() }
  }

  /// Log of the probability density function.
  ///
  /// Uses the numerically stable formula `-z/scale - log(scale)` to avoid the
  /// exp/log round-trip of `log(pdf(x))`.
  public func logpdf(_ x: Double) -> Double {
    let z = x - loc
    if z < 0 { return -.infinity }
    return -z / scale - Darwin.log(scale)
  }

  /// Survival function (complementary CDF): `1 - cdf(x)`.
  public func sf(_ x: Double) -> Double {
    let z = x - loc
    if z < 0 { return 1.0 }
    return Darwin.exp(-z / scale)
  }

  /// Inverse survival function: `ppf(1 - p)`.
  public func isf(_ p: Double) -> Double {
    ppf(1.0 - p)
  }

  /// Distribution mean.
  public var mean: Double { loc + scale }

  /// Distribution variance.
  public var variance: Double { scale * scale }

  /// Distribution standard deviation.
  public var std: Double { scale }
}

// MARK: - Student's t Distribution

/// Student's t distribution.
public struct TDistribution {
  /// Degrees of freedom.
  public let df: Double
  /// Location parameter.
  public let loc: Double
  /// Scale parameter.
  public let scale: Double

  public init(df: Double, loc: Double = 0, scale: Double = 1) {
    precondition(df > 0, "TDistribution: df must be positive, got \(df)")
    precondition(scale > 0, "TDistribution: scale must be positive, got \(scale)")
    self.df = df
    self.loc = loc
    self.scale = scale
  }

  /// Probability density function.
  public func pdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    // Use lgamma for numerical stability with large df (tgamma overflows for df > 340)
    let logCoef =
      Darwin.lgamma((df + 1) / 2.0) - Darwin.lgamma(df / 2.0) - 0.5 * Darwin.log(df * .pi)
    return Darwin.exp(logCoef) * Darwin.pow(1.0 + z * z / df, -(df + 1) / 2.0) / scale
  }

  /// Cumulative distribution function using incomplete beta.
  public func cdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    let t2 = z * z
    let p = betainc(df / 2.0, 0.5, df / (df + t2))

    if z >= 0 {
      return 1.0 - 0.5 * p
    } else {
      return 0.5 * p
    }
  }

  /// Percent point function using Newton-Raphson iteration.
  public func ppf(_ p: Double) -> Double {
    if let g = ppfDomainGuard(p, lower: -.infinity, upper: .infinity) { return g }
    if p == 0.5 { return loc }

    // Initial approximation: normal quantile (erfinv now fixed for full precision)
    var x = Darwin.sqrt(2.0) * erfinv(2.0 * p - 1.0)

    // Newton-Raphson iteration with lgamma for numerical stability
    for _ in 0..<100 {
      let t2 = x * x
      let cdfVal = 0.5 + 0.5 * (1.0 - betainc(df / 2.0, 0.5, df / (df + t2))) * (x >= 0 ? 1 : -1)

      // Use lgamma for stability with large df (tgamma overflows for df > 340)
      let logCoef =
        Darwin.lgamma((df + 1) / 2.0) - Darwin.lgamma(df / 2.0) - 0.5 * Darwin.log(df * .pi)
      let pdfVal = Darwin.exp(logCoef) * Darwin.pow(1.0 + t2 / df, -(df + 1) / 2.0)

      let dx = (cdfVal - p) / pdfVal
      x -= dx
      if abs(dx) < 1e-14 * max(abs(x), 1.0) { break }
    }

    return loc + scale * x
  }

  /// Percent point function paired with a limitation diagnostic.
  ///
  /// This is the **self-aware** overload of ``ppf(_:)``. It returns the same
  /// best-effort quantile, but wrapped in a ``Diagnosed`` so the caller can tell
  /// whether the input lies inside the method's accuracy envelope.
  ///
  /// Per the documented limitation (CLAUDE.md *Known Limitations §1*), the
  /// Student-t `ppf` achieves only ~5 significant digits in the **extreme
  /// tails** (`p < 1e-4` or `p > 1 - 1e-4`, i.e. `|p|` beyond `0.9999`); the
  /// central and near-tail regions reach full double precision. When `p` is in
  /// the extreme tail this overload attaches an
  /// ``NumericDiagnostic/outsideEnvelope(method:reason:)`` diagnostic; otherwise
  /// the returned ``Diagnosed/diagnostics`` array is empty.
  ///
  /// The bare ``ppf(_:)`` is unchanged and remains the right entry point when a
  /// caller does not need the envelope signal.
  ///
  /// - Note: This overload exists only on ``TDistribution`` because the
  ///   extreme-tail precision loss is specific to the Student-t quantile (it
  ///   inverts the regularized incomplete beta, whose tail conditioning is the
  ///   documented limitation). ``NormalDistribution``/``ChiSquaredDistribution``
  ///   etc. reach full double precision across their domain via the (now
  ///   accurate) ``erfinv(_:)`` path and so need no `ppfDiagnosed` companion.
  ///
  /// - Parameter p: Requested probability in `[0, 1]`.
  /// - Returns: A ``Diagnosed`` wrapping the quantile and any diagnostic.
  public func ppfDiagnosed(_ p: Double) -> Diagnosed<Double> {
    let value = ppf(p)
    // Extreme-tail envelope: p in (0, ppfTailEnvelope) ∪ (1 − ppfTailEnvelope, 1),
    // i.e. beyond the 0.9999 probability mark on either side. Only flag finite,
    // in-domain probabilities — out-of-domain p is the bare ppf's NaN contract,
    // not an accuracy-envelope concern.
    if p >= 0, p <= 1, p < TDistribution.ppfTailEnvelope || p > 1.0 - TDistribution.ppfTailEnvelope {
      return Diagnosed(
        value,
        diagnostics: [
          .outsideEnvelope(
            method: "TDistribution.ppf",
            reason: "p=\(p) is in the extreme tail (p < \(TDistribution.ppfTailEnvelope) "
              + "or p > \(1.0 - TDistribution.ppfTailEnvelope)) — precision is ~5 digits, not full double"
          )
        ]
      )
    }
    return Diagnosed(value)
  }

  /// Tail distance from 0 or 1 beyond which ``ppfDiagnosed(_:)`` flags the
  /// Student-t quantile as outside its full-precision envelope (`p < 1e-4` or
  /// `p > 1 − 1e-4`, i.e. `|p|` past the 0.9999 mark). Public so a caller can
  /// test its probability against the boundary before calling, mirroring
  /// ``LinAlg/solveConditionEnvelope`` and ``erfinvEnvelopeBoundary``.
  public static let ppfTailEnvelope = 1e-4

  /// Random variate sampling using ratio of normal and chi-squared.
  public func rvs() -> Double {
    let z = randomNormal()
    let chi2 = randomGamma(df / 2.0) * 2.0
    return loc + scale * z / Darwin.sqrt(chi2 / df)
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Double] {
    guard n > 0 else { return [] }
    return (0..<n).map { _ in rvs() }
  }

  /// Log of the probability density function.
  ///
  /// Computed directly in log-space: `log(pdf(x))` returns −Inf once the density
  /// underflows, whereas the closed form stays finite into the far tails.
  public func logpdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    let logCoef =
      Darwin.lgamma((df + 1) / 2.0) - Darwin.lgamma(df / 2.0) - 0.5 * Darwin.log(df * .pi)
    // log(1 + z²/df) via softplus on a = 2·log|z| − log(df) so that z² never
    // overflows to Inf for very large |x| (log(0) = −Inf softplus's to 0 for z = 0).
    let a = 2.0 * Darwin.log(abs(z)) - Darwin.log(df)
    let log1pZ2 = a > 0 ? a + Foundation.log1p(Foundation.exp(-a)) : Foundation.log1p(Foundation.exp(a))
    return logCoef - (df + 1) / 2.0 * log1pZ2 - Darwin.log(scale)
  }

  /// Survival function (complementary CDF): `1 - cdf(x)`.
  public func sf(_ x: Double) -> Double {
    1.0 - cdf(x)
  }

  /// Inverse survival function: `ppf(1 - p)`.
  public func isf(_ p: Double) -> Double {
    ppf(1.0 - p)
  }

  /// Distribution mean (undefined for df <= 1).
  public var mean: Double {
    df > 1 ? loc : .nan
  }

  /// Distribution variance (undefined for df <= 2).
  public var variance: Double {
    if df > 2 {
      return scale * scale * df / (df - 2)
    } else if df > 1 {
      return .infinity
    }
    return .nan
  }

  /// Distribution standard deviation.
  public var std: Double {
    if df > 2 {
      return scale * Darwin.sqrt(df / (df - 2))
    } else if df > 1 {
      return .infinity
    }
    return .nan
  }
}

// MARK: - Chi-squared Distribution

/// Chi-squared distribution.
public struct ChiSquaredDistribution {
  /// Degrees of freedom.
  public let df: Double
  /// Location parameter.
  public let loc: Double
  /// Scale parameter.
  public let scale: Double

  public init(df: Double, loc: Double = 0, scale: Double = 1) {
    precondition(df > 0, "ChiSquaredDistribution: df must be positive, got \(df)")
    precondition(scale > 0, "ChiSquaredDistribution: scale must be positive, got \(scale)")
    self.df = df
    self.loc = loc
    self.scale = scale
  }

  /// Probability density function.
  public func pdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z < 0 { return 0.0 }

    let k2 = df / 2.0

    // Handle z=0 special case
    if z == 0 {
      if df < 2 { return .infinity }
      if df == 2 { return 0.5 / scale }
      return 0.0  // df > 2
    }

    // Log-space: pow(2,k2)·tgamma(k2) overflows to Inf for df > 342, silently
    // zeroing the density. lgamma keeps it finite.
    let logPdf = (k2 - 1) * Foundation.log(z) - z / 2.0 - k2 * Foundation.log(2.0) - lgamma(k2)
    return Foundation.exp(logPdf) / scale
  }

  /// Cumulative distribution function.
  public func cdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z <= 0 { return 0.0 }
    return gammainc(df / 2.0, z / 2.0)
  }

  /// Percent point function using Newton-Raphson.
  public func ppf(_ p: Double) -> Double {
    if let g = ppfDomainGuard(p, lower: loc, upper: .infinity) { return g }
    // Wilson-Hilferty approximation as starting point
    var x =
      df
      * Darwin.pow(
        1.0 - 2.0 / (9.0 * df) + Darwin.sqrt(2.0) * erfinv(2.0 * p - 1.0)
          * Darwin.sqrt(2.0 / (9.0 * df)), 3)
    if x < 0.01 { x = 0.01 }

    let k2 = df / 2.0
    for _ in 0..<50 {
      let cdfVal = gammainc(k2, x / 2.0)
      let pdfVal = Foundation.exp(
        (k2 - 1) * Foundation.log(x) - x / 2.0 - k2 * Foundation.log(2.0) - lgamma(k2))

      let dx = (cdfVal - p) / pdfVal
      x -= dx
      if x < 0.001 { x = 0.001 }
      if abs(dx) < 1e-10 { break }
    }

    return loc + scale * x
  }

  /// Random variate sampling.
  public func rvs() -> Double {
    loc + scale * randomGamma(df / 2.0) * 2.0
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Double] {
    guard n > 0 else { return [] }
    return (0..<n).map { _ in rvs() }
  }

  /// Log of the probability density function.
  ///
  /// Computed in closed form rather than as `log(pdf(x))`: once `pdf` underflows
  /// to `0` (e.g. far in the tail or for large `df`), `log(0) = -inf` discards the
  /// true log-density. The direct form stays finite across the whole support.
  public func logpdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z < 0 { return -.infinity }
    let k2 = df / 2.0
    if z == 0 {
      if df < 2 { return .infinity }
      if df == 2 { return Foundation.log(0.5 / scale) }
      return -.infinity  // df > 2: density 0
    }
    let logPdf = (k2 - 1) * Foundation.log(z) - z / 2.0 - k2 * Foundation.log(2.0) - lgamma(k2)
    return logPdf - Foundation.log(scale)
  }

  /// Survival function (complementary CDF): `1 - cdf(x)`.
  public func sf(_ x: Double) -> Double {
    1.0 - cdf(x)
  }

  /// Inverse survival function: `ppf(1 - p)`.
  public func isf(_ p: Double) -> Double {
    ppf(1.0 - p)
  }

  /// Distribution mean.
  public var mean: Double { loc + scale * df }

  /// Distribution variance.
  public var variance: Double { scale * scale * 2 * df }

  /// Distribution standard deviation.
  public var std: Double { scale * Darwin.sqrt(2 * df) }
}

// MARK: - F Distribution

/// F distribution.
public struct FDistribution {
  /// Numerator degrees of freedom.
  public let dfn: Double
  /// Denominator degrees of freedom.
  public let dfd: Double
  /// Location parameter.
  public let loc: Double
  /// Scale parameter.
  public let scale: Double

  public init(dfn: Double, dfd: Double, loc: Double = 0, scale: Double = 1) {
    precondition(dfn > 0, "FDistribution: dfn must be positive, got \(dfn)")
    precondition(dfd > 0, "FDistribution: dfd must be positive, got \(dfd)")
    precondition(scale > 0, "FDistribution: scale must be positive, got \(scale)")
    self.dfn = dfn
    self.dfd = dfd
    self.loc = loc
    self.scale = scale
  }

  /// Probability density function.
  ///
  /// Evaluated through the log-density and exponentiated. The direct product
  /// form `(dfn·z)^(dfn/2) · dfd^(dfd/2) / (dfn·z+dfd)^((dfn+dfd)/2)` overflows to
  /// `inf/inf = NaN` for moderate-to-large degrees of freedom; the log form,
  /// using `lgamma` for `log B(dfn/2, dfd/2)`, stays finite.
  /// Log of the standard (loc=0, scale=1) F density at `z > 0`. Shared by `pdf`,
  /// `logpdf`, and the `ppf` Newton step so all three are overflow-safe.
  private func logPdfStandard(_ z: Double) -> Double {
    let halfN = dfn / 2.0
    let halfD = dfd / 2.0
    // log B(halfN, halfD) = lgamma(halfN) + lgamma(halfD) - lgamma(halfN + halfD)
    let logBeta = lgamma(halfN) + lgamma(halfD) - lgamma(halfN + halfD)
    return halfN * Foundation.log(dfn * z) + halfD * Foundation.log(dfd)
      - (halfN + halfD) * Foundation.log(dfn * z + dfd)
      - Foundation.log(z) - logBeta
  }

  public func pdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z <= 0 { return 0.0 }
    return Foundation.exp(logPdfStandard(z)) / scale
  }

  /// Cumulative distribution function.
  public func cdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z <= 0 { return 0.0 }

    let u = dfn * z / (dfn * z + dfd)
    return betainc(dfn / 2.0, dfd / 2.0, u)
  }

  /// Percent point function using Newton-Raphson.
  public func ppf(_ p: Double) -> Double {
    if let g = ppfDomainGuard(p, lower: loc, upper: .infinity) { return g }
    // Starting point: use median approximation
    var x = dfd / (dfd - 2) * (dfn > 2 ? (dfn - 2) / dfn : 1.0)
    if x < 0.01 { x = 0.01 }

    for _ in 0..<100 {
      let u = dfn * x / (dfn * x + dfd)
      let cdfVal = betainc(dfn / 2.0, dfd / 2.0, u)

      // Log-space standard density (overflow-safe for large df), same as `pdf`.
      let pdfVal = Foundation.exp(logPdfStandard(x))

      if pdfVal < 1e-30 { break }
      let dx = (cdfVal - p) / pdfVal
      x -= dx
      if x < 0.001 { x = 0.001 }
      if abs(dx) < 1e-10 { break }
    }

    return loc + scale * x
  }

  /// Random variate sampling.
  public func rvs() -> Double {
    let chi1 = randomGamma(dfn / 2.0) * 2.0
    let chi2 = randomGamma(dfd / 2.0) * 2.0
    return loc + scale * (chi1 / dfn) / (chi2 / dfd)
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Double] {
    guard n > 0 else { return [] }
    return (0..<n).map { _ in rvs() }
  }

  /// Log of the probability density function.
  ///
  /// Closed form via the shared log-density, so tail underflow does not collapse
  /// a finite log density to `-inf` (as `log(pdf(x))` would).
  public func logpdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z <= 0 { return -.infinity }
    return logPdfStandard(z) - Foundation.log(scale)
  }

  /// Survival function (complementary CDF): `1 - cdf(x)`.
  public func sf(_ x: Double) -> Double {
    1.0 - cdf(x)
  }

  /// Inverse survival function: `ppf(1 - p)`.
  public func isf(_ p: Double) -> Double {
    ppf(1.0 - p)
  }

  /// Distribution mean (defined for dfd > 2).
  public var mean: Double {
    dfd > 2 ? loc + scale * dfd / (dfd - 2) : .nan
  }

  /// Distribution variance (defined for dfd > 4).
  public var variance: Double {
    if dfd > 4 {
      let num = 2 * dfd * dfd * (dfn + dfd - 2)
      let den = dfn * (dfd - 2) * (dfd - 2) * (dfd - 4)
      return scale * scale * num / den
    }
    return .nan
  }

  /// Distribution standard deviation.
  public var std: Double {
    if dfd > 4 {
      return Darwin.sqrt(variance)
    }
    return .nan
  }
}

// MARK: - Gamma Distribution

/// Gamma distribution.
public struct GammaDistribution {
  /// Shape parameter.
  public let shape: Double
  /// Location parameter.
  public let loc: Double
  /// Scale parameter.
  public let scale: Double

  public init(shape: Double, loc: Double = 0, scale: Double = 1) {
    precondition(shape > 0, "GammaDistribution: shape must be positive, got \(shape)")
    precondition(scale > 0, "GammaDistribution: scale must be positive, got \(scale)")
    self.shape = shape
    self.loc = loc
    self.scale = scale
  }

  /// Probability density function.
  public func pdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z < 0 { return 0.0 }
    // Support boundary z == 0 (SciPy parity): shape < 1 diverges, shape == 1 is
    // the exponential density 1/scale, shape > 1 vanishes.
    if z == 0 {
      if shape < 1 { return .infinity }
      if shape == 1 { return 1.0 / scale }
      return 0.0
    }
    // Log-space: tgamma(shape) overflows to Inf for shape > 171, silently
    // zeroing the density. lgamma keeps it finite.
    let logPdf = (shape - 1) * Foundation.log(z) - z - lgamma(shape)
    return Foundation.exp(logPdf) / scale
  }

  /// Cumulative distribution function.
  public func cdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z <= 0 { return 0.0 }
    return gammainc(shape, z)
  }

  /// Percent point function using Newton-Raphson.
  public func ppf(_ p: Double) -> Double {
    if let g = ppfDomainGuard(p, lower: loc, upper: .infinity) { return g }
    var x = shape  // Mean as starting point
    if x < 0.1 { x = 0.1 }

    for _ in 0..<100 {
      let cdfVal = gammainc(shape, x)
      let pdfVal = Foundation.exp((shape - 1) * Foundation.log(x) - x - lgamma(shape))

      if pdfVal < 1e-30 { break }
      let dx = (cdfVal - p) / pdfVal
      x -= dx
      if x < 0.001 { x = 0.001 }
      if abs(dx) < 1e-10 { break }
    }

    return loc + scale * x
  }

  /// Random variate sampling.
  public func rvs() -> Double {
    loc + scale * randomGamma(shape)
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Double] {
    guard n > 0 else { return [] }
    return (0..<n).map { _ in rvs() }
  }

  /// Log of the probability density function.
  ///
  /// Closed form `(shape−1)·ln z − z − lgamma(shape) − ln(scale)`, so the tail
  /// stays finite where `log(pdf(x))` would collapse to `-inf` after underflow.
  public func logpdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z < 0 { return -.infinity }
    if z == 0 {
      if shape < 1 { return .infinity }
      if shape == 1 { return -Foundation.log(scale) }  // log(1/scale)
      return -.infinity
    }
    return (shape - 1) * Foundation.log(z) - z - lgamma(shape) - Foundation.log(scale)
  }

  /// Survival function (complementary CDF): `1 - cdf(x)`.
  public func sf(_ x: Double) -> Double {
    1.0 - cdf(x)
  }

  /// Inverse survival function: `ppf(1 - p)`.
  public func isf(_ p: Double) -> Double {
    ppf(1.0 - p)
  }

  /// Distribution mean.
  public var mean: Double { loc + scale * shape }

  /// Distribution variance.
  public var variance: Double { scale * scale * shape }

  /// Distribution standard deviation.
  public var std: Double { scale * Darwin.sqrt(shape) }
}

// MARK: - Beta Distribution

/// Beta distribution.
public struct BetaDistribution {
  /// Shape parameter a.
  public let a: Double
  /// Shape parameter b.
  public let b: Double
  /// Location parameter.
  public let loc: Double
  /// Scale parameter.
  public let scale: Double

  public init(a: Double, b: Double, loc: Double = 0, scale: Double = 1) {
    precondition(a > 0, "BetaDistribution: a must be positive, got \(a)")
    precondition(b > 0, "BetaDistribution: b must be positive, got \(b)")
    precondition(scale > 0, "BetaDistribution: scale must be positive, got \(scale)")
    self.a = a
    self.b = b
    self.loc = loc
    self.scale = scale
  }

  /// Probability density function.
  public func pdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z < 0 || z > 1 { return 0.0 }
    // Support boundaries (SciPy parity). At z == 0 the density is governed by a:
    // a < 1 diverges, a == 1 gives 1/B(1,b) = b, a > 1 vanishes. At z == 1 it is
    // governed symmetrically by b. Values are divided by `scale`.
    if z == 0 {
      if a < 1 { return .infinity }
      if a == 1 { return b / scale }
      return 0.0
    }
    if z == 1 {
      if b < 1 { return .infinity }
      if b == 1 { return a / scale }
      return 0.0
    }
    return Darwin.pow(z, a - 1) * Darwin.pow(1 - z, b - 1) / beta(a, b) / scale
  }

  /// Cumulative distribution function.
  public func cdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z <= 0 { return 0.0 }
    if z >= 1 { return 1.0 }
    return betainc(a, b, z)
  }

  /// Percent point function using Newton-Raphson.
  public func ppf(_ p: Double) -> Double {
    if let g = ppfDomainGuard(p, lower: loc, upper: loc + scale) { return g }
    var x = a / (a + b)  // Use mean as starting point

    for _ in 0..<100 {
      let cdfVal = betainc(a, b, x)
      let pdfVal = Darwin.pow(x, a - 1) * Darwin.pow(1 - x, b - 1) / beta(a, b)

      if pdfVal < 1e-30 { break }
      let dx = (cdfVal - p) / pdfVal
      x -= dx
      if x < 0.001 { x = 0.001 }
      if x > 0.999 { x = 0.999 }
      if abs(dx) < 1e-10 { break }
    }

    return loc + scale * x
  }

  /// Random variate sampling using gamma ratio.
  public func rvs() -> Double {
    let g1 = randomGamma(a)
    let g2 = randomGamma(b)
    return loc + scale * g1 / (g1 + g2)
  }

  /// Generate n random variates.
  public func rvs(_ n: Int) -> [Double] {
    guard n > 0 else { return [] }
    return (0..<n).map { _ in rvs() }
  }

  /// Log of the probability density function.
  ///
  /// Closed form `(a−1)·ln z + (b−1)·ln(1−z) − log B(a,b) − ln(scale)` (with
  /// `log B(a,b) = lgamma(a)+lgamma(b)−lgamma(a+b)`), matching the SciPy-parity
  /// boundary values of `pdf` and avoiding tail collapse from `log(pdf(x))`.
  public func logpdf(_ x: Double) -> Double {
    let z = (x - loc) / scale
    if z < 0 || z > 1 { return -.infinity }
    let logScale = Foundation.log(scale)
    if z == 0 {
      if a < 1 { return .infinity }
      if a == 1 { return Foundation.log(b) - logScale }  // log(b/scale)
      return -.infinity
    }
    if z == 1 {
      if b < 1 { return .infinity }
      if b == 1 { return Foundation.log(a) - logScale }  // log(a/scale)
      return -.infinity
    }
    let logBeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    return (a - 1) * Foundation.log(z) + (b - 1) * Foundation.log(1 - z) - logBeta - logScale
  }

  /// Survival function (complementary CDF): `1 - cdf(x)`.
  public func sf(_ x: Double) -> Double {
    1.0 - cdf(x)
  }

  /// Inverse survival function: `ppf(1 - p)`.
  public func isf(_ p: Double) -> Double {
    ppf(1.0 - p)
  }

  /// Distribution mean.
  public var mean: Double { loc + scale * a / (a + b) }

  /// Distribution variance.
  public var variance: Double {
    let apb = a + b
    return scale * scale * a * b / (apb * apb * (apb + 1))
  }

  /// Distribution standard deviation.
  public var std: Double { Darwin.sqrt(variance) }
}

// MARK: - Statistical Tests

/// Result of a statistical test.
public struct TestResult {
  /// Test statistic.
  public let statistic: Double
  /// p-value.
  public let pvalue: Double
}

/// One-sample t-test.
/// Tests whether the mean of a sample differs from a specified value.
public func ttest1Sample(_ sample: [Double], popmean: Double) -> TestResult? {
  guard sample.count >= 2 else { return nil }

  let n = Double(sample.count)
  let sampleMean = Stats.mean(sample)
  let sampleStd = Stats.stddev(sample, ddof: 1)
  let se = sampleStd / Darwin.sqrt(n)

  guard se > 0 else { return TestResult(statistic: .nan, pvalue: .nan) }

  let tStat = (sampleMean - popmean) / se
  let df = n - 1

  // Two-tailed p-value using t-distribution CDF
  let pvalue = 2.0 * (1.0 - tCdf(abs(tStat), df: df))

  return TestResult(statistic: tStat, pvalue: pvalue)
}

/// Independent two-sample t-test.
/// Tests whether the means of two independent samples differ.
public func ttestIndependent(_ sample1: [Double], _ sample2: [Double], equalVariance: Bool = true)
  -> TestResult?
{
  guard sample1.count >= 2, sample2.count >= 2 else { return nil }

  let n1 = Double(sample1.count)
  let n2 = Double(sample2.count)
  let mean1 = Stats.mean(sample1)
  let mean2 = Stats.mean(sample2)
  let var1 = Stats.variance(sample1, ddof: 1)
  let var2 = Stats.variance(sample2, ddof: 1)

  let tStat: Double
  let df: Double

  if equalVariance {
    // Pooled variance
    let sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    let se = Darwin.sqrt(sp2 * (1 / n1 + 1 / n2))
    guard se > 0 else { return TestResult(statistic: .nan, pvalue: .nan) }
    tStat = (mean1 - mean2) / se
    df = n1 + n2 - 2
  } else {
    // Welch's t-test
    let se = Darwin.sqrt(var1 / n1 + var2 / n2)
    guard se > 0 else { return TestResult(statistic: .nan, pvalue: .nan) }
    tStat = (mean1 - mean2) / se
    // Welch-Satterthwaite degrees of freedom
    let num = Darwin.pow(var1 / n1 + var2 / n2, 2)
    let den = Darwin.pow(var1 / n1, 2) / (n1 - 1) + Darwin.pow(var2 / n2, 2) / (n2 - 1)
    df = num / den
  }

  let pvalue = 2.0 * (1.0 - tCdf(abs(tStat), df: df))

  return TestResult(statistic: tStat, pvalue: pvalue)
}

/// Pearson correlation coefficient and p-value.
public func pearsonr(_ x: [Double], _ y: [Double]) -> TestResult? {
  guard x.count == y.count, x.count >= 3 else { return nil }

  let n = Double(x.count)
  let meanX = Stats.mean(x)
  let meanY = Stats.mean(y)

  var sumXY = 0.0
  var sumX2 = 0.0
  var sumY2 = 0.0

  for i in 0..<x.count {
    let dx = x[i] - meanX
    let dy = y[i] - meanY
    sumXY += dx * dy
    sumX2 += dx * dx
    sumY2 += dy * dy
  }

  guard sumX2 > 0, sumY2 > 0 else {
    return TestResult(statistic: .nan, pvalue: .nan)
  }

  let r = sumXY / Darwin.sqrt(sumX2 * sumY2)

  // t-statistic for correlation
  let t = r * Darwin.sqrt((n - 2) / (1 - r * r))
  let df = n - 2
  let pvalue = 2.0 * (1.0 - tCdf(abs(t), df: df))

  return TestResult(statistic: r, pvalue: pvalue)
}

/// Spearman rank correlation coefficient and p-value.
public func spearmanr(_ x: [Double], _ y: [Double]) -> TestResult? {
  guard x.count == y.count, x.count >= 3 else { return nil }

  let rankX = computeRanks(x)
  let rankY = computeRanks(y)

  // Compute Pearson correlation on ranks
  return pearsonr(rankX, rankY)
}

/// Compute ranks with average for ties.
private func computeRanks(_ arr: [Double]) -> [Double] {
  let indexed = arr.enumerated().map { ($0.offset, $0.element) }
  let sorted = indexed.sorted { $0.1 < $1.1 }

  var ranks = [Double](repeating: 0, count: arr.count)
  var i = 0
  while i < sorted.count {
    var j = i
    // Find ties
    while j < sorted.count - 1 && sorted[j].1 == sorted[j + 1].1 {
      j += 1
    }
    // Average rank for ties
    let avgRank = Double(i + j + 2) / 2.0
    for k in i...j {
      ranks[sorted[k].0] = avgRank
    }
    i = j + 1
  }
  return ranks
}

/// t-distribution CDF helper.
private func tCdf(_ t: Double, df: Double) -> Double {
  if t == 0 { return 0.5 }
  let x = df / (df + t * t)
  let p = 0.5 * betainc(df / 2.0, 0.5, x)
  return t > 0 ? 1.0 - p : p
}

// MARK: - Descriptive Statistics Extensions

/// Result of describe function.
public struct DescribeResult {
  public let nobs: Int
  public let min: Double
  public let max: Double
  public let mean: Double
  public let variance: Double
  public let skewness: Double
  public let kurtosis: Double
}

/// Compute descriptive statistics for a dataset.
public func describe(_ data: [Double]) -> DescribeResult? {
  guard !data.isEmpty else { return nil }

  let n = data.count
  let nDouble = Double(n)
  let dataMean = Stats.mean(data)
  let dataVar = Stats.variance(data, ddof: 1)
  let dataStd = Darwin.sqrt(dataVar)

  // Skewness (Fisher's definition)
  var m3 = 0.0
  for x in data {
    m3 += Darwin.pow(x - dataMean, 3)
  }
  m3 /= nDouble
  let skewness =
    n > 2 && dataStd > 0
    ? m3 / Darwin.pow(dataStd * Darwin.sqrt((nDouble - 1) / nDouble), 3)
      * Darwin.sqrt(nDouble * (nDouble - 1)) / (nDouble - 2) : .nan

  // Kurtosis (Fisher's definition, excess kurtosis)
  var m4 = 0.0
  for x in data {
    m4 += Darwin.pow(x - dataMean, 4)
  }
  m4 /= nDouble
  let kurtosis: Double
  if n > 3 && dataVar > 0 {
    let m2 = dataVar * (nDouble - 1) / nDouble
    let g2 = m4 / (m2 * m2) - 3
    kurtosis = (nDouble - 1) / ((nDouble - 2) * (nDouble - 3)) * ((nDouble + 1) * g2 + 6)
  } else {
    kurtosis = .nan
  }

  return DescribeResult(
    nobs: n,
    min: data.min()!,
    max: data.max()!,
    mean: dataMean,
    variance: dataVar,
    skewness: skewness,
    kurtosis: kurtosis
  )
}

/// Compute z-scores (standard scores) for a dataset.
public func zscore(_ data: [Double], ddof: Int = 0) -> [Double] {
  guard data.count > ddof else { return [] }

  let dataMean = Stats.mean(data)
  let dataStd = Stats.stddev(data, ddof: ddof)

  guard dataStd > 0 else {
    return data.map { _ in Double.nan }
  }

  return data.map { ($0 - dataMean) / dataStd }
}

/// Compute skewness (Fisher's definition).
public func skew(_ data: [Double]) -> Double {
  guard data.count >= 3 else { return .nan }

  let n = Double(data.count)
  let dataMean = Stats.mean(data)
  let dataStd = Stats.stddev(data, ddof: 1)

  guard dataStd > 0 else { return .nan }

  var m3 = 0.0
  for x in data {
    m3 += Darwin.pow(x - dataMean, 3)
  }
  m3 /= n

  // Fisher's skewness
  return m3 / Darwin.pow(dataStd * Darwin.sqrt((n - 1) / n), 3) * Darwin.sqrt(n * (n - 1)) / (n - 2)
}

/// Compute kurtosis (Fisher's definition by default).
/// - Parameters:
///   - data: Input data array.
///   - fisher: If true (default), returns excess kurtosis. If false, returns standard kurtosis.
public func kurtosis(_ data: [Double], fisher: Bool = true) -> Double {
  guard data.count >= 4 else { return .nan }

  let n = Double(data.count)
  let dataMean = Stats.mean(data)
  let dataVar = Stats.variance(data, ddof: 1)

  guard dataVar > 0 else { return .nan }

  var m4 = 0.0
  for x in data {
    m4 += Darwin.pow(x - dataMean, 4)
  }
  m4 /= n

  let m2 = dataVar * (n - 1) / n
  let g2 = m4 / (m2 * m2) - 3
  let kurt = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)

  return fisher ? kurt : kurt + 3
}
