//
//  Distributions.swift
//  NumericSwift
//
//  Probability distributions following scipy.stats patterns.
//  Each distribution provides pdf, cdf, ppf (quantile), rvs (random sampling),
//  and moment functions (mean, variance, std).
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Random Number Generators

/// Box-Muller transform for standard normal random variates.
public func randomNormal() -> Double {
    let u1 = Double.random(in: Double.ulpOfOne..<1.0)
    let u2 = Double.random(in: 0..<1.0)
    return Darwin.sqrt(-2.0 * Darwin.log(u1)) * Darwin.cos(2.0 * .pi * u2)
}

/// Generate n standard normal random variates.
public func randomNormal(_ n: Int) -> [Double] {
    (0..<n).map { _ in randomNormal() }
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
    (0..<n).map { _ in randomGamma(shape) }
}

// MARK: - Normal Distribution

/// Normal (Gaussian) distribution.
public struct NormalDistribution {
    /// Location parameter (mean).
    public let loc: Double
    /// Scale parameter (standard deviation).
    public let scale: Double

    public init(loc: Double = 0, scale: Double = 1) {
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
        loc + scale * Darwin.sqrt(2.0) * erfinv(2.0 * p - 1.0)
    }

    /// Random variate sampling.
    public func rvs() -> Double {
        loc + scale * randomNormal()
    }

    /// Generate n random variates.
    public func rvs(_ n: Int) -> [Double] {
        (0..<n).map { _ in rvs() }
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
        loc + scale * p
    }

    /// Random variate sampling.
    public func rvs() -> Double {
        loc + scale * Double.random(in: 0..<1)
    }

    /// Generate n random variates.
    public func rvs(_ n: Int) -> [Double] {
        (0..<n).map { _ in rvs() }
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
        loc - scale * Darwin.log(1.0 - p)
    }

    /// Random variate sampling.
    public func rvs() -> Double {
        loc - scale * Darwin.log(Double.random(in: Double.ulpOfOne..<1.0))
    }

    /// Generate n random variates.
    public func rvs(_ n: Int) -> [Double] {
        (0..<n).map { _ in rvs() }
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
        self.df = df
        self.loc = loc
        self.scale = scale
    }

    /// Probability density function.
    public func pdf(_ x: Double) -> Double {
        let z = (x - loc) / scale
        // Use lgamma for numerical stability with large df (tgamma overflows for df > 340)
        let logCoef = Darwin.lgamma((df + 1) / 2.0) - Darwin.lgamma(df / 2.0) - 0.5 * Darwin.log(df * .pi)
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
        if p <= 0 { return -.infinity }
        if p >= 1 { return .infinity }
        if p == 0.5 { return loc }

        // Initial approximation: normal quantile (erfinv now fixed for full precision)
        var x = Darwin.sqrt(2.0) * erfinv(2.0 * p - 1.0)

        // Newton-Raphson iteration with lgamma for numerical stability
        for _ in 0..<100 {
            let t2 = x * x
            let cdfVal = 0.5 + 0.5 * (1.0 - betainc(df / 2.0, 0.5, df / (df + t2))) * (x >= 0 ? 1 : -1)

            // Use lgamma for stability with large df (tgamma overflows for df > 340)
            let logCoef = Darwin.lgamma((df + 1) / 2.0) - Darwin.lgamma(df / 2.0) - 0.5 * Darwin.log(df * .pi)
            let pdfVal = Darwin.exp(logCoef) * Darwin.pow(1.0 + t2 / df, -(df + 1) / 2.0)

            let dx = (cdfVal - p) / pdfVal
            x -= dx
            if abs(dx) < 1e-14 * max(abs(x), 1.0) { break }
        }

        return loc + scale * x
    }

    /// Random variate sampling using ratio of normal and chi-squared.
    public func rvs() -> Double {
        let z = randomNormal()
        let chi2 = randomGamma(df / 2.0) * 2.0
        return loc + scale * z / Darwin.sqrt(chi2 / df)
    }

    /// Generate n random variates.
    public func rvs(_ n: Int) -> [Double] {
        (0..<n).map { _ in rvs() }
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

        return Darwin.pow(z, k2 - 1) * Darwin.exp(-z / 2.0) / (Darwin.pow(2.0, k2) * Darwin.tgamma(k2)) / scale
    }

    /// Cumulative distribution function.
    public func cdf(_ x: Double) -> Double {
        let z = (x - loc) / scale
        if z <= 0 { return 0.0 }
        return gammainc(df / 2.0, z / 2.0)
    }

    /// Percent point function using Newton-Raphson.
    public func ppf(_ p: Double) -> Double {
        // Wilson-Hilferty approximation as starting point
        var x = df * Darwin.pow(1.0 - 2.0 / (9.0 * df) + Darwin.sqrt(2.0) * erfinv(2.0 * p - 1.0) * Darwin.sqrt(2.0 / (9.0 * df)), 3)
        if x < 0.01 { x = 0.01 }

        let k2 = df / 2.0
        for _ in 0..<50 {
            let cdfVal = gammainc(k2, x / 2.0)
            let pdfVal = Darwin.pow(x, k2 - 1) * Darwin.exp(-x / 2.0) / (Darwin.pow(2.0, k2) * Darwin.tgamma(k2))

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
        (0..<n).map { _ in rvs() }
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
        self.dfn = dfn
        self.dfd = dfd
        self.loc = loc
        self.scale = scale
    }

    /// Probability density function.
    public func pdf(_ x: Double) -> Double {
        let z = (x - loc) / scale
        if z <= 0 { return 0.0 }

        let num = Darwin.pow(dfn * z, dfn / 2.0) * Darwin.pow(dfd, dfd / 2.0)
        let den = Darwin.pow(dfn * z + dfd, (dfn + dfd) / 2.0)
        let coef = 1.0 / (z * beta(dfn / 2.0, dfd / 2.0))

        return coef * num / den / scale
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
        // Starting point: use median approximation
        var x = dfd / (dfd - 2) * (dfn > 2 ? (dfn - 2) / dfn : 1.0)
        if x < 0.01 { x = 0.01 }

        for _ in 0..<100 {
            let u = dfn * x / (dfn * x + dfd)
            let cdfVal = betainc(dfn / 2.0, dfd / 2.0, u)

            let num = Darwin.pow(dfn * x, dfn / 2.0) * Darwin.pow(dfd, dfd / 2.0)
            let den = Darwin.pow(dfn * x + dfd, (dfn + dfd) / 2.0)
            let pdfVal = num / den / (x * beta(dfn / 2.0, dfd / 2.0))

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
        (0..<n).map { _ in rvs() }
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
        self.shape = shape
        self.loc = loc
        self.scale = scale
    }

    /// Probability density function.
    public func pdf(_ x: Double) -> Double {
        let z = (x - loc) / scale
        if z <= 0 { return 0.0 }
        return Darwin.pow(z, shape - 1) * Darwin.exp(-z) / Darwin.tgamma(shape) / scale
    }

    /// Cumulative distribution function.
    public func cdf(_ x: Double) -> Double {
        let z = (x - loc) / scale
        if z <= 0 { return 0.0 }
        return gammainc(shape, z)
    }

    /// Percent point function using Newton-Raphson.
    public func ppf(_ p: Double) -> Double {
        var x = shape // Mean as starting point
        if x < 0.1 { x = 0.1 }

        for _ in 0..<100 {
            let cdfVal = gammainc(shape, x)
            let pdfVal = Darwin.pow(x, shape - 1) * Darwin.exp(-x) / Darwin.tgamma(shape)

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
        (0..<n).map { _ in rvs() }
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
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
    }

    /// Probability density function.
    public func pdf(_ x: Double) -> Double {
        let z = (x - loc) / scale
        if z <= 0 || z >= 1 { return 0.0 }
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
        var x = a / (a + b) // Use mean as starting point

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
        (0..<n).map { _ in rvs() }
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
    let sampleMean = mean(sample)
    let sampleStd = stddev(sample, ddof: 1)
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
public func ttestIndependent(_ sample1: [Double], _ sample2: [Double], equalVariance: Bool = true) -> TestResult? {
    guard sample1.count >= 2, sample2.count >= 2 else { return nil }

    let n1 = Double(sample1.count)
    let n2 = Double(sample2.count)
    let mean1 = mean(sample1)
    let mean2 = mean(sample2)
    let var1 = NumericSwift.variance(sample1, ddof: 1)
    let var2 = NumericSwift.variance(sample2, ddof: 1)

    let tStat: Double
    let df: Double

    if equalVariance {
        // Pooled variance
        let sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        let se = Darwin.sqrt(sp2 * (1/n1 + 1/n2))
        guard se > 0 else { return TestResult(statistic: .nan, pvalue: .nan) }
        tStat = (mean1 - mean2) / se
        df = n1 + n2 - 2
    } else {
        // Welch's t-test
        let se = Darwin.sqrt(var1/n1 + var2/n2)
        guard se > 0 else { return TestResult(statistic: .nan, pvalue: .nan) }
        tStat = (mean1 - mean2) / se
        // Welch-Satterthwaite degrees of freedom
        let num = Darwin.pow(var1/n1 + var2/n2, 2)
        let den = Darwin.pow(var1/n1, 2)/(n1-1) + Darwin.pow(var2/n2, 2)/(n2-1)
        df = num / den
    }

    let pvalue = 2.0 * (1.0 - tCdf(abs(tStat), df: df))

    return TestResult(statistic: tStat, pvalue: pvalue)
}

/// Pearson correlation coefficient and p-value.
public func pearsonr(_ x: [Double], _ y: [Double]) -> TestResult? {
    guard x.count == y.count, x.count >= 3 else { return nil }

    let n = Double(x.count)
    let meanX = mean(x)
    let meanY = mean(y)

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
    let dataMean = mean(data)
    let dataVar = NumericSwift.variance(data, ddof: 1)
    let dataStd = Darwin.sqrt(dataVar)

    // Skewness (Fisher's definition)
    var m3 = 0.0
    for x in data {
        m3 += Darwin.pow(x - dataMean, 3)
    }
    m3 /= nDouble
    let skewness = n > 2 && dataStd > 0 ?
        m3 / Darwin.pow(dataStd * Darwin.sqrt((nDouble-1)/nDouble), 3) * Darwin.sqrt(nDouble*(nDouble-1)) / (nDouble-2) : .nan

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

    let dataMean = mean(data)
    let dataStd = stddev(data, ddof: ddof)

    guard dataStd > 0 else {
        return data.map { _ in Double.nan }
    }

    return data.map { ($0 - dataMean) / dataStd }
}

/// Compute skewness (Fisher's definition).
public func skew(_ data: [Double]) -> Double {
    guard data.count >= 3 else { return .nan }

    let n = Double(data.count)
    let dataMean = mean(data)
    let dataStd = stddev(data, ddof: 1)

    guard dataStd > 0 else { return .nan }

    var m3 = 0.0
    for x in data {
        m3 += Darwin.pow(x - dataMean, 3)
    }
    m3 /= n

    // Fisher's skewness
    return m3 / Darwin.pow(dataStd * Darwin.sqrt((n-1)/n), 3) * Darwin.sqrt(n*(n-1)) / (n-2)
}

/// Compute kurtosis (Fisher's definition by default).
/// - Parameters:
///   - data: Input data array.
///   - fisher: If true (default), returns excess kurtosis. If false, returns standard kurtosis.
public func kurtosis(_ data: [Double], fisher: Bool = true) -> Double {
    guard data.count >= 4 else { return .nan }

    let n = Double(data.count)
    let dataMean = mean(data)
    let dataVar = NumericSwift.variance(data, ddof: 1)

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
