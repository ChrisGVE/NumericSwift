# Probability Distributions

Probability distributions and statistical tests following scipy.stats patterns.

## Overview

The Distributions module provides continuous probability distributions and statistical hypothesis tests. Each distribution implements a consistent interface with PDF, CDF, PPF (quantile function), random sampling, and moment calculations.

## Distribution Interface

All distributions provide these methods:

- `pdf(x)` - Probability density function
- `cdf(x)` - Cumulative distribution function
- `ppf(p)` - Percent point function (quantile/inverse CDF)
- `rvs(n)` - Generate n random variates
- `mean` - Distribution mean
- `variance` - Distribution variance
- `std` - Standard deviation

## Available Distributions

```swift
// Continuous distributions
let norm = NormalDistribution(loc: 0, scale: 1)
let t = TDistribution(df: 10)
let chi2 = ChiSquaredDistribution(df: 5)
let f = FDistribution(dfn: 5, dfd: 10)
let gamma = GammaDistribution(shape: 2, scale: 1)
let beta = BetaDistribution(a: 2, b: 5)
let exp = ExponentialDistribution(scale: 1)
let uniform = UniformDistribution(low: 0, high: 1)
```

## Using Distributions

```swift
let norm = NormalDistribution(loc: 0, scale: 1)

// Evaluate PDF and CDF
let density = norm.pdf(0)       // 0.3989...
let probability = norm.cdf(1.96) // 0.975

// Find quantiles
let q95 = norm.ppf(0.95)        // 1.645

// Generate random samples
let samples = norm.rvs(1000)

// Moments
let mu = norm.mean              // 0
let sigma = norm.std            // 1
```

## Statistical Tests

```swift
// One-sample t-test
let (statistic, pvalue) = ttest1Sample(data, popmean: 0)

// Two-sample t-test
let (t, p) = ttestIndependent(group1, group2)

// Correlation tests
let (r, p) = pearsonr(x, y)     // Pearson correlation
let (rho, p) = spearmanr(x, y)  // Spearman correlation
```

## Descriptive Statistics

```swift
let stats = describe(data)
print(stats.mean)
print(stats.variance)
print(stats.skewness)
print(stats.kurtosis)

// Z-scores
let standardized = zscore(data)
```

## Topics

### Continuous Distributions

- ``NormalDistribution``
- ``TDistribution``
- ``ChiSquaredDistribution``
- ``FDistribution``
- ``GammaDistribution``
- ``BetaDistribution``
- ``ExponentialDistribution``
- ``UniformDistribution``

### Statistical Tests

- ``ttest1Sample(_:popmean:)``
- ``ttestIndependent(_:_:equalVariance:)``
- ``pearsonr(_:_:)``
- ``spearmanr(_:_:)``

### Descriptive Functions

- ``describe(_:)``
- ``zscore(_:)``
- ``skew(_:)``
- ``kurtosis(_:)``
