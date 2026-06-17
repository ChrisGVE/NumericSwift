# Statistics

Descriptive statistics functions.

## Overview

The `Statistics` module provides functions for computing descriptive statistics
on arrays of numerical data.

All functions live under the `Stats` namespace. The old top-level free
functions (e.g. `mean`, `median`, `variance`) are still available as
`@available(*, deprecated)` shims so existing code continues to compile with a
deprecation warning. New code should use the namespaced forms.

> Note: The namespace is `Stats` (not `Statistics`) to keep call sites concise
> and avoid a name clash with the module itself.

## Migration from Top-Level Functions

```swift
// Before (deprecated)
mean(data)
stddev(data, ddof: 1)
percentile(data, 75)

// After
Stats.mean(data)
Stats.stddev(data, ddof: 1)
Stats.percentile(data, 75)
```

## Central Tendency

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

let avg = Stats.mean(data)     // 3.0
let med = Stats.median(data)   // 3.0
let mod = Stats.mode(data)     // mode (smallest on tie, matching SciPy)

// Geometric and harmonic means
let gm = Stats.gmean(data)     // geometric mean
let hm = Stats.hmean(data)     // harmonic mean
```

## Dispersion

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

// Variance (ddof = degrees of freedom adjustment)
let v  = Stats.variance(data, ddof: 0)  // population variance
let s2 = Stats.variance(data, ddof: 1)  // sample variance

// Standard deviation
let sd = Stats.stddev(data, ddof: 1)

// Range
let range = Stats.ptp(data)  // peak-to-peak (max − min)
```

## Percentiles and Quantiles

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

let p25 = Stats.percentile(data, 25)   // 25th percentile
let p50 = Stats.percentile(data, 50)   // median
let p75 = Stats.percentile(data, 75)   // 75th percentile
```

## Min/Max Functions

```swift
let data = [-2.0, 1.0, 3.0, -5.0, 4.0]

let minimum = Stats.amin(data)   // -5.0
let maximum = Stats.amax(data)   // 4.0
let range   = Stats.ptp(data)    // 9.0 (peak-to-peak)
```

## Cumulative Operations

```swift
let data = [1.0, 2.0, 3.0, 4.0]

let cs = Stats.cumsum(data)   // [1, 3, 6, 10]
let cp = Stats.cumprod(data)  // [1, 2, 6, 24]
let d  = Stats.diff(data)     // [1, 1, 1] (first differences)
```

## Rounding and Clipping

```swift
// Scalar operations
let rounded = Stats.round(3.7, decimals: 0)      // 4.0
let truncated = Stats.trunc(3.7)                 // 3.0
let s = Stats.sign(-5.0)                         // -1.0
let clipped = Stats.clip(15.0, min: 0, max: 10)  // 10.0

// Array clipping
let clippedArray = Stats.clip(values, min: 0, max: 1)
```

## Aggregation

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

let total = Stats.sum(data)  // 15.0
```

## Topics

### Namespace

- ``Stats``

### Central Tendency

- ``Stats/mean(_:)``
- ``Stats/median(_:)``
- ``Stats/mode(_:)``
- ``Stats/gmean(_:)``
- ``Stats/hmean(_:)``

### Dispersion

- ``Stats/variance(_:ddof:)``
- ``Stats/stddev(_:ddof:)``
- ``Stats/ptp(_:)``

### Percentiles

- ``Stats/percentile(_:_:)``

### Min/Max

- ``Stats/amin(_:)``
- ``Stats/amax(_:)``

### Cumulative Operations

- ``Stats/cumsum(_:)``
- ``Stats/cumprod(_:)``
- ``Stats/diff(_:)``

### Aggregation

- ``Stats/sum(_:)``

### Rounding and Clipping

- ``Stats/round(_:decimals:)``
- ``Stats/trunc(_:)``
- ``Stats/sign(_:)``
- ``Stats/clip(_:min:max:)->Double``
- ``Stats/clip(_:min:max:)->[Double]``
