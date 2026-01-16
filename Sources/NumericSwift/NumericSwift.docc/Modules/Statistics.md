# Statistics

Descriptive statistics functions.

## Overview

The Statistics module provides functions for computing descriptive statistics on arrays of numerical data.

## Central Tendency

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

let avg = mean(data)     // 3.0
let med = median(data)   // 3.0
let mod = mode(data)     // Mode (most frequent value)

// Geometric and harmonic means
let gm = gmean(data)     // Geometric mean
let hm = hmean(data)     // Harmonic mean
```

## Dispersion

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

// Variance (ddof = degrees of freedom adjustment)
let v = variance(data, ddof: 0)  // Population variance
let s2 = variance(data, ddof: 1) // Sample variance

// Standard deviation
let sd = stddev(data, ddof: 1)

// Range
let range = ptp(data)  // Peak-to-peak (max - min)
```

## Percentiles and Quantiles

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

let p25 = percentile(data, 25)   // 25th percentile
let p50 = percentile(data, 50)   // Median
let p75 = percentile(data, 75)   // 75th percentile
```

## Min/Max Functions

```swift
let data = [-2.0, 1.0, 3.0, -5.0, 4.0]

let minimum = amin(data)   // -5.0
let maximum = amax(data)   // 4.0
let range = ptp(data)      // 9.0 (peak-to-peak)
```

## Cumulative Operations

```swift
let data = [1.0, 2.0, 3.0, 4.0]

let cs = cumsum(data)   // [1, 3, 6, 10]
let cp = cumprod(data)  // [1, 2, 6, 24]
let d = diff(data)      // [1, 1, 1] (differences)
```

## Rounding and Clipping

```swift
// Scalar operations
let rounded = round(3.7, decimals: 0)  // 4.0
let truncated = trunc(3.7)             // 3.0
let s = sign(-5.0)                     // -1.0
let clipped = clip(15.0, min: 0, max: 10)  // 10.0

// Array operations
let clippedArray = clip(values, min: 0, max: 1)
```

## Topics

### Central Tendency

- ``mean(_:)``
- ``median(_:)``
- ``mode(_:)``
- ``gmean(_:)``
- ``hmean(_:)``

### Dispersion

- ``variance(_:ddof:)``
- ``stddev(_:ddof:)``
- ``ptp(_:)``

### Percentiles

- ``percentile(_:_:)``

### Min/Max

- ``amin(_:)``
- ``amax(_:)``

### Cumulative Operations

- ``cumsum(_:)``
- ``cumprod(_:)``
- ``diff(_:)``

### Rounding and Clipping

- ``round(_:decimals:)``
- ``trunc(_:)``
- ``sign(_:)``
- ``clip(_:min:max:)-7x9y2``
