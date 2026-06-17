# Utilities

vDSP-optimized array operations.

## Overview

The `Utilities` module provides array operations optimized using Apple's vDSP
framework from Accelerate. These functions operate efficiently on large arrays.

All functions live under the `ArrayOps` namespace. The old top-level free
functions (e.g. `roundArray`, `sinArray`, `expArray`) are still available as
`@available(*, deprecated)` shims so existing code continues to compile with a
deprecation warning. New code should use the namespaced forms.

> Note: The namespace is `ArrayOps` (not `Utilities`) to describe what the
> members do rather than the module's internal file name.

## Migration from Top-Level Functions

```swift
// Before (deprecated)
sinArray(angles)
expArray(values)
clipArray(values, lo: 0.0, hi: 1.0)

// After
ArrayOps.sinArray(angles)
ArrayOps.expArray(values)
ArrayOps.clipArray(values, lo: 0.0, hi: 1.0)
```

## Scalar Functions

```swift
// Rounding
let r = ArrayOps.roundValue(3.7)         // 4.0
let t = ArrayOps.truncValue(3.7)         // 3.0

// Sign
let s = ArrayOps.signValue(-5.0)         // -1.0

// Clipping
let c = ArrayOps.clipValue(15.0, lo: 0, hi: 10)  // 10.0
```

## Array Math Functions

All array functions use vDSP for optimal performance:

```swift
let values = [1.0, 2.0, 3.0, 4.0]

// Rounding operations
let rounded   = ArrayOps.roundArray(values)
let truncated = ArrayOps.truncArray(values)
let floored   = ArrayOps.floorArray(values)
let ceiled    = ArrayOps.ceilArray(values)

// Sign and absolute value
let signs     = ArrayOps.signArray(values)
let absolutes = ArrayOps.absArray(values)
let negated   = ArrayOps.negArray(values)

// Clipping
let clipped = ArrayOps.clipArray(values, lo: 1.5, hi: 3.5)
```

## Power and Root Functions

```swift
let values = [1.0, 4.0, 9.0, 16.0]

let roots   = ArrayOps.sqrtArray(values)     // [1, 2, 3, 4]
let squares = ArrayOps.squareArray(values)   // [1, 16, 81, 256]

// Element-wise power
let bases     = [2.0, 2.0, 2.0]
let exponents = [1.0, 2.0, 3.0]
let powers    = ArrayOps.powArray(bases, exponents)  // [2, 4, 8]
```

## Exponential and Logarithmic Functions

```swift
let values = [1.0, 2.0, 3.0]

let exponentials = ArrayOps.expArray(values)    // [e, e², e³]
let logs         = ArrayOps.logArray(values)    // [0, ln(2), ln(3)]
let log10s       = ArrayOps.log10Array(values)  // [0, log₁₀(2), log₁₀(3)]
```

## Trigonometric Functions

```swift
let angles = [0.0, .pi/6, .pi/4, .pi/3, .pi/2]

// Basic trig
let sines    = ArrayOps.sinArray(angles)
let cosines  = ArrayOps.cosArray(angles)
let tangents = ArrayOps.tanArray(angles)

// Inverse trig
let asins = ArrayOps.asinArray([0.0, 0.5, 1.0])
let acoss = ArrayOps.acosArray([1.0, 0.5, 0.0])
let atans = ArrayOps.atanArray([0.0, 1.0])
```

## Hyperbolic Functions

```swift
let values = [0.0, 1.0, 2.0]

let sinhs = ArrayOps.sinhArray(values)
let coshs = ArrayOps.coshArray(values)
let tanhs = ArrayOps.tanhArray(values)
```

## Topics

### Namespace

- ``ArrayOps``

### Scalar Functions

- ``ArrayOps/roundValue(_:)``
- ``ArrayOps/truncValue(_:)``
- ``ArrayOps/signValue(_:)``
- ``ArrayOps/clipValue(_:lo:hi:)``

### Array Rounding

- ``ArrayOps/roundArray(_:)``
- ``ArrayOps/truncArray(_:)``
- ``ArrayOps/floorArray(_:)``
- ``ArrayOps/ceilArray(_:)``

### Array Math

- ``ArrayOps/signArray(_:)``
- ``ArrayOps/absArray(_:)``
- ``ArrayOps/negArray(_:)``
- ``ArrayOps/clipArray(_:lo:hi:)``
- ``ArrayOps/sqrtArray(_:)``
- ``ArrayOps/squareArray(_:)``
- ``ArrayOps/powArray(_:_:)``

### Exponential/Logarithmic

- ``ArrayOps/expArray(_:)``
- ``ArrayOps/logArray(_:)``
- ``ArrayOps/log10Array(_:)``

### Trigonometric

- ``ArrayOps/sinArray(_:)``
- ``ArrayOps/cosArray(_:)``
- ``ArrayOps/tanArray(_:)``
- ``ArrayOps/asinArray(_:)``
- ``ArrayOps/acosArray(_:)``
- ``ArrayOps/atanArray(_:)``

### Hyperbolic

- ``ArrayOps/sinhArray(_:)``
- ``ArrayOps/coshArray(_:)``
- ``ArrayOps/tanhArray(_:)``
