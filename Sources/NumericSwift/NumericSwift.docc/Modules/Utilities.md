# Utilities

vDSP-optimized array operations.

## Overview

The Utilities module provides array operations optimized using Apple's vDSP framework from Accelerate. These functions operate efficiently on large arrays.

## Scalar Functions

```swift
// Rounding
let r = roundValue(3.7)         // 4.0
let t = truncValue(3.7)         // 3.0

// Sign
let s = signValue(-5.0)         // -1.0

// Clipping
let c = clipValue(15.0, lo: 0, hi: 10)  // 10.0
```

## Array Math Functions

All array functions use vDSP for optimal performance:

```swift
let values = [1.0, 2.0, 3.0, 4.0]

// Rounding operations
let rounded = roundArray(values)
let truncated = truncArray(values)
let floored = floorArray(values)
let ceiled = ceilArray(values)

// Sign and absolute value
let signs = signArray(values)
let absolutes = absArray(values)
let negated = negArray(values)

// Clipping
let clipped = clipArray(values, lo: 1.5, hi: 3.5)
```

## Power and Root Functions

```swift
let values = [1.0, 4.0, 9.0, 16.0]

let roots = sqrtArray(values)     // [1, 2, 3, 4]
let squares = squareArray(values) // [1, 16, 81, 256]

// Element-wise power
let bases = [2.0, 2.0, 2.0]
let exponents = [1.0, 2.0, 3.0]
let powers = powArray(bases, exponents)  // [2, 4, 8]
```

## Exponential and Logarithmic Functions

```swift
let values = [1.0, 2.0, 3.0]

let exponentials = expArray(values)   // [e, e^2, e^3]
let logs = logArray(values)           // [0, ln(2), ln(3)]
let log10s = log10Array(values)       // [0, log10(2), log10(3)]
```

## Trigonometric Functions

```swift
let angles = [0.0, .pi/6, .pi/4, .pi/3, .pi/2]

// Basic trig
let sines = sinArray(angles)
let cosines = cosArray(angles)
let tangents = tanArray(angles)

// Inverse trig
let asins = asinArray([0.0, 0.5, 1.0])
let acoss = acosArray([1.0, 0.5, 0.0])
let atans = atanArray([0.0, 1.0])
```

## Hyperbolic Functions

```swift
let values = [0.0, 1.0, 2.0]

let sinhs = sinhArray(values)
let coshs = coshArray(values)
let tanhs = tanhArray(values)
```

## Topics

### Scalar Functions

- ``roundValue(_:)``
- ``truncValue(_:)``
- ``signValue(_:)``
- ``clipValue(_:lo:hi:)``

### Array Rounding

- ``roundArray(_:)``
- ``truncArray(_:)``
- ``floorArray(_:)``
- ``ceilArray(_:)``

### Array Math

- ``signArray(_:)``
- ``absArray(_:)``
- ``negArray(_:)``
- ``clipArray(_:lo:hi:)``
- ``sqrtArray(_:)``
- ``squareArray(_:)``
- ``powArray(_:_:)``

### Exponential/Logarithmic

- ``expArray(_:)``
- ``logArray(_:)``
- ``log10Array(_:)``

### Trigonometric

- ``sinArray(_:)``
- ``cosArray(_:)``
- ``tanArray(_:)``
- ``asinArray(_:)``
- ``acosArray(_:)``
- ``atanArray(_:)``

### Hyperbolic

- ``sinhArray(_:)``
- ``coshArray(_:)``
- ``tanhArray(_:)``
