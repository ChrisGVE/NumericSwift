# Complex Numbers

Complex number arithmetic with full operator support.

## Overview

The Complex module provides a `Complex` type representing complex numbers with double-precision components. All standard arithmetic operations, transcendental functions, and mathematical operations are supported.

## Creating Complex Numbers

```swift
// From rectangular coordinates
let z1 = Complex(re: 3, im: 4)

// From polar coordinates
let z2 = Complex.polar(r: 5, theta: .pi/4)

// Real number as complex
let z3 = Complex(3.0)  // 3 + 0i

// Imaginary unit
let i = Complex.i  // 0 + 1i
```

## Arithmetic Operations

Complex numbers support all standard arithmetic:

```swift
let a = Complex(re: 1, im: 2)
let b = Complex(re: 3, im: 4)

let sum = a + b         // Addition
let diff = a - b        // Subtraction
let product = a * b     // Multiplication
let quotient = a / b    // Division
let conj = a.conjugate  // Complex conjugate
```

## Mathematical Functions

```swift
let z = Complex(re: 1, im: 1)

z.abs        // Magnitude |z|
z.arg        // Argument (phase angle)
z.exp        // e^z
z.log        // ln(z)
z.sqrt       // Principal square root
z.pow(3)     // z^3

// Trigonometric functions
z.sin
z.cos
z.tan

// Hyperbolic functions
z.sinh
z.cosh
z.tanh
```

## Complex Square Root of Negative Reals

Use `csqrt` to get complex results from negative real numbers:

```swift
let result = csqrt(-4.0)  // Complex(re: 0, im: 2)
```

## Topics

### Free Functions

- ``csqrt(_:)``
- ``cgamma(_:)``
- ``clgamma(_:)``
- ``czeta(_:)``
