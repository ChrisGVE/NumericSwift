# Special Functions

Mathematical special functions following scipy.special patterns.

## Overview

The SpecialFunctions module provides implementations of mathematical special functions including error functions, gamma functions, Bessel functions, elliptic integrals, and more.

## Error Functions

```swift
// Error function and complement
let y = erf(1.0)      // 0.8427...
let yc = erfc(1.0)    // 0.1573... (1 - erf(1))

// Inverse error function
let x = erfinv(0.5)   // 0.4769...
let xc = erfcinv(0.5) // 0.4769...

// Verify: erf(erfinv(y)) == y
```

## Gamma Functions

```swift
// Gamma function (via tgamma)
let g = tgamma(5.0)   // 24.0 (= 4!)

// Log-gamma (for large arguments)
let lg = lgamma(100.0)

// Digamma (psi) function
let psi = digamma(2.0)  // 0.4228...

// Incomplete gamma functions
let lower = gammainc(2.0, 1.0)   // Lower regularized
let upper = gammaincc(2.0, 1.0)  // Upper regularized
```

## Beta Functions

```swift
// Beta function B(a,b)
let b = beta(2.0, 3.0)

// Incomplete beta function (regularized)
let ib = betainc(2.0, 3.0, 0.5)
```

## Bessel Functions

```swift
// Bessel functions of first kind
let j0_val = j0(1.0)   // J_0(1)
let j1_val = j1(1.0)   // J_1(1)
let jn_val = jn(2, 1.0) // J_2(1)

// Bessel functions of second kind
let y0_val = y0(1.0)   // Y_0(1)
let y1_val = y1(1.0)   // Y_1(1)
let yn_val = yn(2, 1.0) // Y_2(1)

// Modified Bessel functions
let i_val = besseli(0, 1.0)  // I_0(1)
let k_val = besselk(0, 1.0)  // K_0(1)
```

## Elliptic Integrals

```swift
// Complete elliptic integral of first kind K(m)
let k = ellipk(0.5)

// Complete elliptic integral of second kind E(m)
let e = ellipe(0.5)
```

## Other Functions

```swift
// Riemann zeta function
let z = zeta(2.0)     // pi^2/6

// Lambert W function
let w = lambertw(1.0) // 0.5671...
```

## Complex Versions

```swift
let z = Complex(re: 1, im: 1)

// Complex gamma
let cg = cgamma(z)

// Complex log-gamma
let clg = clgamma(z)

// Complex zeta
let cz = czeta(z)
```

## Topics

### Error Functions

- ``erf(_:)``
- ``erfc(_:)``
- ``erfinv(_:)``
- ``erfcinv(_:)``

### Gamma Functions

- ``digamma(_:)``
- ``gammainc(_:_:)``
- ``gammaincc(_:_:)``
- ``cgamma(_:)``
- ``clgamma(_:)``

### Beta Functions

- ``beta(_:_:)``
- ``betainc(_:_:_:)``

### Bessel Functions

- ``j0(_:)``
- ``j1(_:)``
- ``jn(_:_:)``
- ``y0(_:)``
- ``y1(_:)``
- ``yn(_:_:)``
- ``besseli(_:_:)``
- ``besselk(_:_:)``

### Elliptic Integrals

- ``ellipk(_:)``
- ``ellipe(_:)``

### Other Functions

- ``zeta(_:)``
- ``czeta(_:)``
- ``lambertw(_:)``
