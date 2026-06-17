# Number Theory

Number-theoretic functions and primality testing.

## Overview

The `NumberTheory` module provides functions for primality testing, factorization, combinatorics, and other number-theoretic computations.

All functions live under the `NumberTheory` namespace. The old top-level free
functions (e.g. `isPrime`, `gcd`, `factorial`) are still available as
`@available(*, deprecated)` shims so existing code continues to compile with a
deprecation warning. New code should use the namespaced forms.

## Migration from Top-Level Functions

```swift
// Before (deprecated — produces a compiler warning)
isPrime(17)
gcd(12, 18)
factorial(5)

// After
NumberTheory.isPrime(17)
NumberTheory.gcd(12, 18)
NumberTheory.factorial(5)
```

## Primality Testing

```swift
NumberTheory.isPrime(17)      // true
NumberTheory.isPrime(18)      // false

// Generate primes
let primes = NumberTheory.primesUpTo(100)  // [2, 3, 5, 7, 11, ...]

// Prime counting
let count = NumberTheory.primePi(100)      // 25 (number of primes ≤ 100)
```

## Factorization

```swift
// Prime factorization
let factors = NumberTheory.primeFactors(60)  // [(2, 2), (3, 1), (5, 1)]
// Means: 60 = 2² × 3¹ × 5¹
```

## GCD and LCM

```swift
let g = NumberTheory.gcd(12, 18)    // 6
let l = NumberTheory.lcm(4, 6)      // 12

// Extended GCD: ax + by = gcd(a, b)
let (d, x, y) = NumberTheory.extendedGcd(12, 18)
```

## Combinatorics

```swift
// Factorial
let fact = NumberTheory.factorial(5)    // 120

// Permutations P(n, k)
let p = NumberTheory.perm(5, 3)         // 60

// Combinations C(n, k) = "n choose k"
let c = NumberTheory.comb(5, 3)         // 10
let b = NumberTheory.binomial(5, 3)     // 10 (alias)
```

## Arithmetic Functions

```swift
// Euler's totient function
let phi = NumberTheory.eulerPhi(12)        // 4 (numbers coprime to 12)

// Divisor functions
let numDivisors = NumberTheory.divisorSigma(12, k: 0)  // Number of divisors
let sumDivisors = NumberTheory.divisorSigma(12, k: 1)  // Sum of divisors

// Möbius function
let mu = NumberTheory.mobius(12)           // 0

// Liouville function
let lambda = NumberTheory.liouville(12)    // 1

// Carmichael function
let carm = NumberTheory.carmichael(12)     // 2

// von Mangoldt function
let vonM = NumberTheory.vonMangoldt(8)     // log(2)
```

## Prime Counting Functions

```swift
// Prime counting function π(x)
let pi = NumberTheory.primePi(100)         // 25

// Chebyshev functions
let theta = NumberTheory.chebyshevTheta(100)
let psi   = NumberTheory.chebyshevPsi(100)
```

## Modular Arithmetic

```swift
// Modular exponentiation: a^b mod m
let result = NumberTheory.modPow(2, 10, 1000)  // 24

// Modular inverse: a⁻¹ mod m
let inv = NumberTheory.modInverse(3, 11)        // 4 (since 3×4 = 12 ≡ 1 mod 11)
```

## Digit Functions

```swift
let n = 12345

let sum   = NumberTheory.digitSum(n)        // 15
let root  = NumberTheory.digitalRoot(n)     // 6
let count = NumberTheory.digitCount(n)      // 5
```

## Topics

### Namespace

- ``NumberTheory``

### Primality

- ``NumberTheory/isPrime(_:)``
- ``NumberTheory/primesUpTo(_:)``
- ``NumberTheory/primePi(_:)``

### Factorization

- ``NumberTheory/primeFactors(_:)``

### GCD/LCM

- ``NumberTheory/gcd(_:_:)``
- ``NumberTheory/lcm(_:_:)``
- ``NumberTheory/extendedGcd(_:_:)``

### Combinatorics

- ``NumberTheory/factorial(_:)``
- ``NumberTheory/perm(_:_:)``
- ``NumberTheory/comb(_:_:)``
- ``NumberTheory/binomial(_:_:)``

### Arithmetic Functions

- ``NumberTheory/eulerPhi(_:)``
- ``NumberTheory/divisorSigma(_:k:)``
- ``NumberTheory/mobius(_:)``
- ``NumberTheory/liouville(_:)``
- ``NumberTheory/carmichael(_:)``
- ``NumberTheory/vonMangoldt(_:)``

### Prime Counting

- ``NumberTheory/chebyshevTheta(_:)``
- ``NumberTheory/chebyshevPsi(_:)``

### Modular Arithmetic

- ``NumberTheory/modPow(_:_:_:)``
- ``NumberTheory/modInverse(_:_:)``

### Digit Functions

- ``NumberTheory/digitSum(_:base:)``
- ``NumberTheory/digitalRoot(_:)``
- ``NumberTheory/digitCount(_:base:)``
