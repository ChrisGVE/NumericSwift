# Number Theory

Number-theoretic functions and primality testing.

## Overview

The NumberTheory module provides functions for primality testing, factorization, combinatorics, and other number-theoretic computations.

## Primality Testing

```swift
isPrime(17)      // true
isPrime(18)      // false

// Generate primes
let primes = primesUpTo(100)  // [2, 3, 5, 7, 11, ...]

// Prime counting
let count = primePi(100)      // 25 (number of primes <= 100)
```

## Factorization

```swift
// Prime factorization
let factors = primeFactors(60)  // [(2, 2), (3, 1), (5, 1)]
// Means: 60 = 2^2 * 3^1 * 5^1
```

## GCD and LCM

```swift
let g = gcd(12, 18)    // 6
let l = lcm(4, 6)      // 12

// Extended GCD: ax + by = gcd(a,b)
let (d, x, y) = extendedGcd(12, 18)
```

## Combinatorics

```swift
// Factorial
let fact = factorial(5)    // 120

// Permutations P(n, k)
let p = perm(5, 3)         // 60

// Combinations C(n, k) = "n choose k"
let c = comb(5, 3)         // 10
let b = binomial(5, 3)     // 10 (alias)
```

## Arithmetic Functions

```swift
// Euler's totient function
let phi = eulerPhi(12)        // 4 (numbers coprime to 12)

// Divisor functions
let numDivisors = divisorSigma(12, k: 0)  // Number of divisors
let sumDivisors = divisorSigma(12, k: 1)  // Sum of divisors

// Mobius function
let mu = mobius(12)           // 0

// Liouville function
let lambda = liouville(12)    // 1

// Carmichael function
let carmichael = carmichael(12)  // 2

// von Mangoldt function
let vonM = vonMangoldt(8)     // log(2)
```

## Prime Counting Functions

```swift
// Prime counting function pi(x)
let pi = primePi(100)         // 25

// Chebyshev functions
let theta = chebyshevTheta(100)
let psi = chebyshevPsi(100)
```

## Modular Arithmetic

```swift
// Modular exponentiation: a^b mod m
let result = modPow(2, 10, 1000)  // 24

// Modular inverse: a^(-1) mod m
let inv = modInverse(3, 11)       // 4 (since 3*4 = 12 ≡ 1 mod 11)
```

## Digit Functions

```swift
let n = 12345

let sum = digitSum(n)        // 15
let root = digitalRoot(n)    // 6
let count = digitCount(n)    // 5
```

## Topics

### Primality

- ``isPrime(_:)``
- ``primesUpTo(_:)``
- ``primePi(_:)``

### Factorization

- ``primeFactors(_:)``

### GCD/LCM

- ``gcd(_:_:)``
- ``lcm(_:_:)``
- ``extendedGcd(_:_:)``

### Combinatorics

- ``factorial(_:)``
- ``perm(_:_:)``
- ``comb(_:_:)``
- ``binomial(_:_:)``

### Arithmetic Functions

- ``eulerPhi(_:)``
- ``divisorSigma(_:k:)``
- ``mobius(_:)``
- ``liouville(_:)``
- ``carmichael(_:)``
- ``vonMangoldt(_:)``

### Prime Counting

- ``chebyshevTheta(_:)``
- ``chebyshevPsi(_:)``

### Modular Arithmetic

- ``modPow(_:_:_:)``
- ``modInverse(_:_:)``

### Digit Functions

- ``digitSum(_:)``
- ``digitalRoot(_:)``
- ``digitCount(_:)``
