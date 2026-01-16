//
//  NumberTheory.swift
//  NumericSwift
//
//  Number-theoretic arithmetic functions following scipy.special patterns.
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Primality and Factorization

/// Check if a number is prime using trial division.
///
/// Uses 6k±1 optimization for efficient trial division.
/// For very large numbers, consider probabilistic tests.
///
/// - Parameter n: The number to test
/// - Returns: true if n is prime
public func isPrime(_ n: Int) -> Bool {
    if n < 2 { return false }
    if n == 2 { return true }
    if n % 2 == 0 { return false }
    if n == 3 { return true }
    if n % 3 == 0 { return false }
    var i = 5
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 { return false }
        i += 6
    }
    return true
}

/// Get prime factorization of n as [(prime, exponent)] pairs.
///
/// - Parameter n: The number to factorize
/// - Returns: Array of (prime, exponent) tuples
public func primeFactors(_ n: Int) -> [(prime: Int, exponent: Int)] {
    guard n > 1 else { return [] }
    var result: [(Int, Int)] = []
    var remaining = n

    // Factor out 2s
    var count = 0
    while remaining % 2 == 0 {
        count += 1
        remaining /= 2
    }
    if count > 0 { result.append((2, count)) }

    // Factor out odd primes
    var factor = 3
    while factor * factor <= remaining {
        count = 0
        while remaining % factor == 0 {
            count += 1
            remaining /= factor
        }
        if count > 0 { result.append((factor, count)) }
        factor += 2
    }

    // Remaining is prime if > 1
    if remaining > 1 {
        result.append((remaining, 1))
    }

    return result
}

/// Generate primes up to n using Sieve of Eratosthenes.
///
/// - Parameter n: Upper bound (inclusive)
/// - Returns: Array of all primes ≤ n
public func primesUpTo(_ n: Int) -> [Int] {
    guard n >= 2 else { return [] }
    var sieve = [Bool](repeating: true, count: n + 1)
    sieve[0] = false
    sieve[1] = false

    var i = 2
    while i * i <= n {
        if sieve[i] {
            var j = i * i
            while j <= n {
                sieve[j] = false
                j += i
            }
        }
        i += 1
    }

    return sieve.enumerated().compactMap { $0.element ? $0.offset : nil }
}

// MARK: - GCD and LCM

/// Greatest common divisor using Euclidean algorithm.
///
/// - Parameters:
///   - a: First integer
///   - b: Second integer
/// - Returns: GCD of a and b
public func gcd(_ a: Int, _ b: Int) -> Int {
    var a = abs(a), b = abs(b)
    while b != 0 {
        let t = b
        b = a % b
        a = t
    }
    return a
}

/// Least common multiple.
///
/// - Parameters:
///   - a: First integer
///   - b: Second integer
/// - Returns: LCM of a and b
public func lcm(_ a: Int, _ b: Int) -> Int {
    let absA = abs(a)
    let absB = abs(b)
    if absA == 0 || absB == 0 { return 0 }
    return absA / gcd(absA, absB) * absB
}

// MARK: - Arithmetic Functions

/// Euler's totient function φ(n): count of integers 1 ≤ k ≤ n coprime to n.
///
/// φ(n) = n × Π(1 - 1/p) for all prime factors p of n
///
/// - Parameter n: Positive integer
/// - Returns: φ(n), or nil if n < 1
public func eulerPhi(_ n: Int) -> Int? {
    guard n >= 1 else { return nil }
    if n == 1 { return 1 }

    var result = n
    let factors = primeFactors(n)
    for (prime, _) in factors {
        result = result / prime * (prime - 1)
    }
    return result
}

/// Divisor sigma function σ_k(n): sum of k-th powers of divisors of n.
///
/// - σ_0(n) = number of divisors (divisor count)
/// - σ_1(n) = sum of divisors
/// - σ_2(n) = sum of squares of divisors
///
/// - Parameters:
///   - n: Positive integer
///   - k: Power (default 1)
/// - Returns: σ_k(n), or nil if n < 1
public func divisorSigma(_ n: Int, k: Int = 1) -> Double? {
    guard n >= 1 else { return nil }
    if n == 1 { return 1 }

    let factors = primeFactors(n)

    if k == 0 {
        // Number of divisors: Π(e+1)
        var result = 1
        for (_, exp) in factors {
            result *= (exp + 1)
        }
        return Double(result)
    } else {
        // Sum of k-th powers: Π((p^(k*(e+1)) - 1)/(p^k - 1))
        var result = 1.0
        for (prime, exp) in factors {
            let pk = Darwin.pow(Double(prime), Double(k))
            let numerator = Darwin.pow(pk, Double(exp + 1)) - 1.0
            let denominator = pk - 1.0
            result *= numerator / denominator
        }
        return result
    }
}

/// Möbius function μ(n).
///
/// - μ(n) = (-1)^k if n is product of k distinct primes
/// - μ(n) = 0 if n has a squared prime factor
///
/// - Parameter n: Positive integer
/// - Returns: μ(n), or nil if n < 1
public func mobius(_ n: Int) -> Int? {
    guard n >= 1 else { return nil }
    if n == 1 { return 1 }

    let factors = primeFactors(n)

    // Check for squared factors
    for (_, exp) in factors {
        if exp > 1 { return 0 }
    }

    // All exponents are 1, return (-1)^k
    return factors.count % 2 == 0 ? 1 : -1
}

/// Liouville function λ(n) = (-1)^Ω(n).
///
/// Where Ω(n) is the number of prime factors with multiplicity.
///
/// - Parameter n: Positive integer
/// - Returns: λ(n), or nil if n < 1
public func liouville(_ n: Int) -> Int? {
    guard n >= 1 else { return nil }
    if n == 1 { return 1 }

    let factors = primeFactors(n)
    let omega = factors.reduce(0) { $0 + $1.exponent }
    return omega % 2 == 0 ? 1 : -1
}

/// Carmichael function λ(n): smallest positive m such that a^m ≡ 1 (mod n) for all a coprime to n.
///
/// Also known as the reduced totient function.
///
/// - Parameter n: Positive integer
/// - Returns: λ(n), or nil if n < 1
public func carmichael(_ n: Int) -> Int? {
    guard n >= 1 else { return nil }
    if n == 1 { return 1 }

    let factors = primeFactors(n)
    var result = 1

    for (prime, exp) in factors {
        var lambda: Int
        if prime == 2 {
            // λ(2^k) = 2^(k-2) for k ≥ 3, otherwise φ(2^k)
            if exp >= 3 {
                lambda = 1 << (exp - 2)  // 2^(exp-2)
            } else {
                lambda = 1 << max(0, exp - 1)  // φ(2^exp)
            }
        } else {
            // λ(p^k) = φ(p^k) = p^(k-1) * (p-1)
            lambda = Int(Darwin.pow(Double(prime), Double(exp - 1))) * (prime - 1)
        }
        // lcm
        result = result / gcd(result, lambda) * lambda
    }

    return result
}

/// Von Mangoldt function Λ(n).
///
/// - Λ(n) = log(p) if n = p^k for some prime p and k ≥ 1
/// - Λ(n) = 0 otherwise
///
/// - Parameter n: Positive integer
/// - Returns: Λ(n)
public func vonMangoldt(_ n: Int) -> Double {
    if n < 2 { return 0 }

    let factors = primeFactors(n)

    // n must be a prime power (exactly one distinct prime factor)
    if factors.count == 1 {
        return Darwin.log(Double(factors[0].prime))
    }
    return 0
}

// MARK: - Prime Counting Functions

/// Prime counting function π(x): number of primes ≤ x.
///
/// - Parameter x: Upper bound
/// - Returns: Number of primes not exceeding x
public func primePi(_ x: Double) -> Int {
    guard x >= 2 else { return 0 }
    return primesUpTo(Int(x)).count
}

/// Chebyshev theta function θ(x) = Σ log(p) for primes p ≤ x.
///
/// - Parameter x: Upper bound
/// - Returns: θ(x)
public func chebyshevTheta(_ x: Double) -> Double {
    guard x >= 2 else { return 0 }

    let primes = primesUpTo(Int(x))
    var result = 0.0
    for p in primes {
        result += Darwin.log(Double(p))
    }
    return result
}

/// Chebyshev psi function ψ(x) = Σ Λ(n) for n ≤ x.
///
/// Where Λ(n) is the von Mangoldt function.
///
/// - Parameter x: Upper bound
/// - Returns: ψ(x)
public func chebyshevPsi(_ x: Double) -> Double {
    guard x >= 2 else { return 0 }

    var result = 0.0
    for n in 2...Int(x) {
        result += vonMangoldt(n)
    }
    return result
}

// MARK: - Modular Arithmetic

/// Modular exponentiation: compute (base^exp) mod m efficiently.
///
/// Uses binary exponentiation (square-and-multiply).
///
/// - Parameters:
///   - base: Base value
///   - exp: Exponent (non-negative)
///   - m: Modulus
/// - Returns: (base^exp) mod m
public func modPow(_ base: Int, _ exp: Int, _ m: Int) -> Int {
    guard m > 0 else { return 0 }
    guard exp >= 0 else { return 0 }
    if m == 1 { return 0 }

    var result = 1
    var b = base % m
    var e = exp

    while e > 0 {
        if e % 2 == 1 {
            result = (result * b) % m
        }
        e /= 2
        b = (b * b) % m
    }

    return result
}

/// Extended Euclidean algorithm.
///
/// Computes gcd(a, b) and coefficients x, y such that ax + by = gcd(a, b).
///
/// - Parameters:
///   - a: First integer
///   - b: Second integer
/// - Returns: (gcd, x, y) where ax + by = gcd
public func extendedGcd(_ a: Int, _ b: Int) -> (gcd: Int, x: Int, y: Int) {
    if b == 0 {
        return (a, 1, 0)
    }
    let (g, x1, y1) = extendedGcd(b, a % b)
    return (g, y1, x1 - (a / b) * y1)
}

/// Modular multiplicative inverse: find x such that (a * x) mod m = 1.
///
/// - Parameters:
///   - a: Value to invert
///   - m: Modulus
/// - Returns: Modular inverse, or nil if it doesn't exist (gcd(a, m) ≠ 1)
public func modInverse(_ a: Int, _ m: Int) -> Int? {
    let (g, x, _) = extendedGcd(a, m)
    guard g == 1 else { return nil }
    return ((x % m) + m) % m
}

// MARK: - Combinatorics

/// Factorial: n!
///
/// For n ≤ 20, computes exactly using multiplication.
/// For n > 20, uses the gamma function approximation.
///
/// - Parameter n: Non-negative integer
/// - Returns: n!, or NaN if n < 0
public func factorial(_ n: Int) -> Double {
    guard n >= 0 else { return .nan }

    if n <= 1 { return 1.0 }

    if n <= 20 {
        var result: Double = 1.0
        for i in 2...n {
            result *= Double(i)
        }
        return result
    } else {
        return Darwin.exp(Darwin.lgamma(Double(n) + 1))
    }
}

/// Permutations: P(n, k) = n! / (n-k)!
///
/// The number of ways to arrange k items from n distinct items.
///
/// - Parameters:
///   - n: Total number of items
///   - k: Number of items to arrange
/// - Returns: P(n, k), or NaN if inputs are invalid
public func perm(_ n: Int, _ k: Int) -> Double {
    guard n >= 0 && k >= 0 else { return .nan }

    if k > n { return 0 }
    if k == 0 { return 1 }

    if n <= 20 {
        var result: Double = 1.0
        for i in (n - k + 1)...n {
            result *= Double(i)
        }
        return result
    }

    return Darwin.exp(Darwin.lgamma(Double(n) + 1) - Darwin.lgamma(Double(n - k) + 1))
}

/// Combinations: C(n, k) = n! / (k! * (n-k)!)
///
/// The number of ways to choose k items from n distinct items (order doesn't matter).
/// Also known as the binomial coefficient "n choose k".
///
/// - Parameters:
///   - n: Total number of items
///   - k: Number of items to choose
/// - Returns: C(n, k), or NaN if inputs are invalid
public func comb(_ n: Int, _ k: Int) -> Double {
    guard n >= 0 && k >= 0 else { return .nan }

    if k > n { return 0 }
    if k == 0 || k == n { return 1 }

    // Use symmetry for efficiency
    let kUse = min(k, n - k)

    if n <= 20 {
        var result: Double = 1.0
        for i in 0..<kUse {
            result = result * Double(n - i) / Double(i + 1)
        }
        return Darwin.round(result)
    }

    let result = Darwin.exp(
        Darwin.lgamma(Double(n) + 1) -
        Darwin.lgamma(Double(kUse) + 1) -
        Darwin.lgamma(Double(n - kUse) + 1)
    )
    return Darwin.round(result)
}

/// Alias for comb (binomial coefficient).
///
/// - Parameters:
///   - n: Total number of items
///   - k: Number of items to choose
/// - Returns: The binomial coefficient C(n, k)
public func binomial(_ n: Int, _ k: Int) -> Double {
    comb(n, k)
}

// MARK: - Digit Functions

/// Sum of digits of n in given base.
///
/// - Parameters:
///   - n: Non-negative integer
///   - base: Base (default 10)
/// - Returns: Sum of digits
public func digitSum(_ n: Int, base: Int = 10) -> Int {
    guard n >= 0 && base >= 2 else { return 0 }
    var sum = 0
    var remaining = n
    while remaining > 0 {
        sum += remaining % base
        remaining /= base
    }
    return sum
}

/// Digital root (repeated digit sum until single digit).
///
/// - Parameter n: Non-negative integer
/// - Returns: Digital root
public func digitalRoot(_ n: Int) -> Int {
    guard n > 0 else { return 0 }
    return 1 + (n - 1) % 9
}

/// Count of digits in n in given base.
///
/// - Parameters:
///   - n: Positive integer
///   - base: Base (default 10)
/// - Returns: Number of digits
public func digitCount(_ n: Int, base: Int = 10) -> Int {
    guard n > 0 && base >= 2 else { return n == 0 ? 1 : 0 }
    return Int(Darwin.log(Double(n)) / Darwin.log(Double(base))) + 1
}
