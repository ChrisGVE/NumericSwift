//
//  NumberTheory.swift
//  NumericSwift
//
//  Number-theoretic arithmetic functions following scipy.special patterns.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - NumberTheory Namespace

/// Number-theoretic and combinatorial arithmetic functions.
///
/// ## Overview
///
/// ```swift
/// let primes = NumberTheory.primesUpTo(50)
/// let n      = NumberTheory.factorial(10)   // 3628800
/// let g      = NumberTheory.gcd(24, 36)     // 12
/// ```
public enum NumberTheory {

    // MARK: - Primality and Factorization

    /// Check if a number is prime using trial division.
    ///
    /// Uses 6k±1 optimization for efficient trial division.
    ///
    /// - Parameter n: The number to test
    /// - Returns: true if n is prime
    public static func isPrime(_ n: Int) -> Bool {
        if n < 2 { return false }
        if n == 2 { return true }
        if n % 2 == 0 { return false }
        if n == 3 { return true }
        if n % 3 == 0 { return false }
        var i = 5
        while i <= n / i {
            if n % i == 0 || n % (i + 2) == 0 { return false }
            i += 6
        }
        return true
    }

    /// Get prime factorization of n as [(prime, exponent)] pairs.
    ///
    /// - Parameter n: The number to factorize
    /// - Returns: Array of (prime, exponent) tuples
    public static func primeFactors(_ n: Int) -> [(prime: Int, exponent: Int)] {
        guard n > 1 else { return [] }
        var result: [(Int, Int)] = []
        var remaining = n

        var count = 0
        while remaining % 2 == 0 {
            count += 1
            remaining /= 2
        }
        if count > 0 { result.append((2, count)) }

        var factor = 3
        while factor <= remaining / factor {
            count = 0
            while remaining % factor == 0 {
                count += 1
                remaining /= factor
            }
            if count > 0 { result.append((factor, count)) }
            factor += 2
        }

        if remaining > 1 {
            result.append((remaining, 1))
        }

        return result
    }

    /// Generate primes up to n using Sieve of Eratosthenes.
    ///
    /// - Parameter n: Upper bound (inclusive)
    /// - Returns: Array of all primes ≤ n
    public static func primesUpTo(_ n: Int) -> [Int] {
        guard n >= 2 else { return [] }
        var sieve = [Bool](repeating: true, count: n + 1)
        sieve[0] = false
        sieve[1] = false

        var i = 2
        while i <= n / i {
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
    /// GCD over unsigned magnitudes — avoids the `abs(Int.min)` trap (`|Int.min|`
    /// is not representable as `Int`, but is as `UInt`).
    private static func gcdMagnitude(_ a: UInt, _ b: UInt) -> UInt {
        var x = a, y = b
        while y != 0 { (x, y) = (y, x % y) }
        return x
    }

    public static func gcd(_ a: Int, _ b: Int) -> Int {
        // `abs(Int.min)` traps; use `.magnitude` (UInt) and clamp the one
        // unrepresentable result, gcd(Int.min, Int.min) = 2^63, to Int.max.
        Int(clamping: gcdMagnitude(a.magnitude, b.magnitude))
    }

    /// Least common multiple.
    ///
    /// - Parameters:
    ///   - a: First integer
    ///   - b: Second integer
    /// - Returns: LCM of a and b. Non-trapping overflow policy: a result that
    ///   exceeds `Int.max` (including the `abs(Int.min)` edge) is clamped to `Int.max`.
    public static func lcm(_ a: Int, _ b: Int) -> Int {
        let absA = a.magnitude
        let absB = b.magnitude
        if absA == 0 || absB == 0 { return 0 }
        let (product, overflow) = (absA / gcdMagnitude(absA, absB)).multipliedReportingOverflow(by: absB)
        return overflow ? Int.max : Int(clamping: product)
    }

    // MARK: - Arithmetic Functions

    /// Euler's totient function φ(n).
    ///
    /// - Parameter n: Positive integer
    /// - Returns: φ(n), or nil if n < 1
    public static func eulerPhi(_ n: Int) -> Int? {
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
    /// - Parameters:
    ///   - n: Positive integer
    ///   - k: Power (default 1)
    /// - Returns: σ_k(n), or nil if n < 1
    public static func divisorSigma(_ n: Int, k: Int = 1) -> Double? {
        guard n >= 1 else { return nil }
        if n == 1 { return 1 }

        let factors = primeFactors(n)

        if k == 0 {
            var result = 1
            for (_, exp) in factors { result *= (exp + 1) }
            return Double(result)
        } else {
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
    /// - Parameter n: Positive integer
    /// - Returns: μ(n), or nil if n < 1
    public static func mobius(_ n: Int) -> Int? {
        guard n >= 1 else { return nil }
        if n == 1 { return 1 }

        let factors = primeFactors(n)
        for (_, exp) in factors {
            if exp > 1 { return 0 }
        }
        return factors.count % 2 == 0 ? 1 : -1
    }

    /// Liouville function λ(n) = (-1)^Ω(n).
    ///
    /// - Parameter n: Positive integer
    /// - Returns: λ(n), or nil if n < 1
    public static func liouville(_ n: Int) -> Int? {
        guard n >= 1 else { return nil }
        if n == 1 { return 1 }
        let factors = primeFactors(n)
        let omega = factors.reduce(0) { $0 + $1.exponent }
        return omega % 2 == 0 ? 1 : -1
    }

    /// Carmichael function λ(n): reduced totient function.
    ///
    /// - Parameter n: Positive integer
    /// - Returns: λ(n), or nil if n < 1
    public static func carmichael(_ n: Int) -> Int? {
        guard n >= 1 else { return nil }
        if n == 1 { return 1 }

        let factors = primeFactors(n)
        var result = 1

        for (prime, exp) in factors {
            var lambda: Int
            if prime == 2 {
                if exp >= 3 {
                    lambda = 1 << (exp - 2)
                } else {
                    lambda = 1 << max(0, exp - 1)
                }
            } else {
                lambda = Int(Darwin.pow(Double(prime), Double(exp - 1))) * (prime - 1)
            }
            result = result / gcd(result, lambda) * lambda
        }

        return result
    }

    /// Von Mangoldt function Λ(n).
    ///
    /// - Parameter n: Positive integer
    /// - Returns: Λ(n)
    public static func vonMangoldt(_ n: Int) -> Double {
        if n < 2 { return 0 }
        let factors = primeFactors(n)
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
    public static func primePi(_ x: Double) -> Int {
        // `Int(x)` traps for +inf or x > Int.max. `Double(Int.max)` rounds up to
        // 2^63 (unrepresentable as Int), so `x <= Double(Int.max)` still admits a
        // trapping boundary value — `Int(exactly:)` on the floor is exact.
        guard x.isFinite, x >= 2, let xi = Int(exactly: x.rounded(.down)) else { return 0 }
        return primesUpTo(xi).count
    }

    /// Chebyshev theta function θ(x) = Σ log(p) for primes p ≤ x.
    ///
    /// - Parameter x: Upper bound
    /// - Returns: θ(x)
    public static func chebyshevTheta(_ x: Double) -> Double {
        guard x.isFinite, x >= 2, let xi = Int(exactly: x.rounded(.down)) else { return 0 }
        let primes = primesUpTo(xi)
        var result = 0.0
        for p in primes { result += Darwin.log(Double(p)) }
        return result
    }

    /// Chebyshev psi function ψ(x) = Σ Λ(n) for n ≤ x.
    ///
    /// - Parameter x: Upper bound
    /// - Returns: ψ(x)
    public static func chebyshevPsi(_ x: Double) -> Double {
        guard x.isFinite, x >= 2, let xi = Int(exactly: x.rounded(.down)) else { return 0 }
        var result = 0.0
        for n in 2...xi { result += vonMangoldt(n) }
        return result
    }

    // MARK: - Modular Arithmetic

    /// Modular exponentiation: compute (base^exp) mod m efficiently.
    ///
    /// - Parameters:
    ///   - base: Base value
    ///   - exp: Exponent (non-negative)
    ///   - m: Modulus
    /// - Returns: (base^exp) mod m
    public static func modPow(_ base: Int, _ exp: Int, _ m: Int) -> Int {
        guard m > 0 else { return 0 }
        guard exp >= 0 else { return 0 }
        if m == 1 { return 0 }

        var result = 1
        var b = base % m
        if b < 0 { b += m }
        var e = exp

        while e > 0 {
            if e % 2 == 1 { result = mulMod(result, b, m) }
            e /= 2
            b = mulMod(b, b, m)
        }

        return result
    }

    /// Extended Euclidean algorithm.
    ///
    /// - Parameters:
    ///   - a: First integer
    ///   - b: Second integer
    /// - Returns: (gcd, x, y) where ax + by = gcd
    public static func extendedGcd(_ a: Int, _ b: Int) -> (gcd: Int, x: Int, y: Int) {
        if b == 0 { return (a, 1, 0) }
        // `Int.min % -1` and `Int.min / -1` overflow (|Int.min| is unrepresentable).
        // b == -1 always bottoms the recursion at extendedGcd(-1, 0) → (-1, 0, 1)
        // for every a (a·0 + (-1)·1 = -1); short-circuit it to avoid the trap.
        if b == -1 { return (-1, 0, 1) }
        let (g, x1, y1) = extendedGcd(b, a % b)
        return (g, y1, x1 - (a / b) * y1)
    }

    /// Modular multiplicative inverse.
    ///
    /// - Parameters:
    ///   - a: Value to invert
    ///   - m: Modulus
    /// - Returns: Modular inverse, or nil if it doesn't exist (gcd(a, m) ≠ 1)
    public static func modInverse(_ a: Int, _ m: Int) -> Int? {
        // A non-positive modulus is undefined and `x % 0` traps; reject it.
        guard m > 0 else { return nil }
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
    public static func factorial(_ n: Int) -> Double {
        guard n >= 0 else { return .nan }
        if n <= 1 { return 1.0 }
        if n <= 20 {
            var result: Double = 1.0
            for i in 2...n { result *= Double(i) }
            return result
        } else {
            return Darwin.exp(Darwin.lgamma(Double(n) + 1))
        }
    }

    /// Permutations: P(n, k) = n! / (n-k)!
    ///
    /// - Parameters:
    ///   - n: Total number of items
    ///   - k: Number of items to arrange
    /// - Returns: P(n, k), or NaN if inputs are invalid
    public static func perm(_ n: Int, _ k: Int) -> Double {
        guard n >= 0 && k >= 0 else { return .nan }
        if k > n { return 0 }
        if k == 0 { return 1 }
        if n <= 20 {
            var result: Double = 1.0
            for i in (n - k + 1)...n { result *= Double(i) }
            return result
        }
        return Darwin.exp(Darwin.lgamma(Double(n) + 1) - Darwin.lgamma(Double(n - k) + 1))
    }

    /// Combinations: C(n, k) = n! / (k! * (n-k)!)
    ///
    /// - Parameters:
    ///   - n: Total number of items
    ///   - k: Number of items to choose
    /// - Returns: C(n, k), or NaN if inputs are invalid
    public static func comb(_ n: Int, _ k: Int) -> Double {
        guard n >= 0 && k >= 0 else { return .nan }
        if k > n { return 0 }
        if k == 0 || k == n { return 1 }
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

    /// Alias for ``comb(_:_:)`` (binomial coefficient).
    ///
    /// - Parameters:
    ///   - n: Total number of items
    ///   - k: Number of items to choose
    /// - Returns: The binomial coefficient C(n, k)
    public static func binomial(_ n: Int, _ k: Int) -> Double {
        comb(n, k)
    }

    // MARK: - Digit Functions

    /// Sum of digits of n in given base.
    ///
    /// - Parameters:
    ///   - n: Non-negative integer
    ///   - base: Base (default 10)
    /// - Returns: Sum of digits
    public static func digitSum(_ n: Int, base: Int = 10) -> Int {
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
    public static func digitalRoot(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        return 1 + (n - 1) % 9
    }

    /// Count of digits in n in given base.
    ///
    /// - Parameters:
    ///   - n: Positive integer
    ///   - base: Base (default 10)
    /// - Returns: Number of digits
    public static func digitCount(_ n: Int, base: Int = 10) -> Int {
        guard n > 0 && base >= 2 else { return n == 0 ? 1 : 0 }
        return Int(Darwin.log(Double(n)) / Darwin.log(Double(base))) + 1
    }

    // MARK: - Private helpers

    /// Modular multiply (a·b) mod m without intermediate overflow.
    private static func mulMod(_ a: Int, _ b: Int, _ m: Int) -> Int {
        let (high, low) = a.multipliedFullWidth(by: b)
        let (_, remainder) = m.dividingFullWidth((high: high, low: low))
        return remainder
    }
}

// MARK: - Deprecated shims (backward compatibility)

/// - Note: Deprecated. Use ``NumberTheory/isPrime(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.isPrime(_:) instead")
public func isPrime(_ n: Int) -> Bool {
    NumberTheory.isPrime(n)
}

/// - Note: Deprecated. Use ``NumberTheory/primeFactors(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.primeFactors(_:) instead")
public func primeFactors(_ n: Int) -> [(prime: Int, exponent: Int)] {
    NumberTheory.primeFactors(n)
}

/// - Note: Deprecated. Use ``NumberTheory/primesUpTo(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.primesUpTo(_:) instead")
public func primesUpTo(_ n: Int) -> [Int] {
    NumberTheory.primesUpTo(n)
}

/// - Note: Deprecated. Use ``NumberTheory/gcd(_:_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.gcd(_:_:) instead")
public func gcd(_ a: Int, _ b: Int) -> Int {
    NumberTheory.gcd(a, b)
}

/// - Note: Deprecated. Use ``NumberTheory/lcm(_:_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.lcm(_:_:) instead")
public func lcm(_ a: Int, _ b: Int) -> Int {
    NumberTheory.lcm(a, b)
}

/// - Note: Deprecated. Use ``NumberTheory/eulerPhi(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.eulerPhi(_:) instead")
public func eulerPhi(_ n: Int) -> Int? {
    NumberTheory.eulerPhi(n)
}

/// - Note: Deprecated. Use ``NumberTheory/divisorSigma(_:k:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.divisorSigma(_:k:) instead")
public func divisorSigma(_ n: Int, k: Int = 1) -> Double? {
    NumberTheory.divisorSigma(n, k: k)
}

/// - Note: Deprecated. Use ``NumberTheory/mobius(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.mobius(_:) instead")
public func mobius(_ n: Int) -> Int? {
    NumberTheory.mobius(n)
}

/// - Note: Deprecated. Use ``NumberTheory/liouville(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.liouville(_:) instead")
public func liouville(_ n: Int) -> Int? {
    NumberTheory.liouville(n)
}

/// - Note: Deprecated. Use ``NumberTheory/carmichael(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.carmichael(_:) instead")
public func carmichael(_ n: Int) -> Int? {
    NumberTheory.carmichael(n)
}

/// - Note: Deprecated. Use ``NumberTheory/vonMangoldt(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.vonMangoldt(_:) instead")
public func vonMangoldt(_ n: Int) -> Double {
    NumberTheory.vonMangoldt(n)
}

/// - Note: Deprecated. Use ``NumberTheory/primePi(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.primePi(_:) instead")
public func primePi(_ x: Double) -> Int {
    NumberTheory.primePi(x)
}

/// - Note: Deprecated. Use ``NumberTheory/chebyshevTheta(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.chebyshevTheta(_:) instead")
public func chebyshevTheta(_ x: Double) -> Double {
    NumberTheory.chebyshevTheta(x)
}

/// - Note: Deprecated. Use ``NumberTheory/chebyshevPsi(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.chebyshevPsi(_:) instead")
public func chebyshevPsi(_ x: Double) -> Double {
    NumberTheory.chebyshevPsi(x)
}

/// - Note: Deprecated. Use ``NumberTheory/modPow(_:_:_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.modPow(_:_:_:) instead")
public func modPow(_ base: Int, _ exp: Int, _ m: Int) -> Int {
    NumberTheory.modPow(base, exp, m)
}

/// - Note: Deprecated. Use ``NumberTheory/extendedGcd(_:_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.extendedGcd(_:_:) instead")
public func extendedGcd(_ a: Int, _ b: Int) -> (gcd: Int, x: Int, y: Int) {
    NumberTheory.extendedGcd(a, b)
}

/// - Note: Deprecated. Use ``NumberTheory/modInverse(_:_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.modInverse(_:_:) instead")
public func modInverse(_ a: Int, _ m: Int) -> Int? {
    NumberTheory.modInverse(a, m)
}

/// - Note: Deprecated. Use ``NumberTheory/factorial(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.factorial(_:) instead")
public func factorial(_ n: Int) -> Double {
    NumberTheory.factorial(n)
}

/// - Note: Deprecated. Use ``NumberTheory/perm(_:_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.perm(_:_:) instead")
public func perm(_ n: Int, _ k: Int) -> Double {
    NumberTheory.perm(n, k)
}

/// - Note: Deprecated. Use ``NumberTheory/comb(_:_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.comb(_:_:) instead")
public func comb(_ n: Int, _ k: Int) -> Double {
    NumberTheory.comb(n, k)
}

/// - Note: Deprecated. Use ``NumberTheory/binomial(_:_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.binomial(_:_:) instead")
public func binomial(_ n: Int, _ k: Int) -> Double {
    NumberTheory.binomial(n, k)
}

/// - Note: Deprecated. Use ``NumberTheory/digitSum(_:base:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.digitSum(_:base:) instead")
public func digitSum(_ n: Int, base: Int = 10) -> Int {
    NumberTheory.digitSum(n, base: base)
}

/// - Note: Deprecated. Use ``NumberTheory/digitalRoot(_:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.digitalRoot(_:) instead")
public func digitalRoot(_ n: Int) -> Int {
    NumberTheory.digitalRoot(n)
}

/// - Note: Deprecated. Use ``NumberTheory/digitCount(_:base:)`` instead.
@available(*, deprecated, message: "Use NumberTheory.digitCount(_:base:) instead")
public func digitCount(_ n: Int, base: Int = 10) -> Int {
    NumberTheory.digitCount(n, base: base)
}
