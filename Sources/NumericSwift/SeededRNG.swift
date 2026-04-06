//
//  SeededRNG.swift
//  NumericSwift
//
//  Deterministic random-number generator with a fixed seed, implementing
//  Swift's RandomNumberGenerator protocol using a 64-bit xorshift algorithm.
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - SeededRandomNumberGenerator

/// A deterministic pseudo-random number generator seeded with a fixed value.
///
/// Uses a 64-bit xorshift algorithm (Marsaglia 2003).  Conforms to
/// `RandomNumberGenerator` so it can be passed directly to Swift's
/// `random(in:using:)` family of functions.
public struct SeededRandomNumberGenerator: RandomNumberGenerator {
  private var state: UInt64

  /// Create a generator with the given seed.
  public init(seed: UInt64) {
    // Avoid the zero state, which xorshift cannot escape.
    state = seed == 0 ? 1 : seed
  }

  /// Produce the next 64-bit pseudo-random value.
  public mutating func next() -> UInt64 {
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return state
  }
}

// MARK: - Seeded randomNormal overloads

/// Box-Muller standard normal variate drawn from a seeded generator.
///
/// - Parameter rng: The random-number generator to use.
/// - Returns: A standard normal (μ = 0, σ = 1) sample.
public func randomNormal(using rng: inout some RandomNumberGenerator) -> Double {
  let u1 = Double.random(in: Double.ulpOfOne..<1.0, using: &rng)
  let u2 = Double.random(in: 0..<1.0, using: &rng)
  return Darwin.sqrt(-2.0 * Darwin.log(u1)) * Darwin.cos(2.0 * .pi * u2)
}

/// Generate `n` standard normal variates drawn from a seeded generator.
///
/// - Parameters:
///   - n: Number of samples.
///   - rng: The random-number generator to use.
/// - Returns: Array of `n` standard normal samples.
public func randomNormal(
  _ n: Int, using rng: inout some RandomNumberGenerator
) -> [Double] {
  (0..<n).map { _ in randomNormal(using: &rng) }
}
