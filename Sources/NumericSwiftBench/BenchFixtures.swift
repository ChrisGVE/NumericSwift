// BenchFixtures.swift
// Sources/NumericSwiftBench/
//
// Deterministic, pre-computed input matrices and expressions used by the
// bench legs.
//
// All inputs are seeded from a fixed constant so benchmark runs are
// reproducible across machines and compiler versions. No randomness is
// introduced at bench time.

import Foundation
import NumericSwift

// MARK: - Deterministic pseudo-random generator

/// Xorshift64 — a simple, fast, deterministic pseudo-random number generator.
///
/// Used only to generate reproducible benchmark matrices. Not cryptographically
/// secure; that is not a requirement here.
struct Xorshift64 {
  var state: UInt64

  init(seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) {
    state = seed
  }

  /// Returns the next pseudo-random UInt64.
  mutating func next() -> UInt64 {
    var x = state
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    state = x
    return x
  }

  /// Returns the next pseudo-random Double in (0, 1).
  mutating func nextDouble() -> Double {
    // Take the top 53 bits to fill the mantissa.
    let bits = next() >> 11
    return Double(bits) / Double(1 << 53)
  }

  /// Returns an array of `count` pseudo-random Doubles in (0, 1).
  mutating func nextDoubles(count: Int) -> [Double] {
    (0..<count).map { _ in nextDouble() }
  }
}

// MARK: - BenchFixtures

/// Pre-built deterministic inputs shared across bench gates.
struct BenchFixtures {

  // MARK: - Matrix fixtures

  /// A square real matrix of side `n` filled with deterministic values.
  ///
  /// Values are in (0, 1) to keep condition numbers reasonable for the
  /// matrix-function gates (expm, etc.).
  static func realSquareMatrix(n: Int) -> LinAlg.Matrix {
    var rng = Xorshift64()
    let data = rng.nextDoubles(count: n * n)
    return LinAlg.Matrix(rows: n, cols: n, data: data)
  }

  /// A second square real matrix independent of the first (different RNG
  /// seed offset) so matmul A*B involves genuinely different data.
  static func realSquareMatrixB(n: Int) -> LinAlg.Matrix {
    var rng = Xorshift64(seed: 0x1234_ABCD_5678_EF01)
    let data = rng.nextDoubles(count: n * n)
    return LinAlg.Matrix(rows: n, cols: n, data: data)
  }

  /// A square complex matrix of side `n` with deterministic real and
  /// imaginary parts, both in (0, 1).
  static func complexSquareMatrix(n: Int) -> LinAlg.ComplexMatrix {
    var rng1 = Xorshift64(seed: 0xA1B2_C3D4_E5F6_0011)
    var rng2 = Xorshift64(seed: 0xB2C3_D4E5_F6A1_1100)
    let real = rng1.nextDoubles(count: n * n)
    let imag = rng2.nextDoubles(count: n * n)
    return LinAlg.ComplexMatrix(rows: n, cols: n, real: real, imag: imag)
  }

  /// A small square matrix suitable for expm (scaling is bounded for n≤4).
  ///
  /// Values are scaled to [-0.5, 0.5] to avoid excessive scaling-and-squaring
  /// iterations in the Padé approximation, keeping the bench timing realistic
  /// but not pathologically slow.
  static func expmMatrix(n: Int = 4) -> LinAlg.Matrix {
    var rng = Xorshift64(seed: 0xFEED_FACE_DEAD_BEEF)
    let data = rng.nextDoubles(count: n * n).map { $0 - 0.5 }
    return LinAlg.Matrix(rows: n, cols: n, data: data)
  }

  // MARK: - Expression fixtures

  /// A corpus of simple scalar expressions that exercise MathExpr.evaluate.
  ///
  /// Kept lightweight so the legacy-on-both-legs smoke gate completes fast.
  static let scalarExpressions: [(String, [String: Double])] = [
    ("sin(pi / 2)", [:]),
    ("cos(0)", [:]),
    ("exp(1)", [:]),
    ("log(10)", [:]),
    ("sqrt(2) * sqrt(2)", [:]),
    ("2 + 3 * 4", [:]),
    ("atan2(1, 1)", [:]),
    ("hypot(3, 4)", [:]),
    ("x * x + 1", ["x": 3.0]),
    ("a + b", ["a": 1.5, "b": 2.5]),
  ]

  /// Parses each expression in ``scalarExpressions`` once and caches the ASTs
  /// so timed bench loops do not include parse time.
  ///
  /// Returns (ast, variables) pairs ready for MathExpr.evaluate(_:variables:).
  static func parsedScalarExpressions() throws -> [(MathLexExpression, [String: Double])] {
    try scalarExpressions.map { (src, vars) in
      let ast = try MathExpr.parse(src)
      return (ast, vars)
    }
  }
}
