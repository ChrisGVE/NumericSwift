// Gate3_ComplexMatmul.swift
// Sources/NumericSwiftBench/Gates/
//
// Gate 3: complex matmul (unified path) vs real-block-decomposition direct —
//         ratio ≤ 1.10.
//
// Background:
//   Complex matrix multiplication C = A B can be computed in two ways:
//
//   (a) Real-block decomposition (4 real matmuls):
//       C_r = A_r B_r − A_i B_i
//       C_i = A_r B_i + A_i B_r
//   This is the REFERENCE (denominator) leg — it uses LinAlg.dot directly
//   and is the minimal-overhead baseline for complex matmul.
//
//   (b) Future: unified NumericValue pipeline evaluating a complex matmul
//   expression. That path does not exist in Phase 0.
//
// Denominator leg: 4× LinAlg.dot calls on the real and imaginary parts.
// Numerator leg: PLACEHOLDER until Phase 2/3.
//
// Gate state during Phase 0: PENDING.

import Foundation
import NumericSwift

// MARK: - Gate 3

/// Builds Gate 3: complex matmul vs real-block decomposition ratio (≤ 1.10).
///
/// - Parameter config: Bench configuration (provides matrixSize).
/// - Returns: A placeholder ``GateDefinition`` marked PENDING until Phase 2.
func makeGate3(config: BenchConfig) -> GateDefinition {
  let n = config.matrixSize

  // Pre-build complex matrices once; construction excluded from timed loop.
  let A = BenchFixtures.complexSquareMatrix(n: n)
  let B = BenchFixtures.complexSquareMatrix(n: n)

  // Extract real and imaginary parts as LinAlg.Matrix for direct BLAS calls.
  let Ar = LinAlg.Matrix(rows: n, cols: n, data: A.real)
  let Ai = LinAlg.Matrix(rows: n, cols: n, data: A.imag)
  let Br = LinAlg.Matrix(rows: n, cols: n, data: B.real)
  let Bi = LinAlg.Matrix(rows: n, cols: n, data: B.imag)

  // Denominator: real-block decomposition — 4 BLAS dgemm calls.
  // C_r = A_r B_r − A_i B_i
  // C_i = A_r B_i + A_i B_r
  let denominator: () -> Void = {
    let Cr = LinAlg.dot(Ar, Br) - LinAlg.dot(Ai, Bi)
    let Ci = LinAlg.dot(Ar, Bi) + LinAlg.dot(Ai, Br)
    // Reconstruct the ComplexMatrix so the result is fully materialised.
    blackhole(LinAlg.ComplexMatrix(rows: n, cols: n, real: Cr.data, imag: Ci.data))
  }

  // Numerator: PLACEHOLDER.
  //
  // Replace with the unified-pipeline complex matmul when it is available,
  // for example:
  //   let result = try NumericValue.evaluate(
  //     "A @ B", variables: ["A": .complexMatrix(A), "B": .complexMatrix(B)])
  //   blackhole(result)
  return .placeholder(
    name: "gate3: complex-matmul vs real-block (≤1.10)",
    threshold: 1.10,
    denominatorDescription: "4× LinAlg.dot real-block decomposition (\(n)×\(n) complex)",
    numeratorDescription: "unified-pipeline complex matmul (\(n)×\(n))",
    denominator: denominator,
    reason: "unified NumericValue pipeline complex matmul not yet implemented (Phase 2/3)"
  )
}
