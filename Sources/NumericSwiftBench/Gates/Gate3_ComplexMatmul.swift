// Gate3_ComplexMatmul.swift
// Sources/NumericSwiftBench/Gates/
//
// Gate 3: complex matmul (unified evaluator path) vs real-block-decomposition
//         direct — ratio ≤ 1.10.
//
// Background:
//   Complex matrix multiplication C = A B can be computed in two ways:
//
//   (a) Real-block decomposition (4 real matmuls) — DENOMINATOR:
//       C_r = A_r B_r − A_i B_i
//       C_i = A_r B_i + A_i B_r
//   This is the reference leg: 4 direct LinAlg.dot calls, no evaluator overhead.
//
//   (b) Unified evaluator path — NUMERATOR:
//   MathExpr.evaluateUnified("A * B", values: ["A": .complexMatrix, "B": .complexMatrix])
//   dispatches through NumericDispatch.applyMul → evalComplexMatrixMulComplexMatrix
//   → complexMatmul, which also performs the same real-block decomposition
//   internally via 4 LinAlg.dot calls. The measured overhead is the evaluator
//   dispatch layer on top of the same underlying computation.
//
// Both legs perform mathematically equivalent work; the ratio captures only
// the dispatch cost of routing through the NumericValue pipeline.
//
// Gate state: ACTIVE (Phase 3 unified evaluator shipped).

import Foundation
import NumericSwift

// MARK: - Gate 3

/// Builds Gate 3: complex matmul vs real-block decomposition ratio (≤ 1.10).
///
/// The denominator leg computes the 4-LinAlg.dot real-block decomposition
/// directly. The numerator leg routes the same computation through
/// `MathExpr.evaluateUnified("A * B")` with complex-matrix variables.
/// Both paths ultimately call 4 LinAlg.dot internally; the ratio measures
/// only evaluator-dispatch overhead.
///
/// - Parameter config: Bench configuration (provides matrixSize).
/// - Returns: A ``GateDefinition`` with both legs wired to real implementations.
func makeGate3(config: BenchConfig) -> GateDefinition {
  let n = config.matrixSize

  // Pre-build complex matrices once; construction excluded from timed loop.
  let A = BenchFixtures.complexSquareMatrix(n: n)
  let B = BenchFixtures.complexSquareMatrix(n: n)

  // Extract real and imaginary parts as LinAlg.Matrix for the direct-path leg.
  let Ar = LinAlg.Matrix(rows: n, cols: n, data: A.real)
  let Ai = LinAlg.Matrix(rows: n, cols: n, data: A.imag)
  let Br = LinAlg.Matrix(rows: n, cols: n, data: B.real)
  let Bi = LinAlg.Matrix(rows: n, cols: n, data: B.imag)

  // Pre-parse "A * B" once; parse cost excluded from numerator timing.
  let ast: MathLexExpression
  do {
    ast = try MathExpr.parse("A * B")
  } catch {
    fputs("FATAL: Gate 3 expression parse failed: \(error)\n", stderr)
    exit(2)
  }

  // Pre-wrap complex matrices in NumericValue; wrapping cost excluded from loop.
  let values: [String: NumericValue] = ["A": .complexMatrix(A), "B": .complexMatrix(B)]

  // Denominator: real-block decomposition — 4 direct BLAS dgemm calls.
  // C_r = A_r B_r − A_i B_i
  // C_i = A_r B_i + A_i B_r
  let denominator: () -> Void = {
    let Cr = LinAlg.dot(Ar, Br) - LinAlg.dot(Ai, Bi)
    let Ci = LinAlg.dot(Ar, Bi) + LinAlg.dot(Ai, Br)
    // Reconstruct the ComplexMatrix so the result is fully materialised.
    blackhole(LinAlg.ComplexMatrix(rows: n, cols: n, real: Cr.data, imag: Ci.data))
  }

  // Numerator: unified evaluator — dispatches through NumericDispatch.applyMul
  // → evalComplexMatrixMulComplexMatrix → complexMatmul, which internally uses
  // the same real-block decomposition via LinAlg.dot. Only the dispatch overhead
  // is extra vs the denominator.
  let numerator: () -> Void = {
    blackhole(try? MathExpr.evaluateUnified(ast, values: values))
  }

  return GateDefinition(
    name: "gate3: complex-matmul vs real-block (≤1.10)",
    threshold: 1.10,
    denominatorDescription: "4× LinAlg.dot real-block decomposition (\(n)×\(n) complex)",
    numeratorDescription: "MathExpr.evaluateUnified(\"A * B\", complexMatrix values, \(n)×\(n))",
    denominator: denominator,
    numerator: numerator
  )
}
