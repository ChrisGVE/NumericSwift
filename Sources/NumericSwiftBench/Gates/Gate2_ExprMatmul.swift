// Gate2_ExprMatmul.swift
// Sources/NumericSwiftBench/Gates/
//
// Gate 2: expression-driven matmul vs direct LinAlg.dot call — ratio ≤ 1.10.
//
// Denominator leg: LinAlg.dot(A, B) — the direct, BLAS-accelerated matrix
//   multiply. This is the ground-truth reference cost for matmul.
//
// Numerator leg: MathExpr.evaluateUnified parsing "A * B" with matrix
//   variables, routing through NumericDispatch.applyMul → LinAlg.dot.
//   The overhead is the evaluator dispatch layer (variable lookup, AST
//   traversal, NumericValue wrapping) on top of the same BLAS call.
//
// Gate state: ACTIVE (Phase 3 unified evaluator shipped).
//
// Implementation note:
//   The matrix size is configurable via BenchConfig.matrixSize (default 64).
//   A 64×64 double matrix fits in L2 cache on most hardware, giving stable
//   timing that is representative of the workload mix expected in practice.
//
//   "A * B" is pre-parsed once; parse cost is excluded from the timed loop.
//   Matrix NumericValue wrappers are also pre-built outside the loop so only
//   evaluator dispatch + the BLAS call are timed.

import Foundation
import NumericSwift

// MARK: - Gate 2

/// Builds Gate 2: expression matmul vs direct-dot ratio (≤ 1.10).
///
/// Both legs use the same deterministic matrices so the BLAS work is identical.
/// The ratio captures the evaluator-dispatch overhead introduced by routing
/// through `MathExpr.evaluateUnified`.
///
/// - Parameter config: Bench configuration (provides matrixSize).
/// - Returns: A ``GateDefinition`` with both legs wired to real implementations.
func makeGate2(config: BenchConfig) -> GateDefinition {
  let n = config.matrixSize
  // Pre-build matrices once; construction excluded from timed loop.
  let A = BenchFixtures.realSquareMatrix(n: n)
  let B = BenchFixtures.realSquareMatrixB(n: n)

  // Pre-parse "A * B" once; parse cost must not inflate the numerator timing.
  let ast: MathLexExpression
  do {
    ast = try MathExpr.parse("A * B")
  } catch {
    fputs("FATAL: Gate 2 expression parse failed: \(error)\n", stderr)
    exit(2)
  }

  // Pre-wrap matrices in NumericValue; wrapping cost excluded from loop.
  let values: [String: NumericValue] = ["A": .matrix(A), "B": .matrix(B)]

  // Denominator: direct LinAlg.dot — BLAS-accelerated real matrix multiply.
  // This is the denominator (reference) because it is the minimal-overhead path.
  let denominator: () -> Void = {
    blackhole(LinAlg.dot(A, B))
  }

  // Numerator: unified evaluator — dispatches through NumericDispatch.applyMul
  // → LinAlg.dot. The measured overhead is the evaluator dispatch layer only;
  // the underlying BLAS call is shared.
  let numerator: () -> Void = {
    blackhole(try? MathExpr.evaluateUnified(ast, values: values))
  }

  return GateDefinition(
    name: "gate2: expr-matmul vs direct-dot (≤1.10)",
    threshold: 1.10,
    denominatorDescription: "LinAlg.dot (BLAS \(n)×\(n) real matmul, direct call)",
    numeratorDescription: "MathExpr.evaluateUnified(\"A * B\", matrix values, \(n)×\(n))",
    denominator: denominator,
    numerator: numerator
  )
}
