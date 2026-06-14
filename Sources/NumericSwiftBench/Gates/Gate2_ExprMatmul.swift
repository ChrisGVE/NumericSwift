// Gate2_ExprMatmul.swift
// Sources/NumericSwiftBench/Gates/
//
// Gate 2: expression-driven matmul (future: NumericValue pipeline) vs direct
//         LinAlg.dot call — ratio ≤ 1.10.
//
// Denominator leg: LinAlg.dot(A, B) — the direct, BLAS-accelerated matrix
//   multiply. This is the ground-truth reference cost for matmul.
//
// Numerator leg: PLACEHOLDER — the expression-driven matmul (i.e. evaluating
//   "A @ B" through the unified NumericValue pipeline) does not exist yet.
//   When Phase 2/3 ships the pipeline, replace the placeholder with a real call.
//
// Gate state during Phase 0: PENDING.
//
// Implementation note:
//   The matrix size is configurable via BenchConfig.matrixSize (default 64).
//   A 64×64 double matrix fits in L2 cache on most hardware, giving stable
//   timing that is representative of the workload mix expected in practice.

import Foundation
import NumericSwift

// MARK: - Gate 2

/// Builds Gate 2: expression matmul vs direct-dot ratio (≤ 1.10).
///
/// - Parameter config: Bench configuration (provides matrixSize).
/// - Returns: A placeholder ``GateDefinition`` marked PENDING until Phase 2.
func makeGate2(config: BenchConfig) -> GateDefinition {
  let n = config.matrixSize
  // Pre-build matrices once so construction cost is excluded from the timed loop.
  let A = BenchFixtures.realSquareMatrix(n: n)
  let B = BenchFixtures.realSquareMatrixB(n: n)

  // Denominator: direct LinAlg.dot — BLAS-accelerated real matrix multiply.
  let denominator: () -> Void = {
    blackhole(LinAlg.dot(A, B))
  }

  // Numerator: PLACEHOLDER.
  //
  // Replace with the unified-pipeline expression matmul when it is available,
  // for example:
  //   let result = try NumericValue.evaluate("A @ B", variables: ["A": .matrix(A), "B": .matrix(B)])
  //   blackhole(result)
  return .placeholder(
    name: "gate2: expr-matmul vs direct-dot (≤1.10)",
    threshold: 1.10,
    denominatorDescription: "LinAlg.dot (BLAS \(n)×\(n) real matmul, direct call)",
    numeratorDescription: "unified-pipeline matmul expression (\(n)×\(n))",
    denominator: denominator,
    reason: "unified NumericValue pipeline matmul not yet implemented (Phase 2/3)"
  )
}
