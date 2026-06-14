// Gate4_ExpmViaExpr.swift
// Sources/NumericSwiftBench/Gates/
//
// Gate 4: expm via expression (unified path) vs expm direct — ratio ≤ 1.10.
//
// Denominator leg: LinAlg.expm(M) — the direct Padé-approximation matrix
//   exponential. This is the ground-truth reference cost.
//
// Numerator leg: PLACEHOLDER — evaluating "expm(M)" through the unified
//   NumericValue pipeline does not exist yet in Phase 0.
//
// Gate state during Phase 0: PENDING.
//
// Matrix choice:
//   A 4×4 matrix is used (BenchFixtures.expmMatrix(n:4)) with values in
//   [-0.5, 0.5] to keep the scaling-and-squaring iteration count low and
//   produce stable, representative timing.

import Foundation
import NumericSwift

// MARK: - Gate 4

/// Builds Gate 4: expm-via-expression vs expm-direct ratio (≤ 1.10).
///
/// - Parameter config: Bench configuration.
/// - Returns: A placeholder ``GateDefinition`` marked PENDING until Phase 2.
func makeGate4(config: BenchConfig) -> GateDefinition {
  // Small matrix: expm is O(n³) in the squaring phase; a 4×4 keeps it fast.
  let M = BenchFixtures.expmMatrix(n: 4)

  // Denominator: direct LinAlg.expm call (Padé approximation).
  let denominator: () -> Void = {
    blackhole(try? LinAlg.expm(M))
  }

  // Numerator: PLACEHOLDER.
  //
  // Replace with the unified-pipeline expm expression when it is available,
  // for example:
  //   let result = try NumericValue.evaluate(
  //     "expm(M)", variables: ["M": .matrix(M)])
  //   blackhole(result)
  return .placeholder(
    name: "gate4: expm-via-expr vs expm-direct (≤1.10)",
    threshold: 1.10,
    denominatorDescription: "LinAlg.expm (Padé 4×4 direct call)",
    numeratorDescription: "unified-pipeline expm expression (4×4)",
    denominator: denominator,
    reason: "unified NumericValue pipeline expm not yet implemented (Phase 2/3)"
  )
}
