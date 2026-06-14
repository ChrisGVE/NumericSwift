// Gate4_ExpmViaExpr.swift
// Sources/NumericSwiftBench/Gates/
//
// Gate 4: expm via expression (unified evaluator path) vs expm direct —
//         ratio ≤ 1.10.
//
// Denominator leg: LinAlg.expm(M) — the direct Padé-approximation matrix
//   exponential. This is the ground-truth reference cost.
//
// Numerator leg: MathExpr.evaluateUnified evaluating "exp(M)" with a matrix
//   variable. The function registry maps exp(matrix) → Group-B expm, calling
//   LinAlg.expm internally. The overhead is the evaluator dispatch layer
//   (variable lookup, function dispatch) on top of the same Padé call.
//
// Gate state: ACTIVE (Phase 3 unified evaluator shipped).
//
// Matrix choice:
//   A 4×4 matrix is used (BenchFixtures.expmMatrix(n:4)) with values in
//   [-0.5, 0.5] to keep the scaling-and-squaring iteration count low and
//   produce stable, representative timing.
//
// Expression note:
//   The parser-available function name is "exp", not "expm". The function
//   registry in NumericDispatch+FunctionRegistry routes exp(matrix) → LinAlg.expm
//   via the Group-B handler in NumericDispatch+FunctionDispatchers.

import Foundation
import NumericSwift

// MARK: - Gate 4

/// Builds Gate 4: expm-via-expression vs expm-direct ratio (≤ 1.10).
///
/// The denominator calls `LinAlg.expm` directly. The numerator evaluates
/// `exp(M)` through `MathExpr.evaluateUnified` which dispatches to the same
/// `LinAlg.expm` internally via the function registry's Group-B handler.
/// The ratio measures only evaluator-dispatch overhead.
///
/// - Parameter config: Bench configuration (matrixSize ignored; 4×4 is fixed).
/// - Returns: A ``GateDefinition`` with both legs wired to real implementations.
func makeGate4(config: BenchConfig) -> GateDefinition {
  // Small matrix: expm is O(n³) in the squaring phase; a 4×4 keeps it fast.
  let M = BenchFixtures.expmMatrix(n: 4)

  // Pre-parse "exp(M)" once; parse cost excluded from numerator timing.
  let ast: MathLexExpression
  do {
    ast = try MathExpr.parse("exp(M)")
  } catch {
    fputs("FATAL: Gate 4 expression parse failed: \(error)\n", stderr)
    exit(2)
  }

  // Pre-wrap matrix in NumericValue; wrapping cost excluded from loop.
  let values: [String: NumericValue] = ["M": .matrix(M)]

  // Denominator: direct LinAlg.expm call (Padé approximation, 4×4).
  let denominator: () -> Void = {
    blackhole(try? LinAlg.expm(M))
  }

  // Numerator: unified evaluator — "exp(M)" dispatches through
  // NumericDispatch.applyExpLogSqrt → LinAlg.expm (Group-B handler).
  // The underlying Padé computation is identical; only the dispatch path differs.
  let numerator: () -> Void = {
    blackhole(try? MathExpr.evaluateUnified(ast, values: values))
  }

  return GateDefinition(
    name: "gate4: expm-via-expr vs expm-direct (≤1.10)",
    threshold: 1.10,
    denominatorDescription: "LinAlg.expm (Padé 4×4 direct call)",
    numeratorDescription: "MathExpr.evaluateUnified(\"exp(M)\", matrix value, 4×4)",
    denominator: denominator,
    numerator: numerator
  )
}
