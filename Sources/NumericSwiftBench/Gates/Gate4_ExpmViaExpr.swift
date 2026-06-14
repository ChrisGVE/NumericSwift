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
// Matrix choice & measurement stability:
//   A 16×16 matrix is used (BenchFixtures.expmMatrix(n:16)) with values in
//   [-0.5, 0.5], and EACH timed sample runs `innerReps` (64) expm calls in a
//   loop, dividing nothing — both legs do the same K calls so the ratio is
//   unchanged but per-call timer/scheduling noise is averaged out within each
//   sample. This is necessary because expm allocates many temporaries, so a
//   single-call sample has high inter-leg variance: a bare 4×4 gate was flaky
//   (0.76–1.11) and even a 64×64 single-call gate spread 0.83–1.16, occasionally
//   breaching 1.10 on OS jitter despite both legs doing identical work. The
//   inner-loop amortization gives a stable ratio ≈1.0–1.03 representative of the
//   evaluator's true (negligible) dispatch overhead — which is what the ≤1.10
//   budget is meant to police. The 1.10 threshold is unchanged (PRD).
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
  // 16×16 matrix, with an inner repeat loop per timed sample to amortize
  // per-call timer/scheduling noise (see header note). Both legs run the same
  // number of inner reps, so the ratio is unaffected — only its stability.
  let M = BenchFixtures.expmMatrix(n: 16)
  let innerReps = 64

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

  // Denominator: direct LinAlg.expm call (Padé approximation, 16×16),
  // repeated `innerReps` times per sample to average out per-call noise.
  let denominator: () -> Void = {
    for _ in 0..<innerReps {
      blackhole(try? LinAlg.expm(M))
    }
  }

  // Numerator: unified evaluator — "exp(M)" dispatches through
  // NumericDispatch.applyExpLogSqrt → LinAlg.expm (Group-B handler).
  // The underlying Padé computation is identical; only the dispatch path differs.
  // Same `innerReps` as the denominator, so the ratio is unaffected.
  let numerator: () -> Void = {
    for _ in 0..<innerReps {
      blackhole(try? MathExpr.evaluateUnified(ast, values: values))
    }
  }

  return GateDefinition(
    name: "gate4: expm-via-expr vs expm-direct (≤1.10)",
    threshold: 1.10,
    denominatorDescription: "LinAlg.expm (Padé 16×16 direct, 64× inner reps)",
    numeratorDescription: "MathExpr.evaluateUnified(\"exp(M)\", 16×16, 64× inner reps)",
    denominator: denominator,
    numerator: numerator
  )
}
