// Gate1_UnifiedVsLegacy.swift
// Sources/NumericSwiftBench/Gates/
//
// Gate 1: unified evaluator vs legacy evaluator — ratio ≤ 1.15.
//
// Denominator leg: evaluate the full scalar segment of the snapshot via
//   MathExpr.evaluate (the frozen legacy path from Task #2).
//
// Numerator leg: MathExpr.evaluateUnified (Phase 3 unified front door).
//   The unified path carries some extra dispatch overhead (NumericDispatch
//   routing through the NumericValue tower instead of direct Double arithmetic),
//   so the threshold is 1.15 — 15% overhead headroom vs the legacy scalar path.
//
// Gate state: ACTIVE (Phase 3 unified evaluator shipped).

import Foundation
import NumericSwift

// MARK: - Gate 1

/// Builds Gate 1: unified-evaluator vs legacy-evaluator ratio (≤ 1.15).
///
/// - Parameter snapshot: The loaded legacy snapshot. Both legs evaluate the
///   scalar segment of the snapshot so the benchmark covers the same corpus.
/// - Returns: A ``GateDefinition`` with both legs wired to real evaluators.
func makeGate1(snapshot: BenchSnapshot) -> GateDefinition {
  // Common: parse all variable-free scalar expressions from the snapshot.
  let scalarEntries = snapshot.entries.filter { $0.evaluator == .scalar }

  let parsed: [(MathLexExpression, [String: Double])]
  do {
    parsed = try scalarEntries.compactMap { entry -> (MathLexExpression, [String: Double])? in
      let desc = entry.description
      guard desc.hasPrefix("scalar: ") else { return nil }
      let exprStr = String(desc.dropFirst("scalar: ".count))
      // Skip entries whose variable bindings are not encoded in the description.
      if exprStr.contains("x") || exprStr.contains(" a ") || exprStr.contains(" b") { return nil }
      let ast = try MathExpr.parse(exprStr)
      return (ast, [:])
    }
  } catch {
    fputs("FATAL: Gate 1 pre-parse failed: \(error)\n", stderr)
    exit(2)
  }

  guard !parsed.isEmpty else {
    fputs("FATAL: Gate 1 has no parseable scalar expressions in snapshot.\n", stderr)
    exit(2)
  }

  // Denominator: legacy MathExpr.evaluate → Double.
  let denominator: () -> Void = {
    for (ast, vars) in parsed {
      blackhole(try? MathExpr.evaluate(ast, variables: vars))
    }
  }

  // Numerator: unified MathExpr.evaluateUnified → NumericValue.
  //
  // The values dict is empty; all variables in these expressions have been
  // filtered out above, so only constant-folding expressions reach here.
  // The result is discarded via blackhole to prevent dead-code elimination.
  let numerator: () -> Void = {
    for (ast, _) in parsed {
      blackhole(try? MathExpr.evaluateUnified(ast, values: [:]))
    }
  }

  return GateDefinition(
    name: "gate1: unified-vs-legacy (≤1.15)",
    threshold: 1.15,
    denominatorDescription: "MathExpr.evaluate (legacy scalar path, \(parsed.count) exprs)",
    numeratorDescription: "MathExpr.evaluateUnified (unified NumericValue path, \(parsed.count) exprs)",
    denominator: denominator,
    numerator: numerator
  )
}
