// Gate1_UnifiedVsLegacy.swift
// Sources/NumericSwiftBench/Gates/
//
// Gate 1: unified evaluator vs legacy evaluator — ratio ≤ 1.15.
//
// Denominator leg: evaluate the full scalar segment of the snapshot via
//   MathExpr.evaluate (the frozen legacy path from Task #2).
//
// Numerator leg: PLACEHOLDER — the unified evaluator (NumericValue pipeline)
//   does not exist yet in Phase 0. The gate is wired here so its threshold,
//   denominator, and description are committed. When Phase 2/3 ships the
//   unified evaluator, replace the placeholder numerator with a real call.
//
// Gate state during Phase 0: PENDING (not PASS, not FAIL).

import Foundation
import NumericSwift

// MARK: - Gate 1

/// Builds Gate 1: unified-evaluator vs legacy-evaluator ratio (≤ 1.15).
///
/// - Parameter snapshot: The loaded legacy snapshot. The denominator leg
///   evaluates only the scalar segment, matching the expected unified-path
///   entry point (scalar evaluation is the primary path for Gate 1).
/// - Returns: A placeholder ``GateDefinition`` marked PENDING until Phase 2.
func makeGate1(snapshot: BenchSnapshot) -> GateDefinition {
  // Denominator: legacy MathExpr.evaluate over the scalar segment.
  let scalarEntries = snapshot.entries.filter { $0.evaluator == .scalar }

  // Pre-parse expressions from the snapshot description field.
  // Entries whose descriptions contain variables are skipped because the
  // snapshot description does not encode the variable bindings.
  let parsedLegacy: [(MathLexExpression, [String: Double])]
  do {
    parsedLegacy = try scalarEntries.compactMap { entry -> (MathLexExpression, [String: Double])? in
      let desc = entry.description
      guard desc.hasPrefix("scalar: ") else { return nil }
      let exprStr = String(desc.dropFirst("scalar: ".count))
      if exprStr.contains("x") || exprStr.contains(" a ") || exprStr.contains(" b") { return nil }
      let ast = try MathExpr.parse(exprStr)
      return (ast, [:])
    }
  } catch {
    fputs("FATAL: Gate 1 denominator pre-parse failed: \(error)\n", stderr)
    exit(2)
  }

  guard !parsedLegacy.isEmpty else {
    fputs("FATAL: Gate 1 has no parseable scalar expressions in snapshot.\n", stderr)
    exit(2)
  }

  let denominator: () -> Void = {
    for (ast, vars) in parsedLegacy {
      blackhole(try? MathExpr.evaluate(ast, variables: vars))
    }
  }

  // Numerator: PLACEHOLDER until the unified evaluator (Phase 2/3).
  //
  // When the unified NumericValue pipeline is implemented, replace this
  // placeholder by calling the unified evaluate API here. The gate
  // threshold (≤ 1.15) is intentionally generous to allow the unified
  // path to carry its protocol-dispatch overhead while still being
  // within acceptable performance bounds.
  return .placeholder(
    name: "gate1: unified-vs-legacy (≤1.15)",
    threshold: 1.15,
    denominatorDescription: "MathExpr.evaluate (legacy scalar path, \(parsedLegacy.count) exprs)",
    numeratorDescription: "NumericValue unified evaluator (scalar path)",
    denominator: denominator,
    reason: "unified NumericValue evaluator not yet implemented (Phase 2/3)"
  )
}
