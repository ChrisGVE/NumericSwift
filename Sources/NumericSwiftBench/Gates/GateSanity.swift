// GateSanity.swift
// Sources/NumericSwiftBench/Gates/
//
// Sanity gate: legacy evaluator on BOTH legs, yielding ratio ≈ 1.0.
//
// Purpose: Validates that the timing infrastructure itself works correctly.
// When the same workload is timed on both the numerator and denominator legs,
// the ratio must be close to 1.0 (within statistical noise). A significantly
// higher ratio would indicate a measurement bug rather than a real regression.
//
// Gate spec (from task #3):
//   ratio = t_leg_A / t_leg_B ≤ 1.15 (15% statistical tolerance)
// Both legs execute identical scalar expression evaluation so the expected
// ratio is indistinguishable from 1.0 up to scheduler jitter.

import Foundation
import NumericSwift

// MARK: - Gate: sanity-legacy-vs-legacy

/// Builds the sanity gate that runs legacy MathExpr.evaluate on both legs.
///
/// Both legs evaluate the same corpus of scalar expressions via MathExpr.evaluate.
/// The expected ratio is ≈ 1.0; the gate threshold is ≤ 1.15 to accommodate
/// scheduler noise. A failure here signals a measurement infrastructure bug.
///
/// - Parameter snapshot: The loaded legacy snapshot (entries validated at
///   startup; this gate uses only the scalar segment).
/// - Returns: A fully-implemented ``GateDefinition`` (not a placeholder).
func makeSanityGate(snapshot: BenchSnapshot) -> GateDefinition {
  // Pull scalar entries from the snapshot to use as workload.
  let scalarEntries = snapshot.entries.filter { $0.evaluator == .scalar }

  // Pre-parse all expressions so parse cost is not included in timed loop.
  // If parsing fails, the bench aborts with a clear error — a corrupt snapshot
  // or incompatible MathExpr API is a configuration error, not a bench result.
  let parsed: [(MathLexExpression, [String: Double])]
  do {
    parsed = try scalarEntries.compactMap { entry -> (MathLexExpression, [String: Double])? in
      // Reconstruct expression string from description field.
      // Format: "scalar: <expr>" — strip the prefix.
      let desc = entry.description
      guard desc.hasPrefix("scalar: ") else { return nil }
      let exprStr = String(desc.dropFirst("scalar: ".count))
      // Skip entries that use variables (description doesn't encode them).
      if exprStr.contains("x") || exprStr.contains("a ") || exprStr.contains("b") { return nil }
      let ast = try MathExpr.parse(exprStr)
      return (ast, [:])
    }
  } catch {
    fputs("FATAL: Sanity gate pre-parse failed: \(error)\n", stderr)
    exit(2)
  }

  guard !parsed.isEmpty else {
    fputs("FATAL: Sanity gate has no parseable scalar expressions in snapshot.\n", stderr)
    exit(2)
  }

  // Both legs perform identical work: evaluate all parsed expressions.
  let workload: () -> Void = {
    for (ast, vars) in parsed {
      blackhole(try? MathExpr.evaluate(ast, variables: vars))
    }
  }

  return GateDefinition(
    name: "sanity: legacy-on-both-legs",
    threshold: 1.15,
    denominatorDescription: "MathExpr.evaluate (scalar corpus, leg A)",
    numeratorDescription: "MathExpr.evaluate (scalar corpus, leg B) — identical workload",
    denominator: workload,
    numerator: workload
  )
}
