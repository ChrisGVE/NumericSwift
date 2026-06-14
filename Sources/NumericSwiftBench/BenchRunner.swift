// BenchRunner.swift
// Sources/NumericSwiftBench/
//
// Gate registry, orchestration, and pass/fail summary reporting.
//
// Usage pattern:
//   1. Create a BenchRunner.
//   2. Register gates via register(_:).
//   3. Call run() to execute all gates and print the summary.
//   4. Inspect exitCode to propagate failures to the shell.

import Foundation

// MARK: - BenchRunner

/// Orchestrates and reports on a collection of ratio gates.
///
/// Responsibilities:
///   • Holds the ordered gate registry.
///   • Runs gates in registration order.
///   • Prints a structured per-gate summary to stdout.
///   • Computes an exit code: 0 when all gates PASS or PENDING, 1 when any
///     gate FAILs or ERRORs.
///
/// PENDING gates are not failures — they signal that the unified evaluator
/// does not yet exist (Phase 0). The exit code is non-zero only for actual
/// FAIL or ERROR outcomes so CI can gate on real regressions.
public final class BenchRunner {

  // MARK: State

  private var gates: [GateDefinition] = []
  private let config: BenchConfig

  // MARK: Init

  public init(config: BenchConfig = BenchConfig.fromEnvironment()) {
    self.config = config
  }

  // MARK: Registration

  /// Append a gate to the run queue.
  public func register(_ gate: GateDefinition) {
    gates.append(gate)
  }

  // MARK: Execution

  /// Run all registered gates and print the summary table.
  ///
  /// - Returns: Exit code: 0 for all-pass/pending, 1 on any fail or error.
  @discardableResult
  public func run() -> Int32 {
    printBanner()

    var results: [GateResult] = []
    for gate in gates {
      print("  Running \(gate.name)…", terminator: "")
      fflush(stdout)
      let result = runGate(gate, config: config)
      results.append(result)
      print(" \(result.state.label)")
    }

    print("")
    printSummaryTable(results)

    let exitCode = computeExitCode(results)
    printFooter(results: results, exitCode: exitCode)
    return exitCode
  }

  // MARK: - Reporting helpers

  private func printBanner() {
    let line = String(repeating: "─", count: 70)
    print(line)
    print("NumericSwiftBench — ratio gate harness")
    print("  warmup=\(config.warmupIterations)  samples=\(config.timedSamples)  matrixSize=\(config.matrixSize)")
    print(line)
    print("")
  }

  private func printSummaryTable(_ results: [GateResult]) {
    let line = String(repeating: "─", count: 70)
    print(line)
    // Use string interpolation — not %s — for Swift String columns.
    let header = padRight("Gate", 35) + "  " + padLeft("Denom ns", 8)
      + "  " + padLeft("Numer ns", 8)
      + "  " + padLeft("Ratio", 7)
      + "  Threshold  State"
    print(header)
    print(line)

    for r in results {
      let denomStr = formatNs(r.denominatorNs)
      let numerStr = r.numeratorNs.map { formatNs($0) } ?? " PENDING"
      let ratioStr: String
      switch r.state {
      case .pass(let ratio), .fail(let ratio):
        ratioStr = String(format: "%7.4f", ratio)
      default:
        ratioStr = "      —"
      }
      let threshStr = String(format: "≤%5.2f", r.threshold)
      let stateStr = r.state.label

      // Gate name column: 35 chars; pad or truncate.
      let name = padRight(r.name, 35)
      print("\(name)  \(denomStr)  \(numerStr)  \(ratioStr)  \(threshStr)  \(stateStr)")
    }
    print(line)
  }

  private func printFooter(results: [GateResult], exitCode: Int32) {
    let passing  = results.filter { $0.state.isPassing }.count
    let failing  = results.filter { if case .fail = $0.state { return true }; return false }.count
    let pending  = results.filter { $0.state.isPending }.count
    let erroring = results.filter { if case .error = $0.state { return true }; return false }.count

    print("")
    print("Summary: \(passing) PASS · \(failing) FAIL · \(pending) PENDING · \(erroring) ERROR")
    print("Exit code: \(exitCode)")
    if exitCode != 0 {
      print("NOTE: Non-zero exit — one or more gates FAILED or ERRORed.")
    }
    if pending > 0 {
      print("NOTE: \(pending) gate(s) PENDING — unified evaluator not yet implemented (Phase 0).")
    }
    print(String(repeating: "─", count: 70))
  }

  // MARK: - Exit code

  /// Compute the process exit code from gate results.
  ///
  /// 0  — all gates passed or are pending
  /// 1  — at least one gate failed or errored
  private func computeExitCode(_ results: [GateResult]) -> Int32 {
    for r in results {
      switch r.state {
      case .fail, .error: return 1
      default: continue
      }
    }
    return 0
  }

  // MARK: - Formatting

  private func formatNs(_ ns: UInt64) -> String {
    if ns >= 1_000_000 {
      return String(format: "%6.1fms", Double(ns) / 1_000_000)
    } else if ns >= 1_000 {
      return String(format: "%6.1fµs", Double(ns) / 1_000)
    } else {
      return String(format: "%6uns", UInt32(min(ns, UInt64(UInt32.max))))
    }
  }

  /// Right-pad `s` with spaces to exactly `width` characters.
  private func padRight(_ s: String, _ width: Int) -> String {
    if s.count >= width { return String(s.prefix(width)) }
    return s + String(repeating: " ", count: width - s.count)
  }

  /// Left-pad `s` with spaces to exactly `width` characters.
  private func padLeft(_ s: String, _ width: Int) -> String {
    if s.count >= width { return String(s.prefix(width)) }
    return String(repeating: " ", count: width - s.count) + s
  }
}
