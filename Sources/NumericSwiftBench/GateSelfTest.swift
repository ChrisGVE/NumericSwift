// GateSelfTest.swift
// Sources/NumericSwiftBench/
//
// Internal self-test utilities that validate the gate framework's correctness
// (state-machine transitions and negative-case detection) without relying on
// deterministic timing from the OS scheduler.
//
// Why here and not in a test target?
//   `GateDefinition`, `GateState`, and `runGate` live in the NumericSwiftBench
//   *executable* target, which cannot be @testable-imported by XCTest.
//   The canonical fix — splitting the gate model into a separate library target
//   — is out of scope for Task 21 (that would touch the Package.swift manifest
//   and all import sites). Instead, these structural proofs live in the
//   executable itself and are invoked when BENCH_SELF_TEST=1 is set.
//
// Negative-test contract (subtask 21.14):
//   A gate whose numerator is artificially *slower* than its denominator must
//   produce `.fail` when the ratio exceeds the threshold. This proves the
//   gate framework detects regressions rather than always emitting `.pass`.
//
// Gate-structure proof (subtask 21.17):
//   The placeholder factory must produce `.pending`, never `.pass` or `.fail`.
//   This validates that the PENDING state correctly blocks gate promotion
//   even when the denominator runs successfully.
//
// Usage (invoked from main.swift when BENCH_SELF_TEST=1):
//   runGateSelfTests()

import Foundation
import NumericSwift

// MARK: - Gate self-test suite

/// Run structural self-tests against the gate framework.
///
/// Invoked when the environment variable `BENCH_SELF_TEST=1` is set.
/// Exits with code 3 on any assertion failure so CI can detect
/// framework bugs independently of real performance results.
func runGateSelfTests() {
  print("Running gate self-tests…")

  testNegativeCase()
  testPendingState()

  print("Gate self-tests: all passed.")
}

// MARK: - Negative-case test (subtask 21.14)

/// Proves that a gate whose numerator is materially slower than its denominator
/// produces `.fail`.
///
/// The denominator does trivial work (one multiply). The numerator spins for
/// a fixed count to guarantee it is measurably slower on any machine.
/// The gate threshold is set to 1.01 (1% headroom) so the artificial 4× slower
/// numerator must trip the gate.
///
/// If the gate produces `.pass` or `.pending` instead of `.fail`, the test
/// aborts with an error message and exits with code 3.
private func testNegativeCase() {
  // Use REAL cross-module LinAlg.expm work at two very different sizes. A
  // self-contained in-module loop is unreliable here: the optimizer either
  // dead-code-eliminates it (blackhole cannot fully anchor a trivial local
  // computation) or solves affine/closed-form recurrences in O(1), collapsing
  // the ratio to a meaningless 1.0. expm is opaque across the module boundary
  // (no DCE, no closed form) and O(n³), so a small vs. large matrix yields a
  // large, robust, real time difference — exactly what a FAIL self-test needs.
  let smallM = BenchFixtures.expmMatrix(n: 8)
  let largeM = BenchFixtures.expmMatrix(n: 32)

  // Denominator: small expm — fast reference.
  let denominator: () -> Void = {
    blackhole(try? LinAlg.expm(smallM))
  }

  // Numerator: much larger expm — O(n³) makes it many× slower than the
  // denominator. No realistic scheduler jitter can bridge that gap.
  let numerator: () -> Void = {
    blackhole(try? LinAlg.expm(largeM))
  }

  let gate = GateDefinition(
    name: "self-test: slow-numerator must FAIL",
    threshold: 1.01,
    denominatorDescription: "LinAlg.expm 8×8 (fast reference)",
    numeratorDescription: "LinAlg.expm 32×32 — O(n³) makes it must exceed 1.01 threshold",
    denominator: denominator,
    numerator: numerator
  )

  let result = runGate(gate, config: BenchConfig(warmupIterations: 3, timedSamples: 11, matrixSize: 0))

  guard case .fail = result.state else {
    fputs("""
      GATE SELF-TEST FAILED: negative-case gate produced \(result.state.label)
        expected: FAIL (ratio > 1.01 from slow numerator)
        actual:   \(result.state.label)
      This indicates a framework bug — the gate model does not detect regressions.
      """, stderr)
    exit(3)
  }

  print("  ✓ negative-case gate correctly produced FAIL (ratio \(String(format: "%.4f", ratioFromState(result.state))))")
}

// MARK: - Pending-state structural test (subtask 21.17 / gate-model proof)

/// Proves that the placeholder factory produces `.pending`, never `.pass` / `.fail`.
///
/// Creates a gate via `.placeholder(...)` and verifies that running it — even
/// when a real denominator is provided — yields `.pending` and not `.pass` or
/// `.fail`. This confirms PENDING correctly blocks gate promotion.
private func testPendingState() {
  let denominator: () -> Void = {
    blackhole(sin(Double.pi / 4))
  }

  let gate = GateDefinition.placeholder(
    name: "self-test: placeholder must be PENDING",
    threshold: 1.10,
    denominatorDescription: "real denominator (sin call)",
    numeratorDescription: "placeholder — no numerator",
    denominator: denominator,
    reason: "self-test placeholder"
  )

  let result = runGate(gate, config: BenchConfig(warmupIterations: 1, timedSamples: 3, matrixSize: 0))

  guard case .pending = result.state else {
    fputs("""
      GATE SELF-TEST FAILED: placeholder gate produced \(result.state.label)
        expected: PENDING
        actual:   \(result.state.label)
      This indicates a framework bug — placeholders must never become PASS or FAIL.
      """, stderr)
    exit(3)
  }

  print("  ✓ placeholder gate correctly produced PENDING")
}

// MARK: - Helper

/// Extract the ratio from a `.pass` or `.fail` GateState, defaulting to 0.
private func ratioFromState(_ state: GateState) -> Double {
  switch state {
  case .pass(let r), .fail(let r): return r
  default: return 0
  }
}
