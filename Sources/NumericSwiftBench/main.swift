// main.swift
// Sources/NumericSwiftBench/
//
// Entry point for the NumericSwiftBench executable target.
//
// This harness measures PAIRED legs and reports RATIOS — not absolute times.
// Ratio gates are CI-portable because they are self-relative: no machine-
// specific baseline file is committed; every gate compares two runs on the
// same machine in the same process.
//
// Build:
//   swift build --product NumericSwiftBench
// Run:
//   .build/debug/NumericSwiftBench
//
// Environment overrides (all optional):
//   BENCH_WARMUP     — warmup iteration count (default 5)
//   BENCH_SAMPLES    — timed sample count (default 31)
//   BENCH_MATSIZE    — matrix side dimension for matrix gates (default 64)
//   BENCH_SELF_TEST  — when set to "1", run gate-framework self-tests
//                      (negative-case FAIL + placeholder PENDING proofs)
//                      and exit; no performance gates are executed.
//
// Gate legend:
//   PASS    — ratio ≤ threshold; gate satisfied.
//   FAIL    — ratio > threshold; performance regression detected.
//   PENDING — unified evaluator not yet implemented (Phase 0 state).
//             Not a failure; signals future work.
//   ERROR   — unexpected exception; treated as failure.
//
// Phase 3 note:
//   All five gates are ACTIVE. Gates 1–4 measure real unified-evaluator ratios;
//   the sanity gate validates the timing infrastructure (ratio ≈ 1.0).
//   Gate 1: unified scalar evaluator vs legacy (≤1.15).
//   Gate 2: expr-driven matmul via evaluateUnified vs LinAlg.dot (≤1.10).
//   Gate 3: complex-matmul via evaluateUnified vs 4× LinAlg.dot (≤1.10).
//   Gate 4: exp(M) via evaluateUnified vs LinAlg.expm direct (≤1.10).

import Foundation
import NumericSwift

// MARK: - Gate self-test mode (BENCH_SELF_TEST=1)

// When BENCH_SELF_TEST=1, run structural gate-framework proofs and exit.
// These verify the gate model correctly detects regressions (negative-case
// FAIL) and correctly blocks placeholder promotion (PENDING proof).
// No performance gates are executed in this mode.
if ProcessInfo.processInfo.environment["BENCH_SELF_TEST"] == "1" {
  runGateSelfTests()
  exit(0)
}

// MARK: - Load frozen corpus

// Fail fast with a clear error message if the snapshot is absent or corrupt.
// The snapshot is committed in Task #2; its absence means the repo is broken.
let snapshot = CorpusLoader.loadOrExit()

// MARK: - Build config (environment-overridable)

let config = BenchConfig.fromEnvironment()

// MARK: - Register gates

let runner = BenchRunner(config: config)

// Sanity gate: same workload on both legs → ratio must be ≈ 1.0.
// This validates the timing infrastructure itself. Must PASS.
runner.register(makeSanityGate(snapshot: snapshot))

// Gate 1: unified-evaluator vs legacy scalar evaluator (≤ 1.15).
// ACTIVE — both legs wired (Task 18 numerator + Task 3 legacy denominator).
runner.register(makeGate1(snapshot: snapshot))

// Gate 2: expression matmul vs direct LinAlg.dot (≤ 1.10).
// ACTIVE — numerator wired in Task 21 (evaluateUnified "A * B" with matrix values).
runner.register(makeGate2(config: config))

// Gate 3: complex matmul (unified) vs real-block decomposition direct (≤ 1.10).
// ACTIVE — numerator wired in Task 21 (evaluateUnified "A * B" with complexMatrix values).
runner.register(makeGate3(config: config))

// Gate 4: expm via expression vs LinAlg.expm direct (≤ 1.10).
// ACTIVE — numerator wired in Task 21 (evaluateUnified "exp(M)" with matrix value).
runner.register(makeGate4(config: config))

// MARK: - Run and exit

let exitCode = runner.run()
exit(exitCode)
