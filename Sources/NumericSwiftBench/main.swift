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
//   BENCH_WARMUP   — warmup iteration count (default 5)
//   BENCH_SAMPLES  — timed sample count (default 31)
//   BENCH_MATSIZE  — matrix side dimension for matrix gates (default 64)
//
// Gate legend:
//   PASS    — ratio ≤ threshold; gate satisfied.
//   FAIL    — ratio > threshold; performance regression detected.
//   PENDING — unified evaluator not yet implemented (Phase 0 state).
//             Not a failure; signals future work.
//   ERROR   — unexpected exception; treated as failure.
//
// Phase 0 note:
//   Gates 1–4 are PENDING because the unified NumericValue evaluator does
//   not exist yet. The sanity gate runs both legs with the legacy evaluator
//   and must PASS (ratio ≈ 1.0). When Phases 2/3 ship the unified path,
//   the placeholder numerator closures in each gate file are replaced with
//   real unified-evaluator calls — no other changes needed.

import Foundation
import NumericSwift

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
// PENDING — unified NumericValue evaluator not yet implemented.
runner.register(makeGate1(snapshot: snapshot))

// Gate 2: expression matmul vs direct LinAlg.dot (≤ 1.10).
// PENDING — unified NumericValue pipeline matmul not yet implemented.
runner.register(makeGate2(config: config))

// Gate 3: complex matmul (unified) vs real-block decomposition direct (≤ 1.10).
// PENDING — unified NumericValue pipeline complex matmul not yet implemented.
runner.register(makeGate3(config: config))

// Gate 4: expm via expression vs LinAlg.expm direct (≤ 1.10).
// PENDING — unified NumericValue pipeline expm not yet implemented.
runner.register(makeGate4(config: config))

// MARK: - Run and exit

let exitCode = runner.run()
exit(exitCode)
