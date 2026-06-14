// main.swift
// Sources/NumericSwiftBench/
//
// Entry point for the NumericSwiftBench executable target.
//
// This harness is the performance benchmark driver for NumericSwift. It imports
// the NumericSwift library and exercises its computational modules so that
// run-to-run timing comparisons can detect regressions. The harness is an
// opt-in executable — it is NOT part of the NumericSwift library product and
// does NOT affect remote SPM consumers (e.g. LuaSwift), which only see the
// "NumericSwift" library product.
//
// Build:
//   swift build --product NumericSwiftBench
// Run:
//   .build/debug/NumericSwiftBench
//
// This file intentionally stays minimal: Phase 0 establishes the scaffolding so
// the executable target compiles and links. Benchmark cases are added in
// subsequent phases once the infrastructure is proven correct.

import NumericSwift

print("NumericSwiftBench — benchmark harness (Phase 0 scaffold)")
