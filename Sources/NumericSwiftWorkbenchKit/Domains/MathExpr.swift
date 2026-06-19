//
//  MathExpr.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: MathExpr (expression-string evaluation).
//
//  This is a **single-strategy correctness** domain (WORKBENCH.md §4, "Single-strategy
//  domains"): the one strategy id is `eval`, the comparison scalar is the evaluated
//  `Double`, and the oracle is the value of the EQUIVALENT Python expression with the
//  same variable bindings (Tools/workbench_oracles/mathexpr.py — FP1 / FP3).
//
//  ## Self-awareness
//
//  The expression evaluator either parses + evaluates an expression (to within
//  floating-point tolerance) or throws — there is no "degraded but plausible"
//  regime to warn about. So EVERY fixture case is in-envelope: there are ZERO
//  out-of-envelope cases, the gate is a pure correctness-vs-Python check, and the
//  `eval` closure returns a ``StrategyResult`` with EMPTY diagnostics (the harness
//  never fabricates one). A negative-radicand `sqrt`/`log` returns NaN — that is the
//  documented *real* contract of `MathExpr.evaluate`, NOT an out-of-envelope signal;
//  the fixture stores the NaN bit pattern and the case stays in-envelope.
//
//  ## Inputs
//
//    expr       → the expression string fed to `MathExpr.eval`.
//    varNames   → variable names (parallel to varValues).
//    varValues  → variable values (parallel to varNames).
//
//  Bindings are two parallel arrays rather than a nested object because the
//  workbench ``InputValue`` decoder models JSON scalars and arrays, not nested
//  objects. They are zipped back into the `[String: Double]` the evaluator expects.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The MathExpr (expression-string evaluation) domain suite.
    public static let mathexprSuite = DomainSuite(
        name: "mathexpr",
        registerStrategies: registerMathExprStrategies,
        makeEnvelopeRegistry: makeMathExprEnvelopeRegistry
    )
}

// MARK: - Input resolver

/// Zip the fixture's `varNames` / `varValues` arrays back into the `[String: Double]`
/// binding map the evaluator expects.
///
/// Returns `nil` when the arrays are missing or mismatched in length — the runner
/// records that as an ERROR rather than a self-awareness verdict. An absent
/// `varNames` is treated as an empty binding set (constant expression).
@Sendable
private func mathexprBindings(_ inputs: [String: InputValue]) -> [String: Double]? {
    let names = inputs["varNames"]?.arrayValue ?? []
    let values = inputs["varValues"]?.arrayValue ?? []
    guard names.count == values.count else { return nil }
    var bindings: [String: Double] = [:]
    for (n, v) in zip(names, values) {
        guard let key = n.stringValue, let val = v.doubleValue else { return nil }
        bindings[key] = val
    }
    return bindings
}

// MARK: - Strategy registrations

/// Populate `registry` with the single MathExpr strategy (`eval`).
@Sendable
public func registerMathExprStrategies(into registry: inout StrategyRegistry) {

    // eval — parse and evaluate the expression string against its variable bindings.
    // A parse/eval throw is surfaced as nil (recorded ERROR); a correct corpus never
    // errors. NaN/inf results (real-domain contract, IEEE edge) flow through as the
    // value and are compared bit-exact against the frozen oracle bits by the runner.
    registry.register(id: "eval") { inputs in
        guard let expr = inputs["expr"]?.stringValue,
              let bindings = mathexprBindings(inputs)
        else { return nil }
        guard let value = try? MathExpr.eval(expr, variables: bindings) else { return nil }
        return StrategyResult(value: value)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelope for the MathExpr domain.
///
/// The evaluator is exact up to floating-point rounding of the underlying `Darwin`
/// math primitives, so the registry envelope is uniformly tight (~1e-9) across every
/// tier. The authoritative per-case bound is the fixture's `tol` (WORKBENCH.md §5):
/// pure-arithmetic cases declare 1e-12, transcendental-heavy ones 1e-9, and the
/// extreme-magnitude edge cases scale their tol to the result magnitude. No strategy
/// has an out-of-envelope regime, hence no `outsideEnvelope` diagnostic is ever
/// expected.
@Sendable
public func makeMathExprEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        reg.register(EnvelopeEntry(
            strategy: "eval",
            tier: tier,
            maxAbsError: 1e-9,
            description: "MathExpr.eval — exact vs Python math (\(tier.rawValue) cases)"))
    }
    return reg
}
