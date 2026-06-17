//
//  OptRoot.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Optimization — scalar root finding.
//
//  Mirrors `Domains/Integration.swift` (the reference pattern):
//    1. `optrootSuite` — a `DomainSuite` wiring the strategy + envelope registries.
//    2. `registerOptRootStrategies(into:)` — one closure per strategy id, mapping
//       the fixture `inputs` bag to a NumericSwift root finder and forwarding the
//       result's `diagnostics`.
//    3. `makeOptRootEnvelopeRegistry()` — per-(strategy, tier) accuracy bounds.
//
//  ## Strategy → library mapping
//
//  | id       | library call                          | bracket / guess |
//  |----------|---------------------------------------|-----------------|
//  | `bisect` | `bisect(f, a:b:)`                     | bracket [a, b]  |
//  | `brentq` | `brentq(f, a:b:)`                     | bracket [a, b]  |
//  | `newton` | `newton(f, fprime:x0:)`               | guess x0        |
//  | `secant` | `secant(f, x0:x1:)`                   | guess x0 (+x1)  |
//
//  The comparison scalar is the found `root` (`RootScalarResult.root`); the oracle
//  is `scipy.optimize.brentq` on a sign-changing bracket.
//
//  ## Self-awareness
//
//  Out-of-envelope cases are tagged `inEnvelope: false` in the fixture. The gate
//  requires the *library* to emit a ``NumericDiagnostic/outsideEnvelope`` for them
//  — the closures only forward `result.diagnostics`, never fabricate one:
//    • bracketing methods (`bisect`, `brentq`) on a bracket with NO sign change
//      (`f(a)·f(b) > 0`) — bracketing root finders are invalid there;
//    • open methods (`newton`, `secant`) that hit a near-zero derivative / secant
//      slope or diverge (exhaust the iteration budget).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Optimization (scalar root finding) domain suite.
    public static let optrootSuite = DomainSuite(
        name: "optroot",
        registerStrategies: registerOptRootStrategies,
        makeEnvelopeRegistry: makeOptRootEnvelopeRegistry
    )
}

// MARK: - Function resolver

/// The function whose root is sought, named by the fixture's `tag` input.
///
/// Closed-form test functions keep fixtures small: the JSON carries the tag plus
/// the bracket / guess, not a sampled function. Tags MUST match the `FUNCS`
/// dictionary in `Tools/workbench_oracles/optroot.py`.
@Sendable
private func optrootFunction(_ tag: String?) -> (@Sendable (Double) -> Double)? {
    switch tag {
    case "x2_minus_2":     return { x in x * x - 2.0 }
    case "cos_minus_x":    return { x in cos(x) - x }
    case "cubic_x3_x_2":   return { x in x * x * x - x - 2.0 }
    case "exp_minus_3":    return { x in exp(x) - 3.0 }
    case "sin":            return { x in sin(x) }
    case "cubic_x3_2x_5":  return { x in x * x * x - 2.0 * x - 5.0 }
    case "log_minus_1":    return { x in log(x) - 1.0 }
    case "quad_x2_2x_3":   return { x in x * x - 2.0 * x - 3.0 }
    default:               return nil
    }
}

/// The analytic derivative for the named function, used by `newton`.
///
/// Supplying the exact derivative (rather than a finite difference) keeps the
/// out-of-envelope `newton` cases deterministic: e.g. `x2_minus_2` has
/// `f'(x) = 2x`, so the `x0 = 0` edge case lands exactly on `f'(0) = 0`.
@Sendable
private func optrootDerivative(_ tag: String?) -> (@Sendable (Double) -> Double)? {
    switch tag {
    case "x2_minus_2":     return { x in 2.0 * x }
    case "cos_minus_x":    return { x in -sin(x) - 1.0 }
    case "cubic_x3_x_2":   return { x in 3.0 * x * x - 1.0 }
    case "exp_minus_3":    return { x in exp(x) }
    case "sin":            return { x in cos(x) }
    case "cubic_x3_2x_5":  return { x in 3.0 * x * x - 2.0 }
    case "log_minus_1":    return { x in 1.0 / x }
    case "quad_x2_2x_3":   return { x in 2.0 * x - 2.0 }
    default:               return nil
    }
}

// MARK: - Strategy registrations

/// Populate `registry` with the Optimization (root finding) strategies.
@Sendable
public func registerOptRootStrategies(into registry: inout StrategyRegistry) {

    // bisect — bracketing. Requires a sign-changing bracket [a, b]; emits an
    // `outsideEnvelope` diagnostic when f(a)·f(b) > 0.
    registry.register(id: "bisect") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = optrootFunction(inputs["tag"]?.stringValue)
        else { return nil }
        let r = bisect(f, a: a, b: b)
        return StrategyResult(value: r.root, diagnostics: r.diagnostics)
    }

    // brentq — bracketing (inverse-quadratic + secant + bisection fallback).
    // SciPy `brentq` analogue; same invalid-bracket envelope as bisect.
    registry.register(id: "brentq") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = optrootFunction(inputs["tag"]?.stringValue)
        else { return nil }
        let r = brentq(f, a: a, b: b)
        return StrategyResult(value: r.root, diagnostics: r.diagnostics)
    }

    // newton — open method. Uses the analytic derivative; emits an
    // `outsideEnvelope` diagnostic on a near-zero derivative or divergence.
    registry.register(id: "newton") { inputs in
        guard let x0 = inputs["x0"]?.doubleValue,
              let f = optrootFunction(inputs["tag"]?.stringValue)
        else { return nil }
        let fprime = optrootDerivative(inputs["tag"]?.stringValue)
        let r = newton(f, fprime: fprime, x0: x0)
        return StrategyResult(value: r.root, diagnostics: r.diagnostics)
    }

    // secant — open method. Reads x0 and optional x1; emits an `outsideEnvelope`
    // diagnostic when the secant slope collapses or the iteration diverges.
    registry.register(id: "secant") { inputs in
        guard let x0 = inputs["x0"]?.doubleValue,
              let f = optrootFunction(inputs["tag"]?.stringValue)
        else { return nil }
        let x1 = inputs["x1"]?.doubleValue
        let r = secant(f, x0: x0, x1: x1)
        return StrategyResult(value: r.root, diagnostics: r.diagnostics)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Optimization root-finding domain.
///
/// The library's default convergence test is the absolute step tolerance
/// `xtol = 1e-8`, so every method locates the root to within ~`1e-8`. The declared
/// envelopes sit one order of magnitude above that achieved residual — tight enough
/// to catch a regression, loose enough not to flag the expected step-tolerance
/// floor. Newton's quadratic convergence earns a slightly tighter bound. These
/// fall back only when a fixture case omits a `tol` for the strategy (the per-case
/// `tol` is authoritative — see `Tools/workbench_oracles/optroot.py`).
@Sendable
public func makeOptRootEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let bisectTol: Double = tier == .edge ? 1e-5 : 1e-6
        reg.register(EnvelopeEntry(strategy: "bisect", tier: tier, maxAbsError: bisectTol,
            description: "Bisection (linear convergence, guaranteed on a valid bracket) — \(tier.rawValue) cases"))

        let brentqTol: Double = tier == .edge ? 1e-5 : 1e-6
        reg.register(EnvelopeEntry(strategy: "brentq", tier: tier, maxAbsError: brentqTol,
            description: "Brent's method (inverse-quadratic + bisection fallback) — \(tier.rawValue) cases"))

        let newtonTol: Double = tier == .edge ? 1e-7 : 1e-8
        reg.register(EnvelopeEntry(strategy: "newton", tier: tier, maxAbsError: newtonTol,
            description: "Newton's method (quadratic convergence, needs f'≠0) — \(tier.rawValue) cases"))

        let secantTol: Double = tier == .edge ? 1e-5 : 1e-6
        reg.register(EnvelopeEntry(strategy: "secant", tier: tier, maxAbsError: secantTol,
            description: "Secant method (superlinear convergence) — \(tier.rawValue) cases"))
    }
    return reg
}
