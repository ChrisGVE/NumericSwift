//
//  Integration.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Reference per-domain workbench suite: Integration (quadrature).
//
//  This is the pattern every other domain copies:
//    1. `<domain>Suite` — a `DomainSuite` naming the domain and wiring its
//       strategy + envelope registries.
//    2. `register<Domain>Strategies(into:)` — one closure per strategy id,
//       extracting the fixture `inputs` bag and delegating to NumericSwift.
//       Each closure forwards any ``NumericDiagnostic`` the library emitted.
//    3. `make<Domain>EnvelopeRegistry()` — per-(strategy, tier) accuracy bounds.
//
//  ## Self-awareness
//
//  Out-of-envelope cases are tagged `inEnvelope: false` in the fixture. For the
//  gate to pass, the *library* must emit an ``NumericDiagnostic/outsideEnvelope``
//  for those cases — the strategy closure only forwards what the library
//  produced, it never fabricates a diagnostic. `quad` emits one when its adaptive
//  subdivision limit is reached without meeting tolerance (SciPy-parity
//  `IntegrationWarning`).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Integration (quadrature) domain suite.
    public static let integrationSuite = DomainSuite(
        name: "integration",
        registerStrategies: registerIntegrationStrategies,
        makeEnvelopeRegistry: makeIntegrationEnvelopeRegistry
    )
}

// MARK: - Integrand resolver

/// Resolve the integrand named by the fixture's `tag` input.
///
/// Closed-form integrands keep fixtures small: the JSON carries the tag and the
/// bounds, not a sampled function.
@Sendable
private func integrationIntegrand(_ tag: String?) -> (@Sendable (Double) -> Double)? {
    switch tag {
    case "gaussian_bell":   return { x in exp(-x * x) }
    case "sine":            return { x in sin(x) }
    case "cosine":          return { x in cos(x) }
    case "polynomial_deg2": return { x in x * x + 2 * x + 1 }     // (x+1)^2
    case "exp":             return { x in exp(x) }
    case "runge":           return { x in 1.0 / (1.0 + 25.0 * x * x) }
    case "oscillatory":     return { x in sin(50.0 * x) }
    // Strong endpoint singularity — used for out-of-envelope quad cases.
    case "inverse_sqrt":    return { x in x <= 0 ? 0 : 1.0 / sqrt(x) }
    default:                return nil
    }
}

// MARK: - Strategy registrations

/// Populate `registry` with the Integration (quadrature) strategies.
@Sendable
public func registerIntegrationStrategies(into registry: inout StrategyRegistry) {

    // quad — Gauss-Kronrod 15-point adaptive. Honours optional epsabs/epsrel/limit
    // inputs so out-of-envelope cases can force a tight subdivision budget.
    registry.register(id: "quad") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = integrationIntegrand(inputs["tag"]?.stringValue)
        else { return nil }
        let epsabs = inputs["epsabs"]?.doubleValue ?? quadDefaultEpsAbs
        let epsrel = inputs["epsrel"]?.doubleValue ?? quadDefaultEpsRel
        let limit = inputs["limit"]?.intValue ?? quadDefaultLimit
        let r = quad(f, a, b, epsabs: epsabs, epsrel: epsrel, limit: limit)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }

    // romberg — Richardson-extrapolation quadrature.
    registry.register(id: "romberg") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = integrationIntegrand(inputs["tag"]?.stringValue)
        else { return nil }
        let r = romberg(f, a, b)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }

    // simps — composite Simpson's rule over n+1 samples.
    registry.register(id: "simps") { inputs in
        guard let (xs, ys) = sampledGrid(inputs) else { return nil }
        return StrategyResult(value: simps(ys, x: xs))
    }

    // trapz — composite trapezoidal rule over n+1 samples.
    registry.register(id: "trapz") { inputs in
        guard let (xs, ys) = sampledGrid(inputs) else { return nil }
        return StrategyResult(value: trapz(ys, x: xs))
    }

    // fixed_quad — Gauss-Legendre fixed-order quadrature.
    registry.register(id: "fixed_quad") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = integrationIntegrand(inputs["tag"]?.stringValue)
        else { return nil }
        let order = inputs["order"]?.intValue ?? 5
        return StrategyResult(value: fixedQuad(f, a, b, n: order))
    }
}

/// Build an `(xs, ys)` sample grid for the sample-based strategies (simps/trapz).
///
/// Reads `a`, `b`, `n` (number of subintervals) and the integrand `tag`.
@Sendable
private func sampledGrid(_ inputs: [String: InputValue]) -> ([Double], [Double])? {
    guard let a = inputs["a"]?.doubleValue,
          let b = inputs["b"]?.doubleValue,
          let n = inputs["n"]?.intValue, n > 0,
          let f = integrationIntegrand(inputs["tag"]?.stringValue)
    else { return nil }
    let h = (b - a) / Double(n)
    let xs = (0...n).map { a + Double($0) * h }
    return (xs, xs.map(f))
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Integration domain.
///
///  - `quad` (Gauss-Kronrod adaptive): full double precision on smooth integrands.
///  - `romberg` (Richardson extrapolation): looser on non-smooth integrands.
///  - `simps` (Simpson's rule): exact for polynomials ≤ degree 3.
///  - `trapz` (trapezoidal rule): first-order; exact only for linear functions.
///  - `fixed_quad` (Gauss-Legendre fixed-point): order-dependent.
@Sendable
public func makeIntegrationEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let quadTol: Double = tier == .trivial ? 1e-12 : tier == .hard ? 1e-8 : 1e-5
        reg.register(EnvelopeEntry(strategy: "quad", tier: tier, maxAbsError: quadTol,
            description: "Gauss-Kronrod 15-point adaptive quadrature — \(tier.rawValue) cases"))

        let rombergTol: Double = tier == .trivial ? 1e-10 : tier == .hard ? 1e-6 : 1e-3
        reg.register(EnvelopeEntry(strategy: "romberg", tier: tier, maxAbsError: rombergTol,
            description: "Romberg (Richardson extrapolation) — \(tier.rawValue) cases"))

        let simpsTol: Double = tier == .trivial ? 1e-9 : tier == .hard ? 1e-3 : 1e-1
        reg.register(EnvelopeEntry(strategy: "simps", tier: tier, maxAbsError: simpsTol,
            description: "Composite Simpson's rule — \(tier.rawValue) cases"))

        let trapzTol: Double = tier == .trivial ? 1e-6 : tier == .hard ? 1e-1 : 1e0
        reg.register(EnvelopeEntry(strategy: "trapz", tier: tier, maxAbsError: trapzTol,
            description: "Composite trapezoidal rule — \(tier.rawValue) cases"))

        let fixedTol: Double = tier == .trivial ? 1e-9 : tier == .hard ? 1e-4 : 1e-2
        reg.register(EnvelopeEntry(strategy: "fixed_quad", tier: tier, maxAbsError: fixedTol,
            description: "Gauss-Legendre fixed quadrature — \(tier.rawValue) cases"))
    }
    return reg
}
