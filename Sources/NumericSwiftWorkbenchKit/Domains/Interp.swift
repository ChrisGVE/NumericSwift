//
//  Interp.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Interpolation (1-D interpolants).
//
//  Mirrors the reference `Integration.swift` suite:
//    1. `interpSuite` — the `DomainSuite` wiring the strategy + envelope registries.
//    2. `registerInterpStrategies(into:)` — one closure per interpolation strategy.
//       Each closure builds the interpolant from the fixture's sample (x, y)
//       arrays and evaluates it at the query point `xq`. The comparison scalar is
//       that interpolated value. Each closure delegates to the library's
//       `*Diagnosed` evaluator and FORWARDS the ``NumericDiagnostic`` it produced
//       — it never fabricates one.
//    3. `makeInterpEnvelopeRegistry()` — per-(strategy, tier) accuracy bounds.
//
//  ## Self-awareness
//
//  Out-of-envelope cases (tagged `inEnvelope: false` in the fixture) are
//  extrapolation queries: `xq` outside the knot range `[x[0], x[-1]]`. There the
//  interpolant's accuracy guarantee does not hold, so the library MUST emit a
//  ``NumericDiagnostic/outsideEnvelope`` — which the `*Diagnosed` evaluators do.
//  Interior in-envelope cases must NOT warn.
//
//  Inputs (carried through the JSON `inputs` bag, see ``InputValue``):
//    • `xs`  — array of sample abscissae (strictly increasing).
//    • `ys`  — array of sample ordinates.
//    • `xq`  — scalar query point.
//    • `func`— sample-function tag (`runge`, `sine`, `exp`, …); present for
//              provenance so the oracle and Swift sample the SAME function. The
//              ys array is the authoritative sample; the tag is informational.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Interpolation domain suite.
    public static let interpSuite = DomainSuite(
        name: "interp",
        registerStrategies: registerInterpStrategies,
        makeEnvelopeRegistry: makeInterpEnvelopeRegistry
    )
}

// MARK: - Sample resolver

/// Extract the `(xs, ys, xq)` sample triple from a fixture inputs bag.
///
/// Returns `nil` when any required key is missing or malformed, so the runner
/// records an ERROR rather than a spurious self-awareness verdict.
@Sendable
private func interpSample(_ inputs: [String: InputValue]) -> (xs: [Double], ys: [Double], xq: Double)? {
    guard let xsRaw = inputs["xs"]?.arrayValue,
          let ysRaw = inputs["ys"]?.arrayValue,
          let xq = inputs["xq"]?.doubleValue
    else { return nil }
    let xs = xsRaw.compactMap(\.doubleValue)
    let ys = ysRaw.compactMap(\.doubleValue)
    guard xs.count == xsRaw.count, ys.count == ysRaw.count,
          xs.count >= 2, xs.count == ys.count
    else { return nil }
    return (xs, ys, xq)
}

// MARK: - Strategy registrations

/// Populate `registry` with the Interpolation strategies.
///
/// The three cubic strategies share `evalCubicSplineDiagnosed`; they differ only
/// in the boundary condition passed to `computeSplineCoeffs`.
@Sendable
public func registerInterpStrategies(into registry: inout StrategyRegistry) {

    // cubic_natural — CubicSpline(bc_type="natural").
    registry.register(id: "cubic_natural") { inputs in
        guard let s = interpSample(inputs) else { return nil }
        let coeffs = computeSplineCoeffs(x: s.xs, y: s.ys, bc: .natural)
        let r = evalCubicSplineDiagnosed(x: s.xs, coeffs: coeffs, xNew: s.xq)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }

    // cubic_clamped — CubicSpline(bc_type="clamped"), zero-slope ends.
    registry.register(id: "cubic_clamped") { inputs in
        guard let s = interpSample(inputs) else { return nil }
        let coeffs = computeSplineCoeffs(x: s.xs, y: s.ys, bc: .clamped)
        let r = evalCubicSplineDiagnosed(x: s.xs, coeffs: coeffs, xNew: s.xq)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }

    // cubic_notaknot — CubicSpline(bc_type="not-a-knot") (SciPy default).
    registry.register(id: "cubic_notaknot") { inputs in
        guard let s = interpSample(inputs) else { return nil }
        let coeffs = computeSplineCoeffs(x: s.xs, y: s.ys, bc: .notAKnot)
        let r = evalCubicSplineDiagnosed(x: s.xs, coeffs: coeffs, xNew: s.xq)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }

    // pchip — PchipInterpolator (monotone Hermite).
    registry.register(id: "pchip") { inputs in
        guard let s = interpSample(inputs) else { return nil }
        let d = computePchipDerivatives(x: s.xs, y: s.ys)
        let r = evalPchipDiagnosed(x: s.xs, y: s.ys, d: d, xNew: s.xq)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }

    // akima — Akima1DInterpolator.
    registry.register(id: "akima") { inputs in
        guard let s = interpSample(inputs) else { return nil }
        let coeffs = computeAkimaCoeffs(x: s.xs, y: s.ys)
        let r = evalAkimaDiagnosed(x: s.xs, coeffs: coeffs, xNew: s.xq)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }

    // barycentric — BarycentricInterpolator (global polynomial).
    registry.register(id: "barycentric") { inputs in
        guard let s = interpSample(inputs) else { return nil }
        let w = computeBarycentricWeights(x: s.xs)
        let r = evalBarycentricDiagnosed(x: s.xs, y: s.ys, w: w, xNew: s.xq)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Interpolation domain.
///
///  - cubic (natural/clamped/notAKnot): C² piecewise-cubic; tight on smooth
///    interior queries, looser near steep features (Runge) at the boundary tiers.
///  - pchip: monotone, no overshoot, but only C¹ — a touch looser than cubic.
///  - akima: low-overshoot but discontinuous third derivative; looser still.
///  - barycentric: global polynomial; superb on few-point smooth data but prone
///    to interior Runge oscillation on equispaced many-point grids, so it gets
///    the widest envelope at the hard/edge tiers.
@Sendable
public func makeInterpEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let cubicTol: Double = tier == .trivial ? 1e-9 : tier == .hard ? 1e-1 : 1e0
        for cubic in ["cubic_natural", "cubic_clamped", "cubic_notaknot"] {
            reg.register(EnvelopeEntry(strategy: cubic, tier: tier, maxAbsError: cubicTol,
                description: "Piecewise-cubic spline (\(cubic)) — \(tier.rawValue) cases"))
        }

        let pchipTol: Double = tier == .trivial ? 1e-2 : tier == .hard ? 5e-1 : 1e0
        reg.register(EnvelopeEntry(strategy: "pchip", tier: tier, maxAbsError: pchipTol,
            description: "PCHIP monotone Hermite — \(tier.rawValue) cases"))

        let akimaTol: Double = tier == .trivial ? 1e-2 : tier == .hard ? 5e-1 : 1e0
        reg.register(EnvelopeEntry(strategy: "akima", tier: tier, maxAbsError: akimaTol,
            description: "Akima interpolation — \(tier.rawValue) cases"))

        let baryTol: Double = tier == .trivial ? 1e-6 : tier == .hard ? 1e1 : 1e2
        reg.register(EnvelopeEntry(strategy: "barycentric", tier: tier, maxAbsError: baryTol,
            description: "Barycentric global polynomial — \(tier.rawValue) cases"))
    }
    return reg
}
