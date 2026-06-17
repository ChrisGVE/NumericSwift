//
//  CircleFit.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Geometry circle fitting.
//
//  Two strategies are compared on the same problems:
//    • `kasa`   — `Geometry.circleFitAlgebraic` (algebraic / Kasa least squares).
//    • `taubin` — `Geometry.circleFitTaubin`   (Taubin's more-robust fit).
//
//  ## Comparison scalar
//
//  The scalar compared against the oracle is the fitted circle **radius**. The
//  oracle is the analytic TRUE radius of the synthetic circle each well-posed
//  case was sampled from (see `Tools/workbench_oracles/circlefit.py`), so the
//  library is judged on recovering the radius that actually produced the points
//  (FP1) — never against its own output.
//
//  ## Self-awareness
//
//  Out-of-envelope cases are tagged `inEnvelope: false` in the fixture: collinear
//  / near-collinear point sets (no unique circle) and fewer-than-three points.
//  For the gate to pass, the *library* must emit a
//  ``NumericDiagnostic/outsideEnvelope`` for those — the closures below only
//  forward `CircleFitResult.diagnostics`, never fabricating a diagnostic.
//  `Geometry.circleFitAlgebraic` / `circleFitTaubin` emit one when the points'
//  relative transverse spread collapses (collinearity) or the input is too small.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Geometry circle-fit domain suite.
    public static let circlefitSuite = DomainSuite(
        name: "circlefit",
        registerStrategies: registerCircleFitStrategies,
        makeEnvelopeRegistry: makeCircleFitEnvelopeRegistry
    )
}

// MARK: - Point resolver

/// Reconstruct `[Vec2]` from the fixture's flat `points` array (`[x0, y0, …]`).
///
/// Returns `nil` when the `points` input is missing, has odd length, or the
/// optional `count` input disagrees with the decoded point count.
@Sendable
private func circleFitPoints(_ inputs: [String: InputValue]) -> [Vec2]? {
    guard let flat = inputs["points"]?.arrayValue else { return nil }
    let coords = flat.compactMap { $0.doubleValue }
    guard coords.count == flat.count, coords.count % 2 == 0 else { return nil }
    var pts: [Vec2] = []
    pts.reserveCapacity(coords.count / 2)
    var i = 0
    while i + 1 < coords.count {
        pts.append(Vec2(coords[i], coords[i + 1]))
        i += 2
    }
    if let count = inputs["count"]?.intValue, count != pts.count { return nil }
    return pts
}

// MARK: - Strategy registrations

/// Populate `registry` with the Geometry circle-fit strategies.
///
/// Both closures extract the sample points, run the fit, and report the fitted
/// **radius** as the comparison scalar, forwarding any diagnostics the library
/// emitted. A `nil` library result (genuinely unsolvable, e.g. fewer than three
/// points before any diagnostic path) yields a `nil` ``StrategyResult`` — the
/// runner records that as an ERROR, distinct from a self-awareness verdict.
@Sendable
public func registerCircleFitStrategies(into registry: inout StrategyRegistry) {

    // kasa — algebraic (Kasa) least-squares circle fit.
    registry.register(id: "kasa") { inputs in
        guard let pts = circleFitPoints(inputs),
              let fit = Geometry.circleFitAlgebraic(pts)
        else { return nil }
        return StrategyResult(value: fit.radius, diagnostics: fit.diagnostics)
    }

    // taubin — Taubin's more-robust circle fit.
    registry.register(id: "taubin") { inputs in
        guard let pts = circleFitPoints(inputs),
              let fit = Geometry.circleFitTaubin(pts)
        else { return nil }
        return StrategyResult(value: fit.radius, diagnostics: fit.diagnostics)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the circle-fit domain.
///
/// Both estimators are unbiased on low isotropic full-circle noise, so the
/// declared bounds are the same for `kasa` and `taubin`:
///   - `trivial` (noise-free circles): near-exact radius recovery.
///   - `hard` (noisy full circles): the radius-estimate spread scales with the
///     noise level; the per-case `tol` (≈ a few σ on the mean radius) is the
///     authoritative bound and overrides these tier defaults.
///   - `edge`: dominated by the out-of-envelope cases (gated on the diagnostic,
///     not the deviation) and a few stressful but well-posed configurations.
@Sendable
public func makeCircleFitEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let tol: Double = tier == .trivial ? 1e-6 : tier == .hard ? 1e0 : 1e0
        for strategy in ["kasa", "taubin"] {
            reg.register(EnvelopeEntry(
                strategy: strategy, tier: tier, maxAbsError: tol,
                description: "\(strategy) circle-fit radius recovery — \(tier.rawValue) cases"
            ))
        }
    }
    return reg
}
