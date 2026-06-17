//
//  Regression.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Regression (statsmodels-inspired).
//
//  Strategies compared (WORKBENCH.md §4): `ols` (OLS / dgels), `wls` (weighted
//  least squares), `glm` (IRLS, family + canonical link), `arima` (CSS).
//
//  ## Comparison scalar
//
//  Each strategy estimates a coefficient vector; the workbench compares a single
//  scalar against the statsmodels oracle:
//    • ols / wls / glm : the **slope** `params[1]` — the first non-intercept
//      coefficient. Every fixture builds X with a leading intercept column
//      (`addConstant`-style), so `params[0]` is the intercept and `params[1]` is
//      the first regressor's effect — the estimate a user actually interprets.
//    • arima           : the **AR(1) coefficient** `arParams[0]` — the leading
//      autoregressive parameter (every ARIMA fixture requests p ≥ 1).
//
//  These match `regression.py`, which returns the same scalar from
//  `sm.OLS`, `sm.WLS`, `sm.GLM`, and `sm.tsa.ARIMA`.
//
//  ## Input reconstruction
//
//  Fixtures carry the design matrix and response via ``InputValue/array(_:)``:
//    • ols   : `X` (flat row-major, n×k), `n`, `k`, `y` (length n).
//    • wls   : as ols, plus `weights` (length n).
//    • glm   : as ols, plus `family` (string: gaussian|binomial|poisson|gamma).
//              The canonical link for the family is used (matching the oracle).
//    • arima : `y` (the series), `p`, `d`, `q`.
//
//  ## Self-awareness
//
//  Out-of-envelope cases are tagged `inEnvelope: false`. For the gate to pass the
//  *library* must emit a ``NumericDiagnostic/outsideEnvelope`` (ols/wls/glm/arima)
//  — the closures only forward `result.diagnostics`; they never fabricate one.
//  The out-of-envelope regimes the library detects:
//    • ols/wls/glm : ill-conditioned / multicollinear design matrix
//      (cond(X) > ``regressionConditionEnvelope`` = 1e10).
//    • arima       : series too short for the requested (p,d,q) — fewer than
//      3·(p+q)+1 effective observations after differencing.
//  (GLM IRLS non-convergence is reported as a `nonConvergence` diagnostic; the
//  collinear-design case is the one wired as an `outsideEnvelope` gate trigger.)
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Regression domain suite.
    public static let regressionSuite = DomainSuite(
        name: "regression",
        registerStrategies: registerRegressionStrategies,
        makeEnvelopeRegistry: makeRegressionEnvelopeRegistry
    )
}

// MARK: - Input reconstruction

/// Read a flat `Double` array from an ``InputValue/array(_:)`` input.
@Sendable
private func regressionDoubleArray(_ value: InputValue?) -> [Double]? {
    guard let elements = value?.arrayValue else { return nil }
    let doubles = elements.compactMap(\.doubleValue)
    return doubles.count == elements.count ? doubles : nil
}

/// Reconstruct the row-major design matrix `X` as `[[Double]]` (one row per
/// observation) from a flat array and explicit `n`×`k` shape.
@Sendable
private func designMatrix(_ inputs: [String: InputValue]) -> (y: [Double], X: [[Double]])? {
    guard let n = inputs["n"]?.intValue, n > 0,
          let k = inputs["k"]?.intValue, k > 0,
          let flat = regressionDoubleArray(inputs["X"]), flat.count == n * k,
          let y = regressionDoubleArray(inputs["y"]), y.count == n
    else { return nil }
    var X = [[Double]](repeating: [Double](repeating: 0.0, count: k), count: n)
    for i in 0..<n {
        for j in 0..<k {
            X[i][j] = flat[i * k + j]
        }
    }
    return (y, X)
}

/// The comparison scalar for ols/wls/glm: the slope `params[1]`.
@Sendable
private func slope(_ params: [Double]) -> Double? {
    params.count >= 2 ? params[1] : nil
}

/// Map a fixture `family` string to the library ``GLMFamily``.
@Sendable
private func glmFamily(_ name: String?) -> GLMFamily? {
    switch name {
    case "gaussian": return .gaussian
    case "binomial": return .binomial
    case "poisson":  return .poisson
    case "gamma":    return .gamma
    default:         return nil
    }
}

// MARK: - Strategy registrations

/// Populate `registry` with the Regression strategies.
@Sendable
public func registerRegressionStrategies(into registry: inout StrategyRegistry) {

    // ols — ordinary least squares (dgels). Diagnostic on ill-conditioned design.
    registry.register(id: "ols") { inputs in
        guard let (y, X) = designMatrix(inputs),
              let result = ols(y, X),
              let value = slope(result.params) else { return nil }
        return StrategyResult(value: value, diagnostics: result.diagnostics)
    }

    // wls — weighted least squares. Diagnostic on ill-conditioned design.
    registry.register(id: "wls") { inputs in
        guard let (y, X) = designMatrix(inputs),
              let weights = regressionDoubleArray(inputs["weights"]), weights.count == y.count,
              let result = wls(y, X, weights: weights),
              let value = slope(result.params) else { return nil }
        return StrategyResult(value: value, diagnostics: result.diagnostics)
    }

    // glm — IRLS with the family's canonical link. Diagnostic on ill-conditioned
    // design (outsideEnvelope) or IRLS non-convergence (nonConvergence).
    registry.register(id: "glm") { inputs in
        guard let (y, X) = designMatrix(inputs),
              let family = glmFamily(inputs["family"]?.stringValue),
              let result = glm(y, X, family: family),
              let value = slope(result.params) else { return nil }
        return StrategyResult(value: value, diagnostics: result.diagnostics)
    }

    // arima — CSS ARIMA(p,d,q). Comparison scalar = AR(1) coefficient arParams[0].
    // Diagnostic when the series is too short for the requested order.
    registry.register(id: "arima") { inputs in
        guard let y = regressionDoubleArray(inputs["y"]),
              let p = inputs["p"]?.intValue,
              let d = inputs["d"]?.intValue,
              let q = inputs["q"]?.intValue,
              let result = arima(y, p: p, d: d, q: q),
              let value = result.arParams.first else { return nil }
        return StrategyResult(value: value, diagnostics: result.diagnostics)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Regression domain.
///
/// `ols`/`wls`/`glm` route to LAPACK `dgels`/IRLS and recover the slope to near
/// machine precision on well-conditioned designs; the looser `hard`/`edge`
/// bounds absorb the round-off that grows with the (still in-envelope) condition
/// number of the random designs and the GLM IRLS tolerance. `arima` uses CSS,
/// which only approximates the exact-likelihood AR coefficient statsmodels
/// reports, so it carries a wider envelope. Out-of-envelope edge cases are not
/// accuracy-scored (WORKBENCH.md §5) — only the diagnostic gates them.
@Sendable
public func makeRegressionEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let lsTol: Double = tier == .trivial ? 1e-9 : tier == .hard ? 1e-6 : 1e-5
        for strategy in ["ols", "wls"] {
            reg.register(EnvelopeEntry(
                strategy: strategy, tier: tier, maxAbsError: lsTol,
                description: "least-squares \(strategy) slope — \(tier.rawValue) cases (cond ≤ 1e10 envelope)"))
        }

        let glmTol: Double = tier == .trivial ? 1e-6 : tier == .hard ? 1e-4 : 1e-3
        reg.register(EnvelopeEntry(
            strategy: "glm", tier: tier, maxAbsError: glmTol,
            description: "GLM (IRLS) slope — \(tier.rawValue) cases (cond ≤ 1e10 envelope)"))

        let arimaTol: Double = tier == .trivial ? 1e-2 : tier == .hard ? 1e-1 : 5e-1
        reg.register(EnvelopeEntry(
            strategy: "arima", tier: tier, maxAbsError: arimaTol,
            description: "ARIMA (CSS) AR(1) coefficient — \(tier.rawValue) cases"))
    }
    return reg
}
