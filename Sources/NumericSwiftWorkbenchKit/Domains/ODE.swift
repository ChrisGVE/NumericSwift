//
//  ODE.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Integration (ODE / initial-value problems).
//
//  Mirrors the reference `Domains/Integration.swift` pattern exactly:
//    1. `odeSuite` — a `DomainSuite` wiring the strategy + envelope registries.
//    2. `registerODEStrategies(into:)` — one closure per strategy id, extracting
//       the fixture `inputs` bag and delegating to NumericSwift. Each closure
//       forwards any ``NumericDiagnostic`` the library emitted.
//    3. `makeODEEnvelopeRegistry()` — per-(strategy, tier) accuracy bounds.
//
//  ## Comparison scalar
//
//  The workbench compares the **first solution component `y[0]` at the final
//  time `tf`** against the scipy oracle. The oracle generator (`ode.py`) extracts
//  the same scalar (`sol.y[0][-1]`), so the two are directly comparable.
//
//  ## Strategy → method map
//
//  | strategy | NumericSwift call          | scipy oracle method |
//  |----------|----------------------------|---------------------|
//  | rk45     | `solveIVP(.rk45)`          | RK45                |
//  | rk23     | `solveIVP(.rk23)`          | RK23                |
//  | bdf      | `solveIVP(.bdf)`           | BDF                 |
//  | odeint   | `odeint(...)`              | LSODA               |
//
//  NumericSwift has no DOP853 method, so the `dop853` strategy listed in
//  WORKBENCH.md §4 is deliberately omitted (FP1: no fabricated mapping).
//
//  ## Self-awareness
//
//  Out-of-envelope cases are STIFF systems (Van der Pol with large μ, Robertson)
//  integrated with an EXPLICIT method (`rk45`/`rk23`), tagged `inEnvelope: false`.
//  For the gate to pass, the *library* must emit a
//  ``NumericDiagnostic/outsideEnvelope`` for those calls — the explicit step
//  controller collapses on a stiff system and `solveIVP` reports it (see
//  `detectExplicitStiffness` in `Sources/NumericSwift/Integration.swift`). The
//  strategy closure only forwards what the library produced, never fabricating a
//  diagnostic. The implicit `bdf` and the auto-switching `odeint` solvers handle
//  stiffness and must NOT warn (false-positive guard).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Integration (ODE) domain suite.
    public static let odeSuite = DomainSuite(
        name: "ode",
        registerStrategies: registerODEStrategies,
        makeEnvelopeRegistry: makeODEEnvelopeRegistry
    )
}

// MARK: - System resolver

/// Resolve the ODE right-hand side named by the fixture's `tag` input.
///
/// Closed-form systems keep fixtures small: the JSON carries the tag plus the
/// system's scalar parameters (`mu`, `k`, `r`, `kk`), not a sampled trajectory.
/// MUST stay in lockstep with the `rhs` resolver in `Tools/workbench_oracles/ode.py`.
@Sendable
private func odeSystem(_ inputs: [String: InputValue]) -> (@Sendable ([Double], Double) -> [Double])? {
    guard let tag = inputs["tag"]?.stringValue else { return nil }
    switch tag {
    case "exp_decay":
        let k = inputs["k"]?.doubleValue ?? 1.0
        return { y, _ in [-k * y[0]] }
    case "harmonic":
        return { y, _ in [y[1], -y[0]] }
    case "logistic":
        let r = inputs["r"]?.doubleValue ?? 1.0
        let kk = inputs["kk"]?.doubleValue ?? 1.0
        return { y, _ in [r * y[0] * (1.0 - y[0] / kk)] }
    case "linear2d":
        return { y, _ in [-y[0] + y[1], -y[0] - y[1]] }
    case "vdp":
        guard let mu = inputs["mu"]?.doubleValue else { return nil }
        return { y, _ in [y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]] }
    case "robertson":
        let k1 = 0.04, k2 = 3.0e7, k3 = 1.0e4
        return { y, _ in
            [
                -k1 * y[0] + k3 * y[1] * y[2],
                k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] * y[1],
                k2 * y[1] * y[1],
            ]
        }
    default:
        return nil
    }
}

/// Extract the `(y0, tf)` problem definition from the fixture inputs.
@Sendable
private func odeProblem(_ inputs: [String: InputValue]) -> (y0: [Double], tf: Double)? {
    guard let tf = inputs["tf"]?.doubleValue,
          let y0Raw = inputs["y0"]?.arrayValue
    else { return nil }
    let y0 = y0Raw.compactMap { $0.doubleValue }
    guard y0.count == y0Raw.count, !y0.isEmpty else { return nil }
    return (y0, tf)
}

// MARK: - Strategy registrations

/// Populate `registry` with the Integration (ODE) strategies.
@Sendable
public func registerODEStrategies(into registry: inout StrategyRegistry) {

    // The explicit + implicit `solveIVP` methods share one closure factory; only
    // the `ODEMethod` differs. Each returns y[0] at the final time and forwards
    // the result's diagnostics (the stiffness `outsideEnvelope` signal lives there).
    func registerSolveIVP(id: String, method: ODEMethod) {
        registry.register(id: id) { inputs in
            guard let f = odeSystem(inputs),
                  let (y0, tf) = odeProblem(inputs)
            else { return nil }
            let r = solveIVP(f, tSpan: (0.0, tf), y0: y0, method: method)
            guard let last = r.y.last, !last.isEmpty else { return nil }
            return StrategyResult(value: last[0], diagnostics: r.diagnostics)
        }
    }

    registerSolveIVP(id: "rk45", method: .rk45)
    registerSolveIVP(id: "rk23", method: .rk23)
    registerSolveIVP(id: "bdf", method: .bdf)

    // odeint — single-pass LSODA-style solve; compares y[0] at the final time.
    // `odeint` returns a bare trajectory and carries no diagnostics surface, so
    // it is only ever exercised on in-envelope (stiff-safe) cases.
    registry.register(id: "odeint") { inputs in
        guard let f = odeSystem(inputs),
              let (y0, tf) = odeProblem(inputs)
        else { return nil }
        let ys = odeint(f, y0: y0, t: [0.0, tf])
        guard let last = ys.last, !last.isEmpty else { return nil }
        return StrategyResult(value: last[0])
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the ODE domain.
///
///  - `rk45` (Dormand-Prince explicit 5(4)): a few digits at default tolerances.
///  - `rk23` (Bogacki-Shampine explicit 3(2)): lower order, looser.
///  - `bdf`  (fixed-order BDF-1 / implicit Euler): O(h) global error — widest.
///  - `odeint` (LSODA, tight tolerances): near machine precision on smooth systems.
///
/// These registry entries are the fallback envelope used when a fixture case
/// omits an explicit per-strategy `tol`; the ODE fixtures always declare `tol`,
/// so the registry mainly documents the per-strategy accuracy contract.
@Sendable
public func makeODEEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let rk45Tol: Double = tier == .trivial ? 1e-4 : tier == .hard ? 1e-3 : 1e-2
        reg.register(EnvelopeEntry(strategy: "rk45", tier: tier, maxAbsError: rk45Tol,
            description: "Dormand-Prince explicit RK45 — \(tier.rawValue) cases"))

        let rk23Tol: Double = tier == .trivial ? 1e-3 : tier == .hard ? 1e-2 : 1e-1
        reg.register(EnvelopeEntry(strategy: "rk23", tier: tier, maxAbsError: rk23Tol,
            description: "Bogacki-Shampine explicit RK23 — \(tier.rawValue) cases"))

        let bdfTol: Double = tier == .trivial ? 1e-1 : tier == .hard ? 5e-1 : 1e0
        reg.register(EnvelopeEntry(strategy: "bdf", tier: tier, maxAbsError: bdfTol,
            description: "Fixed-order BDF-1 (implicit Euler) — \(tier.rawValue) cases"))

        let odeintTol: Double = tier == .trivial ? 1e-8 : tier == .hard ? 1e-6 : 1e-3
        reg.register(EnvelopeEntry(strategy: "odeint", tier: tier, maxAbsError: odeintTol,
            description: "odeint (LSODA) — \(tier.rawValue) cases"))
    }
    return reg
}
