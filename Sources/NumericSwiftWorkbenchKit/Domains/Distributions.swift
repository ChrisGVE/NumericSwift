//
//  Distributions.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Distributions (probability functions).
//
//  Mirrors the reference `Integration.swift` suite, but this is a
//  SINGLE-STRATEGY-PER-FUNCTION correctness domain (WORKBENCH.md §4): each
//  distribution function is its own strategy id — `normal_cdf`, `t_ppf`,
//  `chi2_pdf`, … — and the comparison scalar is that function's scalar output,
//  checked against scipy.stats.
//
//    1. `distributionsSuite` — the `DomainSuite` wiring the strategy + envelope
//       registries.
//    2. `registerDistributionsStrategies(into:)` — one closure per strategy id.
//       Each reads the fixture inputs bag (`family`, `func`, params, and the
//       evaluation point `x` or `p`), constructs the matching NumericSwift
//       distribution, and calls the requested function. Closures FORWARD any
//       ``NumericDiagnostic`` the library emitted — they never fabricate one.
//    3. `makeDistributionsEnvelopeRegistry()` — per-(strategy, tier) bounds.
//
//  ## Self-awareness
//
//  The one documented limitation (CLAUDE.md *Known Limitations §1*) is the
//  Student-t `ppf` extreme tails: only ~5 digits for `p < 1e-4` or `p > 0.9999`.
//  Those cases are tagged `inEnvelope: {"t_ppf": false}` in the fixture. The
//  `t_ppf` closure calls ``TDistribution/ppfDiagnosed(_:)`` (NOT the bare
//  ``TDistribution/ppf(_:)``) so the tail diagnostic propagates; the library —
//  not the harness — decides when to emit it. Every other function/case is
//  in-envelope (a correctness-vs-scipy gate), and t_ppf central cases must NOT
//  warn.
//
//  Inputs (carried through the JSON `inputs` bag, see ``InputValue``):
//    • `family` — distribution family tag (`normal`, `t`, `chi2`, `f`, `gamma`,
//                 `beta`, `expon`, `uniform`).
//    • `func`   — function name (`cdf`, `pdf`, `ppf`).
//    • `x` / `p`— scalar evaluation point (`x` for cdf/pdf, `p` for ppf).
//    • params   — distribution parameters: `df`, `dfn`, `dfd`, `shape`, `a`,
//                 `b`, `loc`, `scale` (each optional, family-dependent).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Distributions domain suite.
    public static let distributionsSuite = DomainSuite(
        name: "distributions",
        registerStrategies: registerDistributionsStrategies,
        makeEnvelopeRegistry: makeDistributionsEnvelopeRegistry
    )
}

// MARK: - Input resolver

/// Common parameters pulled from a fixture inputs bag.
private struct DistInputs {
    let family: String
    let funcName: String
    let arg: Double          // x for cdf/pdf, p for ppf
    let loc: Double
    let scale: Double
    let df: Double?
    let dfn: Double?
    let dfd: Double?
    let shape: Double?
    let a: Double?
    let b: Double?
}

/// Extract the distribution inputs from a fixture bag, or `nil` when malformed.
@Sendable
private func distInputs(_ inputs: [String: InputValue]) -> DistInputs? {
    guard let family = inputs["family"]?.stringValue,
          let funcName = inputs["func"]?.stringValue
    else { return nil }
    // cdf/pdf carry `x`; ppf carries `p`.
    let arg: Double?
    if funcName == "ppf" {
        arg = inputs["p"]?.doubleValue
    } else {
        arg = inputs["x"]?.doubleValue
    }
    guard let argValue = arg else { return nil }
    return DistInputs(
        family: family,
        funcName: funcName,
        arg: argValue,
        loc: inputs["loc"]?.doubleValue ?? 0.0,
        scale: inputs["scale"]?.doubleValue ?? 1.0,
        df: inputs["df"]?.doubleValue,
        dfn: inputs["dfn"]?.doubleValue,
        dfd: inputs["dfd"]?.doubleValue,
        shape: inputs["shape"]?.doubleValue,
        a: inputs["a"]?.doubleValue,
        b: inputs["b"]?.doubleValue
    )
}

// MARK: - Per-family evaluation

/// Evaluate a non-t family's cdf/pdf/ppf, returning a bare scalar (no library
/// diagnostics defined for these — they are all in-envelope correctness cases).
@Sendable
private func evalNonT(_ d: DistInputs) -> Double? {
    switch d.family {
    case "normal":
        let dist = NormalDistribution(loc: d.loc, scale: d.scale)
        return scalarFor(d.funcName, dist.cdf, dist.pdf, dist.ppf, d.arg)
    case "uniform":
        let dist = UniformDistribution(loc: d.loc, scale: d.scale)
        return scalarFor(d.funcName, dist.cdf, dist.pdf, dist.ppf, d.arg)
    case "expon":
        let dist = ExponentialDistribution(loc: d.loc, scale: d.scale)
        return scalarFor(d.funcName, dist.cdf, dist.pdf, dist.ppf, d.arg)
    case "chi2":
        guard let df = d.df else { return nil }
        let dist = ChiSquaredDistribution(df: df, loc: d.loc, scale: d.scale)
        return scalarFor(d.funcName, dist.cdf, dist.pdf, dist.ppf, d.arg)
    case "f":
        guard let dfn = d.dfn, let dfd = d.dfd else { return nil }
        let dist = FDistribution(dfn: dfn, dfd: dfd, loc: d.loc, scale: d.scale)
        return scalarFor(d.funcName, dist.cdf, dist.pdf, dist.ppf, d.arg)
    case "gamma":
        guard let shape = d.shape else { return nil }
        let dist = GammaDistribution(shape: shape, loc: d.loc, scale: d.scale)
        return scalarFor(d.funcName, dist.cdf, dist.pdf, dist.ppf, d.arg)
    case "beta":
        guard let a = d.a, let b = d.b else { return nil }
        let dist = BetaDistribution(a: a, b: b, loc: d.loc, scale: d.scale)
        return scalarFor(d.funcName, dist.cdf, dist.pdf, dist.ppf, d.arg)
    default:
        return nil
    }
}

/// Pick the cdf/pdf/ppf member by name and apply it to `arg`.
@inline(__always)
private func scalarFor(
    _ funcName: String,
    _ cdf: (Double) -> Double,
    _ pdf: (Double) -> Double,
    _ ppf: (Double) -> Double,
    _ arg: Double
) -> Double? {
    switch funcName {
    case "cdf": return cdf(arg)
    case "pdf": return pdf(arg)
    case "ppf": return ppf(arg)
    default: return nil
    }
}

// MARK: - Strategy registrations

/// Populate `registry` with the Distributions strategies.
///
/// Strategy ids are `<family>_<func>`. The non-t families share `evalNonT`;
/// the t family splits ppf (the diagnosed path) from cdf/pdf.
@Sendable
public func registerDistributionsStrategies(into registry: inout StrategyRegistry) {

    // ── Non-t families: one closure per (family, func) — all bare scalars. ────
    let nonTFamilies = ["normal", "uniform", "expon", "chi2", "f", "gamma", "beta"]
    for family in nonTFamilies {
        for funcName in ["cdf", "pdf", "ppf"] {
            registry.register(id: "\(family)_\(funcName)") { inputs in
                guard let d = distInputs(inputs),
                      d.family == family, d.funcName == funcName,
                      let v = evalNonT(d)
                else { return nil }
                return StrategyResult(value: v)
            }
        }
    }

    // ── Student-t cdf/pdf: bare scalars (no documented envelope). ─────────────
    registry.register(id: "t_cdf") { inputs in
        guard let d = distInputs(inputs), d.funcName == "cdf", let df = d.df else { return nil }
        let dist = TDistribution(df: df, loc: d.loc, scale: d.scale)
        return StrategyResult(value: dist.cdf(d.arg))
    }
    registry.register(id: "t_pdf") { inputs in
        guard let d = distInputs(inputs), d.funcName == "pdf", let df = d.df else { return nil }
        let dist = TDistribution(df: df, loc: d.loc, scale: d.scale)
        return StrategyResult(value: dist.pdf(d.arg))
    }

    // ── Student-t ppf: the DIAGNOSED path — forwards the extreme-tail
    //    outsideEnvelope diagnostic the library emits for |p| > 0.9999. ────────
    registry.register(id: "t_ppf") { inputs in
        guard let d = distInputs(inputs), d.funcName == "ppf", let df = d.df else { return nil }
        let dist = TDistribution(df: df, loc: d.loc, scale: d.scale)
        let r = dist.ppfDiagnosed(d.arg)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Distributions domain.
///
/// cdf/pdf are closed-form-ish (incomplete beta/gamma, Darwin transcendentals)
/// and reach near-machine precision; ppf is Newton-Raphson and a touch looser.
/// The fixture's per-case `tol` is authoritative (WORKBENCH.md §5); this
/// registry is the fallback when a case omits a tol for a strategy.
@Sendable
public func makeDistributionsEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    let families = ["normal", "uniform", "expon", "t", "chi2", "f", "gamma", "beta"]
    for tier: CaseTier in [.trivial, .hard, .edge] {
        for family in families {
            // cdf / pdf — tight everywhere.
            let cdfTol: Double = tier == .trivial ? 1e-12 : 1e-9
            reg.register(EnvelopeEntry(strategy: "\(family)_cdf", tier: tier, maxAbsError: cdfTol,
                description: "\(family) cdf vs scipy.stats — \(tier.rawValue) cases"))
            reg.register(EnvelopeEntry(strategy: "\(family)_pdf", tier: tier, maxAbsError: cdfTol,
                description: "\(family) pdf vs scipy.stats — \(tier.rawValue) cases"))

            // ppf — Newton-Raphson; full precision in-envelope, ~5 digits in the
            // documented t-ppf extreme tail (edge tier).
            let ppfTol: Double
            if family == "t", tier == .edge {
                ppfTol = 1e-4   // documented extreme-tail accuracy
            } else {
                ppfTol = tier == .trivial ? 1e-9 : 1e-6
            }
            reg.register(EnvelopeEntry(strategy: "\(family)_ppf", tier: tier, maxAbsError: ppfTol,
                description: "\(family) ppf vs scipy.stats — \(tier.rawValue) cases"))
        }
    }
    return reg
}
