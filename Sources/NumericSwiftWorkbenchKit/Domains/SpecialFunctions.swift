//
//  SpecialFunctions.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: SpecialFunctions (scalar special functions).
//
//  Single-strategy-per-function domain: each special function is its own
//  strategy id and the comparison scalar is that function's output at a scalar
//  argument. The oracle is scipy.special, so this is a correctness-vs-scipy gate
//  (FP1) — non-vacuous because no value is ever sourced from NumericSwift.
//
//  Mirrors the reference `Integration.swift` suite:
//    1. `specialfunctionsSuite` — the `DomainSuite` wiring the strategy +
//       envelope registries.
//    2. `registerSpecialFunctionsStrategies(into:)` — one closure per function.
//       Each closure extracts its argument(s) from the fixture `inputs` bag and
//       delegates to NumericSwift, FORWARDING whatever ``NumericDiagnostic`` the
//       library produced — it never fabricates one.
//    3. `makeSpecialFunctionsEnvelopeRegistry()` — per-(strategy, tier) bounds.
//
//  ## Self-awareness
//
//  Exactly one strategy has out-of-envelope cases: `erfinv` in its documented
//  extreme far tail (|x| > 1 − 1e-11; CLAUDE.md "Known Limitations" §1). There
//  the `erfinv` closure delegates to ``erfinvDiagnosed(_:)`` and forwards the
//  ``NumericDiagnostic/outsideEnvelope(method:reason:)`` it emits. All other
//  functions are accurate vs scipy across their whole tested domain, so every
//  one of their cases is in-envelope and they emit no diagnostic.
//
//  Inputs (carried through the JSON `inputs` bag, see ``InputValue``):
//    • `func`        — the strategy/function name (provenance; also selects the
//                      integer Bessel order for the n-ary Bessel functions).
//    • `x`           — the scalar argument (1-ary functions).
//    • `a`, `b`, `x` — beta / incomplete-beta arguments.
//    • `a`, `x`      — incomplete-gamma arguments.
//    • `n`, `x`      — integer-order Bessel arguments.
//    • `m`           — elliptic-integral parameter.
//    • `s`           — zeta argument.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The SpecialFunctions domain suite.
    public static let specialfunctionsSuite = DomainSuite(
        name: "specialfunctions",
        registerStrategies: registerSpecialFunctionsStrategies,
        makeEnvelopeRegistry: makeSpecialFunctionsEnvelopeRegistry
    )
}

// MARK: - Strategy registrations

/// Populate `registry` with the SpecialFunctions strategies.
///
/// Each closure reads exactly the inputs its function needs and returns the
/// scalar result. Only `erfinv` carries a diagnostic (its documented far-tail
/// limitation); every other function returns an empty diagnostic list.
@Sendable
public func registerSpecialFunctionsStrategies(into registry: inout StrategyRegistry) {

    // ── Error functions ───────────────────────────────────────────────────────
    // NOTE: `erf`/`erfc`/`j0`/`j1`/`jn`/`y0`/`y1`/`yn` are also declared by
    // Darwin/Foundation, so they are fully qualified (`NumericSwift.…`) to bind
    // the library overload.
    registry.register(id: "erf") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: NumericSwift.erf(x))
    }
    registry.register(id: "erfc") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: NumericSwift.erfc(x))
    }
    // erfinv — delegates to the diagnosed overload so the documented extreme
    // far-tail limitation surfaces as an `outsideEnvelope` diagnostic.
    registry.register(id: "erfinv") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        let r = erfinvDiagnosed(x)
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }
    registry.register(id: "erfcinv") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: erfcinv(x))
    }

    // ── Beta family ─────────────────────────────────────────────────────────
    registry.register(id: "beta") { inputs in
        guard let a = inputs["a"]?.doubleValue, let b = inputs["b"]?.doubleValue
        else { return nil }
        return StrategyResult(value: beta(a, b))
    }
    registry.register(id: "betainc") { inputs in
        guard let a = inputs["a"]?.doubleValue, let b = inputs["b"]?.doubleValue,
              let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: betainc(a, b, x))
    }

    // ── Gamma family ──────────────────────────────────────────────────────────
    registry.register(id: "digamma") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: digamma(x))
    }
    registry.register(id: "gammainc") { inputs in
        guard let a = inputs["a"]?.doubleValue, let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: gammainc(a, x))
    }
    registry.register(id: "gammaincc") { inputs in
        guard let a = inputs["a"]?.doubleValue, let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: gammaincc(a, x))
    }

    // ── Bessel functions (first/second kind, integer order) ───────────────────
    registry.register(id: "besselj0") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: NumericSwift.j0(x))
    }
    registry.register(id: "besselj1") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: NumericSwift.j1(x))
    }
    registry.register(id: "besseljn") { inputs in
        guard let n = inputs["n"]?.intValue, let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: NumericSwift.jn(n, x))
    }
    registry.register(id: "bessely0") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: NumericSwift.y0(x))
    }
    registry.register(id: "bessely1") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: NumericSwift.y1(x))
    }
    registry.register(id: "besselyn") { inputs in
        guard let n = inputs["n"]?.intValue, let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: NumericSwift.yn(n, x))
    }

    // ── Modified Bessel functions (integer order) ─────────────────────────────
    registry.register(id: "besseli") { inputs in
        guard let n = inputs["n"]?.intValue, let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: besseli(n, x))
    }
    registry.register(id: "besselk") { inputs in
        guard let n = inputs["n"]?.intValue, let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: besselk(n, x))
    }

    // ── Elliptic integrals ────────────────────────────────────────────────────
    registry.register(id: "ellipk") { inputs in
        guard let m = inputs["m"]?.doubleValue else { return nil }
        return StrategyResult(value: ellipk(m))
    }
    registry.register(id: "ellipe") { inputs in
        guard let m = inputs["m"]?.doubleValue else { return nil }
        return StrategyResult(value: ellipe(m))
    }

    // ── Riemann zeta ──────────────────────────────────────────────────────────
    registry.register(id: "zeta") { inputs in
        guard let s = inputs["s"]?.doubleValue else { return nil }
        return StrategyResult(value: zeta(s))
    }

    // ── Lambert W (principal branch) ──────────────────────────────────────────
    registry.register(id: "lambertw") { inputs in
        guard let x = inputs["x"]?.doubleValue else { return nil }
        return StrategyResult(value: lambertw(x))
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the SpecialFunctions domain.
///
/// NumericSwift matches scipy.special to ~12 digits for most of these functions;
/// the looser bounds reflect series / AGM / continued-fraction conditioning at
/// the harder arguments, NOT a self-awareness limitation. Those deviations are
/// reported flags only — the hard gate is the `erfinv` extreme-tail diagnostic.
/// Tiers widen progressively (trivial → hard → edge) to keep the reported
/// in-envelope checks honest without masking a real regression.
@Sendable
public func makeSpecialFunctionsEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()

    // (trivial, hard, edge) absolute-error bounds per strategy.
    let bounds: [String: (Double, Double, Double)] = [
        "erf": (1e-13, 1e-12, 1e-12),
        "erfc": (1e-13, 1e-12, 1e-12),
        "erfinv": (1e-11, 1e-9, 1e-7),
        "erfcinv": (1e-11, 1e-9, 1e-7),
        "beta": (1e-10, 1e-9, 1e-8),
        "betainc": (1e-11, 1e-10, 1e-9),
        "digamma": (1e-10, 1e-9, 1e-8),
        "gammainc": (1e-11, 1e-10, 1e-9),
        "gammaincc": (1e-11, 1e-10, 1e-9),
        "besselj0": (1e-12, 1e-12, 1e-11),
        "besselj1": (1e-12, 1e-12, 1e-11),
        "besseljn": (1e-12, 1e-12, 1e-11),
        "bessely0": (1e-12, 1e-12, 1e-11),
        "bessely1": (1e-12, 1e-12, 1e-11),
        "besselyn": (1e-12, 1e-12, 1e-11),
        "besseli": (1e-8, 1e-6, 1e-5),
        "besselk": (1e-10, 1e-9, 1e-8),
        "ellipk": (1e-12, 1e-11, 1e-10),
        "ellipe": (1e-12, 1e-11, 1e-10),
        "zeta": (1e-8, 1e-5, 1e-4),
        "lambertw": (1e-10, 1e-9, 1e-8),
    ]

    for (strategy, b) in bounds {
        for tier: CaseTier in [.trivial, .hard, .edge] {
            let tol = tier == .trivial ? b.0 : tier == .hard ? b.1 : b.2
            reg.register(EnvelopeEntry(
                strategy: strategy, tier: tier, maxAbsError: tol,
                description: "\(strategy) vs scipy.special — \(tier.rawValue) cases"))
        }
    }
    return reg
}
