//
//  Complex.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Complex (arithmetic + transcendental functions).
//
//  This is a **single-strategy-per-component correctness** domain (WORKBENCH.md
//  §4, "Single-strategy domains"). A ``Complex`` result has TWO components, so
//  each operation contributes TWO strategy ids — the real part suffixed `_re`
//  and the imaginary part suffixed `_im` (e.g. `cexp_re`, `cexp_im`). The
//  comparison scalar for a `*_re` strategy is `result.re`; for `*_im` it is
//  `result.im`. Real-valued ops (`abs`, `arg`) are a single strategy id whose
//  scalar is the `Double` result. The oracle is Python `cmath`
//  (Tools/workbench_oracles/complex.py), split into `.real` / `.imag`.
//
//  ## Branch conventions
//
//  NumericSwift's ``Complex`` follows the numpy/SciPy **principal** branch (C99
//  Annex G), which is exactly `cmath`'s convention — `sqrt(-1) = +i`,
//  `log(-1) = +iπ` (CLAUDE.md design-philosophy #1, Known Limitations §2). So
//  cmath is a faithful oracle for the library's documented conventions, and the
//  comparison is genuine correctness, never vacuous (FP1 / FP3).
//
//  ## Self-awareness
//
//  Complex arithmetic + transcendentals are correctness-vs-cmath (exact up to a
//  few ULP). There is NO documented limitation envelope — principal-branch
//  results are well-defined correctness, not limitations. Accordingly EVERY
//  fixture case is in-envelope, there are ZERO out-of-envelope cases, and every
//  strategy closure returns a ``StrategyResult`` with EMPTY diagnostics — the
//  ``Complex`` operators have no `outsideEnvelope` regime to surface here, and
//  the harness never fabricates a diagnostic.
//
//  Strategy ids ↔ Sources/NumericSwift/Complex.swift:
//
//    add_re / add_im     → lhs + rhs
//    sub_re / sub_im     → lhs - rhs
//    mul_re / mul_im     → lhs * rhs
//    div_re / div_im     → lhs / rhs
//    abs                 → z.abs   (real scalar)
//    arg                 → z.arg   (real scalar)
//    cexp_re / cexp_im   → z.exp
//    clog_re / clog_im   → z.log
//    csqrt_re / csqrt_im → z.sqrt
//    csin_re / csin_im   → z.sin
//    ccos_re / ccos_im   → z.cos
//    ctan_re / ctan_im   → z.tan
//    csinh_re / csinh_im → z.sinh
//    ccosh_re / ccosh_im → z.cosh
//    ctanh_re / ctanh_im → z.tanh
//    cpow_re / cpow_im   → z.pow(w)
//
//  Inputs: `z_re`, `z_im` (unary); plus `w_re`, `w_im` for the binary ops
//  (`add`/`sub`/`mul`/`div`/`cpow`).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Complex (arithmetic + transcendental) domain suite.
    public static let complexSuite = DomainSuite(
        name: "complex",
        registerStrategies: registerComplexStrategies,
        makeEnvelopeRegistry: makeComplexEnvelopeRegistry
    )
}

// MARK: - Input resolvers

/// Extract the primary operand `z` from a fixture case's `inputs` bag.
///
/// Returns `nil` when `z_re`/`z_im` are missing — the runner records that as an
/// ERROR rather than a self-awareness verdict.
@Sendable
private func complexZ(_ inputs: [String: InputValue]) -> Complex? {
    guard let re = inputs["z_re"]?.doubleValue,
          let im = inputs["z_im"]?.doubleValue
    else { return nil }
    return Complex(re: re, im: im)
}

/// Extract the second operand `w` (binary ops only) from a fixture case.
@Sendable
private func complexW(_ inputs: [String: InputValue]) -> Complex? {
    guard let re = inputs["w_re"]?.doubleValue,
          let im = inputs["w_im"]?.doubleValue
    else { return nil }
    return Complex(re: re, im: im)
}

// MARK: - Strategy registrations

/// Populate `registry` with the Complex strategies.
///
/// Each complex-valued op registers a `_re` and an `_im` strategy that pick the
/// matching component of the same computed result. All closures return empty
/// diagnostics: the domain has no `outsideEnvelope` regime (see the file header).
@Sendable
public func registerComplexStrategies(into registry: inout StrategyRegistry) {

    // ── Helpers: register the `_re`/`_im` pair for a complex-valued op ────────

    /// Register `<id>_re` + `<id>_im` for a unary op `z -> Complex`.
    func registerUnary(_ id: String, _ op: @escaping @Sendable (Complex) -> Complex) {
        registry.register(id: "\(id)_re") { inputs in
            guard let z = complexZ(inputs) else { return nil }
            return StrategyResult(value: op(z).re)
        }
        registry.register(id: "\(id)_im") { inputs in
            guard let z = complexZ(inputs) else { return nil }
            return StrategyResult(value: op(z).im)
        }
    }

    /// Register `<id>_re` + `<id>_im` for a binary op `(z, w) -> Complex`.
    func registerBinary(_ id: String, _ op: @escaping @Sendable (Complex, Complex) -> Complex) {
        registry.register(id: "\(id)_re") { inputs in
            guard let z = complexZ(inputs), let w = complexW(inputs) else { return nil }
            return StrategyResult(value: op(z, w).re)
        }
        registry.register(id: "\(id)_im") { inputs in
            guard let z = complexZ(inputs), let w = complexW(inputs) else { return nil }
            return StrategyResult(value: op(z, w).im)
        }
    }

    // ── Arithmetic (binary) ───────────────────────────────────────────────────
    registerBinary("add") { $0 + $1 }
    registerBinary("sub") { $0 - $1 }
    registerBinary("mul") { $0 * $1 }
    registerBinary("div") { $0 / $1 }

    // ── Modulus / argument (real-valued, single id) ─────────────────────────────
    registry.register(id: "abs") { inputs in
        guard let z = complexZ(inputs) else { return nil }
        return StrategyResult(value: z.abs)
    }
    registry.register(id: "arg") { inputs in
        guard let z = complexZ(inputs) else { return nil }
        return StrategyResult(value: z.arg)
    }

    // ── Transcendentals (unary) ─────────────────────────────────────────────────
    registerUnary("cexp") { $0.exp }
    registerUnary("clog") { $0.log }
    registerUnary("csqrt") { $0.sqrt }
    registerUnary("csin") { $0.sin }
    registerUnary("ccos") { $0.cos }
    registerUnary("ctan") { $0.tan }
    registerUnary("csinh") { $0.sinh }
    registerUnary("ccosh") { $0.cosh }
    registerUnary("ctanh") { $0.tanh }

    // ── Power (binary, complex exponent: z^w = exp(w·log z)) ────────────────────
    registerBinary("cpow") { $0.pow($1) }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Complex domain.
///
/// Every Complex op here is exact up to a few ULP against `cmath`, so the
/// envelopes are uniformly tight. Arithmetic (`add`/`sub`/`mul`/`div`) is the
/// tightest (~1e-12); transcendentals carry a touch more libm rounding; `cpow`
/// compounds `exp(w·log z)` rounding so it is the loosest; the `edge` tier is
/// relaxed for large-magnitude / branch-cut cases. No strategy has an
/// out-of-envelope regime, hence no `outsideEnvelope` diagnostic is ever
/// expected (WORKBENCH.md §5).
@Sendable
public func makeComplexEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()

    let arithmetic = ["add_re", "add_im", "sub_re", "sub_im",
                      "mul_re", "mul_im", "div_re", "div_im"]
    let transcendental = ["abs", "arg",
                          "cexp_re", "cexp_im", "clog_re", "clog_im",
                          "csqrt_re", "csqrt_im",
                          "csin_re", "csin_im", "ccos_re", "ccos_im",
                          "ctan_re", "ctan_im",
                          "csinh_re", "csinh_im", "ccosh_re", "ccosh_im",
                          "ctanh_re", "ctanh_im"]
    let power = ["cpow_re", "cpow_im"]

    for tier: CaseTier in [.trivial, .hard, .edge] {
        let edgeRelax: Double = tier == .edge ? 1e-9 : 1e-12
        for strategy in arithmetic {
            reg.register(EnvelopeEntry(
                strategy: strategy, tier: tier, maxAbsError: edgeRelax,
                description: "\(strategy) — Complex arithmetic, exact vs cmath (\(tier.rawValue))"))
        }
        for strategy in transcendental {
            let tol: Double = tier == .edge ? 1e-9 : 1e-12
            reg.register(EnvelopeEntry(
                strategy: strategy, tier: tier, maxAbsError: tol,
                description: "\(strategy) — Complex transcendental vs cmath (\(tier.rawValue))"))
        }
        for strategy in power {
            // cpow = exp(w·log z): compounded rounding → a touch looser.
            let tol: Double = tier == .edge ? 1e-8 : 1e-9
            reg.register(EnvelopeEntry(
                strategy: strategy, tier: tier, maxAbsError: tol,
                description: "\(strategy) — Complex power exp(w·log z) vs cmath (\(tier.rawValue))"))
        }
    }
    return reg
}
