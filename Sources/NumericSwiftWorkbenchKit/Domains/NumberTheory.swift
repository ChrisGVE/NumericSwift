//
//  NumberTheory.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: NumberTheory (number-theoretic & combinatorial).
//
//  This is a **single-strategy-per-function correctness** domain (WORKBENCH.md
//  §4): each `NumberTheory.*` function IS a strategy id, the comparison scalar is
//  that function's output for the case's integer argument(s), and the oracle is
//  the matching sympy / math reference (Tools/workbench_oracles/numbertheory.py).
//
//  ## Self-awareness
//
//  Number theory is EXACT integer arithmetic, so EVERY fixture case is
//  in-envelope. There are ZERO out-of-envelope cases: the gate is a pure
//  correctness-vs-sympy check. Accordingly, every strategy closure returns a
//  ``StrategyResult`` with EMPTY diagnostics — the `NumberTheory.*` functions
//  have no documented limitation envelope to surface here, and the harness never
//  fabricates a diagnostic.
//
//  ## List-returning functions — chosen comparison scalar
//
//  `primeFactors` and `primesUpTo` return arrays; the workbench compares one
//  deterministic scalar (matching the oracle in numbertheory.py):
//    • primeFactorsCount → Ω(n), the sum of exponents (count WITH multiplicity).
//    • primesUpToCount   → π(n), the number of primes ≤ n.
//
//  Strategy ids ↔ `NumberTheory.*` (Sources/NumericSwift/NumberTheory.swift):
//
//    isPrime           → NumberTheory.isPrime           (1.0 / 0.0)
//    primeFactorsCount → NumberTheory.primeFactors      (Σ exponents)
//    primesUpToCount   → NumberTheory.primesUpTo        (count)
//    factorial         → NumberTheory.factorial
//    gcd               → NumberTheory.gcd
//    lcm               → NumberTheory.lcm
//    eulerPhi          → NumberTheory.eulerPhi
//    divisorSigma      → NumberTheory.divisorSigma(_:k:)
//    mobius            → NumberTheory.mobius            (+1 / 0 / -1)
//    liouville         → NumberTheory.liouville         (+1 / -1)
//    carmichael        → NumberTheory.carmichael
//    vonMangoldt       → NumberTheory.vonMangoldt       (float; tol 1e-9)
//    modPow            → NumberTheory.modPow
//    comb              → NumberTheory.comb
//    perm              → NumberTheory.perm
//    modInverse        → NumberTheory.modInverse
//    digitSum          → NumberTheory.digitSum
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The NumberTheory (number-theoretic & combinatorial) domain suite.
    public static let numbertheorySuite = DomainSuite(
        name: "numbertheory",
        registerStrategies: registerNumberTheoryStrategies,
        makeEnvelopeRegistry: makeNumberTheoryEnvelopeRegistry
    )
}

// MARK: - Strategy registrations

/// Populate `registry` with the NumberTheory strategies (one per function).
@Sendable
public func registerNumberTheoryStrategies(into registry: inout StrategyRegistry) {

    // ── Primality & factorization ─────────────────────────────────────────
    registry.register(id: "isPrime") { inputs in
        guard let n = inputs["n"]?.intValue else { return nil }
        return StrategyResult(value: NumberTheory.isPrime(n) ? 1.0 : 0.0)
    }

    // primeFactorsCount → Ω(n): sum of exponents (count with multiplicity).
    registry.register(id: "primeFactorsCount") { inputs in
        guard let n = inputs["n"]?.intValue else { return nil }
        let omega = NumberTheory.primeFactors(n).reduce(0) { $0 + $1.exponent }
        return StrategyResult(value: Double(omega))
    }

    // primesUpToCount → π(n): number of primes ≤ n.
    registry.register(id: "primesUpToCount") { inputs in
        guard let n = inputs["n"]?.intValue else { return nil }
        return StrategyResult(value: Double(NumberTheory.primesUpTo(n).count))
    }

    // ── GCD / LCM ─────────────────────────────────────────────────────────
    registry.register(id: "gcd") { inputs in
        guard let a = inputs["a"]?.intValue, let b = inputs["b"]?.intValue else { return nil }
        return StrategyResult(value: Double(NumberTheory.gcd(a, b)))
    }

    registry.register(id: "lcm") { inputs in
        guard let a = inputs["a"]?.intValue, let b = inputs["b"]?.intValue else { return nil }
        return StrategyResult(value: Double(NumberTheory.lcm(a, b)))
    }

    // ── Arithmetic functions ──────────────────────────────────────────────
    registry.register(id: "eulerPhi") { inputs in
        guard let n = inputs["n"]?.intValue, let phi = NumberTheory.eulerPhi(n) else { return nil }
        return StrategyResult(value: Double(phi))
    }

    // divisorSigma honours the optional `k` input (default 1).
    registry.register(id: "divisorSigma") { inputs in
        guard let n = inputs["n"]?.intValue else { return nil }
        let k = inputs["k"]?.intValue ?? 1
        guard let sigma = NumberTheory.divisorSigma(n, k: k) else { return nil }
        return StrategyResult(value: sigma)
    }

    registry.register(id: "mobius") { inputs in
        guard let n = inputs["n"]?.intValue, let mu = NumberTheory.mobius(n) else { return nil }
        return StrategyResult(value: Double(mu))
    }

    registry.register(id: "liouville") { inputs in
        guard let n = inputs["n"]?.intValue, let lambda = NumberTheory.liouville(n) else { return nil }
        return StrategyResult(value: Double(lambda))
    }

    registry.register(id: "carmichael") { inputs in
        guard let n = inputs["n"]?.intValue, let lambda = NumberTheory.carmichael(n) else { return nil }
        return StrategyResult(value: Double(lambda))
    }

    // vonMangoldt is float-valued (log p for prime powers, else 0).
    registry.register(id: "vonMangoldt") { inputs in
        guard let n = inputs["n"]?.intValue else { return nil }
        return StrategyResult(value: NumberTheory.vonMangoldt(n))
    }

    // ── Modular arithmetic ─────────────────────────────────────────────────
    registry.register(id: "modPow") { inputs in
        guard let base = inputs["base"]?.intValue,
              let exp = inputs["exp"]?.intValue,
              let m = inputs["m"]?.intValue
        else { return nil }
        return StrategyResult(value: Double(NumberTheory.modPow(base, exp, m)))
    }

    registry.register(id: "modInverse") { inputs in
        guard let a = inputs["a"]?.intValue, let m = inputs["m"]?.intValue,
              let inv = NumberTheory.modInverse(a, m)
        else { return nil }
        return StrategyResult(value: Double(inv))
    }

    // ── Combinatorics ──────────────────────────────────────────────────────
    registry.register(id: "factorial") { inputs in
        guard let n = inputs["n"]?.intValue else { return nil }
        return StrategyResult(value: NumberTheory.factorial(n))
    }

    registry.register(id: "comb") { inputs in
        guard let n = inputs["n"]?.intValue, let k = inputs["k"]?.intValue else { return nil }
        return StrategyResult(value: NumberTheory.comb(n, k))
    }

    registry.register(id: "perm") { inputs in
        guard let n = inputs["n"]?.intValue, let k = inputs["k"]?.intValue else { return nil }
        return StrategyResult(value: NumberTheory.perm(n, k))
    }

    // ── Digit functions ────────────────────────────────────────────────────
    registry.register(id: "digitSum") { inputs in
        guard let n = inputs["n"]?.intValue else { return nil }
        return StrategyResult(value: Double(NumberTheory.digitSum(n)))
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the NumberTheory domain.
///
/// Every function here is exact integer arithmetic against sympy / math, so the
/// envelopes are bit-exact (0.0) for the integer-valued strategies. The two
/// float-valued strategies — `vonMangoldt` (a transcendental `log p`) and
/// `divisorSigma` (built from `pow`) — get a small absolute tolerance (1e-9) to
/// absorb benign floating-point rounding. No strategy has an out-of-envelope
/// regime, hence no `outsideEnvelope` diagnostic is ever expected
/// (WORKBENCH.md §5).
@Sendable
public func makeNumberTheoryEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    // Float-valued strategies tolerate benign rounding; the rest are bit-exact.
    let floatValued: Set<String> = ["vonMangoldt", "divisorSigma"]
    let strategies = [
        "isPrime", "primeFactorsCount", "primesUpToCount", "gcd", "lcm",
        "eulerPhi", "divisorSigma", "mobius", "liouville", "carmichael",
        "vonMangoldt", "modPow", "modInverse", "factorial", "comb", "perm",
        "digitSum",
    ]
    for strategy in strategies {
        let tol = floatValued.contains(strategy) ? 1e-9 : 0.0
        for tier: CaseTier in [.trivial, .hard, .edge] {
            reg.register(EnvelopeEntry(
                strategy: strategy,
                tier: tier,
                maxAbsError: tol,
                description: "\(strategy) — exact vs sympy/math (\(tier.rawValue) cases)"))
        }
    }
    return reg
}
