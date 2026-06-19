//
//  Series.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Series (polynomials, Taylor series, summation).
//
//  Mirrors the reference `Integration.swift` / `Statistics.swift` suites:
//    1. `seriesSuite` — the `DomainSuite` wiring the strategy + envelope registries.
//    2. `registerSeriesStrategies(into:)` — one closure per `Series.*` function.
//       Each closure extracts the fixture `inputs` bag and delegates to the
//       NAMESPACED `Series.*` API. The comparison scalar is the function's scalar
//       output; for the array-returning polynomial functions (polyder, polyint,
//       polyadd, polymul) the resulting polynomial is evaluated at the fixture's
//       `x` and THAT scalar is compared (WORKBENCH.md §4). Closures forward any
//       ``NumericDiagnostic`` the library produced — they never fabricate one.
//    3. `makeSeriesEnvelopeRegistry()` — per-(strategy, tier) accuracy bounds.
//
//  ## Self-awareness
//
//  This domain is mostly **single-strategy-per-function correctness** vs numpy /
//  sympy. It has ONE out-of-envelope source: the `taylor` strategy applied to
//  `tan` with more terms than the generator supports. `Series.knownTaylorSeries`
//  hard-codes `tan` coefficients only to index 11 (12 terms); requesting more
//  silently returns 0 for every higher index — including the genuinely non-zero
//  x¹³ coefficient — so the truncated series is materially wrong near x = ±π/2
//  (CLAUDE.md "Code Review Findings → Series.swift"). The `taylor` closure
//  delegates to `Series.taylorEvalDiagnosed`, which emits an
//  ``NumericDiagnostic/outsideEnvelope`` for those over-the-limit requests and
//  stays silent for well-supported ones. All other strategies are exact /
//  closed-form and never warn.
//
//  Strategy ids ↔ `Series.*` (Sources/NumericSwift/Series.swift):
//
//    polyval    → Series.polyval(coeffs, at: x)              [ascending coeffs]
//    polyder    → Series.polyval(Series.polyder(coeffs), at: x)
//    polyint    → Series.polyval(Series.polyint(coeffs), at: x)
//    polyadd    → Series.polyval(Series.polyadd(p, q), at: x)
//    polymul    → Series.polyval(Series.polymul(p, q), at: x)
//    seriesSum  → Series.seriesSum(from:to:term:).sum        [term tag → generator]
//    divdiff    → Series.dividedDifferences(xs:ys:).last     [leading coefficient]
//    taylor     → Series.taylorEvalDiagnosed(func, at: x, terms:)  [out-of-envelope]
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Series (polynomials, Taylor series, summation) domain suite.
    public static let seriesSuite = DomainSuite(
        name: "series",
        registerStrategies: registerSeriesStrategies,
        makeEnvelopeRegistry: makeSeriesEnvelopeRegistry
    )
}

// MARK: - Input resolvers

/// Extract a `[Double]` coefficient array from a fixture inputs bag under `key`.
///
/// Returns `nil` when the key is missing or any element is non-numeric — the
/// runner records that as an ERROR rather than a self-awareness verdict.
@Sendable
private func seriesArray(_ inputs: [String: InputValue], _ key: String) -> [Double]? {
    guard let raw = inputs[key]?.arrayValue else { return nil }
    let values = raw.compactMap(\.doubleValue)
    return values.count == raw.count ? values : nil
}

/// Term generators for the `seriesSum` strategy, keyed by the fixture `tag`.
///
/// Each must match the corresponding closed-form series in `series.py`
/// (`series_sum_oracle`) exactly: same index convention, same summand.
@Sendable
private func seriesSumTerm(_ tag: String?) -> (@Sendable (Int) -> Double)? {
    switch tag {
    case "inv_square":     return { n in 1.0 / (Double(n) * Double(n)) }
    case "geometric_half": return { n in Foundation.pow(0.5, Double(n)) }
    case "inv_factorial":  return { n in 1.0 / NumberTheory.factorial(n) }
    case "alt_harmonic":   return { n in (n % 2 == 1 ? 1.0 : -1.0) / Double(n) }
    case "inv_fourth":     return { n in 1.0 / Foundation.pow(Double(n), 4) }
    default:               return nil
    }
}

// MARK: - Strategy registrations

/// Populate `registry` with the Series strategies (one per `Series.*` function).
@Sendable
public func registerSeriesStrategies(into registry: inout StrategyRegistry) {

    // polyval — Horner evaluation of an ascending-order coefficient array at x.
    registry.register(id: "polyval") { inputs in
        guard let coeffs = seriesArray(inputs, "coeffs"),
              let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: Series.polyval(coeffs, at: x))
    }

    // polyder — differentiate, then evaluate the derivative polynomial at x.
    registry.register(id: "polyder") { inputs in
        guard let coeffs = seriesArray(inputs, "coeffs"),
              let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: Series.polyval(Series.polyder(coeffs), at: x))
    }

    // polyint — integrate (constant 0), then evaluate the integral polynomial at x.
    registry.register(id: "polyint") { inputs in
        guard let coeffs = seriesArray(inputs, "coeffs"),
              let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: Series.polyval(Series.polyint(coeffs), at: x))
    }

    // polyadd — add two polynomials, then evaluate the sum at x.
    registry.register(id: "polyadd") { inputs in
        guard let p = seriesArray(inputs, "p"),
              let q = seriesArray(inputs, "q"),
              let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: Series.polyval(Series.polyadd(p, q), at: x))
    }

    // polymul — multiply two polynomials, then evaluate the product at x.
    registry.register(id: "polymul") { inputs in
        guard let p = seriesArray(inputs, "p"),
              let q = seriesArray(inputs, "q"),
              let x = inputs["x"]?.doubleValue
        else { return nil }
        return StrategyResult(value: Series.polyval(Series.polymul(p, q), at: x))
    }

    // seriesSum — finite sum of a tagged convergent series from `from` to `to`.
    registry.register(id: "seriesSum") { inputs in
        guard let from = inputs["from"]?.intValue,
              let to = inputs["to"]?.intValue,
              let term = seriesSumTerm(inputs["tag"]?.stringValue)
        else { return nil }
        let r = Series.seriesSum(from: from, to: to, term: term)
        return StrategyResult(value: r.sum)
    }

    // divdiff — leading (top-order) Newton divided-difference coefficient.
    registry.register(id: "divdiff") { inputs in
        guard let xs = seriesArray(inputs, "xs"),
              let ys = seriesArray(inputs, "ys")
        else { return nil }
        let dd = Series.dividedDifferences(xs: xs, ys: ys)
        guard let leading = dd.last else { return nil }
        return StrategyResult(value: leading)
    }

    // taylor — evaluate a known Taylor series at x using `terms` coefficients.
    // The diagnosed overload emits `outsideEnvelope` when `terms` exceeds the
    // generator's support (tan: 12); the closure forwards that diagnostic.
    registry.register(id: "taylor") { inputs in
        guard let function = inputs["func"]?.stringValue,
              let x = inputs["x"]?.doubleValue,
              let terms = inputs["terms"]?.intValue,
              let r = Series.taylorEvalDiagnosed(function, at: x, terms: terms)
        else { return nil }
        return StrategyResult(value: r.value, diagnostics: r.diagnostics)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Series domain.
///
///  - polyval/polyder/polyint/polyadd/polymul: exact polynomial arithmetic vs
///    `numpy.polynomial` — tight, with a slightly looser hard tier for the
///    higher-degree random polynomials (Horner rounding) and a wide edge tier
///    for large-x cancellation.
///  - seriesSum: exact partial sum vs a high-precision sympy reference for the
///    SAME finite N — limited only by floating-point summation order.
///  - divdiff: divided-difference recursion accumulates rounding; a touch looser.
///  - taylor: matches the true function value when the series is well-supported
///    and inside its radius of convergence; the out-of-envelope tan cases are
///    judged by the self-awareness gate (diagnostic presence), not numeric tol,
///    so their per-case tol is huge in the fixture.
@Sendable
public func makeSeriesEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {

        let polyTol: Double = tier == .trivial ? 1e-12 : tier == .hard ? 1e-9 : 1e-3
        for poly in ["polyval", "polyder", "polyint", "polyadd", "polymul"] {
            reg.register(EnvelopeEntry(strategy: poly, tier: tier, maxAbsError: polyTol,
                description: "Polynomial arithmetic (\(poly)) — \(tier.rawValue) cases"))
        }

        let sumTol: Double = tier == .trivial ? 1e-12 : tier == .hard ? 1e-9 : 1e-9
        reg.register(EnvelopeEntry(strategy: "seriesSum", tier: tier, maxAbsError: sumTol,
            description: "Finite series summation — \(tier.rawValue) cases"))

        let ddTol: Double = tier == .trivial ? 1e-12 : tier == .hard ? 1e-8 : 1e-6
        reg.register(EnvelopeEntry(strategy: "divdiff", tier: tier, maxAbsError: ddTol,
            description: "Newton divided differences (leading) — \(tier.rawValue) cases"))

        let taylorTol: Double = tier == .trivial ? 1e-9 : tier == .hard ? 1e-6 : 1e-3
        reg.register(EnvelopeEntry(strategy: "taylor", tier: tier, maxAbsError: taylorTol,
            description: "Known Taylor-series evaluation — \(tier.rawValue) cases"))
    }
    return reg
}
