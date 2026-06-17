//
//  Strategy.swift
//  NumericSwiftWorkbench
//
//  Strategy-registry types for the NumericSwift E2E functional workbench.
//
//  A "strategy" is an identified NumericSwift algorithm that can be applied to
//  a ``WorkbenchCase`` input bag and returns a scalar `Double` result plus any
//  ``NumericDiagnostic`` entries the algorithm emitted.
//
//  ## Architecture
//
//  The registry is a `[String: StrategyFn]` dictionary. Each domain suite
//  (wave 2, under `Sources/NumericSwiftWorkbench/Domains/`) populates the
//  global registry via ``StrategyRegistry/register(_:id:)``.
//
//  Wave 1 provides:
//    • `StrategyResult` — the return type of every strategy closure.
//    • `StrategyFn` — the closure type.
//    • `StrategyRegistry` — the dictionary wrapper.
//    • Integration example registrations.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

// MARK: - StrategyResult

/// The result of running one strategy on one ``WorkbenchCase``.
public struct StrategyResult: Sendable {

    /// Scalar result value.
    ///
    /// For multi-output strategies (e.g. ODE solvers, regression), only the
    /// primary scalar of interest is captured here — the domain suite picks
    /// which component to extract for comparison against the oracle.
    public let value: Double

    /// Diagnostics emitted by the strategy during computation.
    ///
    /// These are forwarded from result-struct `diagnostics` fields (wave 1)
    /// or from per-strategy detection logic (wave 2).
    public let diagnostics: [NumericDiagnostic]

    public init(value: Double, diagnostics: [NumericDiagnostic] = []) {
        self.value = value
        self.diagnostics = diagnostics
    }
}

// MARK: - StrategyFn

/// A closure that applies one algorithm to a fixture case's inputs.
///
/// - Parameter inputs: The `inputs` dictionary from the fixture case.
/// - Returns: A ``StrategyResult``, or `nil` when the inputs are missing
///   required keys or the algorithm returns `nil` (e.g. singular matrix).
public typealias StrategyFn = @Sendable ([String: InputValue]) -> StrategyResult?

// MARK: - StrategyRegistry

/// A registry mapping strategy identifiers to their implementation closures.
///
/// ### Usage
///
/// ```swift
/// var registry = StrategyRegistry()
/// registry.register(id: "quad") { inputs in
///     // extract inputs, call NumericSwift, return StrategyResult
/// }
/// let fn = registry["quad"]
/// let result = fn?(case.inputs)
/// ```
public struct StrategyRegistry: Sendable {

    private var strategies: [String: StrategyFn] = [:]

    public init() {}

    /// Register a strategy.
    ///
    /// Later registrations overwrite earlier ones for the same `id`.
    public mutating func register(id: String, _ fn: @escaping StrategyFn) {
        strategies[id] = fn
    }

    /// Look up a strategy by identifier.
    public subscript(_ id: String) -> StrategyFn? {
        strategies[id]
    }

    /// All registered strategy identifiers.
    public var registeredIDs: [String] {
        strategies.keys.sorted()
    }
}

// MARK: - Integration strategy registrations (wave 1 worked example)

/// Populate `registry` with the Integration (quadrature) strategies.
///
/// These are the wave-1 worked examples. Each closure extracts the integration
/// bounds `a` and `b` from the inputs dictionary and delegates to the
/// corresponding NumericSwift API.
///
/// The function identifier (`inputs["tag"]`) selects which integrand to use.
/// Wave 1 supports `"gaussian_bell"` (∫exp(-x²)) as the worked example; wave 2
/// will extend this to the full fixture corpus.
///
/// - Parameter registry: The registry to populate (mutated in place).
public func registerIntegrationStrategies(into registry: inout StrategyRegistry) {

    // Resolve integrand from the "tag" input key.
    @Sendable func makeIntegrand(_ inputs: [String: InputValue]) -> (@Sendable (Double) -> Double)? {
        switch inputs["tag"]?.stringValue {
        case "gaussian_bell":
            return { x in exp(-x * x) }
        case "sine":
            return { x in sin(x) }
        case "polynomial_deg2":
            // integrand: x^2 + 2x + 1 = (x+1)^2
            return { x in x * x + 2 * x + 1 }
        default:
            return nil
        }
    }

    // quad — Gauss-Kronrod 15-point adaptive
    registry.register(id: "quad") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = makeIntegrand(inputs)
        else { return nil }
        let result = quad(f, a, b)
        return StrategyResult(value: result.value, diagnostics: result.diagnostics)
    }

    // romberg — Richardson-extrapolation quadrature
    registry.register(id: "romberg") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = makeIntegrand(inputs)
        else { return nil }
        let result = romberg(f, a, b)
        return StrategyResult(value: result.value, diagnostics: result.diagnostics)
    }

    // simps — Simpson's rule
    registry.register(id: "simps") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = makeIntegrand(inputs),
              let nRaw = inputs["n"]?.intValue
        else { return nil }
        let xs = stride(from: a, through: b, by: (b - a) / Double(nRaw)).map { $0 }
        let ys = xs.map(f)
        let value = simps(ys, x: xs)
        return StrategyResult(value: value)
    }

    // trapz — trapezoidal rule
    registry.register(id: "trapz") { inputs in
        guard let a = inputs["a"]?.doubleValue,
              let b = inputs["b"]?.doubleValue,
              let f = makeIntegrand(inputs),
              let nRaw = inputs["n"]?.intValue
        else { return nil }
        let xs = stride(from: a, through: b, by: (b - a) / Double(nRaw)).map { $0 }
        let ys = xs.map(f)
        let value = trapz(ys, x: xs)
        return StrategyResult(value: value)
    }
}
