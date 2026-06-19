//
//  Statistics.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Statistics (descriptive statistics).
//
//  This is a **single-strategy-per-function correctness** domain (WORKBENCH.md
//  §4): each descriptive statistic IS a strategy id, the comparison scalar is
//  that statistic's output over the case's input array, and the oracle is the
//  matching numpy/scipy reference (Tools/workbench_oracles/statistics.py).
//
//  ## Self-awareness
//
//  Every statistic here is exact / closed-form, so EVERY fixture case is
//  in-envelope. There are ZERO out-of-envelope cases: the gate is a pure
//  correctness-vs-numpy check. Accordingly, every strategy closure returns a
//  ``StrategyResult`` with EMPTY diagnostics — the `Stats.*` functions have no
//  documented limitation envelope to surface here, and the harness never
//  fabricates a diagnostic.
//
//  Strategy ids ↔ `Stats.*` (Sources/NumericSwift/Statistics.swift):
//
//    mean        → Stats.mean
//    median      → Stats.median
//    variance    → Stats.variance(ddof:)
//    stddev      → Stats.stddev(ddof:)
//    percentile  → Stats.percentile(_:_:)   (q in 0...100, linear interpolation)
//    gmean       → Stats.gmean
//    hmean       → Stats.hmean
//    mode        → Stats.mode               (smallest value on tie)
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Statistics (descriptive statistics) domain suite.
    public static let statisticsSuite = DomainSuite(
        name: "statistics",
        registerStrategies: registerStatisticsStrategies,
        makeEnvelopeRegistry: makeStatisticsEnvelopeRegistry
    )
}

// MARK: - Input resolver

/// Extract the `data` array from a fixture case's `inputs` bag.
///
/// Returns `nil` when the array is missing — the runner records that as an
/// ERROR rather than a self-awareness verdict.
@Sendable
private func statisticsData(_ inputs: [String: InputValue]) -> [Double]? {
    guard let raw = inputs["data"]?.arrayValue else { return nil }
    let values = raw.compactMap { $0.doubleValue }
    return values.count == raw.count ? values : nil
}

// MARK: - Strategy registrations

/// Populate `registry` with the Statistics strategies (one per statistic).
@Sendable
public func registerStatisticsStrategies(into registry: inout StrategyRegistry) {

    registry.register(id: "mean") { inputs in
        guard let data = statisticsData(inputs) else { return nil }
        return StrategyResult(value: Stats.mean(data))
    }

    registry.register(id: "median") { inputs in
        guard let data = statisticsData(inputs) else { return nil }
        return StrategyResult(value: Stats.median(data))
    }

    // variance / stddev honour the optional `ddof` input (default 0, matching
    // np.var/np.std with ddof unspecified).
    registry.register(id: "variance") { inputs in
        guard let data = statisticsData(inputs) else { return nil }
        let ddof = inputs["ddof"]?.intValue ?? 0
        return StrategyResult(value: Stats.variance(data, ddof: ddof))
    }

    registry.register(id: "stddev") { inputs in
        guard let data = statisticsData(inputs) else { return nil }
        let ddof = inputs["ddof"]?.intValue ?? 0
        return StrategyResult(value: Stats.stddev(data, ddof: ddof))
    }

    // percentile takes q in 0...100 (linear interpolation), matching np.percentile.
    registry.register(id: "percentile") { inputs in
        guard let data = statisticsData(inputs),
              let q = inputs["q"]?.doubleValue
        else { return nil }
        return StrategyResult(value: Stats.percentile(data, q))
    }

    registry.register(id: "gmean") { inputs in
        guard let data = statisticsData(inputs) else { return nil }
        return StrategyResult(value: Stats.gmean(data))
    }

    registry.register(id: "hmean") { inputs in
        guard let data = statisticsData(inputs) else { return nil }
        return StrategyResult(value: Stats.hmean(data))
    }

    registry.register(id: "mode") { inputs in
        guard let data = statisticsData(inputs) else { return nil }
        return StrategyResult(value: Stats.mode(data))
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Statistics domain.
///
/// All descriptive statistics here are exact / closed-form against numpy/scipy,
/// so the envelopes are uniformly tight (~1e-12) across every tier — the only
/// deviation from a bit-exact match is benign floating-point summation order.
/// No strategy has an out-of-envelope regime, hence no `outsideEnvelope`
/// diagnostic is ever expected (WORKBENCH.md §5).
@Sendable
public func makeStatisticsEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    let strategies = ["mean", "median", "variance", "stddev",
                      "percentile", "gmean", "hmean", "mode"]
    for strategy in strategies {
        for tier: CaseTier in [.trivial, .hard, .edge] {
            reg.register(EnvelopeEntry(
                strategy: strategy,
                tier: tier,
                maxAbsError: 1e-12,
                description: "\(strategy) — exact vs numpy/scipy (\(tier.rawValue) cases)"))
        }
    }
    return reg
}
