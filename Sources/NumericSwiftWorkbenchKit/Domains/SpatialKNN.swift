//
//  SpatialKNN.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Spatial nearest-neighbour search (k-NN).
//
//  Mirrors the reference `Integration.swift` suite:
//    1. `spatialknnSuite` — the `DomainSuite` wiring the strategy + envelope registries.
//    2. `registerSpatialKNNStrategies(into:)` — one closure per k-NN strategy.
//       Each closure builds the search structure from the fixture's flat point
//       array (`points` + `dims`), queries it at `query` for `k` neighbours, and
//       reports the **distance to the k-th nearest neighbour** as the comparison
//       scalar. Each closure delegates to the library's `*Diagnosed` entry point
//       and FORWARDS the ``NumericDiagnostic`` it produced — it never fabricates one.
//    3. `makeSpatialKNNEnvelopeRegistry()` — per-(strategy, tier) accuracy bounds.
//
//  ## Comparison scalar — the k-th-neighbour distance
//
//  Both strategies are EXACT k-NN searches, so on any well-posed query they must
//  agree with each other and with the scipy oracle to machine precision. The
//  single scalar compared is `distances[k - 1]` — the distance from the query
//  point to its k-th nearest neighbour (1-indexed: `k = 1` is the nearest
//  neighbour). The oracle is `scipy.spatial.cKDTree(...).query(q, k)[0]` taking
//  the k-th returned distance. `k` is carried per case in the fixture inputs.
//
//  ## Self-awareness
//
//  Out-of-envelope cases (tagged `inEnvelope: false` in the fixture) are
//  **degenerate queries** that cannot return `k` valid neighbours: `k` greater
//  than the point count, an empty point set, or `k <= 0`. There the k-th-neighbour
//  distance is meaningless, so the library MUST emit a
//  ``NumericDiagnostic/outsideEnvelope`` — which ``KDTree/queryDiagnosed(_:k:)``
//  and ``Spatial/bruteForceKNNDiagnosed(points:query:k:)`` do. Well-posed queries
//  (`0 < k <= n`) must NOT warn, and `kdTree` must match `bruteForce` exactly.
//
//  Inputs (carried through the JSON `inputs` bag, see ``InputValue``):
//    • `points` — flat array of point coordinates, row-major (point 0's `dims`
//                 coordinates, then point 1's, …).
//    • `dims`   — coordinate dimensionality `d`; `points.count == n * d`.
//    • `query`  — flat array of the query point's `d` coordinates.
//    • `k`      — requested neighbour count.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Spatial nearest-neighbour (k-NN) domain suite.
    public static let spatialknnSuite = DomainSuite(
        name: "spatialknn",
        registerStrategies: registerSpatialKNNStrategies,
        makeEnvelopeRegistry: makeSpatialKNNEnvelopeRegistry
    )
}

// MARK: - Sample resolver

/// Extract the `(points, query, k)` triple from a fixture inputs bag.
///
/// `points` arrives as a flat row-major array; it is reshaped into `n` rows of
/// `dims` coordinates. Returns `nil` when any key is missing or the flat array
/// length is not a multiple of `dims`, so the runner records an ERROR rather than
/// a spurious self-awareness verdict.
@Sendable
private func spatialKNNSample(
    _ inputs: [String: InputValue]
) -> (points: [[Double]], query: [Double], k: Int)? {
    guard let flatRaw = inputs["points"]?.arrayValue,
          let dims = inputs["dims"]?.intValue, dims > 0,
          let queryRaw = inputs["query"]?.arrayValue,
          let k = inputs["k"]?.intValue
    else { return nil }
    let flat = flatRaw.compactMap(\.doubleValue)
    let query = queryRaw.compactMap(\.doubleValue)
    guard flat.count == flatRaw.count, query.count == queryRaw.count,
          query.count == dims,
          flat.count % dims == 0
    else { return nil }
    let n = flat.count / dims
    let points = (0..<n).map { row in Array(flat[(row * dims)..<((row + 1) * dims)]) }
    return (points, query, k)
}

/// Reduce a k-NN result to the comparison scalar: the k-th-neighbour distance.
///
/// On a well-posed query the search returns exactly `k` distances and the k-th
/// one is `distances.last`. On a degenerate query the search returns fewer than
/// `k` (or zero) distances; the scalar is then `NaN`, which the oracle also
/// freezes for those cases — the self-awareness gate (not the numeric value) is
/// what these cases test.
private func kthNeighbourDistance(_ r: KNNResult, k: Int) -> Double {
    guard k > 0, r.distances.count >= k else { return .nan }
    return r.distances[k - 1]
}

// MARK: - Strategy registrations

/// Populate `registry` with the Spatial k-NN strategies.
///
/// Both strategies are exact; they differ only in the search structure
/// (`KDTree` vs. exhaustive scan) and must agree on every well-posed query.
@Sendable
public func registerSpatialKNNStrategies(into registry: inout StrategyRegistry) {

    // kdTree — KDTree(points).queryDiagnosed(query, k:): O(log n)-amortised search.
    registry.register(id: "kdTree") { inputs in
        guard let s = spatialKNNSample(inputs) else { return nil }
        let tree = KDTree(s.points)
        let r = tree.queryDiagnosed(s.query, k: s.k)
        return StrategyResult(
            value: kthNeighbourDistance(r.value, k: s.k),
            diagnostics: r.diagnostics)
    }

    // bruteForce — exhaustive Spatial.bruteForceKNNDiagnosed: O(n d) reference scan.
    registry.register(id: "bruteForce") { inputs in
        guard let s = spatialKNNSample(inputs) else { return nil }
        let r = Spatial.bruteForceKNNDiagnosed(points: s.points, query: s.query, k: s.k)
        return StrategyResult(
            value: kthNeighbourDistance(r.value, k: s.k),
            diagnostics: r.diagnostics)
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Spatial k-NN domain.
///
/// Both `kdTree` and `bruteForce` are EXACT k-NN searches: their k-th-neighbour
/// distance equals the oracle's to floating-point round-off (the only error is
/// the order of additions inside the Euclidean norm). The envelopes are therefore
/// uniformly tight across tiers — a deviation beyond a few ULP signals a real bug,
/// not an algorithmic limitation.
@Sendable
public func makeSpatialKNNEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let exactTol = 1e-12
        reg.register(EnvelopeEntry(strategy: "kdTree", tier: tier, maxAbsError: exactTol,
            description: "KDTree exact k-NN — \(tier.rawValue) cases"))
        reg.register(EnvelopeEntry(strategy: "bruteForce", tier: tier, maxAbsError: exactTol,
            description: "Brute-force exact k-NN — \(tier.rawValue) cases"))
    }
    return reg
}
