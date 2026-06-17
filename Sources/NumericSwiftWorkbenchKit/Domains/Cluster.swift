//
//  Cluster.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Cluster (clustering).
//
//  Mirrors the reference `Integration.swift` suite:
//    1. `clusterSuite` — the `DomainSuite` wiring the strategy + envelope registries.
//    2. `registerClusterStrategies(into:)` — one closure per clustering strategy.
//       Each closure reshapes the fixture's flat `points` array into data rows,
//       calls the NumericSwift `Cluster` API, reduces the clustering to a single
//       deterministic comparison scalar, and FORWARDS the ``NumericDiagnostic``
//       the library produced — it never fabricates one.
//    3. `makeClusterEnvelopeRegistry()` — per-(strategy, tier) accuracy bounds.
//
//  ## Determinism (FP3 — non-vacuous gate)
//
//  Clustering is otherwise non-deterministic, so each scalar is reproducible and
//  genuinely discriminating:
//    • `kmeans`         → final INERTIA from FIXED initial centroids. The same
//      `init` centroids are passed to both the library
//      (``Cluster/kmeans(_:initialCentroids:maxIterations:tolerance:)``) and the
//      sklearn oracle (`KMeans(init=<array>, n_init=1)`), so the Lloyd
//      trajectories — and the inertia — coincide.
//    • `dbscan`         → number of clusters found at fixed `eps` / `minPts`.
//    • `hierarchical_*` → size of the LARGEST cluster at a fixed `nClusters` cut.
//      The cluster *count* alone is vacuous (a cut into `n` always yields `n`
//      groups); the largest-group size depends on the actual merge structure and
//      therefore discriminates a correct linkage from a broken one.
//
//  ## Self-awareness
//
//  Out-of-envelope cases (tagged `inEnvelope: false`) are degenerate requests:
//  kmeans with `k > n` / `k <= 0` / empty input; dbscan parameters that label
//  every point as noise, or empty input; hierarchical `nClusters > n`. There the
//  library MUST emit a ``NumericDiagnostic/outsideEnvelope`` — which the
//  `Cluster` result structs now carry. Valid clustering cases must NOT warn.
//
//  Input encoding (carried through the JSON `inputs` bag, see ``InputValue``):
//    • `points` — flat array of `n * dims` coordinates (row-major).
//    • `dims`   — dimensionality `d` (so `n = points.count / d`).
//    • `k`, `init`        — cluster count and flat initial centroids (kmeans).
//    • `eps`, `minPts`    — neighbourhood radius and density floor (dbscan).
//    • `nClusters`        — cut size (hierarchical).
//    • `maxIter`, `tol`   — kmeans Lloyd controls.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Cluster (clustering) domain suite.
    public static let clusterSuite = DomainSuite(
        name: "cluster",
        registerStrategies: registerClusterStrategies,
        makeEnvelopeRegistry: makeClusterEnvelopeRegistry
    )
}

// MARK: - Input reshaping

/// Reshape the flat `points` / `dims` inputs into data rows.
///
/// Returns `nil` when the keys are missing or the length is not a multiple of
/// `dims`, so the runner records an ERROR rather than a spurious verdict. An
/// empty `points` array with a positive `dims` yields an empty row set (a valid
/// degenerate input the library detects).
@Sendable
private func clusterPoints(_ inputs: [String: InputValue]) -> [[Double]]? {
    guard let flat = inputs["points"]?.arrayValue,
          let dims = inputs["dims"]?.intValue, dims > 0
    else { return nil }
    let values = flat.compactMap(\.doubleValue)
    guard values.count == flat.count, values.count % dims == 0 else { return nil }
    return stride(from: 0, to: values.count, by: dims).map {
        Array(values[$0 ..< $0 + dims])
    }
}

/// Reshape a flat centroid array (`k * dims`) into centroid rows.
@Sendable
private func clusterInit(_ inputs: [String: InputValue], dims: Int) -> [[Double]]? {
    guard let flat = inputs["init"]?.arrayValue else { return nil }
    let values = flat.compactMap(\.doubleValue)
    guard values.count == flat.count, dims > 0, values.count % dims == 0 else { return nil }
    return stride(from: 0, to: values.count, by: dims).map {
        Array(values[$0 ..< $0 + dims])
    }
}

/// Size of the largest cluster among 0-indexed labels (the hierarchical scalar).
private func largestClusterSize(_ labels: [Int]) -> Double {
    guard !labels.isEmpty else { return 0 }
    var counts: [Int: Int] = [:]
    for l in labels { counts[l, default: 0] += 1 }
    return Double(counts.values.max() ?? 0)
}

// MARK: - Strategy registrations

/// Populate `registry` with the Cluster strategies.
@Sendable
public func registerClusterStrategies(into registry: inout StrategyRegistry) {

    // kmeans — Lloyd's algorithm from FIXED initial centroids; scalar = inertia.
    registry.register(id: "kmeans") { inputs in
        guard let data = clusterPoints(inputs),
              let dims = inputs["dims"]?.intValue,
              let initCentroids = clusterInit(inputs, dims: dims)
        else { return nil }
        let maxIter = inputs["maxIter"]?.intValue ?? 300
        let tol = inputs["tol"]?.doubleValue ?? 1e-4
        let r = Cluster.kmeans(data, initialCentroids: initCentroids,
                               maxIterations: maxIter, tolerance: tol)
        return StrategyResult(value: r.inertia, diagnostics: r.diagnostics)
    }

    // dbscan — density clustering; scalar = number of clusters found.
    registry.register(id: "dbscan") { inputs in
        guard let data = clusterPoints(inputs),
              let eps = inputs["eps"]?.doubleValue,
              let minPts = inputs["minPts"]?.intValue
        else { return nil }
        let r = Cluster.dbscan(data, eps: eps, minSamples: minPts)
        return StrategyResult(value: Double(r.nClusters), diagnostics: r.diagnostics)
    }

    // hierarchical (single / complete / average / ward) — agglomerative
    // clustering cut at `nClusters`; scalar = size of the largest cluster.
    for (id, linkage): (String, LinkageMethod) in [
        ("hierarchical_single", .single),
        ("hierarchical_complete", .complete),
        ("hierarchical_average", .average),
        ("hierarchical_ward", .ward),
    ] {
        registry.register(id: id) { inputs in
            guard let data = clusterPoints(inputs),
                  let nc = inputs["nClusters"]?.intValue
            else { return nil }
            let r = Cluster.hierarchicalClustering(data, linkage: linkage, nClusters: nc)
            return StrategyResult(value: largestClusterSize(r.labels ?? []),
                                  diagnostics: r.diagnostics)
        }
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Cluster domain.
///
///  - `kmeans` (Lloyd from fixed init): inertia matches sklearn to ~6 digits
///    once both converge to the same fixed point; the edge tier is looser.
///  - `dbscan` / `hierarchical_*`: the scalars are integer counts/sizes, so an
///    exact match is required (envelope 0) — any deviation is a real divergence.
@Sendable
public func makeClusterEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    for tier: CaseTier in [.trivial, .hard, .edge] {
        let kmeansTol: Double = tier == .edge ? 1e-3 : 1e-6
        reg.register(EnvelopeEntry(strategy: "kmeans", tier: tier, maxAbsError: kmeansTol,
            description: "k-means inertia from fixed initial centroids — \(tier.rawValue) cases"))

        reg.register(EnvelopeEntry(strategy: "dbscan", tier: tier, maxAbsError: 0,
            description: "DBSCAN cluster count (integer) — \(tier.rawValue) cases"))

        for (id, name) in [
            ("hierarchical_single", "single"),
            ("hierarchical_complete", "complete"),
            ("hierarchical_average", "average"),
            ("hierarchical_ward", "ward"),
        ] {
            reg.register(EnvelopeEntry(strategy: id, tier: tier, maxAbsError: 0,
                description: "Agglomerative (\(name)) largest-cluster size (integer) — \(tier.rawValue) cases"))
        }
    }
    return reg
}
