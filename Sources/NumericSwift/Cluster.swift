//
//  Cluster.swift
//  NumericSwift
//
//  Clustering algorithms following sklearn.cluster patterns.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

// MARK: - Result Types (public, top-level)

/// Result of k-means clustering.
public struct KMeansResult: Sendable {
    /// Cluster labels for each data point (0-indexed).
    public let labels: [Int]
    /// Final cluster centroids.
    public let centroids: [[Double]]
    /// Sum of squared distances to nearest centroid.
    public let inertia: Double
    /// Number of iterations run.
    public let iterations: Int
    /// Non-fatal diagnostics produced during clustering.
    ///
    /// Empty for a well-posed request. A degenerate request — `k <= 0`,
    /// `k` greater than the number of points, or empty input — attaches a
    /// ``NumericDiagnostic/outsideEnvelope(method:reason:)`` so a self-aware
    /// caller can tell that the (best-effort, possibly empty) result lies
    /// outside k-means' valid envelope. See the workbench self-awareness gate.
    public let diagnostics: [NumericDiagnostic]

    public init(
        labels: [Int],
        centroids: [[Double]],
        inertia: Double,
        iterations: Int,
        diagnostics: [NumericDiagnostic] = []
    ) {
        self.labels = labels
        self.centroids = centroids
        self.inertia = inertia
        self.iterations = iterations
        self.diagnostics = diagnostics
    }
}

/// Result of hierarchical clustering.
public struct HierarchicalResult: Sendable {
    /// Linkage matrix (n-1 × 4): [cluster1, cluster2, distance, size]
    public let linkageMatrix: [[Double]]
    /// Cluster labels if tree was cut (0-indexed), nil otherwise.
    public let labels: [Int]?
    /// Number of original samples.
    public let nLeaves: Int
    /// Non-fatal diagnostics produced during clustering.
    ///
    /// Empty for a well-posed request. A degenerate cut request —
    /// `nClusters <= 0`, `nClusters` greater than the number of leaves, or
    /// empty input — attaches a ``NumericDiagnostic/outsideEnvelope(method:reason:)``.
    public let diagnostics: [NumericDiagnostic]

    public init(
        linkageMatrix: [[Double]],
        labels: [Int]?,
        nLeaves: Int,
        diagnostics: [NumericDiagnostic] = []
    ) {
        self.linkageMatrix = linkageMatrix
        self.labels = labels
        self.nLeaves = nLeaves
        self.diagnostics = diagnostics
    }
}

/// Linkage method for hierarchical clustering.
public enum LinkageMethod: String {
    /// Minimum distance between clusters (nearest neighbor).
    case single
    /// Maximum distance between clusters (farthest neighbor).
    case complete
    /// Average distance between all pairs.
    case average
    /// Ward's method: minimize within-cluster variance.
    case ward
}

/// Result of DBSCAN clustering.
public struct DBSCANResult: Sendable {
    /// Cluster labels (-1 for noise, 0+ for clusters).
    public let labels: [Int]
    /// Indices of core samples.
    public let coreSamples: [Int]
    /// Number of clusters found.
    public let nClusters: Int
    /// Non-fatal diagnostics produced during clustering.
    ///
    /// Empty for a well-posed request. A degenerate request — empty input or
    /// parameters that label every point as noise (zero clusters) — attaches a
    /// ``NumericDiagnostic/outsideEnvelope(method:reason:)``.
    public let diagnostics: [NumericDiagnostic]

    public init(
        labels: [Int],
        coreSamples: [Int],
        nClusters: Int,
        diagnostics: [NumericDiagnostic] = []
    ) {
        self.labels = labels
        self.coreSamples = coreSamples
        self.nClusters = nClusters
        self.diagnostics = diagnostics
    }
}

/// Result of elbow method analysis.
public struct ElbowResult {
    /// Inertias for each k value (index 0 = k=1).
    public let inertias: [Double]
    /// Suggested optimal k based on elbow heuristic.
    public let suggestedK: Int
}

// MARK: - Cluster Namespace

/// Clustering algorithms.
///
/// Groups data into clusters using various methods including k-means,
/// hierarchical clustering, and DBSCAN.
///
/// ## Overview
///
/// ```swift
/// let result = Cluster.kmeans(data, k: 3)
/// let labels  = result.labels
/// ```
public enum Cluster {

    // MARK: - K-Means Clustering

    /// K-means clustering algorithm.
    ///
    /// Partitions data into k clusters by minimizing within-cluster variance.
    /// Uses k-means++ initialization by default for better convergence.
    ///
    /// - Parameters:
    ///   - data: Array of data points (each point is an array of coordinates)
    ///   - k: Number of clusters
    ///   - maxIterations: Maximum iterations (default 300)
    ///   - tolerance: Convergence tolerance for centroid movement (default 1e-4)
    ///   - nInit: Number of times to run with different initializations (default 10)
    ///   - initMethod: Initialization method ("kmeans++" or "random")
    /// - Returns: KMeansResult with labels, centroids, inertia, and iteration count
    public static func kmeans(
        _ data: [[Double]],
        k: Int,
        maxIterations: Int = 300,
        tolerance: Double = 1e-4,
        nInit: Int = 10,
        initMethod: String = "kmeans++"
    ) -> KMeansResult {
        if let diag = kmeansEnvelopeDiagnostic(n: data.count, k: k) {
            return KMeansResult(labels: [], centroids: [], inertia: 0, iterations: 0,
                                diagnostics: [diag])
        }

        let dim = data[0].count

        // Run n_init times and keep best result
        var bestLabels: [Int] = []
        var bestCentroids: [[Double]] = []
        var bestInertia = Double.infinity
        var bestIterations = 0

        for _ in 0..<nInit {
            let initialCentroids: [[Double]]
            if initMethod == "kmeans++" {
                initialCentroids = kMeansPlusPlusInit(data: data, k: k, dim: dim)
            } else {
                initialCentroids = randomInit(data: data, k: k)
            }

            let result = runKMeans(
                data: data, k: k, initialCentroids: initialCentroids,
                maxIter: maxIterations, tol: tolerance
            )

            if result.inertia < bestInertia {
                bestLabels = result.labels
                bestCentroids = result.centroids
                bestInertia = result.inertia
                bestIterations = result.iterations
            }
        }

        return KMeansResult(
            labels: bestLabels,
            centroids: bestCentroids,
            inertia: bestInertia,
            iterations: bestIterations
        )
    }

    /// K-means clustering from a caller-supplied set of initial centroids.
    ///
    /// Unlike ``kmeans(_:k:maxIterations:tolerance:nInit:initMethod:)`` — which
    /// seeds randomly (k-means++ or random) and is therefore non-deterministic —
    /// this overload runs a single Lloyd iteration sequence from the exact
    /// `initialCentroids` provided, so the result (labels, centroids, inertia) is
    /// fully reproducible. It mirrors `sklearn.cluster.KMeans(init=<array>,
    /// n_init=1)`, which makes the two directly comparable for parity testing.
    ///
    /// - Parameters:
    ///   - data: Array of data points (each point is an array of coordinates).
    ///   - initialCentroids: The exact starting centroids; `k` is `initialCentroids.count`.
    ///   - maxIterations: Maximum Lloyd iterations (default 300).
    ///   - tolerance: Convergence tolerance for centroid movement (default 1e-4).
    /// - Returns: ``KMeansResult``. A degenerate request (`k <= 0`, `k` greater
    ///   than the number of points, or empty input) returns an empty best-effort
    ///   result carrying an ``NumericDiagnostic/outsideEnvelope(method:reason:)``.
    public static func kmeans(
        _ data: [[Double]],
        initialCentroids: [[Double]],
        maxIterations: Int = 300,
        tolerance: Double = 1e-4
    ) -> KMeansResult {
        let k = initialCentroids.count
        if let diag = kmeansEnvelopeDiagnostic(n: data.count, k: k) {
            return KMeansResult(labels: [], centroids: [], inertia: 0, iterations: 0,
                                diagnostics: [diag])
        }

        let result = runKMeans(
            data: data, k: k, initialCentroids: initialCentroids,
            maxIter: maxIterations, tol: tolerance
        )
        return KMeansResult(
            labels: result.labels,
            centroids: result.centroids,
            inertia: result.inertia,
            iterations: result.iterations
        )
    }

    /// Degenerate-request detector shared by both `kmeans` entry points.
    ///
    /// Returns an ``NumericDiagnostic/outsideEnvelope(method:reason:)`` when the
    /// request lies outside k-means' valid envelope (empty data, `k <= 0`, or
    /// more clusters than points), else `nil`.
    static func kmeansEnvelopeDiagnostic(n: Int, k: Int) -> NumericDiagnostic? {
        if n == 0 {
            return .outsideEnvelope(method: "kmeans",
                reason: "empty input — no points to cluster")
        }
        if k <= 0 {
            return .outsideEnvelope(method: "kmeans",
                reason: "k=\(k) is not positive — clustering is undefined")
        }
        if k > n {
            return .outsideEnvelope(method: "kmeans",
                reason: "k=\(k) exceeds the number of points (\(n)) — at least one "
                    + "cluster must stay empty")
        }
        return nil
    }

    // MARK: - Hierarchical Clustering

    /// Hierarchical (agglomerative) clustering.
    ///
    /// Builds a hierarchy of clusters by iteratively merging the closest pairs.
    /// The linkage matrix can be used to create dendrograms.
    ///
    /// - Parameters:
    ///   - data: Array of data points
    ///   - linkage: Linkage method (default .ward)
    ///   - nClusters: Number of clusters to extract (optional)
    ///   - distanceThreshold: Distance threshold for cutting tree (optional)
    /// - Returns: HierarchicalResult with linkage matrix and optional labels
    public static func hierarchicalClustering(
        _ data: [[Double]],
        linkage: LinkageMethod = .ward,
        nClusters: Int? = nil,
        distanceThreshold: Double? = nil
    ) -> HierarchicalResult {
        let n = data.count
        guard n > 0 else {
            return HierarchicalResult(linkageMatrix: [], labels: nil, nLeaves: 0,
                diagnostics: [.outsideEnvelope(method: "hierarchical",
                    reason: "empty input — no points to cluster")])
        }

        var clusters: [HierarchicalCluster] = []
        for i in 0..<n {
            clusters.append(HierarchicalCluster(
                id: i,
                points: [i],
                centroid: data[i],
                size: 1,
                active: true
            ))
        }

        var linkageMatrix: [[Double]] = []
        var nextClusterId = n

        for _ in 0..<(n - 1) {
            var minDist = Double.infinity
            var bestI = -1, bestJ = -1

            for i in 0..<clusters.count {
                guard clusters[i].active else { continue }
                for j in (i + 1)..<clusters.count {
                    guard clusters[j].active else { continue }

                    let d = clusterDistance(
                        c1: clusters[i], c2: clusters[j],
                        data: data, method: linkage
                    )
                    if d < minDist {
                        minDist = d
                        bestI = i
                        bestJ = j
                    }
                }
            }

            guard bestI >= 0 else { break }

            let c1 = clusters[bestI]
            let c2 = clusters[bestJ]

            var mergedPoints = c1.points
            mergedPoints.append(contentsOf: c2.points)

            let newSize = c1.size + c2.size
            var newCentroid = [Double](repeating: 0, count: c1.centroid.count)
            for d in 0..<newCentroid.count {
                newCentroid[d] = (c1.centroid[d] * Double(c1.size) + c2.centroid[d] * Double(c2.size)) / Double(newSize)
            }

            linkageMatrix.append([Double(c1.id), Double(c2.id), minDist, Double(newSize)])

            clusters[bestI].active = false
            clusters[bestJ].active = false

            clusters.append(HierarchicalCluster(
                id: nextClusterId,
                points: mergedPoints,
                centroid: newCentroid,
                size: newSize,
                active: true
            ))
            nextClusterId += 1
        }

        var diagnostics: [NumericDiagnostic] = []
        var labels: [Int]? = nil
        if nClusters != nil || distanceThreshold != nil {
            let cutIdx: Int
            if let nc = nClusters {
                if nc <= 0 {
                    diagnostics.append(.outsideEnvelope(method: "hierarchical",
                        reason: "nClusters=\(nc) is not positive — a cut must leave at "
                            + "least one cluster"))
                } else if nc > n {
                    diagnostics.append(.outsideEnvelope(method: "hierarchical",
                        reason: "nClusters=\(nc) exceeds the number of leaves (\(n)) — "
                            + "the tree cannot be cut into that many clusters"))
                }
                cutIdx = max(0, n - nc)
            } else if let dt = distanceThreshold {
                var idx = 0
                for (i, merge) in linkageMatrix.enumerated() {
                    if merge[2] > dt {
                        idx = i
                        break
                    }
                    idx = i + 1
                }
                cutIdx = idx
            } else {
                cutIdx = 0
            }

            labels = cutHierarchy(linkageMatrix: linkageMatrix, n: n, cutIdx: cutIdx)
        }

        return HierarchicalResult(
            linkageMatrix: linkageMatrix,
            labels: labels,
            nLeaves: n,
            diagnostics: diagnostics
        )
    }

    // MARK: - DBSCAN Clustering

    /// DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    ///
    /// Finds clusters based on density. Points in dense regions are grouped together,
    /// sparse points are marked as noise (-1).
    ///
    /// This implementation uses a `KDTree` for O(n log n) average-case neighbourhood
    /// queries and an index-pointer idiom for the BFS queue to avoid O(n) `Array.removeFirst()`.
    ///
    /// Reference: Ester et al., "A Density-Based Algorithm for Discovering Clusters in Large
    /// Spatial Databases with Noise", KDD 1996.
    ///
    /// - Parameters:
    ///   - data: Array of data points; all points must have the same dimensionality.
    ///   - eps: Maximum distance for neighbourhood (default 0.5).
    ///   - minSamples: Minimum points (including the point itself) to form a dense region (default 5).
    /// - Returns: `DBSCANResult` with per-point labels, core-sample indices, and cluster count.
    public static func dbscan(
        _ data: [[Double]],
        eps: Double = 0.5,
        minSamples: Int = 5
    ) -> DBSCANResult {
        let n = data.count
        guard n > 0 else {
            return DBSCANResult(labels: [], coreSamples: [], nClusters: 0,
                diagnostics: [.outsideEnvelope(method: "dbscan",
                    reason: "empty input — no points to cluster")])
        }

        let tree = KDTree(data)

        var neighboursCache: [[Int]] = Array(repeating: [], count: n)
        var isCore = [Bool](repeating: false, count: n)

        for i in 0..<n {
            let (indices, _) = tree.queryRadius(data[i], radius: eps)
            neighboursCache[i] = indices
            isCore[i] = indices.count >= minSamples
        }

        var labels = [Int](repeating: -2, count: n)
        var currentCluster = -1
        var coreSamples: [Int] = []

        for i in 0..<n {
            if labels[i] != -2 {
                continue
            }
            if !isCore[i] {
                labels[i] = -1
                continue
            }

            currentCluster += 1
            coreSamples.append(i)
            labels[i] = currentCluster

            var queue = [i]
            var queueStart = 0

            while queueStart < queue.count {
                let current = queue[queueStart]
                queueStart += 1

                for neighbour in neighboursCache[current] {
                    if labels[neighbour] == -1 {
                        labels[neighbour] = currentCluster
                    } else if labels[neighbour] == -2 {
                        labels[neighbour] = currentCluster
                        if isCore[neighbour] {
                            queue.append(neighbour)
                            coreSamples.append(neighbour)
                        }
                    }
                }
            }
        }

        let nClusters = currentCluster + 1
        // All-noise: eps/minSamples are too strict to form a single dense region.
        // The result (every point labelled -1) is technically correct but lies
        // outside DBSCAN's useful envelope, so flag it for a self-aware caller.
        var diagnostics: [NumericDiagnostic] = []
        if nClusters == 0 {
            diagnostics.append(.outsideEnvelope(method: "dbscan",
                reason: "no clusters formed — every point is noise; eps=\(eps) / "
                    + "minSamples=\(minSamples) are too strict for this data"))
        }

        return DBSCANResult(
            labels: labels,
            coreSamples: coreSamples,
            nClusters: nClusters,
            diagnostics: diagnostics
        )
    }

    // MARK: - Cluster Evaluation

    /// Silhouette score for evaluating clustering quality.
    ///
    /// Measures how similar points are to their own cluster compared to other clusters.
    /// Ranges from -1 (poor clustering) to +1 (excellent clustering).
    ///
    /// - Parameters:
    ///   - data: Array of data points
    ///   - labels: Cluster labels (negative values are ignored as noise)
    /// - Returns: Mean silhouette coefficient
    public static func silhouetteScore(_ data: [[Double]], labels: [Int]) -> Double {
        let n = data.count
        guard n >= 2, labels.count == n else { return 0 }

        var clusters: [Int: [Int]] = [:]
        for (i, label) in labels.enumerated() {
            if label >= 0 {
                clusters[label, default: []].append(i)
            }
        }

        guard clusters.count >= 2 else { return 0 }

        var totalSilhouette: Double = 0
        var count = 0

        for i in 0..<n {
            let labelI = labels[i]
            if labelI < 0 { continue }

            guard let sameCluster = clusters[labelI] else { continue }

            var aI: Double = 0
            if sameCluster.count > 1 {
                for j in sameCluster {
                    if j != i {
                        aI += euclideanDistance(data[i], data[j])
                    }
                }
                aI /= Double(sameCluster.count - 1)
            }

            var bI = Double.infinity
            for (otherLabel, otherCluster) in clusters {
                if otherLabel != labelI {
                    var meanDist: Double = 0
                    for j in otherCluster {
                        meanDist += euclideanDistance(data[i], data[j])
                    }
                    meanDist /= Double(otherCluster.count)
                    if meanDist < bI { bI = meanDist }
                }
            }

            let maxAB = max(aI, bI)
            let sI = maxAB > 0 ? (bI - aI) / maxAB : 0

            totalSilhouette += sI
            count += 1
        }

        return count > 0 ? totalSilhouette / Double(count) : 0
    }

    /// Elbow method for determining optimal number of clusters.
    ///
    /// Runs k-means for k=1 to maxK and returns inertias.
    /// The "elbow" point where inertia stops decreasing rapidly suggests optimal k.
    ///
    /// - Parameters:
    ///   - data: Array of data points
    ///   - maxK: Maximum k to test (default 10)
    /// - Returns: ElbowResult with inertias and suggested k
    public static func elbowMethod(_ data: [[Double]], maxK: Int = 10) -> ElbowResult {
        guard !data.isEmpty else {
            return ElbowResult(inertias: [], suggestedK: 1)
        }

        let actualMaxK = min(maxK, data.count)
        var inertias: [Double] = []

        for k in 1...actualMaxK {
            let result = kmeans(data, k: k, nInit: 3)
            inertias.append(result.inertia)
        }

        var suggestedK = 1
        var maxDiffRatio: Double = 0

        for k in 2..<(actualMaxK - 1) {
            let diff1 = inertias[k - 2] - inertias[k - 1]
            let diff2 = inertias[k - 1] - inertias[k]
            if diff2 > 0 {
                let ratio = diff1 / diff2
                if ratio > maxDiffRatio {
                    maxDiffRatio = ratio
                    suggestedK = k
                }
            }
        }

        return ElbowResult(inertias: inertias, suggestedK: suggestedK)
    }

    // MARK: - Distance / centroid helpers (delegate to Spatial, keep centroid private)

    /// Euclidean distance — delegates to ``Spatial/euclideanDistance(_:_:)``.
    internal static func euclideanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        Spatial.euclideanDistance(p1, p2)
    }

    /// Squared Euclidean distance — delegates to ``Spatial/squaredEuclideanDistance(_:_:)``.
    internal static func squaredEuclideanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        Spatial.squaredEuclideanDistance(p1, p2)
    }

    /// Compute centroid of a set of points using vDSP.
    internal static func centroid(_ points: [[Double]]) -> [Double]? {
        guard !points.isEmpty else { return nil }
        let dim = points[0].count
        var centroidVec = [Double](repeating: 0, count: dim)

        for point in points {
            vDSP_vaddD(centroidVec, 1, point, 1, &centroidVec, 1, vDSP_Length(dim))
        }

        var divisor = Double(points.count)
        vDSP_vsdivD(centroidVec, 1, &divisor, &centroidVec, 1, vDSP_Length(dim))

        return centroidVec
    }

    // MARK: - Private helpers

    private static func kMeansPlusPlusInit(data: [[Double]], k: Int, dim: Int) -> [[Double]] {
        let n = data.count
        guard n > 0, k > 0 else { return [] }

        var centroids: [[Double]] = []

        let firstIdx = Int.random(in: 0..<n)
        centroids.append(data[firstIdx])

        for _ in 1..<k {
            var distances = [Double](repeating: 0, count: n)
            var totalDist: Double = 0

            for i in 0..<n {
                var minDist = Double.infinity
                for centroid in centroids {
                    let d = squaredEuclideanDistance(data[i], centroid)
                    if d < minDist { minDist = d }
                }
                distances[i] = minDist
                totalDist += minDist
            }

            guard totalDist > 0 else { break }
            let r = Double.random(in: 0..<totalDist)
            var cumsum: Double = 0
            var chosenIdx = n - 1

            for i in 0..<n {
                cumsum += distances[i]
                if cumsum >= r {
                    chosenIdx = i
                    break
                }
            }

            centroids.append(data[chosenIdx])
        }

        return centroids
    }

    private static func randomInit(data: [[Double]], k: Int) -> [[Double]] {
        let n = data.count
        var used = Set<Int>()
        var centroids: [[Double]] = []

        while centroids.count < k && used.count < n {
            let idx = Int.random(in: 0..<n)
            if !used.contains(idx) {
                used.insert(idx)
                centroids.append(data[idx])
            }
        }

        return centroids
    }

    private static func runKMeans(
        data: [[Double]], k: Int, initialCentroids: [[Double]],
        maxIter: Int, tol: Double
    ) -> (labels: [Int], centroids: [[Double]], inertia: Double, iterations: Int) {
        let n = data.count
        var centroids = initialCentroids
        var labels = [Int](repeating: 0, count: n)
        var nIter = 0

        for iter in 1...maxIter {
            nIter = iter

            var clusters: [[Int]] = Array(repeating: [], count: k)

            for i in 0..<n {
                var minDist = Double.infinity
                var bestC = 0
                for c in 0..<k {
                    let d = squaredEuclideanDistance(data[i], centroids[c])
                    if d < minDist {
                        minDist = d
                        bestC = c
                    }
                }
                labels[i] = bestC
                clusters[bestC].append(i)
            }

            var newCentroids: [[Double]] = []
            for c in 0..<k {
                if clusters[c].isEmpty {
                    newCentroids.append(centroids[c])
                } else {
                    let clusterPoints = clusters[c].map { data[$0] }
                    if let c = centroid(clusterPoints) {
                        newCentroids.append(c)
                    } else {
                        newCentroids.append(centroids[c])
                    }
                }
            }

            var maxShift: Double = 0
            for c in 0..<k {
                let shift = euclideanDistance(centroids[c], newCentroids[c])
                if shift > maxShift { maxShift = shift }
            }

            centroids = newCentroids

            if maxShift < tol {
                break
            }
        }

        var finalInertia: Double = 0
        for i in 0..<n {
            finalInertia += squaredEuclideanDistance(data[i], centroids[labels[i]])
        }

        return (labels, centroids, finalInertia, nIter)
    }

    private static func clusterDistance(
        c1: HierarchicalCluster, c2: HierarchicalCluster,
        data: [[Double]], method: LinkageMethod
    ) -> Double {
        switch method {
        case .single:
            var minD = Double.infinity
            for i in c1.points {
                for j in c2.points {
                    let d = squaredEuclideanDistance(data[i], data[j])
                    if d < minD { minD = d }
                }
            }
            return Darwin.sqrt(minD)

        case .complete:
            var maxD: Double = 0
            for i in c1.points {
                for j in c2.points {
                    let d = squaredEuclideanDistance(data[i], data[j])
                    if d > maxD { maxD = d }
                }
            }
            return Darwin.sqrt(maxD)

        case .average:
            var total: Double = 0
            var count = 0
            for i in c1.points {
                for j in c2.points {
                    total += euclideanDistance(data[i], data[j])
                    count += 1
                }
            }
            return total / Double(count)

        case .ward:
            let n1 = Double(c1.size)
            let n2 = Double(c2.size)
            let centroidDist = squaredEuclideanDistance(c1.centroid, c2.centroid)
            return Darwin.sqrt(2 * n1 * n2 / (n1 + n2) * centroidDist)
        }
    }

    private static func cutHierarchy(linkageMatrix: [[Double]], n: Int, cutIdx: Int) -> [Int] {
        var clusterMap = [Int](0..<n)
        var nextLabel = n

        for i in 0..<cutIdx {
            let c1Id = Int(linkageMatrix[i][0])
            let c2Id = Int(linkageMatrix[i][1])

            for p in 0..<n {
                if clusterMap[p] == c1Id || clusterMap[p] == c2Id {
                    clusterMap[p] = nextLabel
                }
            }
            nextLabel += 1
        }

        var labelMap: [Int: Int] = [:]
        var nextNewLabel = 0
        var resultLabels = [Int](repeating: 0, count: n)

        for i in 0..<n {
            let oldLabel = clusterMap[i]
            if labelMap[oldLabel] == nil {
                labelMap[oldLabel] = nextNewLabel
                nextNewLabel += 1
            }
            resultLabels[i] = labelMap[oldLabel]!
        }

        return resultLabels
    }
}

// MARK: - Helper struct (private to file)

private struct HierarchicalCluster {
    let id: Int
    var points: [Int]
    var centroid: [Double]
    var size: Int
    var active: Bool
}

// MARK: - Deprecated shims (backward compatibility)

/// - Note: Deprecated. Use ``Cluster/kmeans(_:k:maxIterations:tolerance:nInit:initMethod:)`` instead.
@available(*, deprecated, message: "Use Cluster.kmeans(_:k:maxIterations:tolerance:nInit:initMethod:) instead")
public func kmeans(
    _ data: [[Double]],
    k: Int,
    maxIterations: Int = 300,
    tolerance: Double = 1e-4,
    nInit: Int = 10,
    initMethod: String = "kmeans++"
) -> KMeansResult {
    Cluster.kmeans(data, k: k, maxIterations: maxIterations, tolerance: tolerance, nInit: nInit, initMethod: initMethod)
}

/// - Note: Deprecated. Use ``Cluster/hierarchicalClustering(_:linkage:nClusters:distanceThreshold:)`` instead.
@available(*, deprecated, message: "Use Cluster.hierarchicalClustering(_:linkage:nClusters:distanceThreshold:) instead")
public func hierarchicalClustering(
    _ data: [[Double]],
    linkage: LinkageMethod = .ward,
    nClusters: Int? = nil,
    distanceThreshold: Double? = nil
) -> HierarchicalResult {
    Cluster.hierarchicalClustering(data, linkage: linkage, nClusters: nClusters, distanceThreshold: distanceThreshold)
}

/// - Note: Deprecated. Use ``Cluster/dbscan(_:eps:minSamples:)`` instead.
@available(*, deprecated, message: "Use Cluster.dbscan(_:eps:minSamples:) instead")
public func dbscan(
    _ data: [[Double]],
    eps: Double = 0.5,
    minSamples: Int = 5
) -> DBSCANResult {
    Cluster.dbscan(data, eps: eps, minSamples: minSamples)
}

/// - Note: Deprecated. Use ``Cluster/silhouetteScore(_:labels:)`` instead.
@available(*, deprecated, message: "Use Cluster.silhouetteScore(_:labels:) instead")
public func silhouetteScore(_ data: [[Double]], labels: [Int]) -> Double {
    Cluster.silhouetteScore(data, labels: labels)
}

/// - Note: Deprecated. Use ``Cluster/elbowMethod(_:maxK:)`` instead.
@available(*, deprecated, message: "Use Cluster.elbowMethod(_:maxK:) instead")
public func elbowMethod(_ data: [[Double]], maxK: Int = 10) -> ElbowResult {
    Cluster.elbowMethod(data, maxK: maxK)
}

/// - Note: Deprecated. Use ``Cluster/euclideanDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Cluster.euclideanDistance(_:_:) instead")
public func euclideanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    Cluster.euclideanDistance(p1, p2)
}

/// - Note: Deprecated. Use ``Cluster/squaredEuclideanDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Cluster.squaredEuclideanDistance(_:_:) instead")
public func squaredEuclideanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    Cluster.squaredEuclideanDistance(p1, p2)
}

/// - Note: Deprecated. Use ``Cluster/centroid(_:)`` instead.
///
/// Formerly public as `computeCentroid(_:)`. Renamed to `centroid(_:)` and moved to `Cluster`.
@available(*, deprecated, message: "Use Cluster.centroid(_:) instead")
public func computeCentroid(_ points: [[Double]]) -> [Double]? {
    Cluster.centroid(points)
}

/// - Note: Deprecated. This function was an internal implementation detail and should not have been public.
/// Use ``Cluster/kmeans(_:k:maxIterations:tolerance:nInit:initMethod:)`` instead.
@available(*, deprecated, message: "This was an internal implementation detail. Use Cluster.kmeans instead.")
public func kmeansPlusPlusInit(data: [[Double]], k: Int, dim: Int) -> [[Double]] {
    // Delegate to the private implementation via the namespace.
    // We replicate the logic here since kMeansPlusPlusInit is private.
    guard data.count > 0, k > 0 else { return [] }
    var centroids: [[Double]] = []
    let firstIdx = Int.random(in: 0..<data.count)
    centroids.append(data[firstIdx])
    for _ in 1..<k {
        var distances = [Double](repeating: 0, count: data.count)
        var totalDist: Double = 0
        for i in 0..<data.count {
            var minDist = Double.infinity
            for centroid in centroids {
                let d = Cluster.squaredEuclideanDistance(data[i], centroid)
                if d < minDist { minDist = d }
            }
            distances[i] = minDist
            totalDist += minDist
        }
        guard totalDist > 0 else { break }
        let r = Double.random(in: 0..<totalDist)
        var cumsum: Double = 0
        var chosenIdx = data.count - 1
        for i in 0..<data.count {
            cumsum += distances[i]
            if cumsum >= r { chosenIdx = i; break }
        }
        centroids.append(data[chosenIdx])
    }
    return centroids
}

/// - Note: Deprecated. Use ``Cluster/euclideanDistance(_:_:)`` for computing
/// pairwise distances in a loop, or the `Spatial` distance utilities.
@available(*, deprecated, message: "Use Cluster.euclideanDistance(_:_:) in a loop, or Spatial distance utilities.")
public func pairwiseDistances(_ data: [[Double]]) -> [[Double]] {
    let n = data.count
    var distances = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
    for i in 0..<n {
        for j in (i + 1)..<n {
            let d = Cluster.euclideanDistance(data[i], data[j])
            distances[i][j] = d
            distances[j][i] = d
        }
    }
    return distances
}
