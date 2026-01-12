//
//  Cluster.swift
//  NumericSwift
//
//  Clustering algorithms following sklearn.cluster patterns.
//
//  Licensed under the MIT License.
//

import Foundation
import Accelerate

// MARK: - K-Means Clustering

/// Result of k-means clustering.
public struct KMeansResult {
    /// Cluster labels for each data point (0-indexed).
    public let labels: [Int]
    /// Final cluster centroids.
    public let centroids: [[Double]]
    /// Sum of squared distances to nearest centroid.
    public let inertia: Double
    /// Number of iterations run.
    public let iterations: Int
}

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
public func kmeans(
    _ data: [[Double]],
    k: Int,
    maxIterations: Int = 300,
    tolerance: Double = 1e-4,
    nInit: Int = 10,
    initMethod: String = "kmeans++"
) -> KMeansResult {
    guard !data.isEmpty, k > 0 else {
        return KMeansResult(labels: [], centroids: [], inertia: 0, iterations: 0)
    }

    let dim = data[0].count

    // Run n_init times and keep best result
    var bestLabels: [Int] = []
    var bestCentroids: [[Double]] = []
    var bestInertia = Double.infinity
    var bestIterations = 0

    for _ in 0..<nInit {
        // Initialize centroids
        let initialCentroids: [[Double]]
        if initMethod == "kmeans++" {
            initialCentroids = kmeansPlusPlusInit(data: data, k: k, dim: dim)
        } else {
            initialCentroids = randomInit(data: data, k: k)
        }

        // Run k-means
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

/// K-means++ initialization for better initial centroids.
///
/// Selects initial centroids with probability proportional to squared distance
/// from nearest existing centroid. This leads to better convergence.
///
/// - Parameters:
///   - data: Data points
///   - k: Number of centroids to select
///   - dim: Dimensionality
/// - Returns: Array of initial centroids
public func kmeansPlusPlusInit(data: [[Double]], k: Int, dim: Int) -> [[Double]] {
    let n = data.count
    guard n > 0, k > 0 else { return [] }

    var centroids: [[Double]] = []

    // Choose first centroid randomly
    let firstIdx = Int.random(in: 0..<n)
    centroids.append(data[firstIdx])

    for _ in 1..<k {
        // Compute D(x)^2 for each point (squared distance to nearest centroid)
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

        // Choose next centroid with probability proportional to D(x)^2
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

/// Random initialization for k-means.
private func randomInit(data: [[Double]], k: Int) -> [[Double]] {
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

/// Run single k-means clustering.
private func runKMeans(
    data: [[Double]], k: Int, initialCentroids: [[Double]],
    maxIter: Int, tol: Double
) -> (labels: [Int], centroids: [[Double]], inertia: Double, iterations: Int) {
    let n = data.count
    var centroids = initialCentroids
    var labels = [Int](repeating: 0, count: n)
    var nIter = 0

    for iter in 1...maxIter {
        nIter = iter

        // Assignment step
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

        // Update step
        var newCentroids: [[Double]] = []
        for c in 0..<k {
            if clusters[c].isEmpty {
                newCentroids.append(centroids[c])
            } else {
                let clusterPoints = clusters[c].map { data[$0] }
                if let centroid = computeCentroid(clusterPoints) {
                    newCentroids.append(centroid)
                } else {
                    newCentroids.append(centroids[c])
                }
            }
        }

        // Check convergence
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

    // Compute final inertia
    var finalInertia: Double = 0
    for i in 0..<n {
        finalInertia += squaredEuclideanDistance(data[i], centroids[labels[i]])
    }

    return (labels, centroids, finalInertia, nIter)
}

// MARK: - Hierarchical Clustering

/// Result of hierarchical clustering.
public struct HierarchicalResult {
    /// Linkage matrix (n-1 × 4): [cluster1, cluster2, distance, size]
    public let linkageMatrix: [[Double]]
    /// Cluster labels if tree was cut (0-indexed), nil otherwise.
    public let labels: [Int]?
    /// Number of original samples.
    public let nLeaves: Int
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
public func hierarchicalClustering(
    _ data: [[Double]],
    linkage: LinkageMethod = .ward,
    nClusters: Int? = nil,
    distanceThreshold: Double? = nil
) -> HierarchicalResult {
    let n = data.count
    guard n > 0 else {
        return HierarchicalResult(linkageMatrix: [], labels: nil, nLeaves: 0)
    }

    // Initialize clusters
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

    // Build linkage matrix
    var linkageMatrix: [[Double]] = []
    var nextClusterId = n

    for _ in 0..<(n - 1) {
        // Find closest pair
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

        // Merge clusters
        var mergedPoints = c1.points
        mergedPoints.append(contentsOf: c2.points)

        let newSize = c1.size + c2.size
        var newCentroid = [Double](repeating: 0, count: c1.centroid.count)
        for d in 0..<newCentroid.count {
            newCentroid[d] = (c1.centroid[d] * Double(c1.size) + c2.centroid[d] * Double(c2.size)) / Double(newSize)
        }

        // Record in linkage matrix
        linkageMatrix.append([Double(c1.id), Double(c2.id), minDist, Double(newSize)])

        // Deactivate merged clusters
        clusters[bestI].active = false
        clusters[bestJ].active = false

        // Create new cluster
        clusters.append(HierarchicalCluster(
            id: nextClusterId,
            points: mergedPoints,
            centroid: newCentroid,
            size: newSize,
            active: true
        ))
        nextClusterId += 1
    }

    // Cut tree if requested
    var labels: [Int]? = nil
    if nClusters != nil || distanceThreshold != nil {
        let cutIdx: Int
        if let nc = nClusters {
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
        nLeaves: n
    )
}

/// Helper struct for hierarchical clustering.
private struct HierarchicalCluster {
    let id: Int
    var points: [Int]
    var centroid: [Double]
    var size: Int
    var active: Bool
}

/// Compute distance between clusters based on linkage method.
private func clusterDistance(
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

/// Cut hierarchical tree to produce labels.
private func cutHierarchy(linkageMatrix: [[Double]], n: Int, cutIdx: Int) -> [Int] {
    // Start with each point as its own cluster
    var clusterMap = [Int](0..<n)
    var nextLabel = n

    // Apply merges up to cutIdx
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

    // Convert to consecutive labels starting from 0
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

// MARK: - DBSCAN Clustering

/// Result of DBSCAN clustering.
public struct DBSCANResult {
    /// Cluster labels (-1 for noise, 0+ for clusters).
    public let labels: [Int]
    /// Indices of core samples.
    public let coreSamples: [Int]
    /// Number of clusters found.
    public let nClusters: Int
}

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
///
/// Finds clusters based on density. Points in dense regions are grouped together,
/// sparse points are marked as noise (-1).
///
/// - Parameters:
///   - data: Array of data points
///   - eps: Maximum distance for neighborhood (default 0.5)
///   - minSamples: Minimum points to form a dense region (default 5)
/// - Returns: DBSCANResult with labels, core samples, and cluster count
public func dbscan(
    _ data: [[Double]],
    eps: Double = 0.5,
    minSamples: Int = 5
) -> DBSCANResult {
    let n = data.count
    guard n > 0 else {
        return DBSCANResult(labels: [], coreSamples: [], nClusters: 0)
    }

    // Find neighbors for each point
    var neighborsCache: [[Int]] = []
    var isCore = [Bool](repeating: false, count: n)

    for i in 0..<n {
        var neighbors: [Int] = []
        for j in 0..<n {
            if euclideanDistance(data[i], data[j]) <= eps {
                neighbors.append(j)
            }
        }
        neighborsCache.append(neighbors)
        isCore[i] = neighbors.count >= minSamples
    }

    // Assign clusters
    var labels = [Int](repeating: -2, count: n)  // -2 = unvisited
    var currentCluster = -1
    var coreSamples: [Int] = []

    for i in 0..<n {
        if labels[i] != -2 {
            continue
        }
        if !isCore[i] {
            labels[i] = -1  // Noise
            continue
        }

        // Start new cluster
        currentCluster += 1
        coreSamples.append(i)

        // BFS expansion
        var queue = [i]
        labels[i] = currentCluster

        while !queue.isEmpty {
            let current = queue.removeFirst()

            for neighbor in neighborsCache[current] {
                if labels[neighbor] == -1 {
                    // Was noise, now border point
                    labels[neighbor] = currentCluster
                } else if labels[neighbor] == -2 {
                    labels[neighbor] = currentCluster
                    if isCore[neighbor] {
                        queue.append(neighbor)
                        coreSamples.append(neighbor)
                    }
                }
            }
        }
    }

    return DBSCANResult(
        labels: labels,
        coreSamples: coreSamples,
        nClusters: currentCluster + 1
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
public func silhouetteScore(_ data: [[Double]], labels: [Int]) -> Double {
    let n = data.count
    guard n >= 2, labels.count == n else { return 0 }

    // Group points by cluster (excluding noise)
    var clusters: [Int: [Int]] = [:]
    for (i, label) in labels.enumerated() {
        if label >= 0 {
            clusters[label, default: []].append(i)
        }
    }

    // Need at least 2 clusters
    guard clusters.count >= 2 else { return 0 }

    var totalSilhouette: Double = 0
    var count = 0

    for i in 0..<n {
        let labelI = labels[i]
        if labelI < 0 { continue }

        guard let sameCluster = clusters[labelI] else { continue }

        // a(i): mean distance to same cluster
        var aI: Double = 0
        if sameCluster.count > 1 {
            for j in sameCluster {
                if j != i {
                    aI += euclideanDistance(data[i], data[j])
                }
            }
            aI /= Double(sameCluster.count - 1)
        }

        // b(i): min mean distance to other clusters
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

        // Silhouette coefficient
        let maxAB = max(aI, bI)
        let sI = maxAB > 0 ? (bI - aI) / maxAB : 0

        totalSilhouette += sI
        count += 1
    }

    return count > 0 ? totalSilhouette / Double(count) : 0
}

/// Result of elbow method analysis.
public struct ElbowResult {
    /// Inertias for each k value (index 0 = k=1).
    public let inertias: [Double]
    /// Suggested optimal k based on elbow heuristic.
    public let suggestedK: Int
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
public func elbowMethod(_ data: [[Double]], maxK: Int = 10) -> ElbowResult {
    guard !data.isEmpty else {
        return ElbowResult(inertias: [], suggestedK: 1)
    }

    let actualMaxK = min(maxK, data.count)
    var inertias: [Double] = []

    for k in 1...actualMaxK {
        let result = kmeans(data, k: k, nInit: 3)  // Reduced nInit for speed
        inertias.append(result.inertia)
    }

    // Find elbow using maximum curvature heuristic
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

// MARK: - Distance Functions (Accelerate-optimized)

/// Euclidean distance between two points using BLAS.
///
/// - Parameters:
///   - p1: First point
///   - p2: Second point
/// - Returns: Euclidean distance
public func euclideanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    let n = min(p1.count, p2.count)
    guard n > 0 else { return 0 }

    var diff = [Double](repeating: 0, count: n)
    vDSP_vsubD(p2, 1, p1, 1, &diff, 1, vDSP_Length(n))
    return cblas_dnrm2(Int32(n), diff, 1)
}

/// Squared Euclidean distance between two points using vDSP.
///
/// More efficient when only comparing distances (avoids sqrt).
///
/// - Parameters:
///   - p1: First point
///   - p2: Second point
/// - Returns: Squared Euclidean distance
public func squaredEuclideanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    let n = min(p1.count, p2.count)
    guard n > 0 else { return 0 }

    var diff = [Double](repeating: 0, count: n)
    vDSP_vsubD(p2, 1, p1, 1, &diff, 1, vDSP_Length(n))
    var result: Double = 0
    vDSP_dotprD(diff, 1, diff, 1, &result, vDSP_Length(n))
    return result
}

/// Compute centroid of a set of points using vDSP.
///
/// - Parameter points: Array of points
/// - Returns: Centroid coordinates, or nil if empty
public func computeCentroid(_ points: [[Double]]) -> [Double]? {
    guard !points.isEmpty else { return nil }
    let dim = points[0].count
    var centroid = [Double](repeating: 0, count: dim)

    for point in points {
        vDSP_vaddD(centroid, 1, point, 1, &centroid, 1, vDSP_Length(dim))
    }

    var divisor = Double(points.count)
    vDSP_vsdivD(centroid, 1, &divisor, &centroid, 1, vDSP_Length(dim))

    return centroid
}

/// Compute pairwise distance matrix between all points.
///
/// - Parameter data: Array of points
/// - Returns: Distance matrix (symmetric, diagonal = 0)
public func pairwiseDistances(_ data: [[Double]]) -> [[Double]] {
    let n = data.count
    var distances = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)

    for i in 0..<n {
        for j in (i + 1)..<n {
            let d = euclideanDistance(data[i], data[j])
            distances[i][j] = d
            distances[j][i] = d
        }
    }

    return distances
}
