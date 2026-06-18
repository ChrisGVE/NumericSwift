//
//  SpatialDistance.swift
//  NumericSwift
//
//  Distance metrics and batch distance functions following scipy.spatial.distance patterns.
//
//  Licensed under the Apache License, Version 2.0.
//

import Accelerate
import Foundation

// MARK: - Distance Metric Enumeration

/// Distance metric enumeration.
public enum DistanceMetric: String {
  case euclidean
  case sqeuclidean
  case cityblock
  case manhattan
  case chebyshev
  case cosine
  case correlation
  case jaccard
  case hamming
  case canberra
  case braycurtis
}

// MARK: - Spatial Namespace

/// Spatial data structures and algorithms.
///
/// Provides distance metrics, batch distance computation, nearest-neighbour
/// search, Delaunay triangulation, Voronoi diagrams, and convex hulls.
///
/// ## Overview
///
/// ```swift
/// let d = Spatial.euclideanDistance([0, 0], [3, 4])   // 5.0
/// let pairs = Spatial.cdist(setA, setB)
/// let tri = Spatial.delaunay(points)
/// ```
public enum Spatial {

    // MARK: - Core Distance Functions

    /// Euclidean distance between two points using BLAS.
    ///
    /// - Parameters:
    ///   - p1: First point
    ///   - p2: Second point
    /// - Returns: Euclidean distance
    public static func euclideanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        let n = min(p1.count, p2.count)
        guard n > 0 else { return 0 }
        var diff = [Double](repeating: 0, count: n)
        vDSP_vsubD(p2, 1, p1, 1, &diff, 1, vDSP_Length(n))
        return cblas_dnrm2(Int32(n), diff, 1)
    }

    /// Squared Euclidean distance between two points using vDSP.
    ///
    /// More efficient than ``euclideanDistance(_:_:)`` when only comparing distances (avoids sqrt).
    ///
    /// - Parameters:
    ///   - p1: First point
    ///   - p2: Second point
    /// - Returns: Squared Euclidean distance
    public static func squaredEuclideanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        let n = min(p1.count, p2.count)
        guard n > 0 else { return 0 }
        var diff = [Double](repeating: 0, count: n)
        vDSP_vsubD(p2, 1, p1, 1, &diff, 1, vDSP_Length(n))
        var result: Double = 0
        vDSP_dotprD(diff, 1, diff, 1, &result, vDSP_Length(n))
        return result
    }

    /// Manhattan (cityblock/L1) distance using vDSP.
    ///
    /// - Parameters:
    ///   - p1: First point
    ///   - p2: Second point
    /// - Returns: Manhattan distance
    public static func manhattanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        let n = min(p1.count, p2.count)
        guard n > 0 else { return 0 }
        var diff = [Double](repeating: 0, count: n)
        var absDiff = [Double](repeating: 0, count: n)
        vDSP_vsubD(p2, 1, p1, 1, &diff, 1, vDSP_Length(n))
        vDSP_vabsD(diff, 1, &absDiff, 1, vDSP_Length(n))
        var result: Double = 0
        vDSP_sveD(absDiff, 1, &result, vDSP_Length(n))
        return result
    }

    /// Alias for ``manhattanDistance(_:_:)``.
    public static func cityblockDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        manhattanDistance(p1, p2)
    }

    /// Chebyshev (L∞) distance using vDSP.
    ///
    /// - Parameters:
    ///   - p1: First point
    ///   - p2: Second point
    /// - Returns: Chebyshev distance (maximum absolute difference)
    public static func chebyshevDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        let n = min(p1.count, p2.count)
        guard n > 0 else { return 0 }
        var diff = [Double](repeating: 0, count: n)
        var absDiff = [Double](repeating: 0, count: n)
        vDSP_vsubD(p2, 1, p1, 1, &diff, 1, vDSP_Length(n))
        vDSP_vabsD(diff, 1, &absDiff, 1, vDSP_Length(n))
        var result: Double = 0
        vDSP_maxvD(absDiff, 1, &result, vDSP_Length(n))
        return result
    }

    /// Minkowski distance (generalized Lp norm).
    ///
    /// - Parameters:
    ///   - p1: First point
    ///   - p2: Second point
    ///   - p: Order of the norm (default 2 = Euclidean)
    /// - Returns: Minkowski distance
    public static func minkowskiDistance(_ p1: [Double], _ p2: [Double], p: Double = 2) -> Double {
        let n = min(p1.count, p2.count)
        guard n > 0, p > 0 else { return 0 }
        if p == 1 { return manhattanDistance(p1, p2) }
        if p == 2 { return euclideanDistance(p1, p2) }
        if p == .infinity { return chebyshevDistance(p1, p2) }
        var sum: Double = 0
        for i in 0..<n {
            sum += Darwin.pow(abs(p1[i] - p2[i]), p)
        }
        return Darwin.pow(sum, 1.0 / p)
    }

    /// Cosine distance using BLAS.
    ///
    /// Cosine distance = 1 - cosine similarity.
    ///
    /// - Parameters:
    ///   - p1: First point
    ///   - p2: Second point
    /// - Returns: Cosine distance (0 = identical direction, 2 = opposite)
    public static func cosineDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        let n = Int32(min(p1.count, p2.count))
        guard n > 0 else { return 1.0 }
        let dot = cblas_ddot(n, p1, 1, p2, 1)
        let norm1 = cblas_dnrm2(n, p1, 1)
        let norm2 = cblas_dnrm2(n, p2, 1)
        let denom = norm1 * norm2
        if denom < 1e-15 { return 1.0 }
        return 1.0 - dot / denom
    }

    /// Correlation distance using BLAS.
    ///
    /// Correlation distance = 1 - Pearson correlation coefficient.
    ///
    /// - Parameters:
    ///   - p1: First point
    ///   - p2: Second point
    /// - Returns: Correlation distance
    public static func correlationDistance(_ p1: [Double], _ p2: [Double]) -> Double {
        let n = min(p1.count, p2.count)
        guard n > 0 else { return 1.0 }
        var mean1: Double = 0
        var mean2: Double = 0
        vDSP_meanvD(p1, 1, &mean1, vDSP_Length(n))
        vDSP_meanvD(p2, 1, &mean2, vDSP_Length(n))
        var centered1 = [Double](repeating: 0, count: n)
        var centered2 = [Double](repeating: 0, count: n)
        var negMean1 = -mean1
        var negMean2 = -mean2
        vDSP_vsaddD(p1, 1, &negMean1, &centered1, 1, vDSP_Length(n))
        vDSP_vsaddD(p2, 1, &negMean2, &centered2, 1, vDSP_Length(n))
        let dot = cblas_ddot(Int32(n), centered1, 1, centered2, 1)
        let norm1 = cblas_dnrm2(Int32(n), centered1, 1)
        let norm2 = cblas_dnrm2(Int32(n), centered2, 1)
        let denom = norm1 * norm2
        if denom < 1e-15 { return 1.0 }
        return 1.0 - dot / denom
    }

    /// Jaccard distance for binary vectors.
    ///
    /// Values are treated as binary: > 0 maps to 1, ≤ 0 maps to 0.
    /// Returns 0 when both vectors are all-zero.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Jaccard distance in [0, 1]
    public static func jaccardDistance(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var intersection = 0
        var union = 0
        for i in 0..<n {
            let ai = a[i] > 0
            let bi = b[i] > 0
            if ai || bi { union += 1 }
            if ai && bi { intersection += 1 }
        }
        guard union > 0 else { return 0 }
        return 1.0 - Double(intersection) / Double(union)
    }

    /// Hamming distance (fraction of differing elements).
    ///
    /// Returns the proportion of positions where the two vectors differ.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Hamming distance in [0, 1]
    public static func hammingDistance(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var diffCount = 0
        for i in 0..<n where a[i] != b[i] {
            diffCount += 1
        }
        return Double(diffCount) / Double(n)
    }

    /// Canberra distance.
    ///
    /// Computes `sum(|a_i - b_i| / (|a_i| + |b_i|))`.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Canberra distance (≥ 0)
    public static func canberraDistance(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var sum = 0.0
        for i in 0..<n {
            let denom = abs(a[i]) + abs(b[i])
            guard denom > 0 else { continue }
            sum += abs(a[i] - b[i]) / denom
        }
        return sum
    }

    /// Bray-Curtis distance.
    ///
    /// Computes `sum(|a_i - b_i|) / sum(|a_i + b_i|)`.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Bray-Curtis distance in [0, 1]
    public static func braycurtisDistance(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var numerator = 0.0
        var denominator = 0.0
        for i in 0..<n {
            numerator += abs(a[i] - b[i])
            denominator += abs(a[i] + b[i])
        }
        guard denominator > 0 else { return 0 }
        return numerator / denominator
    }

    /// Mahalanobis distance using a precomputed inverse covariance matrix.
    ///
    /// Computes `sqrt((a - b)^T * invCov * (a - b))`.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    ///   - invCov: Inverse of the covariance matrix (d × d)
    /// - Returns: Mahalanobis distance (≥ 0)
    public static func mahalanobisDistance(_ a: [Double], _ b: [Double], invCov: [[Double]]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0, invCov.count == n, invCov.allSatisfy({ $0.count == n }) else { return 0 }
        var diff = [Double](repeating: 0, count: n)
        vDSP_vsubD(b, 1, a, 1, &diff, 1, vDSP_Length(n))
        var tmp = [Double](repeating: 0, count: n)
        let flatInvCov = invCov.flatMap { $0 }
        cblas_dgemv(CblasRowMajor, CblasNoTrans, Int32(n), Int32(n), 1.0, flatInvCov, Int32(n), diff, 1, 0.0, &tmp, 1)
        let dist2 = cblas_ddot(Int32(n), diff, 1, tmp, 1)
        return Darwin.sqrt(max(dist2, 0))
    }

    // MARK: - Distance Function Lookup

    /// Get distance function by metric type.
    ///
    /// - Note: Mahalanobis distance requires an inverse covariance matrix and is not
    ///   available through this interface. Use ``mahalanobisDistance(_:_:invCov:)`` directly.
    public static func distanceFunction(for metric: DistanceMetric) -> ([Double], [Double]) -> Double {
        switch metric {
        case .euclidean: return euclideanDistance
        case .sqeuclidean: return squaredEuclideanDistance
        case .cityblock, .manhattan: return manhattanDistance
        case .chebyshev: return chebyshevDistance
        case .cosine: return cosineDistance
        case .correlation: return correlationDistance
        case .jaccard: return jaccardDistance
        case .hamming: return hammingDistance
        case .canberra: return canberraDistance
        case .braycurtis: return braycurtisDistance
        }
    }

    // MARK: - Batch Distance Functions

    /// Compute pairwise distances between two sets of points.
    ///
    /// - Parameters:
    ///   - XA: First set of points (m × d)
    ///   - XB: Second set of points (n × d)
    ///   - metric: Distance metric (default .euclidean)
    /// - Returns: Distance matrix (m × n)
    public static func cdist(
        _ XA: [[Double]],
        _ XB: [[Double]],
        metric: DistanceMetric = .euclidean
    ) -> [[Double]] {
        let m = XA.count
        let n = XB.count
        let distFunc = distanceFunction(for: metric)

        var result = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)

        if m * n > 1000 {
            // Each iteration writes a distinct row. Mutating a shared Swift
            // `[[Double]]` from `concurrentPerform` is a data race (copy-on-write
            // touches the shared buffer/refcount across threads). Write through an
            // `UnsafeMutableBufferPointer`, where distinct-index stores from
            // different threads are well-defined, and build each row locally first.
            result.withUnsafeMutableBufferPointer { buf in
                DispatchQueue.concurrentPerform(iterations: m) { i in
                    var row = [Double](repeating: 0, count: n)
                    let xa = XA[i]
                    for j in 0..<n {
                        row[j] = distFunc(xa, XB[j])
                    }
                    buf[i] = row
                }
            }
        } else {
            for i in 0..<m {
                for j in 0..<n {
                    result[i][j] = distFunc(XA[i], XB[j])
                }
            }
        }

        return result
    }

    /// Compute pairwise distances within a set of points (condensed form).
    ///
    /// Returns distances in condensed form: for n points, returns n*(n-1)/2 distances
    /// representing the upper triangle of the distance matrix.
    ///
    /// - Parameters:
    ///   - X: Set of points
    ///   - metric: Distance metric (default .euclidean)
    /// - Returns: Condensed distance array
    public static func pdist(_ X: [[Double]], metric: DistanceMetric = .euclidean) -> [Double] {
        let n = X.count
        let numPairs = n * (n - 1) / 2
        let distFunc = distanceFunction(for: metric)

        var result = [Double](repeating: 0, count: numPairs)
        var k = 0
        for i in 0..<(n - 1) {
            for j in (i + 1)..<n {
                result[k] = distFunc(X[i], X[j])
                k += 1
            }
        }

        return result
    }

    /// Convert a square distance matrix to condensed form.
    ///
    /// - Parameter X: Square symmetric distance matrix (n × n)
    /// - Returns: Condensed distance array (n*(n-1)/2 elements)
    public static func squareform(_ X: [[Double]]) -> [Double] {
        let n = X.count
        var result: [Double] = []
        for i in 0..<(n - 1) {
            for j in (i + 1)..<n {
                result.append(X[i][j])
            }
        }
        return result
    }

    /// Convert a condensed distance array to a square symmetric matrix.
    ///
    /// - Parameter condensed: Condensed distance array (n*(n-1)/2 elements)
    /// - Returns: Square distance matrix (n × n)
    public static func squareformToMatrix(_ condensed: [Double]) -> [[Double]] {
        let m = condensed.count
        let n = Int((1.0 + Darwin.sqrt(1.0 + 8.0 * Double(m))) / 2.0)
        // A valid condensed array has exactly n(n-1)/2 elements. If `m` is not a
        // triangular number the derived `n` is wrong and condensed[k] would
        // over-read; reject such input with an empty result rather than trap.
        guard m == n * (n - 1) / 2 else { return [] }
        var result = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        var k = 0
        for i in 0..<(n - 1) {
            for j in (i + 1)..<n {
                result[i][j] = condensed[k]
                result[j][i] = condensed[k]
                k += 1
            }
        }
        return result
    }
}

// MARK: - Deprecated shims (backward compatibility)

/// - Note: Deprecated. Use ``Spatial/distanceFunction(for:)`` instead.
@available(*, deprecated, message: "Use Spatial.distanceFunction(for:) instead")
public func distanceFunction(for metric: DistanceMetric) -> ([Double], [Double]) -> Double {
    Spatial.distanceFunction(for: metric)
}

/// - Note: Deprecated. Use ``Spatial/manhattanDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.manhattanDistance(_:_:) instead")
public func manhattanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    Spatial.manhattanDistance(p1, p2)
}

/// - Note: Deprecated. Use ``Spatial/cityblockDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.cityblockDistance(_:_:) instead")
public func cityblockDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    Spatial.cityblockDistance(p1, p2)
}

/// - Note: Deprecated. Use ``Spatial/chebyshevDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.chebyshevDistance(_:_:) instead")
public func chebyshevDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    Spatial.chebyshevDistance(p1, p2)
}

/// - Note: Deprecated. Use ``Spatial/minkowskiDistance(_:_:p:)`` instead.
@available(*, deprecated, message: "Use Spatial.minkowskiDistance(_:_:p:) instead")
public func minkowskiDistance(_ p1: [Double], _ p2: [Double], p: Double = 2) -> Double {
    Spatial.minkowskiDistance(p1, p2, p: p)
}

/// - Note: Deprecated. Use ``Spatial/cosineDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.cosineDistance(_:_:) instead")
public func cosineDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    Spatial.cosineDistance(p1, p2)
}

/// - Note: Deprecated. Use ``Spatial/correlationDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.correlationDistance(_:_:) instead")
public func correlationDistance(_ p1: [Double], _ p2: [Double]) -> Double {
    Spatial.correlationDistance(p1, p2)
}

/// - Note: Deprecated. Use ``Spatial/jaccardDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.jaccardDistance(_:_:) instead")
public func jaccardDistance(_ a: [Double], _ b: [Double]) -> Double {
    Spatial.jaccardDistance(a, b)
}

/// - Note: Deprecated. Use ``Spatial/hammingDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.hammingDistance(_:_:) instead")
public func hammingDistance(_ a: [Double], _ b: [Double]) -> Double {
    Spatial.hammingDistance(a, b)
}

/// - Note: Deprecated. Use ``Spatial/canberraDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.canberraDistance(_:_:) instead")
public func canberraDistance(_ a: [Double], _ b: [Double]) -> Double {
    Spatial.canberraDistance(a, b)
}

/// - Note: Deprecated. Use ``Spatial/braycurtisDistance(_:_:)`` instead.
@available(*, deprecated, message: "Use Spatial.braycurtisDistance(_:_:) instead")
public func braycurtisDistance(_ a: [Double], _ b: [Double]) -> Double {
    Spatial.braycurtisDistance(a, b)
}

/// - Note: Deprecated. Use ``Spatial/mahalanobisDistance(_:_:invCov:)`` instead.
@available(*, deprecated, message: "Use Spatial.mahalanobisDistance(_:_:invCov:) instead")
public func mahalanobisDistance(_ a: [Double], _ b: [Double], invCov: [[Double]]) -> Double {
    Spatial.mahalanobisDistance(a, b, invCov: invCov)
}

/// - Note: Deprecated. Use ``Spatial/cdist(_:_:metric:)`` instead.
@available(*, deprecated, message: "Use Spatial.cdist(_:_:metric:) instead")
public func cdist(
    _ XA: [[Double]],
    _ XB: [[Double]],
    metric: DistanceMetric = .euclidean
) -> [[Double]] {
    Spatial.cdist(XA, XB, metric: metric)
}

/// - Note: Deprecated. Use ``Spatial/pdist(_:metric:)`` instead.
@available(*, deprecated, message: "Use Spatial.pdist(_:metric:) instead")
public func pdist(_ X: [[Double]], metric: DistanceMetric = .euclidean) -> [Double] {
    Spatial.pdist(X, metric: metric)
}

/// - Note: Deprecated. Use ``Spatial/squareform(_:)`` instead.
@available(*, deprecated, message: "Use Spatial.squareform(_:) instead")
public func squareform(_ X: [[Double]]) -> [Double] {
    Spatial.squareform(X)
}

/// - Note: Deprecated. Use ``Spatial/squareformToMatrix(_:)`` instead.
@available(*, deprecated, message: "Use Spatial.squareformToMatrix(_:) instead")
public func squareformToMatrix(_ condensed: [Double]) -> [[Double]] {
    Spatial.squareformToMatrix(condensed)
}
