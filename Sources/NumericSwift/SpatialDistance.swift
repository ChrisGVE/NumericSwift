//
//  SpatialDistance.swift
//  NumericSwift
//
//  Distance metrics and batch distance functions following scipy.spatial.distance patterns.
//
//  Licensed under the MIT License.
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

/// Get distance function by metric type.
///
/// Note: Mahalanobis distance requires an inverse covariance matrix and is not
/// available through this interface. Use `mahalanobisDistance(_:_:invCov:)` directly.
public func distanceFunction(for metric: DistanceMetric) -> ([Double], [Double]) -> Double {
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

// MARK: - Lp Norm Distances

/// Manhattan (cityblock/L1) distance using vDSP.
///
/// - Parameters:
///   - p1: First point
///   - p2: Second point
/// - Returns: Manhattan distance
public func manhattanDistance(_ p1: [Double], _ p2: [Double]) -> Double {
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

/// Alias for manhattanDistance.
public func cityblockDistance(_ p1: [Double], _ p2: [Double]) -> Double {
  return manhattanDistance(p1, p2)
}

/// Chebyshev (L∞) distance using vDSP.
///
/// - Parameters:
///   - p1: First point
///   - p2: Second point
/// - Returns: Chebyshev distance (maximum absolute difference)
public func chebyshevDistance(_ p1: [Double], _ p2: [Double]) -> Double {
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
public func minkowskiDistance(_ p1: [Double], _ p2: [Double], p: Double = 2) -> Double {
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

// MARK: - Angular / Similarity-Based Distances

/// Cosine distance using BLAS.
///
/// Cosine distance = 1 - cosine similarity
///
/// - Parameters:
///   - p1: First point
///   - p2: Second point
/// - Returns: Cosine distance (0 = identical direction, 2 = opposite)
public func cosineDistance(_ p1: [Double], _ p2: [Double]) -> Double {
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
/// Correlation distance = 1 - Pearson correlation coefficient
///
/// - Parameters:
///   - p1: First point
///   - p2: Second point
/// - Returns: Correlation distance
public func correlationDistance(_ p1: [Double], _ p2: [Double]) -> Double {
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

// MARK: - Set / Binary Distances

/// Jaccard distance for binary vectors.
///
/// Values are treated as binary: > 0 maps to 1, <= 0 maps to 0.
/// Jaccard distance = 1 - |intersection| / |union|.
/// Returns 0 when both vectors are all-zero.
///
/// - Parameters:
///   - a: First vector
///   - b: Second vector
/// - Returns: Jaccard distance in [0, 1]
public func jaccardDistance(_ a: [Double], _ b: [Double]) -> Double {
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
/// Values are compared directly (not binarized).
///
/// - Parameters:
///   - a: First vector
///   - b: Second vector
/// - Returns: Hamming distance in [0, 1]
public func hammingDistance(_ a: [Double], _ b: [Double]) -> Double {
  let n = min(a.count, b.count)
  guard n > 0 else { return 0 }

  var diffCount = 0
  for i in 0..<n where a[i] != b[i] {
    diffCount += 1
  }
  return Double(diffCount) / Double(n)
}

// MARK: - Weighted / Ratio Distances

/// Canberra distance.
///
/// Computes `sum(|a_i - b_i| / (|a_i| + |b_i|))`.
/// Terms where both values are zero are skipped.
///
/// - Parameters:
///   - a: First vector
///   - b: Second vector
/// - Returns: Canberra distance (>= 0)
public func canberraDistance(_ a: [Double], _ b: [Double]) -> Double {
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
/// Returns 0 when both vectors are all-zero.
///
/// - Parameters:
///   - a: First vector
///   - b: Second vector
/// - Returns: Bray-Curtis distance in [0, 1]
public func braycurtisDistance(_ a: [Double], _ b: [Double]) -> Double {
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

// MARK: - Mahalanobis Distance

/// Mahalanobis distance using a precomputed inverse covariance matrix.
///
/// Computes `sqrt((a - b)^T * invCov * (a - b))`.
/// The inverse covariance matrix must be positive semi-definite and have
/// dimensions matching the vector length.
///
/// - Parameters:
///   - a: First vector
///   - b: Second vector
///   - invCov: Inverse of the covariance matrix (d × d)
/// - Returns: Mahalanobis distance (>= 0)
public func mahalanobisDistance(_ a: [Double], _ b: [Double], invCov: [[Double]]) -> Double {
  let n = min(a.count, b.count)
  guard n > 0, invCov.count == n, invCov.allSatisfy({ $0.count == n }) else { return 0 }

  // diff = a - b
  var diff = [Double](repeating: 0, count: n)
  vDSP_vsubD(b, 1, a, 1, &diff, 1, vDSP_Length(n))

  // tmp = invCov * diff  (matrix-vector product)
  var tmp = [Double](repeating: 0, count: n)
  let flatInvCov = invCov.flatMap { $0 }
  cblas_dgemv(
    CblasRowMajor, CblasNoTrans,
    Int32(n), Int32(n),
    1.0, flatInvCov, Int32(n),
    diff, 1,
    0.0, &tmp, 1
  )

  // result = diff . tmp
  let dist2 = cblas_ddot(Int32(n), diff, 1, tmp, 1)
  return Darwin.sqrt(max(dist2, 0))
}

// MARK: - Batch Distance Functions

/// Compute pairwise distances between two sets of points.
///
/// - Parameters:
///   - XA: First set of points (m × d)
///   - XB: Second set of points (n × d)
///   - metric: Distance metric (default .euclidean)
/// - Returns: Distance matrix (m × n)
public func cdist(
  _ XA: [[Double]],
  _ XB: [[Double]],
  metric: DistanceMetric = .euclidean
) -> [[Double]] {
  let m = XA.count
  let n = XB.count
  let distFunc = distanceFunction(for: metric)

  var result = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)

  if m * n > 1000 {
    DispatchQueue.concurrentPerform(iterations: m) { i in
      for j in 0..<n {
        result[i][j] = distFunc(XA[i], XB[j])
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
public func pdist(_ X: [[Double]], metric: DistanceMetric = .euclidean) -> [Double] {
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
public func squareform(_ X: [[Double]]) -> [Double] {
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
public func squareformToMatrix(_ condensed: [Double]) -> [[Double]] {
  let m = condensed.count
  let n = Int((1.0 + Darwin.sqrt(1.0 + 8.0 * Double(m))) / 2.0)
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
