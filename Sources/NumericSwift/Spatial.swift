//
//  Spatial.swift
//  NumericSwift
//
//  Spatial data structures: KDTree, Delaunay triangulation, Voronoi diagram, convex hull.
//  Distance metrics and batch functions are in SpatialDistance.swift.
//
//  Licensed under the Apache License, Version 2.0.
//

import Accelerate
import Foundation

// MARK: - KDTree (public types, top-level)

/// KDTree node for efficient spatial queries.
public class KDTreeNode {
  public let index: Int
  public let point: [Double]
  public let axis: Int
  public var left: KDTreeNode?
  public var right: KDTreeNode?

  public init(index: Int, point: [Double], axis: Int) {
    self.index = index
    self.point = point
    self.axis = axis
  }
}

/// K-dimensional tree for efficient nearest neighbor queries.
public class KDTree {
  /// Original points.
  public let points: [[Double]]
  /// Dimensionality.
  public let dim: Int
  /// Root node.
  public let root: KDTreeNode?

  /// Build a KDTree from points.
  ///
  /// A zero-dimensional (`[[]]`) or ragged point set builds an empty tree rather
  /// than trapping: `depth % dim` would divide by zero when `dim == 0`, and a
  /// ragged row would trap the median sort at `points[$0][axis]`. Queries on an
  /// empty tree return no neighbours.
  public init(_ points: [[Double]]) {
    let d = points.first?.count ?? 0
    guard d > 0, points.allSatisfy({ $0.count == d }) else {
      self.points = []
      self.dim = 0
      self.root = nil
      return
    }
    self.points = points
    self.dim = d

    var indices = Array(0..<points.count)
    self.root = KDTree.buildTree(points: points, indices: &indices, depth: 0, dim: d)
  }

  private static func buildTree(
    points: [[Double]],
    indices: inout [Int],
    depth: Int,
    dim: Int
  ) -> KDTreeNode? {
    guard !indices.isEmpty else { return nil }

    let axis = depth % dim

    indices.sort { points[$0][axis] < points[$1][axis] }

    let mid = indices.count / 2
    let node = KDTreeNode(index: indices[mid], point: points[indices[mid]], axis: axis)

    var leftIndices = Array(indices[0..<mid])
    var rightIndices = Array(indices[(mid + 1)...])

    node.left = buildTree(points: points, indices: &leftIndices, depth: depth + 1, dim: dim)
    node.right = buildTree(points: points, indices: &rightIndices, depth: depth + 1, dim: dim)

    return node
  }

  /// Find k nearest neighbors to a query point.
  ///
  /// - Parameters:
  ///   - point: Query point
  ///   - k: Number of neighbors
  /// - Returns: Tuple of (indices, distances) sorted by distance
  public func query(_ point: [Double], k: Int = 1) -> (indices: [Int], distances: [Double]) {
    // A query vector shorter than the tree dimension would trap at
    // point[node.axis]; an empty tree (root == nil) yields no neighbours.
    guard point.count >= dim, dim > 0 else { return ([], []) }
    var best: [(idx: Int, dist: Double)] = []

    func search(_ node: KDTreeNode?) {
      guard let node = node else { return }

      let dist = Spatial.euclideanDistance(point, node.point)

      if best.count < k || dist < best.last!.dist {
        var pos = best.count
        for i in 0..<best.count {
          if dist < best[i].dist {
            pos = i
            break
          }
        }
        best.insert((node.index, dist), at: pos)
        if best.count > k {
          best.removeLast()
        }
      }

      let diff = point[node.axis] - node.point[node.axis]
      let near = diff < 0 ? node.left : node.right
      let far = diff < 0 ? node.right : node.left

      search(near)

      if best.count < k || abs(diff) < best.last!.dist {
        search(far)
      }
    }

    search(root)

    return (best.map { $0.idx }, best.map { $0.dist })
  }

  /// Find all points within a radius.
  ///
  /// - Parameters:
  ///   - point: Query point
  ///   - radius: Search radius
  /// - Returns: Tuple of (indices, distances) sorted by distance
  public func queryRadius(_ point: [Double], radius: Double) -> (
    indices: [Int], distances: [Double]
  ) {
    // Guard against a short query vector (point[node.axis] would trap) and an
    // empty tree.
    guard point.count >= dim, dim > 0 else { return ([], []) }
    var result: [(idx: Int, dist: Double)] = []

    func search(_ node: KDTreeNode?) {
      guard let node = node else { return }

      let dist = Spatial.euclideanDistance(point, node.point)
      if dist <= radius {
        result.append((node.index, dist))
      }

      let diff = point[node.axis] - node.point[node.axis]

      if diff - radius <= 0 {
        search(node.left)
      }
      if diff + radius >= 0 {
        search(node.right)
      }
    }

    search(root)

    result.sort { $0.dist < $1.dist }

    return (result.map { $0.idx }, result.map { $0.dist })
  }

  /// Find all pairs of points within a radius.
  ///
  /// - Parameter radius: Maximum distance between pairs
  /// - Returns: Array of (index1, index2, distance) tuples
  public func queryPairs(radius: Double) -> [(Int, Int, Double)] {
    var pairs: [(Int, Int, Double)] = []

    for i in 0..<points.count {
      let (indices, distances) = queryRadius(points[i], radius: radius)
      for (j, idx) in indices.enumerated() {
        if idx > i {
          pairs.append((i, idx, distances[j]))
        }
      }
    }

    return pairs
  }
}

// MARK: - k-NN limitation diagnostics (additive, source-compatible)

/// A k-nearest-neighbour query result: parallel `indices` / `distances` lists.
///
/// `indices` and `distances` are the same best-effort lists the bare
/// ``KDTree/query(_:k:)`` returns, sorted by ascending distance. ``KDTree/queryDiagnosed(_:k:)``
/// and ``Spatial/bruteForceKNNDiagnosed(points:query:k:)`` wrap this in a
/// ``Diagnosed`` so a caller can detect a **degenerate query** — one that cannot
/// return `k` valid neighbours — without changing the bare call site.
///
/// A named struct (rather than a bare tuple) so it carries an explicit `Sendable`
/// and `Equatable` contract and a stable, documented shape across the 0.3.0 API.
public struct KNNResult: Sendable, Equatable {
  /// Neighbour indices into the searched point set, nearest first.
  public let indices: [Int]
  /// Distances to those neighbours, ascending and aligned with `indices`.
  public let distances: [Double]

  /// Create a k-NN result from parallel index / distance lists.
  public init(indices: [Int], distances: [Double]) {
    self.indices = indices
    self.distances = distances
  }
}

/// Build the `outsideEnvelope` diagnostic for a degenerate k-NN query, or `nil`
/// when the query is well-posed.
///
/// A k-NN query is degenerate (and its k-th-neighbour distance is meaningless)
/// when any of these holds (matching `scipy.spatial.cKDTree.query` /
/// `sklearn.neighbors.NearestNeighbors` preconditions):
///
/// - the point set is **empty** — there is no neighbour to return;
/// - **`k <= 0`** — a non-positive neighbour count is undefined;
/// - **`k > n`** — there are fewer than `k` points, so the k-th neighbour does
///   not exist.
///
/// - Parameters:
///   - method: Short method identifier embedded in the diagnostic (e.g. `"kdTree.query"`).
///   - pointCount: Number of points `n` in the searched set.
///   - k: Requested neighbour count.
/// - Returns: A single ``NumericDiagnostic/outsideEnvelope(method:reason:)`` when
///   the query is degenerate, else `nil`.
private func knnDegenerateDiagnostic(
  method: String, pointCount n: Int, k: Int
) -> NumericDiagnostic? {
  if n == 0 {
    return .outsideEnvelope(
      method: method,
      reason: "empty point set — no neighbour exists to return")
  }
  if k <= 0 {
    return .outsideEnvelope(
      method: method,
      reason: "k = \(k) <= 0 — a non-positive neighbour count is undefined")
  }
  if k > n {
    return .outsideEnvelope(
      method: method,
      reason: "k = \(k) exceeds the point count n = \(n) — "
        + "fewer than k neighbours exist, so the k-th neighbour does not exist")
  }
  return nil
}

extension KDTree {

  /// Find the `k` nearest neighbours, reporting a degenerate-query diagnostic.
  ///
  /// Additive, source-compatible companion to ``query(_:k:)``: it returns the
  /// same best-effort `(indices, distances)` (sorted by ascending distance)
  /// wrapped in a ``Diagnosed`` so a caller can detect when the query cannot
  /// yield `k` valid neighbours. Inside the valid envelope (`0 < k <= n` on a
  /// non-empty tree) ``Diagnosed/diagnostics`` is empty; outside it carries a
  /// single ``NumericDiagnostic/outsideEnvelope(method:reason:)``. The bare
  /// ``query(_:k:)`` is unchanged and remains the right call when the warning
  /// is unwanted.
  ///
  /// See ``knnDegenerateDiagnostic`` for the precise degeneracy conditions
  /// (empty set, `k <= 0`, `k > n`).
  ///
  /// - Parameters:
  ///   - point: Query point.
  ///   - k: Number of neighbours.
  /// - Returns: The `(indices, distances)` lists with a degeneracy diagnostic
  ///   when the query is out of envelope.
  public func queryDiagnosed(_ point: [Double], k: Int = 1) -> Diagnosed<KNNResult> {
    if let diag = knnDegenerateDiagnostic(method: "kdTree.query", pointCount: points.count, k: k) {
      // Degenerate: the bare `query` assumes 0 < k <= n, so do NOT call it (it
      // force-unwraps an empty best-list). Return what neighbours exist, capped.
      let safeK = max(min(k, points.count), 0)
      let value: KNNResult
      if safeK == 0 {
        value = KNNResult(indices: [], distances: [])
      } else {
        let r = query(point, k: safeK)
        value = KNNResult(indices: r.indices, distances: r.distances)
      }
      return Diagnosed(value, diagnostics: [diag])
    }
    let r = query(point, k: k)
    return Diagnosed(KNNResult(indices: r.indices, distances: r.distances))
  }
}

extension Spatial {

  /// Brute-force `k` nearest neighbours, reporting a degenerate-query diagnostic.
  ///
  /// Computes the Euclidean distance from `query` to every point and returns the
  /// `k` smallest, sorted by ascending distance — the exact reference answer the
  /// ``KDTree/queryDiagnosed(_:k:)`` accelerated search must agree with. Like its
  /// KDTree companion it wraps the result in a ``Diagnosed`` and emits a single
  /// ``NumericDiagnostic/outsideEnvelope(method:reason:)`` for a degenerate query
  /// (empty point set, `k <= 0`, or `k > n`); a well-posed query carries no
  /// diagnostic.
  ///
  /// - Parameters:
  ///   - points: The searched point set (each an equal-length coordinate array).
  ///   - query: The query point.
  ///   - k: Number of neighbours.
  /// - Returns: The `(indices, distances)` lists with a degeneracy diagnostic
  ///   when the query is out of envelope.
  public static func bruteForceKNNDiagnosed(
    points: [[Double]], query: [Double], k: Int = 1
  ) -> Diagnosed<KNNResult> {
    let ranked = points.enumerated()
      .map { (idx: $0.offset, dist: Spatial.euclideanDistance(query, $0.element)) }
      .sorted { $0.dist < $1.dist }
    let take = ranked.prefix(max(min(k, ranked.count), 0))
    let value = KNNResult(indices: take.map { $0.idx }, distances: take.map { $0.dist })
    if let diag = knnDegenerateDiagnostic(method: "bruteForce.knn", pointCount: points.count, k: k) {
      return Diagnosed(value, diagnostics: [diag])
    }
    return Diagnosed(value)
  }
}

// MARK: - Result Types (public, top-level)

/// Result of Delaunay triangulation.
public struct DelaunayResult {
  /// Original points.
  public let points: [[Double]]
  /// Triangle indices (each triangle has 3 vertex indices).
  public let simplices: [[Int]]
  /// Neighbor indices for each triangle.
  public let neighbors: [[Int]]
}

/// Result of Voronoi diagram computation.
///
/// ## Infinite regions
///
/// Generators on the convex hull have Voronoi cells that extend to infinity.
/// Following the convention of `scipy.spatial.Voronoi`, these are represented
/// with the sentinel value `-1`:
///
/// - `ridgeVertices`: an entry `[-1, k]` (or `[k, -1]`) indicates a ridge that
///   extends to infinity on one side; `k` is the finite Voronoi vertex index.
///   A ridge `[-1, -1]` would indicate both endpoints at infinity (degenerate).
/// - `regions[i]`: the vertex-index list for generator `i` contains `-1`
///   whenever that generator's cell has an infinite ray (i.e., the generator
///   lies on the convex hull).
///
/// Callers that only need bounded Voronoi cells can filter out all `-1` entries.
public struct VoronoiResult {
  /// Original points (generators).
  public let points: [[Double]]
  /// Voronoi vertices (circumcenters of Delaunay triangles).
  public let vertices: [[Double]]
  /// Voronoi regions for each generator (indices into `vertices`; `-1` marks infinite rays).
  public let regions: [[Int]]
  /// Ridge vertex index pairs; `-1` denotes a ray extending to infinity.
  public let ridgeVertices: [[Int]]
  /// Ridge generator index pairs (the two generators sharing each ridge).
  public let ridgePoints: [[Int]]
}

/// Result of convex hull computation.
public struct ConvexHullResult {
  /// Original points.
  public let points: [[Double]]
  /// Indices of hull vertices in counter-clockwise order.
  public let vertices: [Int]
  /// Hull edges as pairs of vertex indices.
  public let simplices: [[Int]]
  /// Area of the convex hull.
  public let area: Double
}

// MARK: - Spatial Namespace (free functions)

extension Spatial {

    // MARK: - Delaunay Triangulation

    /// Compute Delaunay triangulation of 2D points.
    ///
    /// Uses the Bowyer-Watson algorithm, which maximizes the minimum angle of all triangles.
    ///
    /// When all input points are collinear, no valid triangulation exists and an empty
    /// simplex list is returned — matching the behaviour of `scipy.spatial.Delaunay`.
    ///
    /// - Parameter points: Array of 2D points
    /// - Returns: ``DelaunayResult`` with triangles and neighbor information
    public static func delaunay(_ points: [[Double]]) -> DelaunayResult {
        let n = points.count

        // Malformed 2D input (a point with < 2 coordinates) would trap on
        // point[0]/point[1]; return an empty triangulation.
        guard points.allSatisfy({ $0.count >= 2 }) else {
            return DelaunayResult(points: points, simplices: [], neighbors: [])
        }

        guard n >= 3 else {
            return DelaunayResult(points: points, simplices: [], neighbors: [])
        }

        if allPointsCollinear(points) {
            return DelaunayResult(points: points, simplices: [], neighbors: [])
        }

        let (simplices, neighbors) = bowyerWatson(points: points)

        return DelaunayResult(
            points: points,
            simplices: simplices,
            neighbors: neighbors
        )
    }

    // MARK: - Voronoi Diagram

    /// Compute Voronoi diagram of 2D points.
    ///
    /// The Voronoi diagram is the dual of the Delaunay triangulation, computed
    /// from circumcenters of Delaunay triangles.
    ///
    /// Generators on the convex hull have cells that extend to infinity.
    /// `ridgeVertices` uses `-1` as a sentinel for the infinite endpoint.
    ///
    /// - Parameter points: Array of 2D points (at least 3 for a non-trivial diagram).
    /// - Returns: ``VoronoiResult`` with vertices, regions, and ridges.
    public static func voronoi(_ points: [[Double]]) -> VoronoiResult {
        guard !points.isEmpty else {
            return VoronoiResult(
                points: [],
                vertices: [],
                regions: [],
                ridgeVertices: [],
                ridgePoints: []
            )
        }

        // Malformed 2D input (a point with < 2 coordinates) would trap downstream;
        // return an empty diagram.
        guard points.allSatisfy({ $0.count >= 2 }) else {
            return VoronoiResult(
                points: points, vertices: [], regions: [], ridgeVertices: [], ridgePoints: [])
        }

        let n = points.count

        let (delaunaySimplices, delaunayNeighbors) = bowyerWatson(points: points)

        var vertices: [[Double]] = []
        var simplexToVertex: [Int: Int] = [:]

        for (i, simplex) in delaunaySimplices.enumerated() {
            guard simplex.count == 3 else { continue }
            let p1 = points[simplex[0]], p2 = points[simplex[1]], p3 = points[simplex[2]]
            let ax = p1[0], ay = p1[1]
            let bx = p2[0], by = p2[1]
            let cx = p3[0], cy = p3[1]

            let d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            if abs(d) > 1e-10 {
                let a2 = ax * ax + ay * ay
                let b2 = bx * bx + by * by
                let c2 = cx * cx + cy * cy
                let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
                let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
                vertices.append([ux, uy])
                simplexToVertex[i] = vertices.count - 1
            }
        }

        var regions: [[Int]] = Array(repeating: [], count: n)

        for (i, simplex) in delaunaySimplices.enumerated() {
            guard let vIdx = simplexToVertex[i] else { continue }
            for ptIdx in simplex {
                regions[ptIdx].append(vIdx)
            }
        }

        var ridgeVertices: [[Int]] = []
        var ridgePoints: [[Int]] = []
        var processedEdges: Set<String> = []

        for (i, simplex) in delaunaySimplices.enumerated() {
            guard let v1 = simplexToVertex[i] else { continue }

            let edges: [[Int]] = [
                [simplex[0], simplex[1]],
                [simplex[1], simplex[2]],
                [simplex[2], simplex[0]],
            ]

            for edge in edges {
                let edgeKey = "\(min(edge[0], edge[1])),\(max(edge[0], edge[1]))"

                let neighbour = delaunayNeighbors[i].first { nIdx in
                    let nSimplex = delaunaySimplices[nIdx]
                    return nSimplex.contains(edge[0]) && nSimplex.contains(edge[1])
                }

                if let nIdx = neighbour {
                    if nIdx > i {
                        let ridgeKey = "\(min(edge[0], edge[1])),\(max(edge[0], edge[1]))-\(min(i, nIdx)),\(max(i, nIdx))"
                        if !processedEdges.contains(ridgeKey) {
                            processedEdges.insert(ridgeKey)
                            if let v2 = simplexToVertex[nIdx] {
                                ridgeVertices.append([v1, v2])
                            } else {
                                ridgeVertices.append([-1, v1])
                                if !regions[edge[0]].contains(-1) { regions[edge[0]].append(-1) }
                                if !regions[edge[1]].contains(-1) { regions[edge[1]].append(-1) }
                            }
                            ridgePoints.append([edge[0], edge[1]])
                        }
                    }
                } else {
                    if !processedEdges.contains(edgeKey) {
                        processedEdges.insert(edgeKey)
                        ridgeVertices.append([-1, v1])
                        ridgePoints.append([edge[0], edge[1]])
                        if !regions[edge[0]].contains(-1) { regions[edge[0]].append(-1) }
                        if !regions[edge[1]].contains(-1) { regions[edge[1]].append(-1) }
                    }
                }
            }
        }

        return VoronoiResult(
            points: points,
            vertices: vertices,
            regions: regions,
            ridgeVertices: ridgeVertices,
            ridgePoints: ridgePoints
        )
    }

    // MARK: - Convex Hull

    /// Compute convex hull of 2D points using Graham scan.
    ///
    /// - Parameter points: Array of 2D points
    /// - Returns: ``ConvexHullResult`` with vertices and edges
    public static func convexHull(_ points: [[Double]]) -> ConvexHullResult {
        let n = points.count

        // These algorithms index point[0]/point[1]; a point with fewer than two
        // coordinates would trap. Return an empty hull for malformed 2D input.
        guard points.allSatisfy({ $0.count >= 2 }) else {
            return ConvexHullResult(points: points, vertices: [], simplices: [], area: 0)
        }

        guard n >= 3 else {
            let vertices = Array(0..<n)
            return ConvexHullResult(
                points: points,
                vertices: vertices,
                simplices: [],
                area: 0
            )
        }

        var startIdx = 0
        for i in 1..<n {
            if points[i][1] < points[startIdx][1]
                || (points[i][1] == points[startIdx][1] && points[i][0] < points[startIdx][0])
            {
                startIdx = i
            }
        }

        let start = points[startIdx]

        var indices = Array(0..<n).filter { $0 != startIdx }
        indices.sort { a, b in
            let angleA = Darwin.atan2(points[a][1] - start[1], points[a][0] - start[0])
            let angleB = Darwin.atan2(points[b][1] - start[1], points[b][0] - start[0])
            if abs(angleA - angleB) < 1e-10 {
                let distA = Spatial.squaredEuclideanDistance(points[a], start)
                let distB = Spatial.squaredEuclideanDistance(points[b], start)
                return distA < distB
            }
            return angleA < angleB
        }

        func ccw(_ p1: [Double], _ p2: [Double], _ p3: [Double]) -> Double {
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        }

        var hull = [startIdx]
        for idx in indices {
            while hull.count >= 2
                && ccw(points[hull[hull.count - 2]], points[hull[hull.count - 1]], points[idx]) <= 0
            {
                hull.removeLast()
            }
            hull.append(idx)
        }

        var simplices: [[Int]] = []
        for i in 0..<hull.count {
            let nextI = (i + 1) % hull.count
            simplices.append([hull[i], hull[nextI]])
        }

        var area: Double = 0
        for i in 0..<hull.count {
            let j = (i + 1) % hull.count
            let p1 = points[hull[i]]
            let p2 = points[hull[j]]
            area += p1[0] * p2[1]
            area -= p2[0] * p1[1]
        }
        area = abs(area) / 2

        return ConvexHullResult(
            points: points,
            vertices: hull,
            simplices: simplices,
            area: area
        )
    }

    // MARK: - Private helpers

    private static func allPointsCollinear(_ pts: [[Double]]) -> Bool {
        var minX = pts[0][0], maxX = pts[0][0]
        var minY = pts[0][1], maxY = pts[0][1]
        for p in pts {
            if p[0] < minX { minX = p[0] }
            if p[0] > maxX { maxX = p[0] }
            if p[1] < minY { minY = p[1] }
            if p[1] > maxY { maxY = p[1] }
        }
        let bboxDiag = (maxX - minX) * (maxX - minX) + (maxY - minY) * (maxY - minY)

        if bboxDiag < 1e-30 { return true }

        let eps = 1e-10 * bboxDiag

        let anchorA: [Double]
        let anchorB: [Double]
        let xRange = maxX - minX
        let yRange = maxY - minY
        if xRange >= yRange {
            anchorA = pts.min(by: { $0[0] < $1[0] })!
            anchorB = pts.max(by: { $0[0] < $1[0] })!
        } else {
            anchorA = pts.min(by: { $0[1] < $1[1] })!
            anchorB = pts.max(by: { $0[1] < $1[1] })!
        }

        let ax = anchorA[0], ay = anchorA[1]
        let dx = anchorB[0] - ax, dy = anchorB[1] - ay

        for p in pts {
            let cross = dx * (p[1] - ay) - dy * (p[0] - ax)
            if cross * cross > eps {
                return false
            }
        }
        return true
    }

    private static func bowyerWatson(points: [[Double]]) -> (simplices: [[Int]], neighbors: [[Int]]) {
        let n = points.count

        var minX = points[0][0]
        var maxX = points[0][0]
        var minY = points[0][1]
        var maxY = points[0][1]

        for p in points {
            minX = min(minX, p[0])
            maxX = max(maxX, p[0])
            minY = min(minY, p[1])
            maxY = max(maxY, p[1])
        }

        let dx = maxX - minX
        let dy = maxY - minY
        let delta: Double = max(dx, dy) * 10.0

        let superTriangle: [[Double]] = [
            [minX - delta, minY - delta],
            [minX + dx / 2.0, maxY + delta * 2.0],
            [maxX + delta, minY - delta],
        ]

        let allPoints = points + superTriangle

        var triangles: [[Int]] = [[n, n + 1, n + 2]]

        for i in 0..<n {
            let px = points[i][0]
            let py = points[i][1]

            var badTriangles: [Int] = []

            for (j, tri) in triangles.enumerated() {
                if inCircumcircle(px: px, py: py, tri: tri, points: allPoints) {
                    badTriangles.append(j)
                }
            }

            var edgeCount: [String: Int] = [:]
            for j in badTriangles {
                let tri = triangles[j]
                let edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]
                for e in edges {
                    let key = "\(min(e[0], e[1])),\(max(e[0], e[1]))"
                    edgeCount[key, default: 0] += 1
                }
            }

            var polygon: [[Int]] = []
            for j in badTriangles {
                let tri = triangles[j]
                let edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]
                for e in edges {
                    let key = "\(min(e[0], e[1])),\(max(e[0], e[1]))"
                    if edgeCount[key] == 1 {
                        polygon.append(e)
                    }
                }
            }

            for j in badTriangles.sorted().reversed() {
                triangles.remove(at: j)
            }

            for e in polygon {
                triangles.append([e[0], e[1], i])
            }
        }

        var finalTriangles: [[Int]] = []
        for tri in triangles {
            var valid = true
            for v in tri {
                if v >= n {
                    valid = false
                    break
                }
            }
            if valid {
                finalTriangles.append(tri)
            }
        }

        var neighbors: [[Int]] = Array(repeating: [], count: finalTriangles.count)
        for i in 0..<finalTriangles.count {
            let triI = finalTriangles[i]
            for j in (i + 1)..<finalTriangles.count {
                let triJ = finalTriangles[j]
                var shared = 0
                for vi in triI {
                    for vj in triJ {
                        if vi == vj { shared += 1 }
                    }
                }
                if shared == 2 {
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                }
            }
        }

        return (finalTriangles, neighbors)
    }

    private static func inCircumcircle(px: Double, py: Double, tri: [Int], points: [[Double]]) -> Bool {
        let ax = points[tri[0]][0]
        let ay = points[tri[0]][1]
        let bx = points[tri[1]][0]
        let by = points[tri[1]][1]
        let cx = points[tri[2]][0]
        let cy = points[tri[2]][1]

        let d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-15 { return false }

        let a2 = ax * ax + ay * ay
        let b2 = bx * bx + by * by
        let c2 = cx * cx + cy * cy

        let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
        let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d

        let r2 = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy)
        let dist2 = (px - ux) * (px - ux) + (py - uy) * (py - uy)

        return dist2 < r2
    }
}

// MARK: - Deprecated shims (backward compatibility)

/// - Note: Deprecated. Use ``Spatial/delaunay(_:)`` instead.
@available(*, deprecated, message: "Use Spatial.delaunay(_:) instead")
public func delaunay(_ points: [[Double]]) -> DelaunayResult {
    Spatial.delaunay(points)
}

/// - Note: Deprecated. Use ``Spatial/voronoi(_:)`` instead.
@available(*, deprecated, message: "Use Spatial.voronoi(_:) instead")
public func voronoi(_ points: [[Double]]) -> VoronoiResult {
    Spatial.voronoi(points)
}

/// - Note: Deprecated. Use ``Spatial/convexHull(_:)`` instead.
@available(*, deprecated, message: "Use Spatial.convexHull(_:) instead")
public func convexHull(_ points: [[Double]]) -> ConvexHullResult {
    Spatial.convexHull(points)
}
