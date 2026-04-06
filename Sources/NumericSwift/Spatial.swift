//
//  Spatial.swift
//  NumericSwift
//
//  Spatial data structures: KDTree, Delaunay triangulation, Voronoi diagram, convex hull.
//  Distance metrics and batch functions are in SpatialDistance.swift.
//
//  Licensed under the MIT License.
//

import Accelerate
import Foundation

// MARK: - KDTree

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
  public init(_ points: [[Double]]) {
    self.points = points
    self.dim = points.first?.count ?? 0

    var indices = Array(0..<points.count)
    self.root = KDTree.buildTree(points: points, indices: &indices, depth: 0, dim: dim)
  }

  private static func buildTree(
    points: [[Double]],
    indices: inout [Int],
    depth: Int,
    dim: Int
  ) -> KDTreeNode? {
    guard !indices.isEmpty else { return nil }

    let axis = depth % dim

    // Sort by axis
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
    var best: [(idx: Int, dist: Double)] = []

    func search(_ node: KDTreeNode?) {
      guard let node = node else { return }

      let dist = euclideanDistance(point, node.point)

      // Insert into best list maintaining sorted order
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

      // Check if we need to search far branch
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
    var result: [(idx: Int, dist: Double)] = []

    func search(_ node: KDTreeNode?) {
      guard let node = node else { return }

      let dist = euclideanDistance(point, node.point)
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

    // Sort by distance
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

// MARK: - Delaunay Triangulation

/// Result of Delaunay triangulation.
public struct DelaunayResult {
  /// Original points.
  public let points: [[Double]]
  /// Triangle indices (each triangle has 3 vertex indices).
  public let simplices: [[Int]]
  /// Neighbor indices for each triangle.
  public let neighbors: [[Int]]
}

/// Compute Delaunay triangulation of 2D points.
///
/// Uses the Bowyer-Watson algorithm to compute the Delaunay triangulation,
/// which maximizes the minimum angle of all triangles.
///
/// - Parameter points: Array of 2D points
/// - Returns: DelaunayResult with triangles and neighbor information
public func delaunay(_ points: [[Double]]) -> DelaunayResult {
  let n = points.count

  guard n >= 3 else {
    return DelaunayResult(points: points, simplices: [], neighbors: [])
  }

  // Check for collinear points
  let p1 = points[0]
  let p2 = points[1]
  let p3 = points[2]
  let cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
  if abs(cross) < 1e-10 {
    // Collinear - return empty triangulation
    return DelaunayResult(points: points, simplices: [], neighbors: [])
  }

  let (simplices, neighbors) = bowyerWatson(points: points)

  return DelaunayResult(
    points: points,
    simplices: simplices,
    neighbors: neighbors
  )
}

/// Bowyer-Watson algorithm for Delaunay triangulation.
private func bowyerWatson(points: [[Double]]) -> (simplices: [[Int]], neighbors: [[Int]]) {
  let n = points.count

  // Find bounding box
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

  // Create super-triangle
  let dx = maxX - minX
  let dy = maxY - minY
  let delta: Double = max(dx, dy) * 10.0

  let superTriangle: [[Double]] = [
    [minX - delta, minY - delta],
    [minX + dx / 2.0, maxY + delta * 2.0],
    [maxX + delta, minY - delta],
  ]

  // All points including super-triangle vertices
  let allPoints = points + superTriangle

  // Initial triangulation with super-triangle (0-indexed)
  var triangles: [[Int]] = [[n, n + 1, n + 2]]

  // Insert each point
  for i in 0..<n {
    let px = points[i][0]
    let py = points[i][1]

    var badTriangles: [Int] = []

    // Find triangles whose circumcircle contains the point
    for (j, tri) in triangles.enumerated() {
      if inCircumcircle(px: px, py: py, tri: tri, points: allPoints) {
        badTriangles.append(j)
      }
    }

    // Find boundary edges of the hole
    var edgeCount: [String: Int] = [:]
    for j in badTriangles {
      let tri = triangles[j]
      let edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]
      for e in edges {
        let key = "\(min(e[0], e[1])),\(max(e[0], e[1]))"
        edgeCount[key, default: 0] += 1
      }
    }

    // Collect boundary edges (appearing only once)
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

    // Remove bad triangles (in reverse order)
    for j in badTriangles.sorted().reversed() {
      triangles.remove(at: j)
    }

    // Create new triangles
    for e in polygon {
      triangles.append([e[0], e[1], i])
    }
  }

  // Remove triangles containing super-triangle vertices
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

  // Build neighbor information
  var neighbors: [[Int]] = Array(repeating: [], count: finalTriangles.count)
  for i in 0..<finalTriangles.count {
    let triI = finalTriangles[i]
    for j in (i + 1)..<finalTriangles.count {
      let triJ = finalTriangles[j]
      // Count shared vertices
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

/// Check if point is inside circumcircle of triangle.
private func inCircumcircle(px: Double, py: Double, tri: [Int], points: [[Double]]) -> Bool {
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

// MARK: - Voronoi Diagram

/// Result of Voronoi diagram computation.
public struct VoronoiResult {
  /// Original points (generators).
  public let points: [[Double]]
  /// Voronoi vertices.
  public let vertices: [[Double]]
  /// Voronoi regions for each point (indices into vertices).
  public let regions: [[Int]]
  /// Ridge vertices (pairs of vertex indices).
  public let ridgeVertices: [[Int]]
  /// Ridge points (pairs of generator indices).
  public let ridgePoints: [[Int]]
}

/// Compute Voronoi diagram of 2D points.
///
/// The Voronoi diagram is the dual of the Delaunay triangulation.
///
/// - Parameter points: Array of 2D points
/// - Returns: VoronoiResult with vertices, regions, and ridges
public func voronoi(_ points: [[Double]]) -> VoronoiResult {
  guard !points.isEmpty else {
    return VoronoiResult(
      points: [],
      vertices: [],
      regions: [],
      ridgeVertices: [],
      ridgePoints: []
    )
  }

  let n = points.count

  // Compute Delaunay first (Voronoi is dual)
  let (delaunaySimplices, delaunayNeighbors) = bowyerWatson(points: points)

  // Compute circumcenters of Delaunay triangles = Voronoi vertices
  var vertices: [[Double]] = []
  var simplexToVertex: [Int: Int] = [:]

  for (i, simplex) in delaunaySimplices.enumerated() {
    if simplex.count == 3 {
      let p1 = points[simplex[0]]
      let p2 = points[simplex[1]]
      let p3 = points[simplex[2]]

      let ax = p1[0]
      let ay = p1[1]
      let bx = p2[0]
      let by = p2[1]
      let cx = p3[0]
      let cy = p3[1]

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
  }

  // Build regions for each input point
  var regions: [[Int]] = Array(repeating: [], count: n)
  for (i, simplex) in delaunaySimplices.enumerated() {
    if let vIdx = simplexToVertex[i] {
      for ptIdx in simplex {
        regions[ptIdx].append(vIdx)
      }
    }
  }

  // Build ridge information
  var ridgeVertices: [[Int]] = []
  var ridgePoints: [[Int]] = []

  for (i, _) in delaunaySimplices.enumerated() {
    if let v1 = simplexToVertex[i] {
      for nIdx in delaunayNeighbors[i] {
        if nIdx > i {
          if let v2 = simplexToVertex[nIdx] {
            // Find shared edge
            var shared: [Int] = []
            for p1 in delaunaySimplices[i] {
              for p2 in delaunaySimplices[nIdx] {
                if p1 == p2 { shared.append(p1) }
              }
            }
            if shared.count >= 2 {
              ridgeVertices.append([v1, v2])
              ridgePoints.append([shared[0], shared[1]])
            }
          }
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

/// Compute convex hull of 2D points using Graham scan.
///
/// - Parameter points: Array of 2D points
/// - Returns: ConvexHullResult with vertices and edges
public func convexHull(_ points: [[Double]]) -> ConvexHullResult {
  let n = points.count

  guard n >= 3 else {
    let vertices = Array(0..<n)
    return ConvexHullResult(
      points: points,
      vertices: vertices,
      simplices: [],
      area: 0
    )
  }

  // Find lowest point (and leftmost if tie)
  var startIdx = 0
  for i in 1..<n {
    if points[i][1] < points[startIdx][1]
      || (points[i][1] == points[startIdx][1] && points[i][0] < points[startIdx][0])
    {
      startIdx = i
    }
  }

  let start = points[startIdx]

  // Sort by polar angle
  var indices = Array(0..<n).filter { $0 != startIdx }
  indices.sort { a, b in
    let angleA = Darwin.atan2(points[a][1] - start[1], points[a][0] - start[0])
    let angleB = Darwin.atan2(points[b][1] - start[1], points[b][0] - start[0])
    if abs(angleA - angleB) < 1e-10 {
      let distA = squaredEuclideanDistance(points[a], start)
      let distB = squaredEuclideanDistance(points[b], start)
      return distA < distB
    }
    return angleA < angleB
  }

  // Graham scan with CCW check
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

  // Build simplices (edges)
  var simplices: [[Int]] = []
  for i in 0..<hull.count {
    let nextI = (i + 1) % hull.count
    simplices.append([hull[i], hull[nextI]])
  }

  // Compute area using shoelace formula
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
