//
//  SpatialDegeneracyTests.swift
//  NumericSwift
//
//  Tests for degenerate and edge-case inputs to spatial algorithms.
//  The goal is to document and verify current behavior — no crashes, predictable results.
//

import XCTest

@testable import NumericSwift

final class SpatialDegeneracyTests: XCTestCase {

  // MARK: - Delaunay: very few points

  func testDelaunayZeroPoints() {
    let result = delaunay([])
    XCTAssertTrue(result.simplices.isEmpty)
    XCTAssertTrue(result.neighbors.isEmpty)
    XCTAssertTrue(result.points.isEmpty)
  }

  func testDelaunayOnePoint() {
    let result = delaunay([[1.0, 2.0]])
    XCTAssertTrue(result.simplices.isEmpty)
    XCTAssertTrue(result.neighbors.isEmpty)
  }

  func testDelaunayTwoPoints() {
    let result = delaunay([[0.0, 0.0], [1.0, 0.0]])
    XCTAssertTrue(result.simplices.isEmpty)
    XCTAssertTrue(result.neighbors.isEmpty)
  }

  // MARK: - Delaunay: collinear points

  func testDelaunayCollinearThreePoints() {
    // Exactly collinear — implementation checks only first 3 points for collinearity
    let points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    let result = delaunay(points)
    // Current behavior: returns empty triangulation for collinear inputs
    XCTAssertTrue(result.simplices.isEmpty, "Collinear points should produce no triangles")
  }

  func testDelaunayCollinearFivePoints() {
    // Five collinear points — collinearity detected from first three
    let points: [[Double]] = (0..<5).map { [Double($0), 0.0] }
    let result = delaunay(points)
    XCTAssertTrue(result.simplices.isEmpty, "Five collinear points should produce no triangles")
  }

  func testDelaunayCollinearDiagonal() {
    // Points along y=x diagonal
    let points = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    let result = delaunay(points)
    XCTAssertTrue(result.simplices.isEmpty, "Diagonal collinear points should produce no triangles")
  }

  // MARK: - Delaunay: near-collinear points

  func testDelaunayNearCollinear() {
    // Almost collinear — cross product is 2e-8, which clears the 1e-10 collinearity guard.
    // However Bowyer-Watson yields no surviving triangle for this extremely thin configuration.
    let pointsThin = [[0.0, 0.0], [1.0, 1e-8], [2.0, 0.0]]
    let resultThin = delaunay(pointsThin)
    // Current behavior: guard passes but no final triangle survives
    XCTAssertTrue(
      resultThin.simplices.isEmpty, "Extremely thin near-collinear triple produces no triangles")

    // With a larger deviation the triangulation does succeed
    let pointsOk = [[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]]
    let resultOk = delaunay(pointsOk)
    XCTAssertEqual(
      resultOk.simplices.count, 1, "Sufficiently non-collinear triple produces one triangle")
  }

  // MARK: - Delaunay: single triangle (exactly 3 non-collinear points)

  func testDelaunayExactlyOneTriangle() {
    let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    let result = delaunay(points)
    XCTAssertEqual(result.simplices.count, 1)
    XCTAssertEqual(result.simplices[0].count, 3)
    let verts = Set(result.simplices[0])
    XCTAssertTrue(verts.contains(0))
    XCTAssertTrue(verts.contains(1))
    XCTAssertTrue(verts.contains(2))
  }

  // MARK: - Delaunay: duplicate points

  func testDelaunayDuplicatePoints() {
    // Duplicate points cause degenerate circumcircles; function should not crash
    let points = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    XCTAssertNoThrow(
      {
        let result = delaunay(points)
        // Result may have 0 or more triangles — just verify structural consistency
        for simplex in result.simplices {
          XCTAssertEqual(simplex.count, 3)
          for idx in simplex {
            XCTAssertTrue(idx >= 0 && idx < points.count)
          }
        }
      }())
  }

  func testDelaunayAllDuplicatePoints() {
    let points = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    XCTAssertNoThrow(
      {
        let result = delaunay(points)
        // All identical = collinear (zero cross product), empty result
        XCTAssertTrue(result.simplices.isEmpty)
      }())
  }

  // MARK: - Voronoi: very few points

  func testVoronoiZeroPoints() {
    let result = voronoi([])
    XCTAssertTrue(result.vertices.isEmpty)
    XCTAssertTrue(result.regions.isEmpty)
    XCTAssertTrue(result.ridgeVertices.isEmpty)
    XCTAssertTrue(result.ridgePoints.isEmpty)
  }

  func testVoronoiOnePoint() {
    let result = voronoi([[0.0, 0.0]])
    // One point — no Delaunay triangles, so no circumcenters (Voronoi vertices)
    XCTAssertEqual(result.points.count, 1)
    XCTAssertTrue(result.vertices.isEmpty)
  }

  func testVoronoiTwoPoints() {
    let result = voronoi([[0.0, 0.0], [2.0, 0.0]])
    // Two points produce no Delaunay triangles — no interior Voronoi vertices
    XCTAssertEqual(result.points.count, 2)
    XCTAssertTrue(result.vertices.isEmpty, "Two points produce no finite Voronoi vertices")
  }

  // MARK: - Voronoi: collinear points

  func testVoronoiCollinearPoints() {
    let points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    XCTAssertNoThrow(
      {
        let result = voronoi(points)
        // bowyerWatson is called directly — may produce degenerate triangles
        // or no triangles for collinear input; verify no crash and structural validity
        XCTAssertEqual(result.points.count, 3)
        for region in result.regions {
          for idx in region {
            XCTAssertTrue(idx >= 0 && idx < result.vertices.count)
          }
        }
      }())
  }

  // MARK: - Voronoi: duplicate points

  func testVoronoiDuplicatePoints() {
    let points = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    XCTAssertNoThrow(
      {
        let result = voronoi(points)
        XCTAssertEqual(result.points.count, 4)
      }())
  }

  // MARK: - Convex hull: zero, one, and two points

  func testConvexHullZeroPoints() {
    let result = convexHull([])
    XCTAssertTrue(result.vertices.isEmpty)
    XCTAssertTrue(result.simplices.isEmpty)
    XCTAssertEqual(result.area, 0.0, accuracy: 1e-15)
  }

  func testConvexHullOnePoint() {
    let result = convexHull([[3.0, 7.0]])
    XCTAssertEqual(result.vertices.count, 1)
    XCTAssertEqual(result.vertices[0], 0)
    XCTAssertTrue(result.simplices.isEmpty)
    XCTAssertEqual(result.area, 0.0, accuracy: 1e-15)
  }

  func testConvexHullTwoPoints() {
    let result = convexHull([[0.0, 0.0], [4.0, 3.0]])
    XCTAssertEqual(result.vertices.count, 2)
    XCTAssertTrue(result.vertices.contains(0))
    XCTAssertTrue(result.vertices.contains(1))
    XCTAssertTrue(result.simplices.isEmpty)
    XCTAssertEqual(result.area, 0.0, accuracy: 1e-15)
  }

  // MARK: - Convex hull: collinear points

  func testConvexHullCollinearThreePoints() {
    // Three collinear points — Graham scan removes interior collinear points via <= 0 CCW check
    let points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    let result = convexHull(points)
    // Only the two endpoints survive the scan
    XCTAssertEqual(result.vertices.count, 2, "Collinear scan keeps only 2 endpoints")
    XCTAssertEqual(result.area, 0.0, accuracy: 1e-15)
  }

  func testConvexHullAllCollinear() {
    // Five points on a horizontal line
    let points: [[Double]] = (0..<5).map { [Double($0), 0.0] }
    let result = convexHull(points)
    // Hull collapses to the two extreme endpoints
    XCTAssertEqual(result.vertices.count, 2)
    XCTAssertEqual(result.area, 0.0, accuracy: 1e-15)
  }

  func testConvexHullCollinearVertical() {
    // Five points with same x coordinate
    let points: [[Double]] = (0..<5).map { [0.0, Double($0)] }
    let result = convexHull(points)
    XCTAssertEqual(result.vertices.count, 2)
    XCTAssertEqual(result.area, 0.0, accuracy: 1e-15)
  }

  func testConvexHullSameXCoordinates() {
    // Points with identical x but varying y, plus one off-axis point
    let points = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]]
    let result = convexHull(points)
    XCTAssertGreaterThan(result.vertices.count, 2)
    XCTAssertGreaterThan(result.area, 0.0)
  }

  func testConvexHullSameYCoordinates() {
    // Points with identical y but varying x, plus one off-axis point
    let points = [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 0.0]]
    let result = convexHull(points)
    XCTAssertGreaterThan(result.vertices.count, 2)
    XCTAssertGreaterThan(result.area, 0.0)
  }

  // MARK: - Convex hull: duplicate points

  func testConvexHullDuplicatePoints() {
    // Duplicates at vertices should not crash
    let points = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    XCTAssertNoThrow(
      {
        let result = convexHull(points)
        XCTAssertGreaterThan(result.vertices.count, 0)
      }())
  }

  func testConvexHullAllDuplicatePoints() {
    let points = [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]
    XCTAssertNoThrow(
      {
        let result = convexHull(points)
        XCTAssertEqual(result.area, 0.0, accuracy: 1e-15)
      }())
  }

  // MARK: - KDTree: empty point set

  func testKDTreeEmptyPoints() {
    let tree = KDTree([])
    XCTAssertEqual(tree.points.count, 0)
    XCTAssertEqual(tree.dim, 0)
    XCTAssertNil(tree.root)
  }

  func testKDTreeEmptyQuery() {
    // Querying an empty tree should return empty results, not crash
    let tree = KDTree([])
    let (indices, distances) = tree.query([0.0, 0.0], k: 1)
    XCTAssertTrue(indices.isEmpty)
    XCTAssertTrue(distances.isEmpty)
  }

  func testKDTreeEmptyRadiusQuery() {
    let tree = KDTree([])
    let (indices, distances) = tree.queryRadius([0.0, 0.0], radius: 10.0)
    XCTAssertTrue(indices.isEmpty)
    XCTAssertTrue(distances.isEmpty)
  }

  // MARK: - KDTree: single point

  func testKDTreeSinglePoint() {
    let tree = KDTree([[3.0, 4.0]])
    XCTAssertNotNil(tree.root)
    XCTAssertEqual(tree.dim, 2)

    let (indices, distances) = tree.query([0.0, 0.0], k: 1)
    XCTAssertEqual(indices.count, 1)
    XCTAssertEqual(indices[0], 0)
    XCTAssertEqual(distances[0], 5.0, accuracy: 1e-10)
  }

  func testKDTreeSinglePointKGreaterThanN() {
    // k=3 but only 1 point — should return 1 result
    let tree = KDTree([[1.0, 1.0]])
    let (indices, distances) = tree.query([0.0, 0.0], k: 3)
    XCTAssertEqual(indices.count, 1, "Returns only available points when k > n")
    XCTAssertEqual(distances.count, 1)
  }

  // MARK: - KDTree: identical points

  func testKDTreeIdenticalPoints() {
    // All points at the same coordinates
    let points = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    let tree = KDTree(points)
    XCTAssertNotNil(tree.root)
    XCTAssertEqual(tree.points.count, 4)

    // All distances to query point should be equal
    let (indices, distances) = tree.query([0.0, 0.0], k: 4)
    XCTAssertEqual(indices.count, 4)
    let expected = euclideanDistance([0.0, 0.0], [1.0, 1.0])
    for d in distances {
      XCTAssertEqual(d, expected, accuracy: 1e-10)
    }
  }

  func testKDTreeIdenticalPointsRadiusQuery() {
    let points = [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]
    let tree = KDTree(points)
    // All points at distance 2.0 from origin
    let (indices, _) = tree.queryRadius([0.0, 0.0], radius: 2.5)
    XCTAssertEqual(indices.count, 3, "All identical points within radius should be returned")
  }

  // MARK: - KDTree: k greater than number of points

  func testKDTreeKGreaterThanN() {
    let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    let tree = KDTree(points)

    // Request more neighbors than exist
    let (indices, distances) = tree.query([0.5, 0.5], k: 10)
    XCTAssertEqual(indices.count, 3, "Returns all available points when k > n")
    XCTAssertEqual(distances.count, 3)
  }

  func testKDTreeKOne() {
    // k=0 causes a fatal crash (force-unwrap of best.last! when best is empty).
    // Document that k >= 1 is the minimum safe value.
    let points = [[0.0, 0.0], [1.0, 0.0]]
    let tree = KDTree(points)
    let (indices, distances) = tree.query([0.0, 0.0], k: 1)
    XCTAssertEqual(indices.count, 1)
    XCTAssertEqual(distances.count, 1)
    XCTAssertEqual(distances[0], 0.0, accuracy: 1e-15)
  }
}
