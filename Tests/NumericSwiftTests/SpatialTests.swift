//
//  SpatialTests.swift
//  NumericSwift
//
//  Tests for spatial algorithms and data structures.
//

import XCTest
@testable import NumericSwift

final class SpatialTests: XCTestCase {

    // MARK: - Distance Function Tests

    func testManhattanDistance() {
        let p1 = [0.0, 0.0]
        let p2 = [3.0, 4.0]
        XCTAssertEqual(manhattanDistance(p1, p2), 7.0, accuracy: 1e-10)

        // Same point
        XCTAssertEqual(manhattanDistance(p1, p1), 0.0, accuracy: 1e-10)
    }

    func testChebyshevDistance() {
        let p1 = [0.0, 0.0]
        let p2 = [3.0, 7.0]
        XCTAssertEqual(chebyshevDistance(p1, p2), 7.0, accuracy: 1e-10)

        let p3 = [1.0, 2.0, 3.0]
        let p4 = [4.0, 6.0, 3.0]
        XCTAssertEqual(chebyshevDistance(p3, p4), 4.0, accuracy: 1e-10)
    }

    func testMinkowskiDistance() {
        let p1 = [0.0, 0.0]
        let p2 = [3.0, 4.0]

        // p=1 should equal Manhattan
        XCTAssertEqual(minkowskiDistance(p1, p2, p: 1), manhattanDistance(p1, p2), accuracy: 1e-10)

        // p=2 should equal Euclidean
        XCTAssertEqual(minkowskiDistance(p1, p2, p: 2), euclideanDistance(p1, p2), accuracy: 1e-10)

        // p=infinity should equal Chebyshev
        XCTAssertEqual(minkowskiDistance(p1, p2, p: .infinity), chebyshevDistance(p1, p2), accuracy: 1e-10)
    }

    func testCosineDistance() {
        // Same direction = distance 0
        let p1 = [1.0, 0.0]
        let p2 = [2.0, 0.0]
        XCTAssertEqual(cosineDistance(p1, p2), 0.0, accuracy: 1e-10)

        // Perpendicular = distance 1
        let p3 = [1.0, 0.0]
        let p4 = [0.0, 1.0]
        XCTAssertEqual(cosineDistance(p3, p4), 1.0, accuracy: 1e-10)

        // Opposite direction = distance 2
        let p5 = [1.0, 0.0]
        let p6 = [-1.0, 0.0]
        XCTAssertEqual(cosineDistance(p5, p6), 2.0, accuracy: 1e-10)
    }

    func testCorrelationDistance() {
        // Perfect positive correlation
        let p1 = [1.0, 2.0, 3.0]
        let p2 = [2.0, 4.0, 6.0]
        XCTAssertEqual(correlationDistance(p1, p2), 0.0, accuracy: 1e-10)

        // Perfect negative correlation
        let p3 = [1.0, 2.0, 3.0]
        let p4 = [3.0, 2.0, 1.0]
        XCTAssertEqual(correlationDistance(p3, p4), 2.0, accuracy: 1e-10)
    }

    // MARK: - Batch Distance Tests

    func testCdist() {
        let XA = [[0.0, 0.0], [1.0, 0.0]]
        let XB = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        let dists = cdist(XA, XB)

        XCTAssertEqual(dists.count, 2)
        XCTAssertEqual(dists[0].count, 3)

        // Check specific distances
        XCTAssertEqual(dists[0][0], 0.0, accuracy: 1e-10)  // (0,0) to (0,0)
        XCTAssertEqual(dists[0][1], 1.0, accuracy: 1e-10)  // (0,0) to (0,1)
        XCTAssertEqual(dists[1][2], 1.0, accuracy: 1e-10)  // (1,0) to (1,1)
    }

    func testPdist() {
        let X = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]

        let dists = pdist(X)

        // n=3 => n*(n-1)/2 = 3 distances
        XCTAssertEqual(dists.count, 3)
        XCTAssertEqual(dists[0], 3.0, accuracy: 1e-10)  // 0-1
        XCTAssertEqual(dists[1], 4.0, accuracy: 1e-10)  // 0-2
        XCTAssertEqual(dists[2], 5.0, accuracy: 1e-10)  // 1-2
    }

    func testSquareform() {
        let matrix = [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0]
        ]

        let condensed = squareform(matrix)
        XCTAssertEqual(condensed.count, 3)
        XCTAssertEqual(condensed[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(condensed[1], 2.0, accuracy: 1e-10)
        XCTAssertEqual(condensed[2], 3.0, accuracy: 1e-10)

        // Round trip
        let back = squareformToMatrix(condensed)
        XCTAssertEqual(back.count, 3)
        for i in 0..<3 {
            for j in 0..<3 {
                XCTAssertEqual(back[i][j], matrix[i][j], accuracy: 1e-10)
            }
        }
    }

    // MARK: - KDTree Tests

    func testKDTreeQuery() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        let tree = KDTree(points)

        // Query nearest to (0.1, 0.1)
        let (indices, distances) = tree.query([0.1, 0.1], k: 1)
        XCTAssertEqual(indices.count, 1)
        XCTAssertEqual(indices[0], 0)  // Closest to (0,0)
        XCTAssertEqual(distances[0], euclideanDistance([0.1, 0.1], [0.0, 0.0]), accuracy: 1e-10)

        // Query 2 nearest
        let (indices2, _) = tree.query([0.5, 0.5], k: 2)
        XCTAssertEqual(indices2.count, 2)
    }

    func testKDTreeQueryRadius() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]
        let tree = KDTree(points)

        // Query within radius 1.5 from origin
        let (indices, _) = tree.queryRadius([0.0, 0.0], radius: 1.5)

        // Should include (0,0), (1,0), (0,1) but not (5,5)
        XCTAssertEqual(indices.count, 3)
        XCTAssertTrue(indices.contains(0))
        XCTAssertTrue(indices.contains(1))
        XCTAssertTrue(indices.contains(2))
    }

    func testKDTreeQueryPairs() {
        let points = [[0.0, 0.0], [0.5, 0.0], [10.0, 10.0]]
        let tree = KDTree(points)

        let pairs = tree.queryPairs(radius: 1.0)

        // Only (0,0) and (0.5,0) are close enough
        XCTAssertEqual(pairs.count, 1)
        XCTAssertEqual(pairs[0].0, 0)
        XCTAssertEqual(pairs[0].1, 1)
        XCTAssertEqual(pairs[0].2, 0.5, accuracy: 1e-10)
    }

    // MARK: - Delaunay Tests

    func testDelaunayTriangle() {
        // Simple triangle
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        let result = delaunay(points)

        XCTAssertEqual(result.simplices.count, 1)
        XCTAssertEqual(result.simplices[0].count, 3)

        // All vertices should be present
        let vertices = Set(result.simplices[0])
        XCTAssertTrue(vertices.contains(0))
        XCTAssertTrue(vertices.contains(1))
        XCTAssertTrue(vertices.contains(2))
    }

    func testDelaunaySquare() {
        // Square should produce 2 triangles
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        let result = delaunay(points)

        XCTAssertEqual(result.simplices.count, 2)
    }

    func testDelaunayEmpty() {
        let result = delaunay([[0.0, 0.0], [1.0, 0.0]])
        XCTAssertTrue(result.simplices.isEmpty)
    }

    // MARK: - Voronoi Tests

    func testVoronoiBasic() {
        let points = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]
        let result = voronoi(points)

        XCTAssertEqual(result.points.count, 3)
        // Should have at least one vertex (circumcenter)
        XCTAssertGreaterThan(result.vertices.count, 0)
    }

    func testVoronoiEmpty() {
        let result = voronoi([])
        XCTAssertTrue(result.vertices.isEmpty)
    }

    // MARK: - Convex Hull Tests

    func testConvexHullTriangle() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        let result = convexHull(points)

        XCTAssertEqual(result.vertices.count, 3)
        XCTAssertEqual(result.simplices.count, 3)
        XCTAssertEqual(result.area, 0.5, accuracy: 1e-10)
    }

    func testConvexHullSquare() {
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        let result = convexHull(points)

        XCTAssertEqual(result.vertices.count, 4)
        XCTAssertEqual(result.simplices.count, 4)
        XCTAssertEqual(result.area, 1.0, accuracy: 1e-10)
    }

    func testConvexHullWithInterior() {
        // Square with center point
        let points = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]]
        let result = convexHull(points)

        // Hull should only contain 4 vertices (not the center)
        XCTAssertEqual(result.vertices.count, 4)
        XCTAssertFalse(result.vertices.contains(4))  // Center point index
        XCTAssertEqual(result.area, 4.0, accuracy: 1e-10)
    }

    // MARK: - Integration Tests

    func testKDTreeWithClustering() {
        // Create clustered data
        var points: [[Double]] = []
        for _ in 0..<20 {
            points.append([Double.random(in: 0...1), Double.random(in: 0...1)])
        }
        for _ in 0..<20 {
            points.append([Double.random(in: 10...11), Double.random(in: 10...11)])
        }

        let tree = KDTree(points)

        // Query from first cluster should find neighbors in same cluster
        let (indices, _) = tree.query([0.5, 0.5], k: 5)
        for idx in indices {
            // All should be from first cluster (index < 20)
            XCTAssertLessThan(idx, 20)
        }
    }

    func testDistanceMetricConsistency() {
        let p1 = [1.0, 2.0, 3.0]
        let p2 = [4.0, 5.0, 6.0]

        // Test via enum
        let eucDist = distanceFunction(for: .euclidean)(p1, p2)
        let manDist = distanceFunction(for: .manhattan)(p1, p2)
        let cheDist = distanceFunction(for: .chebyshev)(p1, p2)

        XCTAssertEqual(eucDist, euclideanDistance(p1, p2), accuracy: 1e-10)
        XCTAssertEqual(manDist, manhattanDistance(p1, p2), accuracy: 1e-10)
        XCTAssertEqual(cheDist, chebyshevDistance(p1, p2), accuracy: 1e-10)
    }
}
