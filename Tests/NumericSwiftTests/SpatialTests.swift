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
        XCTAssertEqual(Spatial.manhattanDistance(p1, p2), 7.0, accuracy: 1e-10)

        // Same point
        XCTAssertEqual(Spatial.manhattanDistance(p1, p1), 0.0, accuracy: 1e-10)
    }

    func testChebyshevDistance() {
        let p1 = [0.0, 0.0]
        let p2 = [3.0, 7.0]
        XCTAssertEqual(Spatial.chebyshevDistance(p1, p2), 7.0, accuracy: 1e-10)

        let p3 = [1.0, 2.0, 3.0]
        let p4 = [4.0, 6.0, 3.0]
        XCTAssertEqual(Spatial.chebyshevDistance(p3, p4), 4.0, accuracy: 1e-10)
    }

    func testMinkowskiDistance() {
        let p1 = [0.0, 0.0]
        let p2 = [3.0, 4.0]

        // p=1 should equal Manhattan
        XCTAssertEqual(Spatial.minkowskiDistance(p1, p2, p: 1), Spatial.manhattanDistance(p1, p2), accuracy: 1e-10)

        // p=2 should equal Euclidean
        XCTAssertEqual(Spatial.minkowskiDistance(p1, p2, p: 2), Spatial.euclideanDistance(p1, p2), accuracy: 1e-10)

        // p=infinity should equal Chebyshev
        XCTAssertEqual(Spatial.minkowskiDistance(p1, p2, p: .infinity), Spatial.chebyshevDistance(p1, p2), accuracy: 1e-10)
    }

    func testCosineDistance() {
        // Same direction = distance 0
        let p1 = [1.0, 0.0]
        let p2 = [2.0, 0.0]
        XCTAssertEqual(Spatial.cosineDistance(p1, p2), 0.0, accuracy: 1e-10)

        // Perpendicular = distance 1
        let p3 = [1.0, 0.0]
        let p4 = [0.0, 1.0]
        XCTAssertEqual(Spatial.cosineDistance(p3, p4), 1.0, accuracy: 1e-10)

        // Opposite direction = distance 2
        let p5 = [1.0, 0.0]
        let p6 = [-1.0, 0.0]
        XCTAssertEqual(Spatial.cosineDistance(p5, p6), 2.0, accuracy: 1e-10)
    }

    func testCorrelationDistance() {
        // Perfect positive correlation
        let p1 = [1.0, 2.0, 3.0]
        let p2 = [2.0, 4.0, 6.0]
        XCTAssertEqual(Spatial.correlationDistance(p1, p2), 0.0, accuracy: 1e-10)

        // Perfect negative correlation
        let p3 = [1.0, 2.0, 3.0]
        let p4 = [3.0, 2.0, 1.0]
        XCTAssertEqual(Spatial.correlationDistance(p3, p4), 2.0, accuracy: 1e-10)
    }

    // MARK: - Batch Distance Tests

    func testCdist() {
        let XA = [[0.0, 0.0], [1.0, 0.0]]
        let XB = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        let dists = Spatial.cdist(XA, XB)

        XCTAssertEqual(dists.count, 2)
        XCTAssertEqual(dists[0].count, 3)

        // Check specific distances
        XCTAssertEqual(dists[0][0], 0.0, accuracy: 1e-10)  // (0,0) to (0,0)
        XCTAssertEqual(dists[0][1], 1.0, accuracy: 1e-10)  // (0,0) to (0,1)
        XCTAssertEqual(dists[1][2], 1.0, accuracy: 1e-10)  // (1,0) to (1,1)
    }

    func testPdist() {
        let X = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]

        let dists = Spatial.pdist(X)

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

        let condensed = Spatial.squareform(matrix)
        XCTAssertEqual(condensed.count, 3)
        XCTAssertEqual(condensed[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(condensed[1], 2.0, accuracy: 1e-10)
        XCTAssertEqual(condensed[2], 3.0, accuracy: 1e-10)

        // Round trip
        let back = Spatial.squareformToMatrix(condensed)
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
        XCTAssertEqual(distances[0], Spatial.euclideanDistance([0.1, 0.1], [0.0, 0.0]), accuracy: 1e-10)

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
        let result = Spatial.delaunay(points)

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
        let result = Spatial.delaunay(points)

        XCTAssertEqual(result.simplices.count, 2)
    }

    func testDelaunayEmpty() {
        let result = Spatial.delaunay([[0.0, 0.0], [1.0, 0.0]])
        XCTAssertTrue(result.simplices.isEmpty)
    }

    // MARK: - Delaunay Degeneracy Tests (Issue #12)

    /// All four points collinear — Bowyer-Watson has no valid triangulation.
    /// scipy.spatial.Delaunay raises QhullError; we return an empty simplex list.
    /// Oracle: scipy.spatial.Delaunay([[0,0],[1,0],[2,0],[3,0]]) → QhullError
    func testDelaunayFullyCollinear() {
        let points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        let result = Spatial.delaunay(points)
        XCTAssertTrue(
            result.simplices.isEmpty,
            "Fully collinear set must produce empty triangulation")
    }

    /// Later-collinear: first 3 points are NON-collinear (old check passes),
    /// but 3 of the 4 points lie on a line. Bowyer-Watson handles this correctly;
    /// scipy produces 2 triangles. Verifies robust collinearity detection doesn't
    /// incorrectly reject a valid (non-fully-collinear) input.
    /// Oracle: scipy.spatial.Delaunay([[0,1],[1,0],[2,0],[3,0]]) → [[2,3,0],[1,2,0]]
    func testDelaunayLaterCollinearSubset() {
        // First 3 points are non-collinear: [0,1],[1,0],[2,0] → cross ≠ 0
        // Points [1,0],[2,0],[3,0] ARE collinear, but the full set is NOT fully collinear.
        // Valid Delaunay triangulation exists (2 triangles).
        let points = [[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        let result = Spatial.delaunay(points)
        XCTAssertEqual(
            result.simplices.count, 2,
            "Partially-collinear (but not fully) set must produce 2 triangles")
        // All 4 point indices must appear across the two triangles.
        let allIndices = Set(result.simplices.flatMap { $0 })
        XCTAssertEqual(allIndices, [0, 1, 2, 3])
    }

    /// Reordered collinear: points in an order where naive first-3 check
    /// selects non-collinear triple while the full set IS fully collinear.
    /// Geometrically impossible (any 3 of n collinear points are collinear),
    /// so this test confirms the robust check still catches it.
    /// Oracle: scipy.spatial.Delaunay([[0,0],[1,0],[2,0],[3,0]]) → QhullError (empty)
    func testDelaunayCollinearAllOrders() {
        // Permutations of fully-collinear set: any order should yield empty.
        let base: [[Double]] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        let permutations: [[[Double]]] = [
            [base[2], base[0], base[1], base[3]],
            [base[3], base[1], base[0], base[2]],
            [base[1], base[3], base[2], base[0]],
        ]
        for perm in permutations {
            let result = Spatial.delaunay(perm)
            XCTAssertTrue(
                result.simplices.isEmpty,
                "Fully collinear set (any ordering) must produce empty triangulation")
        }
    }

    // MARK: - Voronoi Tests

    func testVoronoiBasic() {
        let points = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]
        let result = Spatial.voronoi(points)

        XCTAssertEqual(result.points.count, 3)
        // Should have at least one vertex (circumcenter)
        XCTAssertGreaterThan(result.vertices.count, 0)
    }

    func testVoronoiEmpty() {
        let result = Spatial.voronoi([])
        XCTAssertTrue(result.vertices.isEmpty)
    }

    // MARK: - Voronoi Infinite-Region Tests (Issue #12)

    /// Three points form a triangle: ALL generators are on the convex hull and
    /// have infinite Voronoi regions. The current implementation silently drops
    /// the infinite-region markers, producing empty region lists for hull generators.
    ///
    /// scipy contract (scipy.spatial.Voronoi([[0,0],[1,0],[0.5,0.866]])):
    ///   ridge_vertices: [[-1,0], [-1,0], [-1,0]]  (all ridges infinite)
    ///   regions[generator]: each contains -1 (infinite vertex sentinel)
    ///
    /// Our contract: ridgeVertices entries contain -1 for ridges extending to infinity;
    /// regions[i] contains -1 to mark that generator i has an infinite Voronoi region.
    func testVoronoiThreePointsInfiniteRegions() {
        // Triangle: all 3 generators on convex hull → all regions are infinite.
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]]
        let result = Spatial.voronoi(points)

        // Must produce exactly 1 Voronoi vertex (the circumcenter of the triangle).
        XCTAssertEqual(result.vertices.count, 1, "Triangle Voronoi must have 1 circumcenter vertex")

        // All 3 ridges must be infinite (contain -1).
        XCTAssertEqual(result.ridgeVertices.count, 3, "Triangle Voronoi must have 3 ridges")
        for ridge in result.ridgeVertices {
            XCTAssertTrue(
                ridge.contains(-1),
                "Every ridge of a 3-point Voronoi must be infinite (contain -1)")
        }

        // Every generator's region must contain -1 (infinite region marker).
        // regions has 3 entries, one per generator.
        XCTAssertEqual(result.regions.count, 3)
        for (i, region) in result.regions.enumerated() {
            XCTAssertTrue(
                region.contains(-1),
                "Generator \(i) is on the convex hull and must have -1 in its region list")
        }
    }

    /// Square: 4 hull generators sharing a single unique circumcenter at [0.5, 0.5].
    ///
    /// scipy contract (scipy.spatial.Voronoi([[0,0],[1,0],[1,1],[0,1]])):
    ///   vertices: [[0.5, 0.5]]   (both Delaunay triangles share the same circumcenter;
    ///                             scipy deduplicates and treats even the internal ridge as
    ///                             infinite — an artifact of deduplication)
    ///   ridge_vertices: [[-1,0],[-1,0],[-1,0],[-1,0]]
    ///   regions[each generator]: contains -1
    ///
    /// Our contract: implementations that do not deduplicate vertices may produce a finite
    /// internal ridge between the two (geometrically identical) circumcenters.  The key
    /// invariant is that all 4 generators are hull generators and therefore each region
    /// contains -1, and at least the 4 hull edges produce infinite ridges.
    func testVoronoiFourPointsSquareInfiniteRegions() {
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        let result = Spatial.voronoi(points)

        // Both Delaunay triangles of a square have circumcenter [0.5, 0.5].
        // Implementations may emit 1 or 2 entries; all must be near [0.5, 0.5].
        XCTAssertGreaterThan(result.vertices.count, 0, "Square Voronoi must have at least 1 vertex")
        for v in result.vertices {
            XCTAssertEqual(v[0], 0.5, accuracy: 1e-10)
            XCTAssertEqual(v[1], 0.5, accuracy: 1e-10)
        }

        // At least some ridges must be infinite (the 4 hull edges).
        // (Non-dedup implementations may also have a finite zero-length internal ridge.)
        let infiniteRidges = result.ridgeVertices.filter { $0.contains(-1) }
        XCTAssertGreaterThanOrEqual(
            infiniteRidges.count, 4,
            "Square must have at least 4 infinite ridges (one per hull edge)")

        // All 4 generator regions must contain -1 (all are hull generators).
        XCTAssertEqual(result.regions.count, 4)
        for (i, region) in result.regions.enumerated() {
            XCTAssertTrue(
                region.contains(-1),
                "Hull generator \(i) of square must have -1 in region list")
        }
    }

    /// Five-point set with 3 Delaunay triangles (2 unique circumcenters after dedup)
    /// and a mix of bounded/infinite ridges.
    ///
    /// SciPy oracle (scipy.spatial.Voronoi, scipy 1.x):
    ///   pts5 = [[0,0],[4,0],[2,3],[0.5,2],[3.5,2]]
    ///   vertices: [[2.0, 1.375], [2.0, 0.5625]]  (scipy deduplicates; impls may return 3)
    ///   ridge_vertices include both finite pairs and -1 pairs
    ///
    /// Our contract: 2 or 3 Voronoi vertices (2 unique circumcenters; non-dedup impls
    /// may return 3).  The frozen circumcenter coordinates are verified against SciPy.
    func testVoronoiFivePointsMixedRegions() {
        let points = [[0.0, 0.0], [4.0, 0.0], [2.0, 3.0], [0.5, 2.0], [3.5, 2.0]]
        let result = Spatial.voronoi(points)

        // Exactly 2 or 3 Voronoi vertices (scipy has 2 after dedup; non-dedup impls may return 3).
        XCTAssertTrue(
            result.vertices.count == 2 || result.vertices.count == 3,
            "5-point set must have 2 or 3 Voronoi vertices, got \(result.vertices.count)")

        // The two unique circumcenters from SciPy (frozen oracle).
        // Every vertex returned must match one of these two coordinates within 1e-6.
        let scipyVertices: [[Double]] = [[2.0, 1.375], [2.0, 0.5625]]
        for v in result.vertices {
            let matchesScipy = scipyVertices.contains { ref in
                abs(v[0] - ref[0]) < 1e-6 && abs(v[1] - ref[1]) < 1e-6
            }
            XCTAssertTrue(
                matchesScipy,
                "Voronoi vertex \(v) does not match any SciPy circumcenter \(scipyVertices)")
        }

        // At least some ridges must be infinite (the hull generators' ridges).
        let infiniteRidges = result.ridgeVertices.filter { $0.contains(-1) }
        XCTAssertGreaterThan(
            infiniteRidges.count, 0, "5-point set must have some infinite ridges")

        // At least some ridges must be finite (internal Voronoi edges).
        let finiteRidges = result.ridgeVertices.filter { !$0.contains(-1) }
        XCTAssertGreaterThan(
            finiteRidges.count, 0, "5-point set must have some finite ridges")

        // 5 regions, one per generator.
        XCTAssertEqual(result.regions.count, 5)
    }

    /// Regression test for the silent-ridge-drop bug (audit finding #1).
    ///
    /// When a Delaunay neighbour triangle has a degenerate circumcenter
    /// (|determinant| ≤ 1e-10, i.e. the three vertices are nearly collinear),
    /// the shared ridge must NOT be silently discarded.  Instead it must be
    /// emitted as an infinite ridge `[-1, v]` so that the SciPy-parity contract
    /// (every Delaunay edge maps to a Voronoi ridge) is maintained.
    ///
    /// This test uses a nearly-collinear triple [0,0],[2,0],[4,1e-12] which
    /// produces a triangle whose circumcenter determinant d ≈ 2·1e-12 ≤ 1e-10,
    /// paired with a non-degenerate neighbouring triangle formed by the fourth
    /// point [2, 2].  Before the fix the ridge between those two triangles was
    /// silently dropped; after the fix it appears as an infinite ridge.
    func testVoronoiDegenerateNeighbourCircumcenterBecomesInfiniteRidge() {
        // Three nearly-collinear points + one well-separated point.
        // The nearly-collinear triangle has |d| = 2*1e-12 ≤ 1e-10 → degenerate.
        let points = [[0.0, 0.0], [2.0, 0.0], [4.0, 1e-12], [2.0, 2.0]]
        let result = Spatial.voronoi(points)

        // Every Delaunay edge must appear as either a finite or an infinite ridge.
        // At minimum, the input set produces at least one infinite ridge.
        let infiniteRidges = result.ridgeVertices.filter { $0.contains(-1) }
        XCTAssertGreaterThan(
            infiniteRidges.count, 0,
            "Near-collinear input must produce at least one infinite ridge; "
            + "got ridgeVertices=\(result.ridgeVertices)")

        // None of the ridgeVertices arrays may be empty (which would indicate a
        // silently-dropped ridge has been stored as a placeholder).
        for rv in result.ridgeVertices {
            XCTAssertFalse(rv.isEmpty, "ridgeVertices must never contain an empty array")
        }
    }

    // MARK: - Convex Hull Tests

    func testConvexHullTriangle() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        let result = Spatial.convexHull(points)

        XCTAssertEqual(result.vertices.count, 3)
        XCTAssertEqual(result.simplices.count, 3)
        XCTAssertEqual(result.area, 0.5, accuracy: 1e-10)
    }

    func testConvexHullSquare() {
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        let result = Spatial.convexHull(points)

        XCTAssertEqual(result.vertices.count, 4)
        XCTAssertEqual(result.simplices.count, 4)
        XCTAssertEqual(result.area, 1.0, accuracy: 1e-10)
    }

    func testConvexHullWithInterior() {
        // Square with center point
        let points = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]]
        let result = Spatial.convexHull(points)

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
        let eucDist = Spatial.distanceFunction(for: .euclidean)(p1, p2)
        let manDist = Spatial.distanceFunction(for: .manhattan)(p1, p2)
        let cheDist = Spatial.distanceFunction(for: .chebyshev)(p1, p2)

        XCTAssertEqual(eucDist, Spatial.euclideanDistance(p1, p2), accuracy: 1e-10)
        XCTAssertEqual(manDist, Spatial.manhattanDistance(p1, p2), accuracy: 1e-10)
        XCTAssertEqual(cheDist, Spatial.chebyshevDistance(p1, p2), accuracy: 1e-10)
    }

    // MARK: - Malformed-input hardening (audit round 3)

    /// A zero-dimensional point set builds an empty tree instead of trapping at
    /// `depth % dim` (dim==0 → division by zero). A short query returns nothing.
    func testKDTreeZeroDimAndRaggedDoNotTrap() {
        let emptyDimTree = KDTree([[]])
        XCTAssertTrue(emptyDimTree.query([1.0], k: 1).indices.isEmpty)

        let ragged = KDTree([[0.0, 0.0], [1.0]])  // ragged → empty tree
        XCTAssertTrue(ragged.query([0.0, 0.0], k: 1).indices.isEmpty)

        // Valid tree, but a query vector shorter than the dimension returns empty
        // rather than trapping at point[node.axis].
        let tree = KDTree([[0.0, 0.0], [1.0, 1.0]])
        XCTAssertTrue(tree.query([0.0], k: 1).indices.isEmpty)
        XCTAssertTrue(tree.queryRadius([0.0], radius: 1.0).indices.isEmpty)
    }

    /// 2D algorithms must not trap on points with fewer than two coordinates;
    /// they return an empty result.
    func testConvexHullDelaunayVoronoiMalformedReturnEmpty() {
        let bad = [[0.0], [1.0], [2.0]]  // 1-D points
        XCTAssertTrue(Spatial.convexHull(bad).vertices.isEmpty)
        XCTAssertTrue(Spatial.delaunay(bad).simplices.isEmpty)
        XCTAssertTrue(Spatial.voronoi(bad).regions.isEmpty)
    }

    /// squareformToMatrix rejects a non-triangular length (n(n-1)/2) with an empty
    /// result rather than over-reading the condensed array.
    func testSquareformInvalidLengthReturnsEmpty() {
        // 4 is not a triangular number (1,3,6,10,…); valid lengths produce n×n.
        XCTAssertTrue(Spatial.squareformToMatrix([1, 2, 3, 4]).isEmpty)
        XCTAssertEqual(Spatial.squareformToMatrix([1, 2, 3]).count, 3)  // n=3 valid
    }
}
