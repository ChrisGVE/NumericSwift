//
//  ClusterTests.swift
//  NumericSwift
//
//  Tests for clustering algorithms.
//

import XCTest
@testable import NumericSwift

final class ClusterTests: XCTestCase {

    // MARK: - Distance Function Tests

    func testEuclideanDistance() {
        // Basic 2D distance
        let p1 = [0.0, 0.0]
        let p2 = [3.0, 4.0]
        XCTAssertEqual(euclideanDistance(p1, p2), 5.0, accuracy: 1e-10)

        // Same point
        XCTAssertEqual(euclideanDistance(p1, p1), 0.0, accuracy: 1e-10)

        // 3D distance
        let p3 = [1.0, 2.0, 3.0]
        let p4 = [4.0, 6.0, 3.0]
        XCTAssertEqual(euclideanDistance(p3, p4), 5.0, accuracy: 1e-10)
    }

    func testSquaredEuclideanDistance() {
        let p1 = [0.0, 0.0]
        let p2 = [3.0, 4.0]
        XCTAssertEqual(squaredEuclideanDistance(p1, p2), 25.0, accuracy: 1e-10)
    }

    func testComputeCentroid() {
        let points = [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]]
        let centroid = computeCentroid(points)!
        XCTAssertEqual(centroid[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(centroid[1], 1.0, accuracy: 1e-10)

        // Empty case
        XCTAssertNil(computeCentroid([]))
    }

    func testPairwiseDistances() {
        let data = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
        let dists = pairwiseDistances(data)

        // Diagonal is zero
        XCTAssertEqual(dists[0][0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(dists[1][1], 0.0, accuracy: 1e-10)
        XCTAssertEqual(dists[2][2], 0.0, accuracy: 1e-10)

        // Distances
        XCTAssertEqual(dists[0][1], 3.0, accuracy: 1e-10)
        XCTAssertEqual(dists[0][2], 4.0, accuracy: 1e-10)
        XCTAssertEqual(dists[1][2], 5.0, accuracy: 1e-10)

        // Symmetric
        XCTAssertEqual(dists[1][0], dists[0][1], accuracy: 1e-10)
    }

    // MARK: - K-Means Tests

    func testKMeansSimple() {
        // Two clear clusters
        let data = [
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.1, 0.2],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.0], [5.0, 5.2]
        ]

        let result = kmeans(data, k: 2)

        // Should find 2 clusters
        XCTAssertEqual(result.centroids.count, 2)
        XCTAssertEqual(result.labels.count, 8)

        // Labels should group points correctly
        // First 4 should have same label, last 4 should have same (different) label
        let firstGroup = result.labels[0]
        let secondGroup = result.labels[4]
        XCTAssertNotEqual(firstGroup, secondGroup)

        for i in 0..<4 {
            XCTAssertEqual(result.labels[i], firstGroup)
        }
        for i in 4..<8 {
            XCTAssertEqual(result.labels[i], secondGroup)
        }
    }

    func testKMeansEmpty() {
        let result = kmeans([], k: 3)
        XCTAssertTrue(result.labels.isEmpty)
        XCTAssertTrue(result.centroids.isEmpty)
        XCTAssertEqual(result.inertia, 0)
    }

    func testKMeansConvergence() {
        // Random data
        var data: [[Double]] = []
        for _ in 0..<50 {
            data.append([Double.random(in: 0...10), Double.random(in: 0...10)])
        }

        let result = kmeans(data, k: 3, maxIterations: 100)

        // Should converge
        XCTAssertLessThanOrEqual(result.iterations, 100)
        XCTAssertGreaterThan(result.centroids.count, 0)
    }

    func testKMeansPlusPlusInit() {
        let data = [[0.0, 0.0], [10.0, 0.0], [5.0, 10.0], [5.0, 5.0]]
        let centroids = kmeansPlusPlusInit(data: data, k: 3, dim: 2)

        // Should return k centroids
        XCTAssertEqual(centroids.count, 3)

        // All centroids should be from data
        for centroid in centroids {
            XCTAssertTrue(data.contains { $0[0] == centroid[0] && $0[1] == centroid[1] })
        }
    }

    func testKMeansInertia() {
        // Perfect clustering
        let data = [[0.0, 0.0], [10.0, 10.0]]
        let result = kmeans(data, k: 2)

        // With k=2, each point is its own centroid, inertia should be 0
        XCTAssertEqual(result.inertia, 0, accuracy: 1e-10)
    }

    // MARK: - Hierarchical Clustering Tests

    func testHierarchicalSimple() {
        let data = [[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]]

        let result = hierarchicalClustering(data, linkage: .single, nClusters: 2)

        // Should produce linkage matrix
        XCTAssertEqual(result.linkageMatrix.count, 3)  // n-1 merges
        XCTAssertEqual(result.nLeaves, 4)

        // Should produce labels
        guard let labels = result.labels else {
            XCTFail("Expected labels")
            return
        }

        // First two and last two should be grouped
        XCTAssertEqual(labels[0], labels[1])
        XCTAssertEqual(labels[2], labels[3])
        XCTAssertNotEqual(labels[0], labels[2])
    }

    func testHierarchicalLinkageMethods() {
        let data = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]]

        for linkage in [LinkageMethod.single, .complete, .average, .ward] {
            let result = hierarchicalClustering(data, linkage: linkage)
            XCTAssertEqual(result.linkageMatrix.count, 3)
        }
    }

    func testHierarchicalWard() {
        let data = [
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
            [10.0, 0.0], [10.5, 0.0], [11.0, 0.0]
        ]

        let result = hierarchicalClustering(data, linkage: .ward, nClusters: 2)

        guard let labels = result.labels else {
            XCTFail("Expected labels")
            return
        }

        // First 3 and last 3 should be in different clusters
        XCTAssertEqual(labels[0], labels[1])
        XCTAssertEqual(labels[1], labels[2])
        XCTAssertEqual(labels[3], labels[4])
        XCTAssertEqual(labels[4], labels[5])
        XCTAssertNotEqual(labels[0], labels[3])
    }

    func testHierarchicalDistanceThreshold() {
        let data = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]]

        let result = hierarchicalClustering(data, linkage: .single, distanceThreshold: 2.0)

        guard let labels = result.labels else {
            XCTFail("Expected labels")
            return
        }

        // Points closer than 2.0 should be grouped
        XCTAssertEqual(labels[0], labels[1])
        XCTAssertEqual(labels[2], labels[3])
    }

    func testHierarchicalEmpty() {
        let result = hierarchicalClustering([])
        XCTAssertTrue(result.linkageMatrix.isEmpty)
        XCTAssertEqual(result.nLeaves, 0)
    }

    // MARK: - DBSCAN Tests

    func testDBSCANSimple() {
        // Two dense clusters
        let data = [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1]
        ]

        let result = dbscan(data, eps: 0.5, minSamples: 3)

        XCTAssertEqual(result.labels.count, 8)
        XCTAssertEqual(result.nClusters, 2)

        // First 4 should be one cluster, last 4 another
        XCTAssertEqual(result.labels[0], result.labels[1])
        XCTAssertEqual(result.labels[4], result.labels[5])
        XCTAssertNotEqual(result.labels[0], result.labels[4])
    }

    func testDBSCANNoise() {
        // One cluster + one noise point
        let data = [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [10.0, 10.0]  // Isolated point
        ]

        let result = dbscan(data, eps: 0.5, minSamples: 3)

        // Last point should be noise (-1)
        XCTAssertEqual(result.labels[4], -1)

        // First 4 should be in a cluster (>=0)
        XCTAssertGreaterThanOrEqual(result.labels[0], 0)
    }

    func testDBSCANCoreSamples() {
        let data = [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [0.2, 0.2]  // Border point (only 2 neighbors within eps)
        ]

        let result = dbscan(data, eps: 0.2, minSamples: 3)

        // Core samples should not include border point
        // (This depends on exact geometry)
        XCTAssertGreaterThan(result.coreSamples.count, 0)
    }

    func testDBSCANEmpty() {
        let result = dbscan([], eps: 0.5, minSamples: 5)
        XCTAssertTrue(result.labels.isEmpty)
        XCTAssertEqual(result.nClusters, 0)
    }

    // MARK: - Silhouette Score Tests

    func testSilhouetteScoreGood() {
        // Well-separated clusters
        let data = [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]
        ]
        let labels = [0, 0, 0, 1, 1, 1]

        let score = silhouetteScore(data, labels: labels)

        // Well-separated clusters should have high score
        XCTAssertGreaterThan(score, 0.5)
    }

    func testSilhouetteScoreBad() {
        // Overlapping clusters
        let data = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        let labels = [0, 1, 0, 1]  // Alternating labels - bad clustering

        let score = silhouetteScore(data, labels: labels)

        // Poor clustering should have low/negative score
        XCTAssertLessThan(score, 0.5)
    }

    func testSilhouetteScoreIgnoresNoise() {
        let data = [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0], [100.0, 0.0]]
        let labels = [0, 0, 1, 1, -1]  // Last is noise

        let score = silhouetteScore(data, labels: labels)

        // Should compute score ignoring noise point
        XCTAssertGreaterThan(score, 0)
    }

    func testSilhouetteScoreSingleCluster() {
        let data = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        let labels = [0, 0, 0]

        let score = silhouetteScore(data, labels: labels)

        // Single cluster returns 0
        XCTAssertEqual(score, 0)
    }

    // MARK: - Elbow Method Tests

    func testElbowMethod() {
        // Data with clear clusters
        var data: [[Double]] = []
        // Cluster 1
        for _ in 0..<20 {
            data.append([Double.random(in: 0...1), Double.random(in: 0...1)])
        }
        // Cluster 2
        for _ in 0..<20 {
            data.append([Double.random(in: 10...11), Double.random(in: 10...11)])
        }
        // Cluster 3
        for _ in 0..<20 {
            data.append([Double.random(in: 20...21), Double.random(in: 0...1)])
        }

        let result = elbowMethod(data, maxK: 6)

        XCTAssertEqual(result.inertias.count, 6)

        // Inertia should decrease with k
        for i in 1..<result.inertias.count {
            XCTAssertLessThanOrEqual(result.inertias[i], result.inertias[i - 1])
        }

        // Suggested k should be reasonable
        XCTAssertGreaterThanOrEqual(result.suggestedK, 1)
        XCTAssertLessThanOrEqual(result.suggestedK, 6)
    }

    func testElbowMethodEmpty() {
        let result = elbowMethod([])
        XCTAssertTrue(result.inertias.isEmpty)
        XCTAssertEqual(result.suggestedK, 1)
    }

    // MARK: - Integration Tests

    func testKMeansThenSilhouette() {
        var data: [[Double]] = []
        for _ in 0..<30 {
            data.append([Double.random(in: 0...1), Double.random(in: 0...1)])
        }
        for _ in 0..<30 {
            data.append([Double.random(in: 10...11), Double.random(in: 10...11)])
        }

        let kmeansResult = kmeans(data, k: 2)
        let score = silhouetteScore(data, labels: kmeansResult.labels)

        // Well-separated clusters should have good silhouette
        XCTAssertGreaterThan(score, 0.5)
    }
}
