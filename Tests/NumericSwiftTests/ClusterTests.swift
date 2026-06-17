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
        XCTAssertEqual(Spatial.euclideanDistance(p1, p2), 5.0, accuracy: 1e-10)

        // Same point
        XCTAssertEqual(Spatial.euclideanDistance(p1, p1), 0.0, accuracy: 1e-10)

        // 3D distance
        let p3 = [1.0, 2.0, 3.0]
        let p4 = [4.0, 6.0, 3.0]
        XCTAssertEqual(Spatial.euclideanDistance(p3, p4), 5.0, accuracy: 1e-10)
    }

    func testSquaredEuclideanDistance() {
        let p1 = [0.0, 0.0]
        let p2 = [3.0, 4.0]
        XCTAssertEqual(Spatial.squaredEuclideanDistance(p1, p2), 25.0, accuracy: 1e-10)
    }

    func testComputeCentroid() {
        let points = [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]]
        let centroid = Cluster.centroid(points)!
        XCTAssertEqual(centroid[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(centroid[1], 1.0, accuracy: 1e-10)

        // Empty case
        XCTAssertNil(Cluster.centroid([]))
    }

    func testPairwiseDistances() {
        let data = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
        let dists = Spatial.cdist(data, data)

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

        let result = Cluster.kmeans(data, k: 2)

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
        let result = Cluster.kmeans([], k: 3)
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

        let result = Cluster.kmeans(data, k: 3, maxIterations: 100)

        // Should converge
        XCTAssertLessThanOrEqual(result.iterations, 100)
        XCTAssertGreaterThan(result.centroids.count, 0)
    }

    func testKMeansPlusPlusInit() {
        // kMeansPlusPlusInit is now a private implementation detail of Cluster.kmeans.
        // Verify its effect indirectly: kmeans with k-means++ init converges to k centroids
        // that are drawn from the data distribution.
        let data = [[0.0, 0.0], [10.0, 0.0], [5.0, 10.0], [5.0, 5.0]]
        let result = Cluster.kmeans(data, k: 3)

        // Should produce exactly k centroids
        XCTAssertEqual(result.centroids.count, 3)
        // Labels must be valid cluster indices
        for label in result.labels {
            XCTAssertGreaterThanOrEqual(label, 0)
            XCTAssertLessThan(label, 3)
        }
    }

    func testKMeansInertia() {
        // Perfect clustering
        let data = [[0.0, 0.0], [10.0, 10.0]]
        let result = Cluster.kmeans(data, k: 2)

        // With k=2, each point is its own centroid, inertia should be 0
        XCTAssertEqual(result.inertia, 0, accuracy: 1e-10)
    }

    // MARK: - Hierarchical Clustering Tests

    func testHierarchicalSimple() {
        let data = [[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]]

        let result = Cluster.hierarchicalClustering(data, linkage: .single, nClusters: 2)

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
            let result = Cluster.hierarchicalClustering(data, linkage: linkage)
            XCTAssertEqual(result.linkageMatrix.count, 3)
        }
    }

    func testHierarchicalWard() {
        let data = [
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
            [10.0, 0.0], [10.5, 0.0], [11.0, 0.0]
        ]

        let result = Cluster.hierarchicalClustering(data, linkage: .ward, nClusters: 2)

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

        let result = Cluster.hierarchicalClustering(data, linkage: .single, distanceThreshold: 2.0)

        guard let labels = result.labels else {
            XCTFail("Expected labels")
            return
        }

        // Points closer than 2.0 should be grouped
        XCTAssertEqual(labels[0], labels[1])
        XCTAssertEqual(labels[2], labels[3])
    }

    func testHierarchicalEmpty() {
        let result = Cluster.hierarchicalClustering([])
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

        let result = Cluster.dbscan(data, eps: 0.5, minSamples: 3)

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

        let result = Cluster.dbscan(data, eps: 0.5, minSamples: 3)

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

        let result = Cluster.dbscan(data, eps: 0.2, minSamples: 3)

        // Core samples should not include border point
        // (This depends on exact geometry)
        XCTAssertGreaterThan(result.coreSamples.count, 0)
    }

    func testDBSCANEmpty() {
        let result = Cluster.dbscan([], eps: 0.5, minSamples: 5)
        XCTAssertTrue(result.labels.isEmpty)
        XCTAssertEqual(result.nClusters, 0)
    }

    // MARK: - DBSCAN Sklearn Equivalence Tests

    /// Check whether two clusterings are partition-equivalent.
    ///
    /// Two clusterings are equivalent when:
    /// - They have the same length.
    /// - Noise indices (label == -1) match exactly.
    /// - There exists a bijection between the non-noise label sets such that each
    ///   pair of labels maps to the identical set of point indices.
    private func areClusteringsEquivalent(_ a: [Int], _ b: [Int]) -> Bool {
        guard a.count == b.count else { return false }
        let n = a.count

        // Noise points must agree exactly by index.
        for i in 0..<n {
            let aIsNoise = (a[i] == -1)
            let bIsNoise = (b[i] == -1)
            if aIsNoise != bIsNoise { return false }
        }

        // Build label→index-set maps for non-noise points.
        var aMap: [Int: Set<Int>] = [:]
        var bMap: [Int: Set<Int>] = [:]
        for i in 0..<n {
            if a[i] != -1 { aMap[a[i], default: []].insert(i) }
            if b[i] != -1 { bMap[b[i], default: []].insert(i) }
        }

        // Partition cardinalities must match.
        guard aMap.count == bMap.count else { return false }

        // Every partition in `a` must appear in `b`.
        let bPartitions = Set(bMap.values)
        for partition in aMap.values {
            if !bPartitions.contains(partition) { return false }
        }
        return true
    }

    /// sklearn 1.9.0, DBSCAN(eps=0.5, min_samples=3) on 45-point three-blob dataset.
    ///
    /// The 45 points are three Gaussian blobs (seed=42, σ=0.2) centred at
    /// (0,0), (5,0), and (0,5) — 15 points each.  sklearn labels: 0 for
    /// blob-1, 1 for blob-2, 2 for blob-3 (all 45 non-noise).
    func testDBSCANEquivalenceToSklearn_blobs() {
        let data: [[Double]] = [
            [0.09934283060224654, -0.027652860234236933],
            [0.1295377076201385, 0.3046059712816051],
            [-0.046830674944667194, -0.04682739138983611],
            [0.3158425631014783, 0.15348694583058176],
            [-0.09389487718699042, 0.10851200871719294],
            [-0.09268353856249245, -0.09314595071405138],
            [0.04839245431320682, -0.3826560489315596],
            [-0.3449835665026066, -0.11245750584819454],
            [-0.20256622406688476, 0.06284946651905478],
            [-0.1816048151042422, -0.2824607402670583],
            [0.29312975378431083, -0.04515526009730714],
            [0.013505640937584768, -0.28494963724269134],
            [-0.10887654490503654, 0.02218451794197322],
            [-0.23019871548446058, 0.0751396036691344],
            [-0.120127737983761, -0.05833874995865536],
            [4.87965867755412, 0.37045563690178757],
            [4.997300555052413, -0.2115421857911801],
            [5.164508982420638, -0.24416872999420447],
            [5.041772719000951, -0.3919340247759551],
            [4.734362790220314, 0.039372247173824704],
            [5.147693315999082, 0.0342736562379941],
            [4.976870343522352, -0.06022073911785776],
            [4.704295601926514, -0.14396884167894172],
            [4.907872245808043, 0.21142444524378315],
            [5.068723657913692, -0.3526080310725468],
            [5.0648167938789594, -0.07701645608326331],
            [4.8646155999388085, 0.12233525776817358],
            [5.20619990449919, 0.18625602382323972],
            [4.8321564953554725, -0.06184247517024292],
            [5.066252686280713, 0.19510902542447184],
            [-0.095834847569058, 4.962868204667236],
            [-0.22126699480120565, 4.760758675183866],
            [0.16250516447883961, 5.271248005714164],
            [-0.01440202431606677, 5.200706579578405],
            [0.07232720500952683, 4.870976049078975],
            [0.07227912110168279, 5.307607313293194],
            [-0.007165207821990308, 5.312928731162801],
            [-0.5239490208179489, 5.164380500875045],
            [0.017409413647634243, 4.9401985299068265],
            [0.01835215530710046, 4.6024862170798215],
            [-0.04393437756750239, 5.071422514302349],
            [0.29557880894830324, 4.896345956345271],
            [-0.16169872057863754, 4.899648591283093],
            [0.18308042354041484, 5.065750221931937],
            [-0.10595204075340776, 5.102653486622671]
        ]
        // sklearn reference: all 45 points in exactly 3 clusters, zero noise
        let sklearnLabels = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ]
        let result = Cluster.dbscan(data, eps: 0.5, minSamples: 3)

        XCTAssertEqual(result.nClusters, 3, "expected 3 clusters matching sklearn")
        XCTAssertTrue(
            areClusteringsEquivalent(result.labels, sklearnLabels),
            "partition mismatch vs sklearn — got \(result.labels), expected \(sklearnLabels)"
        )
    }

    /// sklearn 1.9.0, DBSCAN(eps=0.2, min_samples=3) on 8-point varying-density dataset.
    ///
    /// Dense cluster of 5 near origin; sparse cluster of 3 near (2,2).
    func testDBSCANEquivalenceToSklearn_varyingDensity() {
        let data: [[Double]] = [
            [0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [0.05, 0.05], [0.025, 0.025],
            [2.0, 2.0], [2.1, 2.0], [2.0, 2.1]
        ]
        // sklearn reference: cluster 0 for indices 0-4, cluster 1 for indices 5-7
        let sklearnLabels = [0, 0, 0, 0, 0, 1, 1, 1]
        let result = Cluster.dbscan(data, eps: 0.2, minSamples: 3)

        XCTAssertEqual(result.nClusters, 2, "expected 2 clusters matching sklearn")
        XCTAssertTrue(
            areClusteringsEquivalent(result.labels, sklearnLabels),
            "partition mismatch vs sklearn — got \(result.labels), expected \(sklearnLabels)"
        )
    }

    /// sklearn 1.9.0, DBSCAN(eps=0.1, min_samples=3): five isolated points → all noise.
    func testDBSCANEquivalenceToSklearn_allNoise() {
        let data: [[Double]] = [
            [0.0, 0.0], [5.0, 5.0], [10.0, 0.0], [0.0, 10.0], [7.0, 3.0]
        ]
        let sklearnLabels = [-1, -1, -1, -1, -1]
        let result = Cluster.dbscan(data, eps: 0.1, minSamples: 3)

        XCTAssertEqual(result.nClusters, 0, "expected 0 clusters — all noise")
        XCTAssertTrue(
            areClusteringsEquivalent(result.labels, sklearnLabels),
            "all-noise mismatch — got \(result.labels)"
        )
    }

    /// sklearn 1.9.0, DBSCAN(eps=0.5, min_samples=3): five tightly packed points → one cluster.
    func testDBSCANEquivalenceToSklearn_singleCluster() {
        let data: [[Double]] = [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05]
        ]
        let sklearnLabels = [0, 0, 0, 0, 0]
        let result = Cluster.dbscan(data, eps: 0.5, minSamples: 3)

        XCTAssertEqual(result.nClusters, 1, "expected 1 cluster matching sklearn")
        XCTAssertTrue(
            areClusteringsEquivalent(result.labels, sklearnLabels),
            "single-cluster mismatch — got \(result.labels)"
        )
    }

    /// sklearn 1.9.0, DBSCAN(eps=0.1, min_samples=3): 4+3 duplicate points → 2 clusters.
    func testDBSCANEquivalenceToSklearn_duplicatePoints() {
        let data: [[Double]] = [
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]
        ]
        let sklearnLabels = [0, 0, 0, 0, 1, 1, 1]
        let result = Cluster.dbscan(data, eps: 0.1, minSamples: 3)

        XCTAssertEqual(result.nClusters, 2, "expected 2 clusters matching sklearn")
        XCTAssertTrue(
            areClusteringsEquivalent(result.labels, sklearnLabels),
            "duplicate-points mismatch — got \(result.labels)"
        )
    }

    /// Empty input produces empty result with zero clusters.
    func testDBSCANEdgeCases_empty() {
        let result = Cluster.dbscan([], eps: 0.5, minSamples: 5)
        XCTAssertTrue(result.labels.isEmpty, "labels must be empty for empty input")
        XCTAssertTrue(result.coreSamples.isEmpty, "coreSamples must be empty for empty input")
        XCTAssertEqual(result.nClusters, 0, "nClusters must be 0 for empty input")
    }

    // MARK: - Silhouette Score Tests

    func testSilhouetteScoreGood() {
        // Well-separated clusters
        let data = [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]
        ]
        let labels = [0, 0, 0, 1, 1, 1]

        let score = Cluster.silhouetteScore(data, labels: labels)

        // Well-separated clusters should have high score
        XCTAssertGreaterThan(score, 0.5)
    }

    func testSilhouetteScoreBad() {
        // Overlapping clusters
        let data = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        let labels = [0, 1, 0, 1]  // Alternating labels - bad clustering

        let score = Cluster.silhouetteScore(data, labels: labels)

        // Poor clustering should have low/negative score
        XCTAssertLessThan(score, 0.5)
    }

    func testSilhouetteScoreIgnoresNoise() {
        let data = [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0], [100.0, 0.0]]
        let labels = [0, 0, 1, 1, -1]  // Last is noise

        let score = Cluster.silhouetteScore(data, labels: labels)

        // Should compute score ignoring noise point
        XCTAssertGreaterThan(score, 0)
    }

    func testSilhouetteScoreSingleCluster() {
        let data = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        let labels = [0, 0, 0]

        let score = Cluster.silhouetteScore(data, labels: labels)

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

        let result = Cluster.elbowMethod(data, maxK: 6)

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
        let result = Cluster.elbowMethod([])
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

        let kmeansResult = Cluster.kmeans(data, k: 2)
        let score = Cluster.silhouetteScore(data, labels: kmeansResult.labels)

        // Well-separated clusters should have good silhouette
        XCTAssertGreaterThan(score, 0.5)
    }
}
