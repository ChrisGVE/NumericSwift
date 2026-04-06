//
//  SpatialDistanceTests.swift
//  NumericSwiftTests
//
//  Tests for Mahalanobis, Jaccard, Hamming, Canberra, and Bray-Curtis distance metrics.
//
//  Licensed under the MIT License.
//

import XCTest

@testable import NumericSwift

final class SpatialDistanceTests: XCTestCase {

  // MARK: - Helpers

  private let eps = 1e-10

  /// Asserts triangle inequality: d(a,c) <= d(a,b) + d(b,c).
  private func assertTriangleInequality(
    _ dist: ([Double], [Double]) -> Double,
    _ a: [Double], _ b: [Double], _ c: [Double],
    file: StaticString = #file, line: UInt = #line
  ) {
    let dAB = dist(a, b)
    let dBC = dist(b, c)
    let dAC = dist(a, c)
    XCTAssertLessThanOrEqual(
      dAC, dAB + dBC + 1e-12, "Triangle inequality violated", file: file, line: line)
  }

  /// Asserts symmetry: d(a,b) == d(b,a).
  private func assertSymmetry(
    _ dist: ([Double], [Double]) -> Double,
    _ a: [Double], _ b: [Double],
    file: StaticString = #file, line: UInt = #line
  ) {
    XCTAssertEqual(dist(a, b), dist(b, a), accuracy: 1e-12, file: file, line: line)
  }

  // MARK: - Mahalanobis Distance

  func testMahalanobisIdentical() {
    let a = [1.0, 2.0, 3.0]
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    XCTAssertEqual(mahalanobisDistance(a, a, invCov: identity), 0.0, accuracy: eps)
  }

  func testMahalanobisIdentityEqualsEuclidean() {
    // With identity inverse covariance, Mahalanobis == Euclidean
    let a = [1.0, 0.0, 0.0]
    let b = [4.0, 4.0, 0.0]
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    let expected = euclideanDistance(a, b)
    XCTAssertEqual(mahalanobisDistance(a, b, invCov: identity), expected, accuracy: eps)
  }

  func testMahalanobisKnownValue() {
    // 2D example: invCov = [[2,0],[0,1]], diff=[1,1]
    // dist^2 = [1,1] * [[2,0],[0,1]] * [1,1]^T = 2 + 1 = 3 => dist = sqrt(3)
    let a = [1.0, 1.0]
    let b = [0.0, 0.0]
    let invCov = [[2.0, 0.0], [0.0, 1.0]]
    XCTAssertEqual(mahalanobisDistance(a, b, invCov: invCov), 3.0.squareRoot(), accuracy: eps)
  }

  func testMahalanobisSymmetry() {
    let a = [1.0, 2.0, 3.0]
    let b = [4.0, 5.0, 6.0]
    let invCov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    XCTAssertEqual(
      mahalanobisDistance(a, b, invCov: invCov),
      mahalanobisDistance(b, a, invCov: invCov),
      accuracy: eps
    )
  }

  func testMahalanobisEmptyInput() {
    let invCov: [[Double]] = []
    XCTAssertEqual(mahalanobisDistance([], [], invCov: invCov), 0.0, accuracy: eps)
  }

  func testMahalanobisInvalidInvCov() {
    // Mismatched invCov dimensions returns 0
    let a = [1.0, 2.0]
    let b = [3.0, 4.0]
    let invCov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    XCTAssertEqual(mahalanobisDistance(a, b, invCov: invCov), 0.0, accuracy: eps)
  }

  // MARK: - Jaccard Distance

  func testJaccardIdentical() {
    let a = [1.0, 0.0, 1.0, 1.0]
    XCTAssertEqual(jaccardDistance(a, a), 0.0, accuracy: eps)
  }

  func testJaccardKnownValue() {
    // a=[1,1,0,0], b=[1,0,1,0]: intersection=1, union=3 => 1 - 1/3 = 2/3
    let a = [1.0, 1.0, 0.0, 0.0]
    let b = [1.0, 0.0, 1.0, 0.0]
    XCTAssertEqual(jaccardDistance(a, b), 2.0 / 3.0, accuracy: eps)
  }

  func testJaccardDisjoint() {
    // No shared non-zero elements => distance = 1.0
    let a = [1.0, 0.0, 0.0]
    let b = [0.0, 1.0, 0.0]
    XCTAssertEqual(jaccardDistance(a, b), 1.0, accuracy: eps)
  }

  func testJaccardBothZero() {
    // Both all-zero => defined as 0
    XCTAssertEqual(jaccardDistance([0.0, 0.0], [0.0, 0.0]), 0.0, accuracy: eps)
  }

  func testJaccardSymmetry() {
    let a = [1.0, 0.0, 1.0]
    let b = [0.0, 1.0, 1.0]
    assertSymmetry(jaccardDistance, a, b)
  }

  func testJaccardTriangleInequality() {
    let a = [1.0, 0.0, 0.0]
    let b = [1.0, 1.0, 0.0]
    let c = [0.0, 1.0, 1.0]
    assertTriangleInequality(jaccardDistance, a, b, c)
  }

  func testJaccardUnitVectors() {
    // Single-element non-zero vectors: identical direction => 0
    let a = [1.0]
    XCTAssertEqual(jaccardDistance(a, a), 0.0, accuracy: eps)
  }

  // MARK: - Hamming Distance

  func testHammingIdentical() {
    let a = [1.0, 2.0, 3.0, 4.0]
    XCTAssertEqual(hammingDistance(a, a), 0.0, accuracy: eps)
  }

  func testHammingKnownValue() {
    // [1,2,3,4] vs [1,0,3,0]: 2 differ out of 4 => 0.5
    let a = [1.0, 2.0, 3.0, 4.0]
    let b = [1.0, 0.0, 3.0, 0.0]
    XCTAssertEqual(hammingDistance(a, b), 0.5, accuracy: eps)
  }

  func testHammingAllDifferent() {
    let a = [1.0, 2.0, 3.0]
    let b = [4.0, 5.0, 6.0]
    XCTAssertEqual(hammingDistance(a, b), 1.0, accuracy: eps)
  }

  func testHammingSymmetry() {
    let a = [1.0, 2.0, 3.0]
    let b = [1.0, 5.0, 3.0]
    assertSymmetry(hammingDistance, a, b)
  }

  func testHammingTriangleInequality() {
    let a = [1.0, 0.0, 0.0, 0.0]
    let b = [1.0, 1.0, 0.0, 0.0]
    let c = [1.0, 1.0, 1.0, 0.0]
    assertTriangleInequality(hammingDistance, a, b, c)
  }

  func testHammingZeroVectors() {
    XCTAssertEqual(hammingDistance([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0, accuracy: eps)
  }

  func testHammingRangeIsZeroToOne() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0]
    let b = [5.0, 4.0, 3.0, 2.0, 1.0]
    let d = hammingDistance(a, b)
    XCTAssertGreaterThanOrEqual(d, 0.0)
    XCTAssertLessThanOrEqual(d, 1.0)
  }

  // MARK: - Canberra Distance

  func testCanberraIdentical() {
    let a = [1.0, 2.0, 3.0]
    XCTAssertEqual(canberraDistance(a, a), 0.0, accuracy: eps)
  }

  func testCanberraKnownValue() {
    // a=[1,2], b=[3,4]:
    // |1-3|/(1+3) + |2-4|/(2+4) = 2/4 + 2/6 = 0.5 + 0.3333... = 0.8333...
    let a = [1.0, 2.0]
    let b = [3.0, 4.0]
    let expected = 2.0 / 4.0 + 2.0 / 6.0
    XCTAssertEqual(canberraDistance(a, b), expected, accuracy: eps)
  }

  func testCanberraSkipsBothZeroTerms() {
    // Both zero at position 1: only 2 terms contribute
    let a = [1.0, 0.0, 1.0]
    let b = [0.0, 0.0, 0.0]
    // |1-0|/(1+0) + skip + |1-0|/(1+0) = 1 + 1 = 2
    XCTAssertEqual(canberraDistance(a, b), 2.0, accuracy: eps)
  }

  func testCanberraSymmetry() {
    let a = [1.0, 3.0, 5.0]
    let b = [2.0, 4.0, 6.0]
    assertSymmetry(canberraDistance, a, b)
  }

  func testCanberraTriangleInequality() {
    let a = [1.0, 2.0, 3.0]
    let b = [2.0, 3.0, 4.0]
    let c = [4.0, 5.0, 6.0]
    assertTriangleInequality(canberraDistance, a, b, c)
  }

  func testCanberraZeroVector() {
    let a = [0.0, 0.0, 0.0]
    let b = [1.0, 2.0, 3.0]
    // Each term: |0-x|/(0+|x|) = 1 => sum = 3
    XCTAssertEqual(canberraDistance(a, b), 3.0, accuracy: eps)
  }

  // MARK: - Bray-Curtis Distance

  func testBrayCurtisIdentical() {
    let a = [1.0, 2.0, 3.0]
    XCTAssertEqual(braycurtisDistance(a, a), 0.0, accuracy: eps)
  }

  func testBrayCurtisKnownValue() {
    // a=[1,2], b=[3,4]:
    // numerator = |1-3| + |2-4| = 2 + 2 = 4
    // denominator = |1+3| + |2+4| = 4 + 6 = 10
    // distance = 4/10 = 0.4
    let a = [1.0, 2.0]
    let b = [3.0, 4.0]
    XCTAssertEqual(braycurtisDistance(a, b), 0.4, accuracy: eps)
  }

  func testBrayCurtisBothZero() {
    XCTAssertEqual(braycurtisDistance([0.0, 0.0], [0.0, 0.0]), 0.0, accuracy: eps)
  }

  func testBrayCurtisSymmetry() {
    let a = [2.0, 4.0, 6.0]
    let b = [1.0, 3.0, 5.0]
    assertSymmetry(braycurtisDistance, a, b)
  }

  func testBrayCurtisTriangleInequality() {
    let a = [1.0, 0.0, 0.0]
    let b = [0.5, 0.5, 0.0]
    let c = [0.0, 1.0, 0.0]
    assertTriangleInequality(braycurtisDistance, a, b, c)
  }

  func testBrayCurtisRangeIsZeroToOne() {
    let a = [1.0, 0.0, 0.0]
    let b = [0.0, 1.0, 0.0]
    let d = braycurtisDistance(a, b)
    XCTAssertGreaterThanOrEqual(d, 0.0)
    XCTAssertLessThanOrEqual(d, 1.0 + eps)
  }

  func testBrayCurtisUnitVectors() {
    // sum|a-b| = |1-0| + |0-1| = 2; sum|a+b| = |1+0| + |0+1| = 2 => 1.0
    let a = [1.0, 0.0]
    let b = [0.0, 1.0]
    XCTAssertEqual(braycurtisDistance(a, b), 1.0, accuracy: eps)
  }

  // MARK: - DistanceMetric enum integration (cdist / pdist)

  func testCdistJaccard() {
    let XA = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    let XB = [[1.0, 1.0, 0.0]]
    let result = cdist(XA, XB, metric: .jaccard)
    XCTAssertEqual(result.count, 2)
    XCTAssertEqual(result[0].count, 1)
    // Row 0: jaccardDistance([1,0,1],[1,1,0]) = 1 - 1/3 = 2/3
    XCTAssertEqual(result[0][0], jaccardDistance(XA[0], XB[0]), accuracy: eps)
    // Row 1: jaccardDistance([0,1,0],[1,1,0]) = 1 - 1/2 = 0.5
    XCTAssertEqual(result[1][0], jaccardDistance(XA[1], XB[0]), accuracy: eps)
  }

  func testPdistHamming() {
    let X = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    let result = pdist(X, metric: .hamming)
    // 3 points => 3 pairs
    XCTAssertEqual(result.count, 3)
    XCTAssertEqual(result[0], hammingDistance(X[0], X[1]), accuracy: eps)
    XCTAssertEqual(result[1], hammingDistance(X[0], X[2]), accuracy: eps)
    XCTAssertEqual(result[2], hammingDistance(X[1], X[2]), accuracy: eps)
  }

  func testCdistCanberra() {
    let XA = [[1.0, 2.0]]
    let XB = [[3.0, 4.0]]
    let result = cdist(XA, XB, metric: .canberra)
    XCTAssertEqual(result[0][0], canberraDistance([1.0, 2.0], [3.0, 4.0]), accuracy: eps)
  }

  func testCdistBrayCurtis() {
    let XA = [[1.0, 2.0]]
    let XB = [[3.0, 4.0]]
    let result = cdist(XA, XB, metric: .braycurtis)
    XCTAssertEqual(result[0][0], braycurtisDistance([1.0, 2.0], [3.0, 4.0]), accuracy: eps)
  }
}
