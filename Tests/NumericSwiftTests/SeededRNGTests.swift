import XCTest

@testable import NumericSwift

final class SeededRNGTests: XCTestCase {

  func testReproducibility() {
    var rng1 = SeededRandomNumberGenerator(seed: 42)
    var rng2 = SeededRandomNumberGenerator(seed: 42)

    let values1 = (0..<10).map { _ in Double.random(in: 0..<1, using: &rng1) }
    let values2 = (0..<10).map { _ in Double.random(in: 0..<1, using: &rng2) }

    for i in 0..<10 {
      XCTAssertEqual(values1[i], values2[i], "Mismatch at index \(i)")
    }
  }

  func testDifferentSeeds() {
    var rng1 = SeededRandomNumberGenerator(seed: 1)
    var rng2 = SeededRandomNumberGenerator(seed: 2)

    let v1 = Double.random(in: 0..<1, using: &rng1)
    let v2 = Double.random(in: 0..<1, using: &rng2)
    XCTAssertNotEqual(v1, v2)
  }

  func testRandomNormalSeeded() {
    var rng1 = SeededRandomNumberGenerator(seed: 123)
    var rng2 = SeededRandomNumberGenerator(seed: 123)

    let n1 = randomNormal(using: &rng1)
    let n2 = randomNormal(using: &rng2)
    XCTAssertEqual(n1, n2)
  }

  func testRandomNormalArraySeeded() {
    var rng1 = SeededRandomNumberGenerator(seed: 99)
    var rng2 = SeededRandomNumberGenerator(seed: 99)

    let arr1 = randomNormal(5, using: &rng1)
    let arr2 = randomNormal(5, using: &rng2)
    XCTAssertEqual(arr1.count, 5)
    for i in 0..<5 {
      XCTAssertEqual(arr1[i], arr2[i])
    }
  }

  func testUniformRange() {
    var rng = SeededRandomNumberGenerator(seed: 7)
    for _ in 0..<100 {
      let v = Double.random(in: 0..<1, using: &rng)
      XCTAssertGreaterThanOrEqual(v, 0.0)
      XCTAssertLessThan(v, 1.0)
    }
  }

  func testNormalDistribution() {
    // Generate many samples and check mean/std are roughly correct
    var rng = SeededRandomNumberGenerator(seed: 42)
    let samples = randomNormal(10000, using: &rng)
    let m = samples.reduce(0, +) / Double(samples.count)
    let v = samples.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(samples.count)
    XCTAssertEqual(m, 0.0, accuracy: 0.05)
    XCTAssertEqual(v, 1.0, accuracy: 0.1)
  }
}
