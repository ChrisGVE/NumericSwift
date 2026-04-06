import Darwin
import XCTest

@testable import NumericSwift

final class IntegrationCumulativeTests: XCTestCase {

  // MARK: - cumulativeTrapezoid (uniform)

  func testCumulativeTrapezoidConstant() {
    // Integral of constant 2 from 0..4 with dx=1: [2, 4, 6, 8]
    let y = [2.0, 2.0, 2.0, 2.0, 2.0]
    let result = cumulativeTrapezoid(y, dx: 1.0)
    XCTAssertEqual(result.count, 4)
    XCTAssertEqual(result[0], 2.0, accuracy: 1e-14)
    XCTAssertEqual(result[1], 4.0, accuracy: 1e-14)
    XCTAssertEqual(result[2], 6.0, accuracy: 1e-14)
    XCTAssertEqual(result[3], 8.0, accuracy: 1e-14)
  }

  func testCumulativeTrapezoidLinear() {
    // y = x on [0, 4] with dx=1: integral from 0 to k is k^2/2
    let y = [0.0, 1.0, 2.0, 3.0, 4.0]
    let result = cumulativeTrapezoid(y, dx: 1.0)
    XCTAssertEqual(result.count, 4)
    XCTAssertEqual(result[0], 0.5, accuracy: 1e-14)  // 0.5*(0+1)*1
    XCTAssertEqual(result[1], 2.0, accuracy: 1e-14)  // 0.5 + 0.5*(1+2)*1
    XCTAssertEqual(result[2], 4.5, accuracy: 1e-14)
    XCTAssertEqual(result[3], 8.0, accuracy: 1e-14)
  }

  func testCumulativeTrapezoidCustomDx() {
    let y = [0.0, 1.0, 4.0]
    let result = cumulativeTrapezoid(y, dx: 0.5)
    XCTAssertEqual(result.count, 2)
    XCTAssertEqual(result[0], 0.25, accuracy: 1e-14)  // 0.5*(0+1)*0.5
    XCTAssertEqual(result[1], 1.5, accuracy: 1e-14)  // 0.25 + 0.5*(1+4)*0.5
  }

  func testCumulativeTrapezoidLastEqualsTotal() {
    // Last element should equal trapz total
    let y = [1.0, 3.0, 2.0, 5.0, 4.0]
    let cumResult = cumulativeTrapezoid(y, dx: 1.0)
    let totalResult = trapz(y, dx: 1.0)
    XCTAssertEqual(cumResult.last!, totalResult, accuracy: 1e-14)
  }

  func testCumulativeTrapezoidTooFewPoints() {
    XCTAssertTrue(cumulativeTrapezoid([1.0]).isEmpty)
    XCTAssertTrue(cumulativeTrapezoid([Double]()).isEmpty)
  }

  // MARK: - cumulativeTrapezoid (non-uniform)

  func testCumulativeTrapezoidNonUniform() {
    let x = [0.0, 1.0, 3.0, 6.0]
    let y = [0.0, 2.0, 2.0, 2.0]
    let result = cumulativeTrapezoid(y, x: x)
    XCTAssertEqual(result.count, 3)
    XCTAssertEqual(result[0], 1.0, accuracy: 1e-14)  // 0.5*(0+2)*1
    XCTAssertEqual(result[1], 5.0, accuracy: 1e-14)  // 1 + 0.5*(2+2)*2
    XCTAssertEqual(result[2], 11.0, accuracy: 1e-14)  // 5 + 0.5*(2+2)*3
  }

  func testCumulativeTrapezoidNonUniformLastEqualsTotal() {
    let x = [0.0, 0.5, 1.5, 2.0, 3.0]
    let y = [1.0, 2.0, 3.0, 2.0, 1.0]
    let cumResult = cumulativeTrapezoid(y, x: x)
    let totalResult = trapz(y, x: x)
    XCTAssertEqual(cumResult.last!, totalResult, accuracy: 1e-14)
  }

  // MARK: - cumulativeSimpson

  func testCumulativeSimpsonConstant() {
    let y = [3.0, 3.0, 3.0, 3.0, 3.0]
    let result = cumulativeSimpson(y, dx: 1.0)
    XCTAssertEqual(result.count, 4)
    // Constant function: integral = 3 * x
    XCTAssertEqual(result[0], 3.0, accuracy: 1e-14)
    XCTAssertEqual(result[1], 6.0, accuracy: 1e-14)
    XCTAssertEqual(result[2], 9.0, accuracy: 1e-14)
    XCTAssertEqual(result[3], 12.0, accuracy: 1e-14)
  }

  func testCumulativeSimpsonQuadratic() {
    // y = x^2 on [0, 4] with dx=1: Simpson should be exact for quadratics
    // Integral of x^2 from 0 to k = k^3/3
    let y = [0.0, 1.0, 4.0, 9.0, 16.0]
    let result = cumulativeSimpson(y, dx: 1.0)
    XCTAssertEqual(result.count, 4)
    // Last value should be close to 64/3 ≈ 21.333...
    XCTAssertEqual(result[3], 64.0 / 3.0, accuracy: 0.5)
  }

  func testCumulativeSimpsonTooFewPoints() {
    XCTAssertTrue(cumulativeSimpson([1.0]).isEmpty)
    XCTAssertTrue(cumulativeSimpson([Double]()).isEmpty)
  }

  func testCumulativeSimpsonTwoPoints() {
    // With only 2 points, falls back to trapezoidal for first interval
    let y = [0.0, 4.0]
    let result = cumulativeSimpson(y, dx: 1.0)
    XCTAssertEqual(result.count, 1)
    XCTAssertEqual(result[0], 2.0, accuracy: 1e-14)  // 0.5*(0+4)*1
  }
}
