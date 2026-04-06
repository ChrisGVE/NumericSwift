import XCTest

@testable import NumericSwift

final class InterpolationVectorizedTests: XCTestCase {

  func testVectorizedLinearMatchesScalar() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0]
    let y = [0.0, 1.0, 4.0, 9.0, 16.0]
    let queries = [0.5, 1.5, 2.5, 3.5]
    let vectorResult = interp1d(x: x, y: y, xNew: queries, kind: .linear)
    for (i, q) in queries.enumerated() {
      let scalarResult = interp1d(x: x, y: y, xNew: q, kind: .linear)
      XCTAssertEqual(vectorResult[i], scalarResult, accuracy: 1e-15, "at xNew=\(q)")
    }
  }

  func testVectorizedLinearCorrectValues() {
    let x = [0.0, 1.0, 2.0]
    let y = [0.0, 2.0, 6.0]
    let queries = [0.0, 0.5, 1.0, 1.5, 2.0]
    let result = interp1d(x: x, y: y, xNew: queries, kind: .linear)
    XCTAssertEqual(result[0], 0.0, accuracy: 1e-14)
    XCTAssertEqual(result[1], 1.0, accuracy: 1e-14)
    XCTAssertEqual(result[2], 2.0, accuracy: 1e-14)
    XCTAssertEqual(result[3], 4.0, accuracy: 1e-14)
    XCTAssertEqual(result[4], 6.0, accuracy: 1e-14)
  }

  func testVectorizedOutOfBoundsFillValue() {
    let x = [1.0, 2.0, 3.0]
    let y = [10.0, 20.0, 30.0]
    let queries = [0.0, 1.5, 4.0]
    let result = interp1d(x: x, y: y, xNew: queries, kind: .linear, fillValue: -999.0)
    XCTAssertEqual(result[0], -999.0, accuracy: 1e-14)
    XCTAssertEqual(result[1], 15.0, accuracy: 1e-14)
    XCTAssertEqual(result[2], -999.0, accuracy: 1e-14)
  }

  func testVectorizedEmptyQuery() {
    let x = [0.0, 1.0]
    let y = [0.0, 1.0]
    let result = interp1d(x: x, y: y, xNew: [Double](), kind: .linear)
    XCTAssertTrue(result.isEmpty)
  }

  func testVectorizedCubic() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0]
    let y = [0.0, 1.0, 4.0, 9.0, 16.0]
    let coeffs = computeSplineCoeffs(x: x, y: y)
    let queries = [0.5, 1.5, 2.5, 3.5]
    let vectorResult = interp1d(x: x, y: y, xNew: queries, kind: .cubic, coeffs: coeffs)
    for (i, q) in queries.enumerated() {
      let scalarResult = interp1d(x: x, y: y, xNew: q, kind: .cubic, coeffs: coeffs)
      XCTAssertEqual(vectorResult[i], scalarResult, accuracy: 1e-15, "cubic at xNew=\(q)")
    }
  }

  func testVectorizedNearest() {
    let x = [0.0, 1.0, 2.0]
    let y = [10.0, 20.0, 30.0]
    let queries = [0.3, 0.7, 1.4, 1.6]
    let result = interp1d(x: x, y: y, xNew: queries, kind: .nearest)
    XCTAssertEqual(result[0], 10.0, accuracy: 1e-14)  // closer to 0
    XCTAssertEqual(result[1], 20.0, accuracy: 1e-14)  // closer to 1
    XCTAssertEqual(result[2], 20.0, accuracy: 1e-14)  // closer to 1
    XCTAssertEqual(result[3], 30.0, accuracy: 1e-14)  // closer to 2
  }
}
