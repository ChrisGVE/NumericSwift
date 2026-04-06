import XCTest

@testable import NumericSwift

final class RegressionHCTests: XCTestCase {

  // Simple dataset: y = 2 + 3*x with mild heteroscedasticity
  let x: [Double] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let y: [Double] = [5.1, 8.3, 10.9, 14.2, 17.0, 20.1, 23.4, 26.0, 29.5, 32.1]

  private func designMatrix(_ xs: [Double]) -> [[Double]] {
    xs.map { [1.0, $0] }
  }

  func testHC4IsComputed() throws {
    guard let result = ols(y, designMatrix(x)) else {
      XCTFail("OLS failed to fit")
      return
    }
    XCTAssertEqual(result.bseHC4.count, 2, "HC4 should have one SE per parameter")
    XCTAssertFalse(result.bseHC4.allSatisfy { $0 == 0.0 }, "HC4 should not be all zeros")
  }

  func testHC5IsComputed() throws {
    guard let result = ols(y, designMatrix(x)) else {
      XCTFail("OLS failed to fit")
      return
    }
    XCTAssertEqual(result.bseHC5.count, 2, "HC5 should have one SE per parameter")
    XCTAssertFalse(result.bseHC5.allSatisfy { $0 == 0.0 }, "HC5 should not be all zeros")
  }

  func testHC4DiffersFromHC3() throws {
    guard let result = ols(y, designMatrix(x)) else {
      XCTFail("OLS failed to fit")
      return
    }
    // HC4 uses a different omega than HC3 so SEs should differ
    let allEqual = zip(result.bseHC4, result.bseHC3).allSatisfy { abs($0 - $1) < 1e-12 }
    XCTAssertFalse(allEqual, "HC4 should differ from HC3")
  }

  func testHC5DiffersFromHC3() throws {
    guard let result = ols(y, designMatrix(x)) else {
      XCTFail("OLS failed to fit")
      return
    }
    let allEqual = zip(result.bseHC5, result.bseHC3).allSatisfy { abs($0 - $1) < 1e-12 }
    XCTAssertFalse(allEqual, "HC5 should differ from HC3")
  }

  func testHC4AndHC5ArePositive() throws {
    guard let result = ols(y, designMatrix(x)) else {
      XCTFail("OLS failed to fit")
      return
    }
    XCTAssertTrue(result.bseHC4.allSatisfy { $0 > 0 }, "HC4 SEs should be positive")
    XCTAssertTrue(result.bseHC5.allSatisfy { $0 > 0 }, "HC5 SEs should be positive")
  }

  func testHomoscedasticDataHCEstimatorsAreSimilar() throws {
    // y = 1 + 2*x + small noise — OLS SE and all HC SEs should be close
    let xs = stride(from: 1.0, through: 20.0, by: 1.0).map { $0 }
    let ys = xs.enumerated().map { i, xi -> Double in
      // Fixed small perturbations (no random seed needed)
      let noise = [
        0.05, -0.03, 0.02, -0.04, 0.06, -0.01, 0.03, -0.05,
        0.02, 0.04, -0.02, 0.01, -0.06, 0.03, -0.01, 0.05,
        0.02, -0.03, 0.04, -0.02,
      ][i]
      return 1.0 + 2.0 * xi + noise
    }
    let X = xs.map { [1.0, $0] }
    guard let result = ols(ys, X) else {
      XCTFail("OLS failed to fit")
      return
    }
    // All HC estimators should be within a factor of 3 of the OLS SE for homoscedastic data
    for j in 0..<2 {
      let olsSE = result.bse[j]
      XCTAssertLessThan(result.bseHC4[j], olsSE * 3, "HC4[\(j)] too large vs OLS SE")
      XCTAssertLessThan(result.bseHC5[j], olsSE * 3, "HC5[\(j)] too large vs OLS SE")
      XCTAssertGreaterThan(result.bseHC4[j], olsSE / 3, "HC4[\(j)] too small vs OLS SE")
      XCTAssertGreaterThan(result.bseHC5[j], olsSE / 3, "HC5[\(j)] too small vs OLS SE")
    }
  }

  func testSmokeTestSinglePredictor() throws {
    // Minimal regression: 5 observations
    let xs5: [Double] = [1, 2, 3, 4, 5]
    let ys5: [Double] = [2.1, 4.2, 5.9, 8.1, 10.0]
    let X5 = xs5.map { [1.0, $0] }
    guard let result = ols(ys5, X5) else {
      XCTFail("OLS failed")
      return
    }
    XCTAssertEqual(result.bseHC4.count, 2)
    XCTAssertEqual(result.bseHC5.count, 2)
    XCTAssertFalse(result.bseHC4.contains(where: { $0.isNaN }), "HC4 should not contain NaN")
    XCTAssertFalse(result.bseHC5.contains(where: { $0.isNaN }), "HC5 should not contain NaN")
  }
}
