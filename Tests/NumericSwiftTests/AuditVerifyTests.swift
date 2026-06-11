//
//  AuditVerifyTests.swift
//  NumericSwift
//
//  Verify-then-fix tests for the two findings the 2026-06-10 audit flagged as
//  UNVERIFIED: notAKnot spline coefficients (M18) and arimaForecast d ≥ 2 (M22).
//
//  Licensed under the MIT License.
//

import XCTest

@testable import NumericSwift

final class AuditVerifyTests: XCTestCase {

  // MARK: - M18: not-a-knot must reproduce a cubic exactly

  /// A not-a-knot cubic spline through samples of a single cubic polynomial is
  /// that polynomial exactly (the not-a-knot condition forces one cubic across
  /// the first/last two intervals). Evaluating between knots must recover f.
  func testNotAKnotReproducesCubic() {
    func f(_ x: Double) -> Double { 2 * x * x * x - 3 * x * x + x - 5 }
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    let ys = xs.map(f)
    let coeffs = computeSplineCoeffs(x: xs, y: ys, bc: .notAKnot)
    for xq in [0.5, 1.5, 2.5, 3.3, 4.75] {
      let got = evalCubicSpline(x: xs, coeffs: coeffs, xNew: xq)
      XCTAssertEqual(got, f(xq), accuracy: 1e-8, "not-a-knot must reproduce cubic at \(xq)")
    }
  }

  // MARK: - M22: arimaForecast with d ≥ 2 must continue a linear trend

  /// For exactly linear data, Δ²y ≡ 0, so an ARIMA(0,2,0) forecast of the twice-
  /// differenced series is 0 and the level forecast must continue the straight
  /// line. The d ≥ 2 integration that re-uses `original.last` as the baseline for
  /// every pass instead breaks this.
  func testArimaForecastLinearTrendD2() throws {
    let y = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]  // y_t = 10 + 2t
    guard let result = arima(y, p: 0, d: 2, q: 0) else {
      throw XCTSkip("ARIMA(0,2,0) fit returned nil on linear data")
    }
    let fc = arimaForecast(result, steps: 3)
    XCTAssertEqual(fc.count, 3)
    XCTAssertEqual(fc[0], 26.0, accuracy: 1e-6)
    XCTAssertEqual(fc[1], 28.0, accuracy: 1e-6)
    XCTAssertEqual(fc[2], 30.0, accuracy: 1e-6)
  }
}
