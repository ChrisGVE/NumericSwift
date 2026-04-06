//
//  ARIMAForecastTests.swift
//  NumericSwift
//
//  Tests for ARIMA forecasting with confidence intervals.
//

import XCTest

@testable import NumericSwift

final class ARIMAForecastTests: XCTestCase {

  // MARK: - Helpers

  /// Fit a simple AR(1) model on a linear trend series.
  private func fitSimpleARIMA() -> ARIMAResult? {
    let y = (0..<30).map { Double($0) + 0.1 * Double($0 % 3) }
    return arima(y, p: 1, d: 1, q: 0)
  }

  // MARK: - Symmetry

  /// Intervals must be symmetric around the point forecast.
  func testIntervalsSymmetricAroundForecast() throws {
    let result = try XCTUnwrap(fitSimpleARIMA())
    let out = arimaForecastWithIntervals(result, steps: 5)

    XCTAssertEqual(out.forecast.count, 5)
    XCTAssertEqual(out.lower.count, 5)
    XCTAssertEqual(out.upper.count, 5)

    for h in 0..<5 {
      let halfWidth = (out.upper[h] - out.lower[h]) / 2.0
      let center = (out.upper[h] + out.lower[h]) / 2.0
      XCTAssertEqual(
        center, out.forecast[h], accuracy: 1e-10,
        "Interval center should equal point forecast at horizon \(h + 1)")
      XCTAssertGreaterThan(
        halfWidth, 0,
        "Interval half-width should be positive at horizon \(h + 1)")
    }
  }

  // MARK: - Growing width

  /// Interval width must increase monotonically with horizon.
  func testIntervalWidthGrowsWithHorizon() throws {
    let result = try XCTUnwrap(fitSimpleARIMA())
    let out = arimaForecastWithIntervals(result, steps: 6)

    for h in 1..<6 {
      let prevWidth = out.upper[h - 1] - out.lower[h - 1]
      let currWidth = out.upper[h] - out.lower[h]
      XCTAssertGreaterThan(
        currWidth, prevWidth,
        "Width at h=\(h + 1) should exceed width at h=\(h)")
    }
  }

  // MARK: - Point forecast consistency

  /// Point forecasts must match standalone arimaForecast output.
  func testPointForecastMatchesArimaForecast() throws {
    let result = try XCTUnwrap(fitSimpleARIMA())
    let steps = 8
    let standalone = arimaForecast(result, steps: steps)
    let withIntervals = arimaForecastWithIntervals(result, steps: steps)

    XCTAssertEqual(withIntervals.forecast.count, steps)
    for h in 0..<steps {
      XCTAssertEqual(
        withIntervals.forecast[h], standalone[h], accuracy: 1e-12,
        "Point forecast mismatch at horizon \(h + 1)")
    }
  }

  // MARK: - Higher confidence → wider intervals

  /// A higher confidence level must produce wider intervals.
  func testHigherConfidenceLevelProducesWiderIntervals() throws {
    let result = try XCTUnwrap(fitSimpleARIMA())

    let out90 = arimaForecastWithIntervals(result, steps: 4, confidenceLevel: 0.90)
    let out99 = arimaForecastWithIntervals(result, steps: 4, confidenceLevel: 0.99)

    XCTAssertEqual(out90.confidenceLevel, 0.90, accuracy: 1e-12)
    XCTAssertEqual(out99.confidenceLevel, 0.99, accuracy: 1e-12)

    for h in 0..<4 {
      let width90 = out90.upper[h] - out90.lower[h]
      let width99 = out99.upper[h] - out99.lower[h]
      XCTAssertGreaterThan(
        width99, width90,
        "99% interval should be wider than 90% at horizon \(h + 1)")
    }
  }

  // MARK: - Zero steps edge case

  /// Zero steps must return empty arrays without crashing.
  func testZeroStepsReturnsEmpty() throws {
    let result = try XCTUnwrap(fitSimpleARIMA())
    let out = arimaForecastWithIntervals(result, steps: 0)

    XCTAssertTrue(out.forecast.isEmpty)
    XCTAssertTrue(out.lower.isEmpty)
    XCTAssertTrue(out.upper.isEmpty)
    XCTAssertEqual(out.confidenceLevel, 0.95, accuracy: 1e-12)
  }
}
