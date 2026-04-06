//
//  BFGSTests.swift
//  NumericSwift
//
//  Tests for the BFGS quasi-Newton optimizer.
//
//  Licensed under the MIT License.
//

import XCTest

@testable import NumericSwift

final class BFGSTests: XCTestCase {

  // MARK: - Rosenbrock

  /// Classic banana function: minimum at (1, 1) with f = 0.
  func testRosenbrock() {
    func rosenbrock(_ x: [Double]) -> Double {
      let a = 1.0 - x[0]
      let b = x[1] - x[0] * x[0]
      return a * a + 100 * b * b
    }

    let result = bfgs(rosenbrock, x0: [-1.2, 1.0])

    XCTAssertTrue(result.success, "BFGS should converge on Rosenbrock")
    XCTAssertEqual(result.x[0], 1.0, accuracy: 1e-4)
    XCTAssertEqual(result.x[1], 1.0, accuracy: 1e-4)
    XCTAssertEqual(result.fun, 0.0, accuracy: 1e-8)
  }

  // MARK: - Simple quadratic

  /// f(x, y) = x² + y²: minimum at (0, 0) with f = 0.
  func testSimpleQuadratic() {
    func quadratic(_ x: [Double]) -> Double {
      x.reduce(0) { $0 + $1 * $1 }
    }

    let result = bfgs(quadratic, x0: [3.0, -4.0])

    XCTAssertTrue(result.success)
    XCTAssertEqual(result.x[0], 0.0, accuracy: 1e-6)
    XCTAssertEqual(result.x[1], 0.0, accuracy: 1e-6)
    XCTAssertEqual(result.fun, 0.0, accuracy: 1e-10)
  }

  // MARK: - Analytic vs finite-difference gradient

  /// Providing the exact gradient should converge to the same point as
  /// finite differences, and should generally use fewer function evaluations.
  func testAnalyticGradientMatchesFiniteDiff() {
    func f(_ x: [Double]) -> Double {
      (x[0] - 2) * (x[0] - 2) + (x[1] + 3) * (x[1] + 3)
    }
    func g(_ x: [Double]) -> [Double] {
      [2 * (x[0] - 2), 2 * (x[1] + 3)]
    }

    let rFD = bfgs(f, x0: [0.0, 0.0])
    let rAN = bfgs(f, x0: [0.0, 0.0], grad: g)

    XCTAssertTrue(rFD.success)
    XCTAssertTrue(rAN.success)
    XCTAssertEqual(rFD.x[0], rAN.x[0], accuracy: 1e-5)
    XCTAssertEqual(rFD.x[1], rAN.x[1], accuracy: 1e-5)
    // Analytic grad needs fewer or equal function evaluations
    XCTAssertLessThanOrEqual(rAN.nfev, rFD.nfev)
  }

  // MARK: - High-dimensional quadratic

  /// 5-D bowl: f(x) = Σ xᵢ², minimum at origin.
  func testHighDimensionalQuadratic() {
    func f(_ x: [Double]) -> Double { x.reduce(0) { $0 + $1 * $1 } }

    let x0 = [1.0, -2.0, 3.0, -4.0, 5.0]
    let result = bfgs(f, x0: x0)

    XCTAssertTrue(result.success)
    for xi in result.x {
      XCTAssertEqual(xi, 0.0, accuracy: 1e-5)
    }
    XCTAssertEqual(result.fun, 0.0, accuracy: 1e-9)
  }

  // MARK: - Success flag

  /// A very tight gtol that forces max-iter should yield success = false.
  func testSuccessFlagFalseOnMaxIter() {
    // Himmelblau's function has no simple analytic starting direction;
    // force non-convergence by limiting iterations to 1 and using a
    // very tight tolerance.
    func f(_ x: [Double]) -> Double {
      let a = x[0] * x[0] + x[1] - 11
      let b = x[0] + x[1] * x[1] - 7
      return a * a + b * b
    }

    let result = bfgs(f, x0: [0.0, 0.0], maxIter: 1, gtol: 1e-20)
    XCTAssertFalse(result.success)
    XCTAssertEqual(result.nit, 1)
  }

  // MARK: - Diagnostic counters

  /// nfev and njev must be positive after a run.
  func testDiagnosticCounters() {
    func f(_ x: [Double]) -> Double { x[0] * x[0] }

    let result = bfgs(f, x0: [5.0])

    XCTAssertGreaterThan(result.nfev, 0)
    XCTAssertGreaterThan(result.njev, 0)
    XCTAssertGreaterThan(result.nit, 0)
  }
}
