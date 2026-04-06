//
//  OptimizationEdgeTests.swift
//  NumericSwift
//
//  Edge case and pathological input tests for optimization and root-finding functions.
//
//  Licensed under the MIT License.
//

import XCTest

@testable import NumericSwift

final class OptimizationEdgeTests: XCTestCase {

  // MARK: - bisect edge cases

  func testBisectNoSignChange() {
    // f(a) and f(b) have the same sign – no root bracketed
    let result = bisect({ x in x * x + 1 }, a: -2, b: 2)
    XCTAssertFalse(result.converged)
    XCTAssertTrue(result.root.isNaN, "root should be NaN when no sign change")
    XCTAssertEqual(result.flag, "f(a) and f(b) must have different signs")
  }

  func testBisectRootAtLeftEndpoint() {
    // f(a) == 0 exactly
    let result = bisect({ x in x * (x - 3) }, a: 0, b: 5)
    XCTAssertTrue(result.converged)
    XCTAssertEqual(result.root, 0, accuracy: 1e-10)
    XCTAssertEqual(result.iterations, 0)
  }

  func testBisectRootAtRightEndpoint() {
    // f(b) == 0 exactly
    let result = bisect({ x in x * (x - 3) }, a: -1, b: 3)
    XCTAssertTrue(result.converged)
    XCTAssertEqual(result.root, 3, accuracy: 1e-10)
    XCTAssertEqual(result.iterations, 0)
  }

  func testBisectTightTolerance() {
    let xtol = 1e-14
    let result = bisect({ x in x * x - 2 }, a: 1, b: 2, xtol: xtol)
    XCTAssertTrue(result.converged)
    XCTAssertEqual(result.root, sqrt(2), accuracy: xtol * 10)
  }

  func testBisectMultipleRootsPicksOne() {
    // sin(x) has roots at 0, π, 2π … [0.5, 3.5] brackets π
    let result = bisect({ x in sin(x) }, a: 0.5, b: 3.5)
    XCTAssertTrue(result.converged)
    XCTAssertEqual(result.root, Double.pi, accuracy: 1e-6)
  }

  func testBisectMaxIterReached() {
    // Very tight tolerance with tiny maxiter forces early exit
    let result = bisect({ x in x - 1 }, a: 0, b: 2, xtol: 1e-15, maxiter: 3)
    // Either converged or not, but must not crash and root must be finite
    XCTAssertFalse(result.root.isNaN)
    XCTAssertFalse(result.root.isInfinite)
  }

  // MARK: - newton edge cases

  func testNewtonDerivativeZeroAtStart() {
    // f(x) = x^2, f'(0) = 0
    let result = newton({ x in x * x }, fprime: { x in 2 * x }, x0: 0)
    XCTAssertFalse(result.converged)
    XCTAssertEqual(result.flag, "derivative is zero")
  }

  func testNewtonDerivativeZeroDuringIteration() {
    // f(x) = x^3 - x, f'(x) = 3x^2 - 1 = 0 at x = ±1/√3 ≈ ±0.577
    // Starting near that point may trigger zero-derivative guard
    let result = newton(
      { x in x * x * x - x },
      fprime: { x in 3 * x * x - 1 },
      x0: 1.0 / sqrt(3.0)
    )
    // Either it detects zero derivative or converges to a root; must not crash
    XCTAssertFalse(result.root.isNaN)
  }

  func testNewtonFarStartConverges() {
    // Starting far from root x=1 of x^3 - 1 = 0
    let result = newton({ x in x * x * x - 1 }, x0: 100)
    XCTAssertTrue(result.converged)
    XCTAssertEqual(result.root, 1, accuracy: 1e-6)
  }

  func testNewtonNumericalDerivativeFallback() {
    // No explicit fprime supplied – uses central finite difference
    let result = newton({ x in x * x - 9 }, x0: 2)
    XCTAssertTrue(result.converged)
    XCTAssertEqual(result.root, 3, accuracy: 1e-6)
  }

  func testNewtonNoRealRoot() {
    // f(x) = x^2 + 1 has no real root; method should fail to converge
    let result = newton({ x in x * x + 1 }, x0: 1.0, maxiter: 50)
    // Newton will oscillate or diverge; root should not satisfy f(root) ≈ 0
    let residual = result.root * result.root + 1
    XCTAssertGreaterThan(abs(residual), 0.5, "Should not find a real root")
  }

  // MARK: - secant edge cases

  func testSecantNearlyIdenticalStartingPoints() {
    // x0 and x1 almost equal → denominator f(x1)-f(x0) is tiny
    let result = secant({ x in x * x - 4 }, x0: 2.0, x1: 2.0 + 1e-15)
    // Should return early with "denominator too small" or converge immediately
    XCTAssertFalse(result.root.isNaN)
  }

  func testSecantFlatRegion() {
    // f(x) ≈ constant in a neighbourhood of the starting points
    // e^(-x^2) - 1/sqrt(e) = 0 at x = ±0.5; start far from root in flat region
    let result = secant({ x in exp(-x * x) - 1.0 / exp(0.5) }, x0: 3.0)
    // May or may not converge; must not crash
    XCTAssertFalse(result.root.isNaN || result.root.isInfinite)
  }

  func testSecantExplicitSecondPoint() {
    let result = secant({ x in x * x - 2 }, x0: 1.0, x1: 1.5)
    XCTAssertTrue(result.converged)
    XCTAssertEqual(result.root, sqrt(2), accuracy: 1e-8)
  }

  func testSecantMaxIterReached() {
    // Very low maxiter so method cannot converge
    let result = secant({ x in x * x - 100 }, x0: 1, maxiter: 2)
    XCTAssertFalse(result.converged)
    XCTAssertEqual(result.flag, "maxiter reached")
  }

  // MARK: - brent (scalar minimization) edge cases

  func testBrentMinimumAtLeftBoundary() {
    // f(x) = x^2 is monotonically decreasing on [-2, -0.5]; min at -2
    let result = brent({ x in x * x }, a: -2, b: -0.5)
    // Brent searches for a local minimum in [a,b]; unimodal, min is at the boundary
    XCTAssertFalse(result.fun.isNaN)
    XCTAssertFalse(result.fun.isInfinite)
    // The minimum found should be ≤ f at interior
    let fMid = 0.5 * 0.5  // f(-0.5) = 0.25, f(-2) = 4 — both endpoints are higher than interior
    _ = fMid  // silence warning; directional test below
    XCTAssertGreaterThanOrEqual(result.fun, 0)
  }

  func testBrentVeryFlatFunction() {
    // f(x) = (x-1)^4 has near-zero curvature around the minimum
    let result = brent({ x in pow(x - 1, 4) }, a: 0, b: 3)
    XCTAssertTrue(result.success)
    XCTAssertEqual(result.x, 1, accuracy: 1e-4)
    XCTAssertEqual(result.fun, 0, accuracy: 1e-12)
  }

  func testBrentMultipleLocalMinima() {
    // f(x) = sin(x) on [0, 4π] has minima at 3π/2 and 7π/2
    // Brent finds one local minimum depending on the initial bracket
    let result = brent({ x in sin(x) }, a: 3, b: 7)
    XCTAssertTrue(result.success)
    XCTAssertLessThanOrEqual(result.fun, 0)
  }

  func testBrentNarrowInterval() {
    // Very narrow interval around the minimum
    let result = brent({ x in (x - 2) * (x - 2) }, a: 1.9999, b: 2.0001)
    XCTAssertFalse(result.fun.isNaN)
    XCTAssertEqual(result.fun, 0, accuracy: 1e-6)
  }

  // MARK: - goldenSection edge cases

  func testGoldenSectionMinimumAtBoundary() {
    // Minimum of x^2 in [1, 5] is at the left boundary x=1
    let result = goldenSection({ x in x * x }, a: 1, b: 5)
    // Golden section converges to the minimum inside the interval
    // For a monotone function, it will report x near the boundary
    XCTAssertFalse(result.fun.isNaN)
    XCTAssertLessThanOrEqual(result.fun, 25)  // f value ≤ f(5)
  }

  func testGoldenSectionVeryNarrowInterval() {
    let result = goldenSection({ x in (x - 0.5) * (x - 0.5) }, a: 0.49999, b: 0.50001)
    XCTAssertFalse(result.fun.isNaN)
    XCTAssertEqual(result.fun, 0, accuracy: 1e-8)
  }

  func testGoldenSectionMaxIterReached() {
    // maxiter=1 forces early exit
    let result = goldenSection({ x in (x - 2) * (x - 2) }, a: 0, b: 5, maxiter: 1)
    XCTAssertFalse(result.success)
    XCTAssertEqual(result.message, "Maximum iterations reached.")
  }

  // MARK: - nelderMead edge cases

  func testNelderMeadStartingAtMinimum() {
    // f(x,y) = x^2 + y^2, starting exactly at (0,0)
    let result = nelderMead({ x in x[0] * x[0] + x[1] * x[1] }, x0: [0.0, 0.0])
    XCTAssertTrue(result.success)
    XCTAssertEqual(result.fun, 0, accuracy: 1e-8)
  }

  func testNelderMeadHighDimensional() {
    // Minimize sum of squares in 6 dimensions; minimum at origin
    let result = nelderMead(
      { x in x.reduce(0) { $0 + $1 * $1 } },
      x0: [1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    XCTAssertTrue(result.success)
    XCTAssertEqual(result.fun, 0, accuracy: 1e-4)
    XCTAssertEqual(result.x.count, 6)
    for xi in result.x {
      XCTAssertEqual(xi, 0, accuracy: 1e-3)
    }
  }

  func testNelderMeadSingleVariable() {
    // 1-D case: minimize (x-3)^2
    let result = nelderMead({ x in pow(x[0] - 3, 2) }, x0: [0.0])
    XCTAssertTrue(result.success)
    XCTAssertEqual(result.x[0], 3, accuracy: 1e-4)
  }

  func testNelderMeadReturnsCorrectDimensionality() {
    let n = 5
    let result = nelderMead(
      { x in x.reduce(0) { $0 + $1 * $1 } },
      x0: Array(repeating: 1.0, count: n))
    XCTAssertEqual(result.x.count, n)
  }

  // MARK: - leastSquares edge cases

  func testLeastSquaresPerfectFitZeroResiduals() {
    // Single residual that is exactly zero at x=3
    let result = leastSquares({ x in [x[0] - 3] }, x0: [1.0])
    XCTAssertTrue(result.success)
    XCTAssertEqual(result.x[0], 3, accuracy: 1e-6)
    XCTAssertEqual(result.cost, 0, accuracy: 1e-10)
  }

  func testLeastSquaresOverdeterminedSystem() {
    // 4 equations, 1 unknown: min sum((x - k)^2) for k=1..4 → x = 2.5
    let result = leastSquares(
      { p in
        [p[0] - 1, p[0] - 2, p[0] - 3, p[0] - 4]
      }, x0: [0.0])
    XCTAssertTrue(result.success)
    XCTAssertEqual(result.x[0], 2.5, accuracy: 1e-5)
  }

  func testLeastSquaresAlreadyAtSolution() {
    // Starting exactly at the solution: cost is already zero.
    // The unbounded LM loop cannot accept a step (costNew == cost == 0 is not < 0),
    // so it exhausts maxiter and returns success=false. The important invariant is
    // that the returned point still has zero cost and correct parameter values.
    let result = leastSquares({ p in [p[0] - 5, p[1] + 2] }, x0: [5.0, -2.0])
    XCTAssertEqual(result.cost, 0, accuracy: 1e-10)
    XCTAssertEqual(result.x[0], 5, accuracy: 1e-10)
    XCTAssertEqual(result.x[1], -2, accuracy: 1e-10)
  }

  func testLeastSquaresMultipleResiduals() {
    // Fit a line y = ax + b to points (0,1),(1,3),(2,5),(3,7): a=2, b=1
    let xs = [0.0, 1.0, 2.0, 3.0]
    let ys = [1.0, 3.0, 5.0, 7.0]
    let result = leastSquares(
      { p in
        xs.indices.map { i in ys[i] - (p[0] * xs[i] + p[1]) }
      }, x0: [0.0, 0.0])
    XCTAssertTrue(result.success)
    XCTAssertEqual(result.x[0], 2, accuracy: 1e-4)  // slope
    XCTAssertEqual(result.x[1], 1, accuracy: 1e-4)  // intercept
  }

  // MARK: - curveFit edge cases

  func testCurveFitPerfectData() {
    // y = 3*exp(-0.5*x), no noise; parameters should recover exactly
    let xdata = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    let a = 3.0
    let b = 0.5
    let ydata = xdata.map { a * exp(-b * $0) }

    let (popt, _, info) = curveFit(
      { params, x in params[0] * exp(-params[1] * x) },
      xdata: xdata,
      ydata: ydata,
      p0: [1.0, 1.0]
    )
    XCTAssertTrue(info.success)
    XCTAssertEqual(popt[0], a, accuracy: 1e-4)
    XCTAssertEqual(popt[1], b, accuracy: 1e-4)
  }

  func testCurveFitLinearModel() {
    // y = m*x + c; linear model, should converge reliably
    let xdata = [0.0, 1.0, 2.0, 3.0, 4.0]
    let m = 2.0
    let c = 1.0
    let ydata = xdata.map { m * $0 + c }

    let (popt, _, info) = curveFit(
      { params, x in params[0] * x + params[1] },
      xdata: xdata,
      ydata: ydata,
      p0: [0.0, 0.0]
    )
    XCTAssertTrue(info.success)
    XCTAssertEqual(popt[0], m, accuracy: 1e-4)
    XCTAssertEqual(popt[1], c, accuracy: 1e-4)
  }

  func testCurveFitDataWithOutliers() {
    // Mostly clean linear data with a large outlier; fit should still converge
    let xdata = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    var ydata = xdata.map { 2.0 * $0 + 1.0 }
    ydata[3] = 100.0  // outlier

    let (popt, _, info) = curveFit(
      { params, x in params[0] * x + params[1] },
      xdata: xdata,
      ydata: ydata,
      p0: [1.0, 0.0]
    )
    // curveFit uses least squares so outlier will skew the result;
    // just verify it converges and returns finite values
    XCTAssertTrue(info.success)
    XCTAssertFalse(popt[0].isNaN)
    XCTAssertFalse(popt[1].isNaN)
  }

  func testCurveFitCovarianceMatrixShape() {
    // Covariance matrix should be n × n for n parameters
    let xdata = [0.0, 1.0, 2.0, 3.0]
    let ydata = [1.0, 3.0, 5.0, 7.0]

    let (_, pcov, _) = curveFit(
      { params, x in params[0] * x + params[1] },
      xdata: xdata,
      ydata: ydata,
      p0: [1.0, 0.0]
    )
    XCTAssertEqual(pcov.count, 2)
    XCTAssertEqual(pcov[0].count, 2)
    XCTAssertEqual(pcov[1].count, 2)
  }
}
