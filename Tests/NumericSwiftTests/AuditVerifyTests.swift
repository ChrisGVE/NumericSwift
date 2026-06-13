//
//  AuditVerifyTests.swift
//  NumericSwift
//
//  Verify-then-fix tests for the two findings the 2026-06-10 audit flagged as
//  UNVERIFIED: notAKnot spline coefficients (M18) and arimaForecast d ≥ 2 (M22).
//
//  Licensed under the Apache License, Version 2.0.
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

  // MARK: - M26: fallback tokenizer must support the `%` (modulo) operator

  /// The pure-Swift fallback parser previously rejected `%`. MathLex supports it
  /// (`BinaryOp.mod`); the fallback must match so expressions parse identically
  /// whether or not the Rust backend is compiled in.
  func testFallbackModuloOperator() throws {
    XCTAssertEqual(try MathExpr.eval("7 % 3"), 1.0, accuracy: 1e-12)
    XCTAssertEqual(try MathExpr.eval("10 % 4"), 2.0, accuracy: 1e-12)
    // Same precedence as * and /: 2 + 7 % 3 == 2 + (7 % 3) == 3
    XCTAssertEqual(try MathExpr.eval("2 + 7 % 3"), 3.0, accuracy: 1e-12)
  }

  // MARK: - M27: unary minus must emit `.unary(.neg, x)`, not `.binary(.sub, 0, x)`

  /// The fallback parser must produce the same AST shape as the MathLex backend
  /// for negation: a `.unary(.neg, …)` node rather than `0 - x`.
  func testFallbackUnaryNegShape() throws {
    let ast = try MathExpr.parse("-x")
    guard case .unary(let op, let operand) = ast else {
      return XCTFail("expected .unary node for `-x`, got \(ast)")
    }
    XCTAssertEqual(op, .neg)
    guard case .variable(let name) = operand else {
      return XCTFail("expected .variable operand, got \(operand)")
    }
    XCTAssertEqual(name, "x")
  }

  /// Unary minus binds below `^`, so `-x^2` is `-(x^2)`, not `(-x)^2`.
  func testFallbackUnaryNegBindsBelowPow() throws {
    XCTAssertEqual(try MathExpr.eval("-x^2", variables: ["x": 3.0]), -9.0, accuracy: 1e-12)
    XCTAssertEqual(try MathExpr.eval("-3^2"), -9.0, accuracy: 1e-12)
    // Nested negation and binary minus against a unary operand.
    XCTAssertEqual(try MathExpr.eval("2 - -3"), 5.0, accuracy: 1e-12)
    XCTAssertEqual(try MathExpr.eval("--5"), 5.0, accuracy: 1e-12)
  }

  // MARK: - M16: curveFit rank-deficient covariance

  /// Two perfectly collinear parameters make J^T J singular, so the covariance
  /// is unestimable. `curveFit` must signal this the SciPy way — a pcov filled
  /// with infinity — not return garbage finite numbers.
  func testCurveFitRankDeficientCovarianceIsInfinite() {
    // f(a, b, x) = (a + b) * x  →  ∂f/∂a == ∂f/∂b == x  →  rank-deficient J.
    let model: ([Double], Double) -> Double = { p, x in (p[0] + p[1]) * x }
    let xdata = [0.0, 1.0, 2.0, 3.0, 4.0]
    let ydata = xdata.map { 2.0 * $0 }   // true (a + b) = 2
    let (_, pcov, _) = curveFit(model, xdata: xdata, ydata: ydata, p0: [0.5, 0.5])
    for row in pcov {
      for v in row {
        XCTAssertTrue(v.isInfinite, "rank-deficient pcov entry must be infinite, got \(v)")
      }
    }
  }

  // MARK: - M20: no-intercept dfModel + uncentered R²

  /// statsmodels constant-column detection: a constant nonzero column is the
  /// intercept; an ordinary or all-zero column is not.
  func testHasConstantColumnDetection() {
    XCTAssertTrue(hasConstantColumn([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]]))  // col 0 constant
    XCTAssertFalse(hasConstantColumn([[1.0], [2.0], [3.0]]))                // no constant
    XCTAssertFalse(hasConstantColumn([[0.0, 1.0], [0.0, 2.0]]))             // all-zero ≠ intercept
  }

  /// A through-origin design (no constant column) must count every column in
  /// dfModel and report R² against the UNCENTERED total sum of squares (Σy²),
  /// matching statsmodels OLS without a constant.
  func testOlsNoInterceptDfAndUncenteredR2() {
    let X = [[1.0], [2.0], [3.0]]
    let y = [1.0, 2.0, 4.0]
    guard let r = ols(y, X) else { return XCTFail("ols returned nil") }
    XCTAssertEqual(r.dfModel, 1)   // k = 1, kConstant = 0
    XCTAssertEqual(r.dfResid, 2)
    // slope = 17/14; ssr = 0.3571429; Σy² = 21 ⇒ R² = 1 − ssr/Σy².
    XCTAssertEqual(r.rsquared, 0.9829932, accuracy: 1e-5)
  }

  /// With an explicit constant column the intercept is excluded from dfModel.
  func testOlsWithInterceptDf() {
    let X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    let y = [2.0, 4.0, 6.0, 8.0]
    guard let r = ols(y, X) else { return XCTFail("ols returned nil") }
    XCTAssertEqual(r.dfModel, 1)   // k = 2, kConstant = 1
    XCTAssertEqual(r.dfResid, 2)
  }

  /// GLM df_model = rank − 1 only when a constant is present (statsmodels rule);
  /// a no-constant design counts every column.
  func testGlmNoInterceptDfModel() {
    let X = [[1.0], [2.0], [3.0], [4.0]]
    let y = [1.0, 2.0, 3.0, 4.0]
    guard let r = glm(y, X, family: .gaussian) else { return XCTFail("glm returned nil") }
    XCTAssertEqual(r.dfModel, 1)   // k = 1, no constant ⇒ k − 0
    XCTAssertEqual(r.dfResid, 3)
  }
}
