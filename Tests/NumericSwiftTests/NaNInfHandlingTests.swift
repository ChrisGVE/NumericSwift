//
//  NaNInfHandlingTests.swift
//  NumericSwift
//
//  Tests verifying NaN/Inf behavior across all modules.
//  Goal: document current behavior and ensure no crashes.
//
//  Licensed under the MIT License.
//

import XCTest

@testable import NumericSwift

// MARK: - NaN/Inf Handling Tests

final class NaNInfHandlingTests: XCTestCase {

  // MARK: - Statistics: NaN inputs

  func testMeanWithNaN() {
    // NaN propagates through arithmetic sum
    let result = mean([1.0, .nan, 3.0])
    XCTAssertTrue(result.isNaN, "mean with NaN element should propagate NaN")
  }

  func testMeanWithInf() {
    // Inf propagates through arithmetic sum
    let result = mean([1.0, .infinity, 3.0])
    XCTAssertTrue(result.isInfinite, "mean with Inf element should propagate Inf")
  }

  func testMedianWithNaN() {
    // Swift's sort is not stable for NaN (NaN < x and x < NaN are both false).
    // In practice NaN ends up in a position that makes the median NaN.
    // Document actual behavior: NaN propagates through the sorted result.
    let result = median([1.0, .nan, 3.0])
    XCTAssertTrue(result.isNaN, "median with NaN element propagates NaN via sort ordering")
  }

  func testMedianWithInf() {
    let result = median([1.0, .infinity, 3.0])
    // Sorted: [1, 3, Inf]; median → 3.0
    XCTAssertFalse(result.isNaN)
    XCTAssertEqual(result, 3.0, accuracy: 1e-12)
  }

  func testVarianceWithNaN() {
    // mean becomes NaN → squared diffs become NaN → variance is NaN
    let result = variance([1.0, .nan, 3.0])
    XCTAssertTrue(result.isNaN, "variance with NaN element should propagate NaN")
  }

  func testVarianceWithInf() {
    let result = variance([1.0, .infinity, 3.0])
    XCTAssertTrue(
      result.isNaN || result.isInfinite,
      "variance with Inf element should produce NaN or Inf")
  }

  func testStddevWithNaN() {
    let result = stddev([1.0, .nan, 3.0])
    XCTAssertTrue(result.isNaN, "stddev with NaN element should propagate NaN")
  }

  func testStddevWithInf() {
    let result = stddev([1.0, .infinity, 3.0])
    XCTAssertTrue(
      result.isNaN || result.isInfinite,
      "stddev with Inf element should produce NaN or Inf")
  }

  func testPercentileWithNaN() {
    // NaN sort position is undefined; the NaN propagates into the interpolated result.
    // Document actual behavior: no crash, but result is NaN.
    let result = percentile([1.0, .nan, 3.0], 50)
    XCTAssertTrue(
      result.isNaN, "percentile with NaN element propagates NaN via sort ordering")
  }

  func testPercentileWithInf() {
    let result = percentile([1.0, .infinity, 3.0], 100)
    XCTAssertTrue(result.isInfinite, "100th percentile with Inf element should return Inf")
  }

  func testGmeanWithNaN() {
    // gmean checks v <= 0 but NaN > 0 is false, so NaN passes the guard
    // and log(NaN) = NaN propagates
    let result = gmean([1.0, .nan, 3.0])
    XCTAssertTrue(result.isNaN, "gmean with NaN element should propagate NaN")
  }

  func testGmeanWithInf() {
    // Inf passes the positivity check; log(Inf) = Inf; exp(Inf) = Inf
    let result = gmean([1.0, .infinity, 3.0])
    XCTAssertTrue(result.isInfinite, "gmean with Inf element should produce Inf")
  }

  func testHmeanWithNaN() {
    // NaN fails the v <= 0 guard (NaN <= 0 is false) so it passes through;
    // 1/NaN = NaN propagates into the reciprocal sum
    let result = hmean([1.0, .nan, 3.0])
    XCTAssertTrue(result.isNaN, "hmean with NaN element should propagate NaN")
  }

  func testHmeanWithInf() {
    // 1/Inf = 0; harmonic mean of [1, Inf, 3] = 3 / (1 + 0 + 1/3) = 3/(4/3) = 2.25
    let result = hmean([1.0, .infinity, 3.0])
    XCTAssertFalse(result.isNaN)
    XCTAssertFalse(result.isInfinite)
    XCTAssertGreaterThan(result, 0)
  }

  // MARK: - Statistics: all-NaN array

  func testMeanAllNaN() {
    let result = mean([Double.nan, Double.nan])
    XCTAssertTrue(result.isNaN)
  }

  func testVarianceAllNaN() {
    let result = variance([Double.nan, Double.nan])
    XCTAssertTrue(result.isNaN)
  }

  // MARK: - Distributions: NaN/Inf inputs

  func testNormalPdfNaN() {
    let dist = NormalDistribution()
    let result = dist.pdf(.nan)
    XCTAssertTrue(result.isNaN, "Normal PDF at NaN should return NaN")
  }

  func testNormalCdfNaN() {
    let dist = NormalDistribution()
    let result = dist.cdf(.nan)
    XCTAssertTrue(result.isNaN, "Normal CDF at NaN should return NaN")
  }

  func testNormalPpfNaN() {
    let dist = NormalDistribution()
    let result = dist.ppf(.nan)
    XCTAssertTrue(result.isNaN, "Normal PPF at NaN should return NaN")
  }

  func testNormalPdfInf() {
    let dist = NormalDistribution()
    // exp(-∞) → 0
    let resultPos = dist.pdf(.infinity)
    let resultNeg = dist.pdf(-.infinity)
    XCTAssertEqual(resultPos, 0.0, accuracy: 1e-15)
    XCTAssertEqual(resultNeg, 0.0, accuracy: 1e-15)
  }

  func testNormalCdfInf() {
    let dist = NormalDistribution()
    XCTAssertEqual(dist.cdf(.infinity), 1.0, accuracy: 1e-10)
    XCTAssertEqual(dist.cdf(-.infinity), 0.0, accuracy: 1e-10)
  }

  func testNormalPpfInfProbability() {
    let dist = NormalDistribution()
    // ppf(1) → +∞, ppf(0) → -∞
    XCTAssertTrue(dist.ppf(1.0).isInfinite)
    XCTAssertTrue(dist.ppf(0.0).isInfinite)
  }

  func testExponentialPdfNaN() {
    let dist = ExponentialDistribution()
    XCTAssertTrue(dist.pdf(.nan).isNaN, "Exponential PDF at NaN should return NaN")
  }

  func testExponentialCdfNaN() {
    let dist = ExponentialDistribution()
    XCTAssertTrue(dist.cdf(.nan).isNaN, "Exponential CDF at NaN should return NaN")
  }

  func testExponentialCdfInf() {
    let dist = ExponentialDistribution()
    // 1 - exp(-∞) = 1
    XCTAssertEqual(dist.cdf(.infinity), 1.0, accuracy: 1e-15)
  }

  // MARK: - Integration: NaN-returning function

  func testQuadWithNaNFunction() {
    // A function that always returns NaN; quad should not crash.
    let result = quad({ _ in Double.nan }, 0.0, 1.0)
    // NaN propagates into the quadrature sum
    XCTAssertTrue(result.value.isNaN, "quad with NaN-returning function should produce NaN value")
  }

  func testQuadWithInfLimits() {
    // Integrate standard normal density over (-∞, +∞) → should be ~1
    let result = quad(
      { x in Darwin.exp(-0.5 * x * x) / Darwin.sqrt(2 * .pi) },
      -.infinity, .infinity)
    XCTAssertEqual(
      result.value, 1.0, accuracy: 1e-6,
      "Integral of standard normal over R should be 1")
  }

  func testQuadWithPosInfLimit() {
    // Integral of exp(-x) from 0 to +∞ = 1
    let result = quad({ x in Darwin.exp(-x) }, 0.0, .infinity)
    XCTAssertEqual(
      result.value, 1.0, accuracy: 1e-6,
      "Integral of exp(-x) from 0 to +inf should equal 1")
  }

  func testQuadWithNegInfLimit() {
    // Integral of exp(x) from -∞ to 0 = 1
    let result = quad({ x in Darwin.exp(x) }, -.infinity, 0.0)
    XCTAssertEqual(
      result.value, 1.0, accuracy: 1e-6,
      "Integral of exp(x) from -inf to 0 should equal 1")
  }

  // MARK: - Interpolation: NaN in data arrays

  func testComputeSplineCoeffsWithNaNY() {
    // NaN in y data propagates into coefficient arithmetic; no crash expected.
    let x = [0.0, 1.0, 2.0, 3.0]
    let y = [0.0, .nan, 4.0, 9.0]
    let coeffs = computeSplineCoeffs(x: x, y: y)
    // Should return some coefficients (no crash), values may be NaN
    XCTAssertFalse(coeffs.isEmpty, "computeSplineCoeffs should return coefficients even with NaN y")
  }

  func testEvalCubicSplineWithNaNCoeffs() {
    // Verify evalCubicSpline doesn't crash when coefficients contain NaN
    let x = [0.0, 1.0, 2.0, 3.0]
    let y = [0.0, .nan, 4.0, 9.0]
    let coeffs = computeSplineCoeffs(x: x, y: y)
    guard !coeffs.isEmpty else { return }
    let result = evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.5)
    // Result contains NaN due to NaN inputs; main check is no crash
    _ = result
  }

  func testInterp1dWithNaNY() {
    // Linear interpolation with NaN in y; no crash, NaN propagation expected
    let x = [0.0, 1.0, 2.0]
    let y = [0.0, .nan, 4.0]
    let result = interp1d(x: x, y: y, xNew: 0.5)
    XCTAssertTrue(result.isNaN, "interp1d with NaN y values should propagate NaN")
  }

  func testComputePchipWithNaNY() {
    // PCHIP derivative computation with NaN in y; no crash expected
    let x = [0.0, 1.0, 2.0, 3.0]
    let y = [0.0, .nan, 4.0, 9.0]
    let d = computePchipDerivatives(x: x, y: y)
    XCTAssertFalse(d.isEmpty, "computePchipDerivatives should return array even with NaN y")
  }

  // MARK: - LinAlg: NaN elements

  func testMatrixDotWithNaN() {
    // Dot product with NaN element; NaN should propagate
    let a = LinAlg.Matrix([[1.0, .nan], [3.0, 4.0]])
    let b = LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])
    let result = LinAlg.dot(a, b)
    // NaN in first row should propagate
    XCTAssertTrue(
      result.data.contains(where: { $0.isNaN }),
      "Matrix dot with NaN element should propagate NaN")
  }

  func testMatrixAddWithNaN() {
    let a = LinAlg.Matrix([[1.0, .nan], [3.0, 4.0]])
    let b = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
    let result = LinAlg.add(a, b)
    XCTAssertTrue(result.data[1].isNaN, "Matrix add with NaN should propagate NaN in that element")
  }

  func testMatrixTraceWithNaN() {
    let m = LinAlg.Matrix([[.nan, 2.0], [3.0, 4.0]])
    let result = LinAlg.trace(m)
    XCTAssertTrue(result.isNaN, "Matrix trace with NaN on diagonal should return NaN")
  }

  func testMatrixDetWithNaN() {
    let m = LinAlg.Matrix([[.nan, 2.0], [3.0, 4.0]])
    let result = LinAlg.det(m)
    XCTAssertTrue(
      result.isNaN || result == 0 || result.isInfinite,
      "Matrix determinant with NaN should not crash")
  }

  func testMatrixNormWithNaN() {
    let m = LinAlg.Matrix([[1.0, .nan], [3.0, 4.0]])
    let result = LinAlg.norm(m)
    XCTAssertTrue(result.isNaN, "Matrix norm with NaN element should return NaN")
  }

  func testMatrixWithInfElement() {
    let a = LinAlg.Matrix([[1.0, .infinity], [3.0, 4.0]])
    let b = LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])
    let result = LinAlg.dot(a, b)
    XCTAssertTrue(
      result.data.contains(where: { $0.isInfinite }),
      "Matrix dot with Inf element should propagate Inf")
  }

  // MARK: - Optimization: NaN-returning functions

  func testBisectWithNaNFunction() {
    // Function returns NaN; bisect should not crash or loop infinitely.
    // fa*fb > 0 is false for NaN, so the sign-change guard passes.
    // fa*fc < 0 is also always false, so the else branch always fires:
    // a = c each iteration, narrowing the interval until |b-a| <= xtol.
    // bisect therefore reports converged=true even though every f call
    // returned NaN. The root is the final midpoint (a finite value).
    let result = bisect({ _ in Double.nan }, a: -1.0, b: 1.0)
    // Document actual behavior: no crash; interval narrows via else branch;
    // converged=true; root is a finite midpoint value.
    XCTAssertTrue(
      result.converged, "bisect interval narrows via else-branch even with NaN f values")
    XCTAssertFalse(result.root.isNaN, "bisect root is the narrowed midpoint, not NaN")
  }

  func testBisectWithInfBounds() {
    // Finite function evaluated at boundaries; normal case with wide bracket
    let result = bisect({ x in x * x - 4.0 }, a: 0.0, b: 1e10)
    XCTAssertTrue(result.converged)
    XCTAssertEqual(result.root, 2.0, accuracy: 1e-6)
  }

  func testNewtonWithNaNFunction() {
    // Function returns NaN; newton should not crash.
    let result = newton({ _ in Double.nan }, x0: 1.0)
    // fp will be NaN/NaN difference → NaN; abs(NaN) < 1e-14 is false,
    // so dx = NaN/NaN = NaN; x becomes NaN; loop runs to maxiter.
    XCTAssertFalse(result.converged, "newton with NaN function should not converge")
  }

  func testNewtonWithInfStartingPoint() {
    // Starting at Inf; function returns Inf; derivative step is NaN
    let result = newton({ x in x - 2.0 }, x0: .infinity)
    XCTAssertFalse(result.converged, "newton starting at Inf should not converge normally")
  }

  func testNewtonWithNaNDerivative() {
    // Function is valid but derivative always returns NaN
    let result = newton(
      { x in x - 2.0 },
      fprime: { _ in Double.nan },
      x0: 1.0)
    // abs(NaN) < 1e-14 is false → fp accepted as NaN → dx = NaN → no convergence
    XCTAssertFalse(
      result.converged,
      "newton with NaN derivative should not converge")
  }
}
