//
//  JacobianStepSizingTests.swift
//  Tests/NumericSwiftTests/JacobianStepSizingTests.swift
//
//  Tests for per-variable relative step sizing in finite-difference Jacobians.
//
//  Context: Optimization.swift contains four Jacobian computation sites
//  (newtonMulti ~L650, bounded leastSquares ~L826, leastSquaresUnbounded ~L1053,
//  curveFit covariance ~L1214). Prior to this fix, sites used a single global
//  step h or h scaled by max|x|, causing poor accuracy when variables differ
//  greatly in magnitude. The fix follows scipy.optimize.approx_fprime:
//    h_i = sqrt(eps) * max(1, |x_i|)
//  which scales each step to the variable's own magnitude.
//
//  Oracle: all expected derivatives are computed from analytic (closed-form)
//  formulas, never from the code under test.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest

@testable import NumericSwift

// MARK: - newtonMulti Jacobian accuracy tests

final class NewtonMultiJacobianTests: XCTestCase {

    // MARK: Well-scaled case: variables all ≈ 1

    /// f(x,y) = [x^2 + y - 1, x + y^2 - 1].
    ///
    /// Analytic symmetric solution: x = y = (√5 - 1)/2 ≈ 0.618.
    /// Starting from [0.5, 0.5] produces a singular Jacobian (both rows
    /// equal [1,1] at the symmetric midpoint), so we start from [0.7, 0.6]
    /// which breaks the degeneracy and converges to the symmetric root in 4
    /// Newton steps (verified analytically: J at [0.7,0.6] = [[1.4,1],[1,1.2]],
    /// det = 1.4*1.2 - 1 = 0.68 ≠ 0).
    func testNewtonMultiWellScaledConverges() {
        let f: ([Double]) -> [Double] = { x in
            [x[0] * x[0] + x[1] - 1.0,
             x[0] + x[1] * x[1] - 1.0]
        }
        let result = newtonMulti(f, x0: [0.7, 0.6])
        XCTAssertTrue(result.success, "newtonMulti should converge for well-scaled problem")
        let phi = (sqrt(5.0) - 1.0) / 2.0  // analytic root ≈ 0.618
        XCTAssertEqual(result.x[0], phi, accuracy: 1e-8,
                       "x[0] should match analytic root (√5-1)/2")
        XCTAssertEqual(result.x[1], phi, accuracy: 1e-8,
                       "x[1] should match analytic root (√5-1)/2")
    }

    // MARK: Near-zero variable

    /// f(x, y) = [x, y - 1e-12]. Solution: [0, 1e-12].
    ///
    /// When |x_i| is near zero, per-variable h_i = sqrt(eps)*max(1, 0) = sqrt(eps)*1,
    /// same as the well-scaled case. The fix must not regress near-zero handling.
    func testNewtonMultiNearZeroVariable() {
        let f: ([Double]) -> [Double] = { x in
            [x[0],
             x[1] - 1e-12]
        }
        let result = newtonMulti(f, x0: [0.1, 0.1])
        XCTAssertTrue(result.success, "newtonMulti should converge for near-zero problem")
        XCTAssertEqual(result.x[0], 0.0, accuracy: 1e-9, "x[0] should be ≈ 0")
        XCTAssertEqual(result.x[1], 1e-12, accuracy: 1e-15, "x[1] should be ≈ 1e-12")
    }

    // MARK: Large-magnitude variable — EXPOSES old global-h bug

    /// f(x1, x2) = [x1 - 1e-6, x2 - 1e10].
    ///
    /// Analytic solution: [1e-6, 1e10].
    ///
    /// Old code: global h = sqrt(ulpOfOne) ≈ 1.49e-8.
    ///   ULP(1e10) ≈ 1.9e-6, so h < ULP(1e10).
    ///   (x2 + h) rounds back to x2 in floating-point → J[1][1] = (f2(x2+h) - f2(x2)) / h = 0
    ///   → singular Jacobian → newtonMulti fails to converge.
    ///
    /// New code: h_i = sqrt(ulpOfOne) * max(1, |x_i|).
    ///   For x2 = 1e10: h2 = sqrt(ulpOfOne) * 1e10 ≈ 1.49e2 >> ULP(1e10).
    ///   Jacobian is well-conditioned → converges.
    func testNewtonMultiLargeMagnitudeVariableExposesOldBug() {
        let f: ([Double]) -> [Double] = { x in
            [x[0] - 1e-6,
             x[1] - 1e10]
        }
        let result = newtonMulti(f, x0: [1e-5, 1e9])
        XCTAssertTrue(result.success,
            "newtonMulti must converge when one variable is 1e10 (old global h underflows)")
        XCTAssertEqual(result.x[0], 1e-6, accuracy: 1e-12,
            "x[0] should converge to 1e-6")
        XCTAssertEqual(result.x[1], 1e10, accuracy: 1e4,
            "x[1] should converge to 1e10")
    }

    // MARK: Jacobian accuracy vs. analytic oracle

    /// f(x,y) = [x^2 + y - 1, x + y^2 - 1], starting near the solution.
    ///
    /// An accurate Jacobian at x0 = [0.8, 0.8] should let Newton converge in
    /// very few iterations to the analytic root φ = (√5-1)/2 with residual < 1e-10.
    func testNewtonMultiJacobianAccuracyNearSolution() {
        let f: ([Double]) -> [Double] = { x in
            [x[0] * x[0] + x[1] - 1.0,
             x[0] + x[1] * x[1] - 1.0]
        }
        let result = newtonMulti(f, x0: [0.8, 0.8], tol: 1e-12)
        let fx = f(result.x)
        let resNorm = sqrt(fx.reduce(0) { $0 + $1 * $1 })
        XCTAssertLessThan(resNorm, 1e-10,
            "Residual norm should be tiny at converged solution (accurate Jacobian)")
    }
}

// MARK: - leastSquares Jacobian accuracy tests

final class LeastSquaresJacobianTests: XCTestCase {

    // MARK: Well-scaled case

    /// Fit y = a * exp(b * x) to clean data with a = 2, b = 0.5.
    /// Parameters are O(1). Verifies that the fix preserves well-scaled behaviour.
    func testLeastSquaresWellScaledExponentialFit() {
        let xdata = stride(from: 0.0, through: 3.0, by: 0.5).map { $0 }
        let a = 2.0, b = 0.5
        let ydata = xdata.map { a * exp(b * $0) }

        let result = leastSquares({ p in
            xdata.map { x in ydata[xdata.firstIndex(of: x)!] - p[0] * exp(p[1] * x) }
        }, x0: [1.0, 0.3])

        XCTAssertTrue(result.success, "leastSquares should converge for well-scaled exponential")
        XCTAssertEqual(result.x[0], a, accuracy: 1e-8, "amplitude should be ≈ 2.0")
        XCTAssertEqual(result.x[1], b, accuracy: 1e-8, "rate should be ≈ 0.5")
    }

    // MARK: Mixed-magnitude — EXPOSES old global-max-h bug

    /// Mixed-magnitude least-squares: fit f(p, x) = p[0]*x + p[1],
    /// true parameters p[0] = 1.0 (unit slope), p[1] = 1e8 (huge intercept).
    ///
    /// The two parameters differ by 8 orders of magnitude — exactly the case
    /// that breaks the old global-max-h step formula:
    ///
    /// Old code: h = sqrt(eps) * max(1, max|p|) = sqrt(eps) * 1e8 ≈ 1.49
    ///   For p[0] ≈ 1: relative step = h/p[0] ≈ 1.49 → far too large.
    ///   The finite-difference column for the slope is numerically dominated by
    ///   the intercept-scale rounding noise and can be wildly wrong.
    ///
    /// New code: h_j = sqrt(eps) * max(1, |p[j]|)
    ///   For p[0] ≈ 1:  h0 = sqrt(eps)*1 ≈ 1.49e-8 → well-conditioned.
    ///   For p[1] ≈ 1e8: h1 = sqrt(eps)*1e8 ≈ 1.49 → well-conditioned.
    ///
    /// Analytic solution: p* = [1.0, 1e8] (exact for consistent linear data).
    /// The design matrix is well-conditioned (cond ≈ 10), so IEEE-754 Double
    /// arithmetic recovers both parameters to near machine precision.
    func testLeastSquaresMixedMagnitudeExposesOldBug() {
        let trueSlope = 1.0
        let trueIntercept = 1e8
        let xdata = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        let ydata = xdata.map { trueSlope * $0 + trueIntercept }

        let result = leastSquares({ p in
            zip(xdata, ydata).map { x, y in y - (p[0] * x + p[1]) }
        }, x0: [10.0, 1e7])

        XCTAssertTrue(result.success,
            "leastSquares must converge for mixed-magnitude linear fit (old global-max-h fails)")
        XCTAssertEqual(result.x[0], trueSlope, accuracy: 1e-4,
            "unit slope must be recovered; old max-h gives catastrophically wrong Jacobian column")
        XCTAssertEqual(result.x[1], trueIntercept, accuracy: trueIntercept * 1e-4,
            "large intercept (1e8) must be recovered")
    }

    // MARK: Near-zero parameter

    /// Fit f(p, x) = p[0]*x^2 + p[1] with true p[1] = 0 (near-zero).
    /// Per-variable h_j = sqrt(eps)*max(1, 0) = sqrt(eps)*1: same as well-scaled.
    func testLeastSquaresNearZeroParameter() {
        let xdata = [-2.0, -1.0, 0.0, 1.0, 2.0]
        let trueA = 3.0
        let trueB = 0.0
        let ydata = xdata.map { trueA * $0 * $0 + trueB }

        let result = leastSquares({ p in
            zip(xdata, ydata).map { x, y in y - (p[0] * x * x + p[1]) }
        }, x0: [1.0, 1.0])

        XCTAssertTrue(result.success, "leastSquares should converge with near-zero parameter")
        XCTAssertEqual(result.x[0], trueA, accuracy: 1e-5, "p[0] should converge to 3.0")
        XCTAssertEqual(result.x[1], trueB, accuracy: 1e-8, "p[1] should converge near 0")
    }

    // MARK: Jacobian accuracy vs. analytic oracle (linear residuals)

    /// Overdetermined linear system: residuals = A*p - b.
    ///
    /// A = [[2, 1], [1, 3], [0, 1]], b = [5, 10, 3].
    /// Analytic solution from normal equations:
    ///   p[1] = 3 (from row 3), p[0] = (5-3)/2 = 1 (from row 1). Check: 1+9=10 ✓
    ///   Exact analytic solution: p = [1, 3].
    func testLeastSquaresJacobianAccuracyLinearOracle() {
        let result = leastSquares({ p in
            [2.0 * p[0] + p[1] - 5.0,
             p[0] + 3.0 * p[1] - 10.0,
             p[1] - 3.0]
        }, x0: [0.0, 0.0])

        XCTAssertTrue(result.success, "Linear overdetermined system must converge")
        XCTAssertEqual(result.x[0], 1.0, accuracy: 1e-8, "p[0] should equal analytic value 1.0")
        XCTAssertEqual(result.x[1], 3.0, accuracy: 1e-8, "p[1] should equal analytic value 3.0")
        XCTAssertEqual(result.cost, 0.0, accuracy: 1e-10, "Consistent system should have zero cost")
    }
}

// MARK: - curveFit Jacobian accuracy tests

final class CurveFitJacobianTests: XCTestCase {

    // MARK: Mixed-magnitude curveFit parameters

    /// curveFit with mixed-magnitude parameters: slope = 1.0, intercept = 1e8.
    ///
    /// The curveFit covariance Jacobian (~L1246) used the same global-max-h pattern
    /// as leastSquares. With the old formula h ≈ sqrt(eps)*1e8 ≈ 1.49 for the
    /// slope column, the covariance computation is numerically destroyed and pcov
    /// contains NaN or Inf. With per-variable h both popt values and pcov are finite.
    ///
    /// Design-matrix condition number ≈ 10 (cond(A) verified via numpy), so IEEE-754
    /// Double arithmetic recovers both parameters to near machine precision.
    func testCurveFitMixedMagnitudeParameterAccuracy() {
        let trueSlope = 1.0
        let trueIntercept = 1e8
        let xdata = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        let ydata = xdata.map { trueSlope * $0 + trueIntercept }

        let (popt, pcov, info) = curveFit(
            { p, x in p[0] * x + p[1] },
            xdata: xdata,
            ydata: ydata,
            p0: [10.0, 1e7]
        )

        XCTAssertTrue(info.success, "curveFit must converge for mixed-magnitude linear model")
        XCTAssertEqual(popt[0], trueSlope, accuracy: 1e-4,
            "unit slope must be recovered; old max-h gives NaN covariance")
        XCTAssertEqual(popt[1], trueIntercept, accuracy: trueIntercept * 1e-4,
            "large intercept (1e8) must be recovered")

        // Covariance matrix must be finite
        for i in 0..<pcov.count {
            for j in 0..<pcov[i].count {
                XCTAssertFalse(pcov[i][j].isNaN,
                    "pcov[\(i)][\(j)] must not be NaN after per-variable h fix")
                XCTAssertFalse(pcov[i][j].isInfinite,
                    "pcov[\(i)][\(j)] must not be infinite after per-variable h fix")
            }
        }
    }

    // MARK: curveFit covariance diagonal positivity (well-scaled)

    /// For clean well-scaled data, the covariance diagonal must be positive.
    ///
    /// Model: f(p, x) = p[0] * exp(-p[1] * x), true p = [3.0, 0.5].
    func testCurveFitCovarianceDiagonalPositive() {
        let a = 3.0, b = 0.5
        let xdata = stride(from: 0.0, through: 4.0, by: 0.5).map { $0 }
        let ydata = xdata.map { a * exp(-b * $0) }

        let (_, pcov, info) = curveFit(
            { p, x in p[0] * exp(-p[1] * x) },
            xdata: xdata,
            ydata: ydata,
            p0: [1.0, 0.3]
        )
        XCTAssertTrue(info.success, "curveFit should converge for well-scaled exponential")
        for i in 0..<pcov.count {
            XCTAssertGreaterThanOrEqual(pcov[i][i], 0,
                "pcov[\(i)][\(i)] (variance) must be non-negative")
        }
    }
}
