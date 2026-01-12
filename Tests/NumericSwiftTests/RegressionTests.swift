//
//  RegressionTests.swift
//  NumericSwift
//
//  Tests for regression modeling.
//

import XCTest
@testable import NumericSwift

final class RegressionTests: XCTestCase {

    // MARK: - OLS Tests

    func testOLSSimple() {
        // y = 2 + 3*x
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [5.0, 8.0, 11.0, 14.0, 17.0]

        let X = addConstant(x)
        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Check coefficients
        XCTAssertEqual(r.params[0], 2.0, accuracy: 1e-6)  // Intercept
        XCTAssertEqual(r.params[1], 3.0, accuracy: 1e-6)  // Slope

        // Perfect fit should have R^2 = 1
        XCTAssertEqual(r.rsquared, 1.0, accuracy: 1e-6)
    }

    func testOLSWithNoise() {
        // y = 1 + 2*x + noise
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let y = [3.1, 5.2, 6.9, 9.1, 11.0, 12.9, 15.1, 16.8, 19.2, 20.9]

        let X = addConstant(x)
        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Coefficients should be approximately correct
        XCTAssertEqual(r.params[0], 1.0, accuracy: 0.5)  // Intercept
        XCTAssertEqual(r.params[1], 2.0, accuracy: 0.1)  // Slope

        // High R^2 expected
        XCTAssertGreaterThan(r.rsquared, 0.99)
    }

    func testOLSMultipleRegression() {
        // y = 1 + 2*x1 + 3*x2
        let X = [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
            [1.0, 4.0, 2.0],
            [1.0, 5.0, 3.0],
            [1.0, 6.0, 3.0],
            [1.0, 7.0, 4.0],
            [1.0, 8.0, 4.0]
        ]
        let y = [6.0, 8.0, 13.0, 15.0, 20.0, 22.0, 27.0, 29.0]

        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Check coefficients
        XCTAssertEqual(r.params[0], 1.0, accuracy: 1e-6)  // Intercept
        XCTAssertEqual(r.params[1], 2.0, accuracy: 1e-6)  // x1 coefficient
        XCTAssertEqual(r.params[2], 3.0, accuracy: 1e-6)  // x2 coefficient

        XCTAssertEqual(r.rsquared, 1.0, accuracy: 1e-6)
    }

    func testOLSResiduals() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [2.0, 4.0, 5.0, 4.0, 5.0]

        let X = addConstant(x)
        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Residuals should sum to approximately 0
        let residSum: Double = r.residuals.reduce(0.0) { $0 + $1 }
        XCTAssertEqual(residSum, 0.0, accuracy: 1e-6)

        // Fitted + residuals = y
        for i in 0..<y.count {
            XCTAssertEqual(r.fittedValues[i] + r.residuals[i], y[i], accuracy: 1e-6)
        }
    }

    func testOLSStatistics() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let y = [2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0, 18.1, 19.9]

        let X = addConstant(x)
        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Check various statistics are computed
        XCTAssertEqual(r.nobs, 10)
        XCTAssertEqual(r.dfModel, 1)
        XCTAssertEqual(r.dfResid, 8)
        XCTAssertGreaterThan(r.rsquared, 0.99)
        XCTAssertGreaterThan(r.fvalue, 0)
        XCTAssertLessThan(r.fPvalue, 0.001)

        // t-values should be significant
        for tval in r.tvalues {
            XCTAssertNotEqual(tval, 0)
        }
    }

    func testOLSConfidenceIntervals() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let y = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]

        let X = addConstant(x)
        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Confidence intervals should contain the true parameter
        XCTAssertEqual(r.confInt.count, 2)

        // For each parameter, CI should bracket the estimate
        for i in 0..<r.params.count {
            XCTAssertLessThanOrEqual(r.confInt[i][0], r.params[i])
            XCTAssertGreaterThanOrEqual(r.confInt[i][1], r.params[i])
        }
    }

    func testOLSInfluenceDiagnostics() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

        let X = addConstant(x)
        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Hat diagonal should be computed
        XCTAssertEqual(r.hatDiag.count, 10)

        // Hat values should be between 0 and 1
        for h in r.hatDiag {
            XCTAssertGreaterThanOrEqual(h, 0)
            XCTAssertLessThanOrEqual(h, 1)
        }

        // Cook's distance should be computed
        XCTAssertEqual(r.cooksDistance.count, 10)

        // DFFITS should be computed
        XCTAssertEqual(r.dffits.count, 10)
    }

    func testOLSRobustStandardErrors() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let y = [2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0, 18.1, 19.9]

        let X = addConstant(x)
        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // HC0-HC3 should be computed
        XCTAssertEqual(r.bseHC0.count, 2)
        XCTAssertEqual(r.bseHC1.count, 2)
        XCTAssertEqual(r.bseHC2.count, 2)
        XCTAssertEqual(r.bseHC3.count, 2)

        // HC standard errors should be positive
        for se in r.bseHC0 {
            XCTAssertGreaterThanOrEqual(se, 0)
        }
    }

    func testOLSEmpty() {
        let result = ols([], [])
        XCTAssertNil(result)
    }

    func testOLSInsufficientData() {
        // More parameters than observations
        let y = [1.0, 2.0]
        let X = [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]
        let result = ols(y, X)
        XCTAssertNil(result)
    }

    // MARK: - WLS Tests

    func testWLS() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [2.0, 4.0, 6.0, 8.0, 10.0]
        let weights = [1.0, 1.0, 2.0, 2.0, 3.0]  // More weight on later observations

        let X = addConstant(x)
        let result = wls(y, X, weights: weights)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Should still find the perfect linear relationship
        XCTAssertEqual(r.params[0], 0.0, accuracy: 1e-6)  // Intercept
        XCTAssertEqual(r.params[1], 2.0, accuracy: 1e-6)  // Slope
    }

    // MARK: - GLM Tests

    func testGLMGaussian() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let y = [2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0, 18.1, 19.9]

        let X = addConstant(x)
        let result = glm(y, X, family: .gaussian)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Gaussian GLM should give similar results to OLS
        XCTAssertEqual(r.params[0], 0.0, accuracy: 0.5)
        XCTAssertEqual(r.params[1], 2.0, accuracy: 0.1)
        XCTAssertTrue(r.converged)
    }

    func testGLMBinomial() {
        // Logistic regression: y = logit(x)
        let X = [
            [1.0, -2.0], [1.0, -1.5], [1.0, -1.0], [1.0, -0.5],
            [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.0, 2.0]
        ]
        let y = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]

        let result = glm(y, X, family: .binomial, link: .logit)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Should converge
        XCTAssertTrue(r.converged)

        // Mu should be in (0, 1) for binomial
        for mu in r.mu {
            XCTAssertGreaterThan(mu, 0)
            XCTAssertLessThan(mu, 1)
        }
    }

    func testGLMPoisson() {
        // Poisson regression
        let X = [
            [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5],
            [1.0, 2.0], [1.0, 2.5], [1.0, 3.0]
        ]
        let y = [1.0, 2.0, 3.0, 5.0, 7.0, 12.0, 20.0]

        let result = glm(y, X, family: .poisson)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Should converge
        XCTAssertTrue(r.converged)

        // Mu should be positive for Poisson
        for mu in r.mu {
            XCTAssertGreaterThan(mu, 0)
        }

        // Deviance should be computed
        XCTAssertGreaterThanOrEqual(r.deviance, 0)
    }

    func testGLMGamma() {
        // Gamma regression
        let X = [
            [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0],
            [1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [1.0, 8.0]
        ]
        let y = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

        let result = glm(y, X, family: .gamma)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Mu should be positive for Gamma
        for mu in r.mu {
            XCTAssertGreaterThan(mu, 0)
        }
    }

    func testGLMStatistics() {
        let X = addConstant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        let y = [2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0, 18.1, 19.9]

        let result = glm(y, X, family: .gaussian)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Check statistics
        XCTAssertEqual(r.nobs, 10)
        XCTAssertEqual(r.dfModel, 1)
        XCTAssertEqual(r.dfResid, 8)

        // AIC and BIC should be finite
        XCTAssertTrue(r.aic.isFinite)
        XCTAssertTrue(r.bic.isFinite)

        // z-values and p-values should be computed
        XCTAssertEqual(r.zvalues.count, 2)
        XCTAssertEqual(r.pvalues.count, 2)
    }

    func testGLMEmpty() {
        let result = glm([], [], family: .gaussian)
        XCTAssertNil(result)
    }

    // MARK: - ARIMA Tests

    func testARIMASimple() {
        // AR(1) process
        let y = [1.0, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5]

        let result = arima(y, p: 1, d: 0, q: 0)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Should converge
        XCTAssertTrue(r.converged)

        // Should have 1 AR parameter
        XCTAssertEqual(r.arParams.count, 1)
        XCTAssertEqual(r.maParams.count, 0)
    }

    func testARIMAWithDifferencing() {
        // Random walk (d=1 should help)
        let y = [10.0, 10.5, 11.2, 11.8, 12.3, 12.9, 13.5, 14.0, 14.7, 15.2]

        let result = arima(y, p: 1, d: 1, q: 0)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Should converge
        XCTAssertTrue(r.converged)

        // Order should be stored
        XCTAssertEqual(r.order.p, 1)
        XCTAssertEqual(r.order.d, 1)
        XCTAssertEqual(r.order.q, 0)

        // Differenced series should be shorter
        XCTAssertEqual(r.yDiff.count, y.count - 1)
    }

    func testARIMAWithMA() {
        // ARMA(1,1) process
        let y = [1.0, 2.0, 2.5, 3.0, 3.8, 4.2, 5.0, 5.5, 6.0, 6.8,
                 7.2, 7.8, 8.5, 9.0, 9.5, 10.0, 10.8, 11.2, 12.0, 12.5]

        let result = arima(y, p: 1, d: 0, q: 1)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Should have both AR and MA parameters
        XCTAssertEqual(r.arParams.count, 1)
        XCTAssertEqual(r.maParams.count, 1)
    }

    func testARIMAForecast() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        let result = arima(y, p: 1, d: 0, q: 0)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        let forecasts = arimaForecast(r, steps: 5)

        XCTAssertEqual(forecasts.count, 5)

        // Forecasts should be finite
        for f in forecasts {
            XCTAssertTrue(f.isFinite)
        }
    }

    func testARIMAForecastWithDifferencing() {
        let y = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]

        let result = arima(y, p: 0, d: 1, q: 0)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        let forecasts = arimaForecast(r, steps: 3)

        // With d=1, forecasts should be integrated back
        XCTAssertEqual(forecasts.count, 3)

        // For a linear trend with d=1, forecasts should continue roughly at the last level
        for f in forecasts {
            XCTAssertGreaterThan(f, y.last! - 5)  // Reasonable range
        }
    }

    func testARIMAStatistics() {
        let y = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

        let result = arima(y, p: 1, d: 0, q: 0)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Statistics should be computed
        XCTAssertGreaterThan(r.sigma2, 0)
        XCTAssertTrue(r.llf.isFinite)
        XCTAssertTrue(r.aic.isFinite)
        XCTAssertTrue(r.bic.isFinite)

        // Residuals should be computed
        XCTAssertEqual(r.residuals.count, y.count)

        // Fitted values should be computed
        XCTAssertEqual(r.fittedValues.count, y.count)
    }

    func testARIMAEmpty() {
        let result = arima([], p: 1, d: 0, q: 0)
        XCTAssertNil(result)
    }

    func testARIMAInsufficientData() {
        // Too few observations for the model order
        let y = [1.0, 2.0]
        let result = arima(y, p: 5, d: 0, q: 0)
        XCTAssertNil(result)
    }

    // MARK: - Statistical Distribution Tests

    func testTCDF() {
        // t=0 should give 0.5
        XCTAssertEqual(tCDF(0, 10), 0.5, accuracy: 1e-10)

        // CDF should increase with t
        let cdf1 = tCDF(1.0, 10)
        let cdf2 = tCDF(2.0, 10)
        XCTAssertGreaterThan(cdf2, cdf1)

        // Should approach 1 for large t
        XCTAssertGreaterThan(tCDF(10, 10), 0.99)
    }

    func testTPPF() {
        // PPF(0.5) should be 0
        XCTAssertEqual(tPPF(0.5, 10), 0.0, accuracy: 1e-6)

        // PPF and CDF should be inverses
        let t = 1.5
        let df = 10.0
        let cdf = tCDF(t, df)
        let roundTrip = tPPF(cdf, df)
        XCTAssertEqual(roundTrip, t, accuracy: 1e-6)
    }

    func testTPDF() {
        // PDF should be positive
        XCTAssertGreaterThan(tPDF(0, 10), 0)

        // PDF should be symmetric
        let pdf1 = tPDF(-1, 10)
        let pdf2 = tPDF(1, 10)
        XCTAssertEqual(pdf1, pdf2, accuracy: 1e-10)

        // Maximum at t=0
        XCTAssertGreaterThan(tPDF(0, 10), tPDF(1, 10))
    }

    func testFCDF() {
        // F(0) should be 0
        XCTAssertEqual(fCDF(0, 5, 10), 0.0, accuracy: 1e-10)

        // CDF should increase with x
        let cdf1 = fCDF(1.0, 5, 10)
        let cdf2 = fCDF(2.0, 5, 10)
        XCTAssertGreaterThan(cdf2, cdf1)

        // Should approach 1 for large x
        XCTAssertGreaterThan(fCDF(100, 5, 10), 0.99)
    }

    func testStandardNormalCDF() {
        // CDF(0) = 0.5
        XCTAssertEqual(standardNormalCDF(0), 0.5, accuracy: 1e-10)

        // CDF should be symmetric
        let cdf1 = standardNormalCDF(-1)
        let cdf2 = standardNormalCDF(1)
        XCTAssertEqual(cdf1 + cdf2, 1.0, accuracy: 1e-10)

        // Known values
        XCTAssertEqual(standardNormalCDF(1.96), 0.975, accuracy: 0.001)
    }

    // MARK: - Helper Function Tests

    func testAddConstant() {
        // 1D case
        let x1D = [1.0, 2.0, 3.0]
        let X1D = addConstant(x1D)
        XCTAssertEqual(X1D.count, 3)
        XCTAssertEqual(X1D[0], [1.0, 1.0])
        XCTAssertEqual(X1D[1], [1.0, 2.0])
        XCTAssertEqual(X1D[2], [1.0, 3.0])

        // 2D case
        let X2D = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        let XConst = addConstant(X2D)
        XCTAssertEqual(XConst.count, 3)
        XCTAssertEqual(XConst[0], [1.0, 1.0, 2.0])
        XCTAssertEqual(XConst[1], [1.0, 3.0, 4.0])
        XCTAssertEqual(XConst[2], [1.0, 5.0, 6.0])
    }

    // MARK: - Integration Tests

    func testOLSThenPredict() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [3.0, 5.0, 7.0, 9.0, 11.0]  // y = 1 + 2x

        let X = addConstant(x)
        let result = ols(y, X)

        XCTAssertNotNil(result)
        guard let r = result else { return }

        // Manual prediction for x = 6
        let prediction = r.params[0] + r.params[1] * 6.0
        XCTAssertEqual(prediction, 13.0, accuracy: 1e-6)
    }

    func testCompareOLSAndGLMGaussian() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let y = [2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0, 18.1, 19.9]

        let X = addConstant(x)

        let olsResult = ols(y, X)
        let glmResult = glm(y, X, family: .gaussian)

        XCTAssertNotNil(olsResult)
        XCTAssertNotNil(glmResult)

        guard let olsR = olsResult, let glmR = glmResult else { return }

        // Coefficients should be similar
        XCTAssertEqual(olsR.params[0], glmR.params[0], accuracy: 0.1)
        XCTAssertEqual(olsR.params[1], glmR.params[1], accuracy: 0.1)
    }
}
