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

    // MARK: - ARIMA CSS MA-bias fix (issue #10)
    //
    // Oracle: statsmodels 0.14.6 SARIMAX (innovations MLE) on fixed-seed
    // simulated processes. CSS and MLE agree to within ±0.15 on n=200 series.
    // Series generated in Python (see comments per test for exact spec).

    /// AR+MA jointly identified: ARMA(1,1) with phi=0.6, theta=0.4, seed=42, n=200.
    ///
    /// Old code: AR estimated on raw y ignoring MA → biased, AR param inflated toward 1.
    /// Fixed code: Hannan-Rissanen iterated joint OLS → AR and MA near oracle.
    ///
    /// Oracle (statsmodels 0.14.6 SARIMAX, innovations_mle, n=200): AR≈0.4935, MA≈0.4197.
    func testARIMACSS_ARMA11_JointEstimation() {
        // Fixed-seed ARMA(1,1): y[t] = 0.6*y[t-1] + e[t] + 0.4*e[t-1], e~N(0,1), seed=42
        // Series spec: np.random.seed(42); e~N(0,1) n+50; burn-in 50; length 200.
        let y: [Double] = [
            -1.1753181296, -0.9606395704, -1.4073386547, -0.5034957041,  0.9735726156,
             1.9278234975,  0.6899886229, -0.2309062114,  0.0690347542,  1.1494713522,
             0.6007266243, -0.0168926972, -1.1907341830, -2.3531811235, -1.0778655013,
             1.0345310567,  1.0912045239,  1.6294515636,  1.7407201224,  0.5439667288,
             0.4297277410,  1.9404314532,  1.7436474594,  2.5965017158, -0.4359866123,
            -0.4875875046,  0.1232555672, -0.1902351828, -0.1419832734, -2.0360545680,
            -2.2363321945, -1.0725555003,  0.9772057732,  0.6592108635, -0.6202751721,
            -1.1973195880, -0.0036924525,  0.6926964852,  0.0173581312,  0.3117782303,
             0.4894514608,  1.3011468868,  0.4660930344, -0.3288275635, -0.7204695499,
            -2.0526399393, -1.5208696658, -0.5330184165, -0.2102754844, -0.3587070413,
            -1.7244298202, -2.0214515117, -1.7238435527, -1.9736692074, -1.6663981438,
            -0.6603023141,  1.6516248555,  1.9200270866,  1.4793977678,  0.9162129012,
            -1.3988218409, -1.6333154661, -0.9303646199,  1.9291154245,  1.9504051349,
             1.3948460374,  0.9228147897, -0.6288738717,  0.2980272764,  1.3878785244,
             1.9245322747,  0.5617446889,  1.3760861423, -0.0150816530,  0.0170676769,
             2.4354390695,  1.3469093669, -0.1543666395, -0.2194877105, -0.5953077344,
            -2.1092383333, -1.8172453976, -2.1252257624, -1.2264645123, -1.4658659693,
             0.3026451297,  0.0183075475, -0.6243783046,  0.3100656281, -0.7194180526,
            -0.6965366235,  0.9802047540, -0.4965032804, -0.7562614036, -0.1200205045,
             0.8137636868, -0.4359633501, -2.0768149075, -1.2523300241, -0.2456367150,
             0.2219046906,  0.5797881640, -0.1935725394, -0.1558997151,  0.2924341231,
            -0.4216619548,  1.3270367710,  2.0163647880,  0.2080485440,  0.3048613361,
            -0.5291434251,  0.0797258806,  1.5212649489,  0.5555148826,  0.9684121314,
             1.3791786575,  1.8146797253,  3.3144248818,  2.5019840061,  0.6492989929,
            -0.8014294996, -1.6524737566, -1.3949100773, -0.5266347554,  0.0971707360,
             0.9961620104,  0.9415723977,  2.0236782725,  1.5309637612,  3.5328846900,
             3.8334658284,  1.6931888797, -0.3978421928, -0.1845898997, -0.1412277590,
             0.5398787245,  1.0827650569,  0.7661251713, -0.4162501803, -2.1033148201,
            -2.3144427340, -0.7108728269,  0.1301295657, -1.0820235416, -0.9743287106,
            -0.1300074763, -0.8077349701, -0.6844588506, -0.2909765495, -1.2942727402,
            -0.8759644029,  0.1783208288,  1.4143575510,  2.3356370799,  0.4452337008,
            -1.2217525666, -0.5931462887,  0.3639122846,  0.9389094374,  4.6220962276,
             4.8852408435,  4.2950663506,  3.9852678299,  3.4241526546,  1.9997788487,
             1.8327288318,  0.6303997728, -0.1677088289, -0.6807162879, -0.5207010525,
             2.0349875910,  0.2795907886,  0.1071085865, -1.2739466431, -1.8813862001,
            -0.2286538694,  0.3626679362, -0.8344320086, -1.6470608256, -0.5947602301,
            -0.8153836702, -0.5649182652, -0.2067956834, -0.7574490217,  1.4288345373,
             2.3487973804, -0.3622965495, -0.8409806496, -1.0917931286, -0.0673571283,
            -0.4919616815, -0.7269217457,  0.0229396550,  1.0815138987, -0.2050859901,
            -0.9376713928, -1.1713486411, -1.5461165417,  0.5764526222,  1.4570349804,
        ]

        let result = arima(y, p: 1, d: 0, q: 1)
        XCTAssertNotNil(result, "ARMA(1,1) fit should not return nil")
        guard let r = result else { return }

        // Oracle: statsmodels 0.14.6, SARIMAX(order=(1,0,1), trend='n').fit() on 200 obs.
        // AR(1) = 0.4935, MA(1) = 0.4197.
        //
        // Pre-fix symptom: estimating AR on raw y without MA contribution inflates
        // AR toward ρ(1)—the lag-1 autocorrelation—which for ARMA(1,1)(φ=0.6, θ=0.4)
        // is ≈ 0.694 (old AR OLS on this series: 0.6940). The tighter tolerance of
        // ±0.05 below is narrow enough to exclude 0.694 while accepting the
        // Hannan-Rissanen estimate (≈ 0.4941, δ ≈ 0.0006 from oracle).
        let arOracle = 0.4935
        let maOracle = 0.4197
        let tol = 0.05

        // Discriminating guard: the pre-fix symptom is AR inflated toward 1.
        // Old biased estimate on this series: AR ≈ 0.694. The fixed estimate ≈ 0.494.
        XCTAssertLessThan(r.arParams[0], 0.65,
            "AR(1) must be below 0.65; old AR-only estimator gives ≈ 0.694 on this series, " +
            "indicating MA contribution is being ignored.")

        XCTAssertEqual(r.arParams[0], arOracle, accuracy: tol,
            "AR(1) should be near oracle \(arOracle) (±\(tol)); got \(r.arParams[0]). " +
            "Bias indicates AR is being estimated on raw y ignoring MA contribution.")
        XCTAssertEqual(r.maParams[0], maOracle, accuracy: tol,
            "MA(1) should be near oracle \(maOracle) (±\(tol)); got \(r.maParams[0]).")
    }

    /// Pure AR(2) regression must be unchanged after fix: phi1=0.5, phi2=-0.3, seed=123, n=200.
    ///
    /// Oracle (statsmodels 0.14.6 SARIMAX): AR1≈0.5351, AR2≈-0.3064.
    func testARIMACSS_AR2_UnchangedByFix() {
        // Fixed-seed AR(2): y[t] = 0.5*y[t-1] - 0.3*y[t-2] + e[t], e~N(0,1), seed=123
        // Series spec: np.random.seed(123); e~N(0,1) n+50; burn-in 50; length 200.
        let y: [Double] = [
            -0.4431731024, -1.9557264679,  0.8988009219,  0.2380556661, -0.1209292132,
             0.9374346630,  1.3957024868,  2.1715070265,  2.1626869042,  1.4992840139,
            -0.6718727786,  0.0091410743,  0.5204043652, -1.0688055996,  0.7267749371,
             1.4912656830,  0.5730904410, -0.3939265454, -1.5671915497, -0.4658937377,
             0.7056497155, -0.3385620052,  0.7812281319, -0.6050203790, -2.6599789791,
            -0.1087562850,  0.3402495131,  0.0767220567, -0.9012305484, -2.0795946519,
             0.4858092133,  0.1779140184,  1.6041667334,  1.5560173474, -0.0179994930,
            -1.5617073519, -1.5079158147, -1.4979688329,  1.7905036876,  1.5090837239,
             1.3675962982, -1.0362790171, -0.7473832684,  1.1150540097,  0.4467312235,
             0.9199638677, -0.7586053452, -2.0187633775, -0.4023994731,  0.0252528421,
             0.7754009523, -1.5977633080, -0.3192373042,  2.9180142676,  1.5301523436,
            -0.0761859796, -0.3175892082, -1.9979145210, -0.4775338584, -1.2448023169,
            -0.9068205991,  1.1628999451,  0.1182791962,  0.2115186041,  1.0830145972,
             0.7567925734, -1.3174565625, -1.2182413286,  1.7455276468, -0.7868095410,
            -1.1928490784, -0.9124897484,  0.0223572124,  1.0331411479,  2.1185543783,
             0.4791024527,  0.4163262429,  0.5641725305,  0.6315356902, -0.4174078455,
            -1.3954860986, -1.6725638083, -1.1740732839,  0.2364190764,  1.2313809165,
             0.8682335832, -0.4842525795,  1.3033737452,  2.3158282701,  0.4129018987,
            -1.3117289376, -0.6495200843,  1.3360572842,  1.1956496446,  0.7535563420,
            -0.1939968447,  0.1332055703,  1.6693462900,  0.5550426924, -0.0799748083,
             0.0473162647,  0.3313759306, -1.2603957902, -2.6064793301, -1.9447759989,
            -0.0225019050,  1.1260380132,  0.0390950183,  1.0589935881,  0.3745923142,
            -0.1100859211, -0.3613845254, -0.0136396936,  0.8060695848,  1.0727801384,
            -0.6038537469,  0.8999028615, -0.4639189026, -0.4227032957, -0.3465725506,
            -1.0954669639, -0.5188823048, -0.6716148365, -0.1072354835,  0.5509526704,
             1.7795763492,  1.0318865921, -0.6291549490, -1.0157632629, -0.1791570406,
             0.3086112880,  1.6676420244,  2.1365905556,  0.2090667442, -1.0850859227,
            -3.1623175887, -1.8045534309, -0.9316391449, -0.2792780013,  0.5314369852,
             0.5266942222,  0.0739480085,  0.0785478487, -0.1090282514,  0.1189404522,
            -3.1388763064, -1.8744137786, -0.1063947184,  0.1678650582, -0.1020953175,
             0.6019029416, -0.2665252653,  1.8868685841,  1.7116888019,  0.2834765748,
            -0.5784306564, -0.4607805870, -0.9721681665, -0.4430524464,  0.3488077435,
             0.8868612212,  0.9184780667, -0.0816968796, -1.7324741123, -1.5108306185,
             1.3765199672,  2.0375674826,  0.9754473374, -0.8848408208, -0.7314094567,
            -1.3559211677, -1.0104746226, -0.3436643026, -0.2303296964,  0.9445363736,
            -0.8773588183, -1.5874725928, -1.9052166219, -1.7137197458, -0.1612389899,
            -1.1669441043,  0.2187684237,  0.2126516662,  0.1094836385,  0.3135230574,
            -0.3102500831,  0.7832977570,  0.2903811767,  0.5042715165, -0.0340909775,
             0.1225474437,  0.3511636853,  0.3887875490, -0.8852631807, -0.1233920845,
            -0.1150740771,  0.6099686173, -1.8129869126, -2.5546002636, -0.3699584959,
        ]

        let result = arima(y, p: 2, d: 0, q: 0)
        XCTAssertNotNil(result, "AR(2) fit should not return nil")
        guard let r = result else { return }

        // Oracle: statsmodels 0.14.6, SARIMAX(order=(2,0,0), trend='n').fit() on 200 obs.
        // AR1≈0.5351, AR2≈-0.3064. Pure AR: fix must not degrade AR estimation.
        let ar1Oracle = 0.5351
        let ar2Oracle = -0.3064
        let tol = 0.10

        XCTAssertEqual(r.arParams[0], ar1Oracle, accuracy: tol,
            "AR(2) phi1 should be near oracle \(ar1Oracle) (±\(tol)); got \(r.arParams[0]).")
        XCTAssertEqual(r.arParams[1], ar2Oracle, accuracy: tol,
            "AR(2) phi2 should be near oracle \(ar2Oracle) (±\(tol)); got \(r.arParams[1]).")
        XCTAssertEqual(r.maParams.count, 0, "Pure AR(2) should have no MA params.")
    }

    /// ARIMA(1,1,1) with integration: phi=0.7, theta=0.3, seed=7, n=200.
    ///
    /// Oracle (statsmodels 0.14.6 SARIMAX(order=(1,1,1))): AR≈0.6900, MA≈0.2817.
    func testARIMACSS_ARIMA111_JointEstimation() {
        // ARIMA(1,1,1): cumsum of ARMA(1,1) with phi=0.7, theta=0.3, e~N(0,1), seed=7
        // Series spec: np.random.seed(7); burn-in 50; cumsum of ARMA innovations; length 200.
        let y: [Double] = [
              2.0883410713,   4.3610873266,   6.1246342445,   7.6421250182,   8.6477646828,
              9.0025229482,   7.7230473658,   6.8985495414,   6.3771128509,   7.1767604548,
              7.7256210857,   6.0928081083,   4.2783174287,   4.6778280601,   5.0839235682,
              4.3633066260,   2.3983257859,  -0.3852551900,  -2.9489606532,  -6.0135946919,
             -7.0151940315,  -7.5496570706,  -7.8999233973,  -6.6742757938,  -3.8815320947,
             -1.6883489162,   0.1150336341,   2.2120202683,   3.7075634272,   2.9185741474,
              2.4875834994,   3.2766540617,   4.5128117610,   4.5792282782,   4.1526291308,
              3.2044317151,   2.0637513412,   2.4722469009,   4.6768389839,   7.3483455255,
              9.9679706824,  12.6429608177,  14.7062004227,  16.0711320287,  16.3302399271,
             16.2536544497,  18.4432313671,  21.5229586346,  23.5973624971,  24.4748834418,
             24.0830801158,  23.9238411399,  24.3162306551,  23.2652453669,  22.5829543652,
             21.6817789527,  22.3068356395,  23.3292356509,  25.8096989037,  27.6032154026,
             28.4332550475,  29.2276959912,  30.9296954567,  33.0037620609,  33.3918494867,
             33.4763542032,  33.6126224538,  33.2856150573,  32.2798840087,  33.1290179262,
             33.8572983589,  33.4040560803,  33.4701225269,  33.8381258666,  34.0591012313,
             34.9723416377,  36.3346258815,  37.7970024778,  39.8935235799,  41.9339849205,
             42.8304768370,  42.9106623654,  42.3371352314,  41.9890965941,  40.7304012707,
             37.6934931094,  35.1060845311,  31.9848244536,  27.4470081359,  24.0582740678,
             21.0871062591,  19.2900982564,  17.6501765293,  15.2210128065,  11.9404919702,
              8.7561449080,   6.0992409914,   3.3339333238,   0.8700720185,  -2.4750655652,
             -5.2630036177,  -9.0038623128, -12.3544143388, -14.0606635424, -13.9539109497,
            -11.8372222472, -11.3975137283, -10.7978788586,  -8.9442346955,  -6.3042421382,
             -5.0914040678,  -4.1094177320,  -1.4384325753,  -0.5079999063,   0.1704585695,
              1.9005712984,   2.7487463386,   3.7164618653,   3.4978613170,   2.2113645994,
              0.2500740758,  -1.1756456660,  -2.2069384416,  -2.0020225566,  -2.5126259890,
             -3.2834077761,  -2.9012612585,  -1.7846585848,  -2.1594448971,  -2.1539840315,
             -2.4290181999,  -2.3796809438,  -1.9288484881,  -0.3037427607,   1.3161868535,
              0.6312996938,  -1.7741227984,  -3.1377220733,  -4.1312115833,  -3.7661700801,
             -2.8958084920,  -2.9535894708,  -2.1363491171,  -2.0469565231,  -4.2792365200,
             -4.3938276193,  -5.7656516011,  -6.6627401926,  -6.1609265801,  -5.3809356575,
             -6.4742190964,  -6.7295216213,  -8.0452726247, -10.7423900069, -13.3951290566,
            -14.5720423111, -14.8592163113, -14.1594356539, -13.8700025386, -13.9726757935,
            -15.2333388461, -16.7738094726, -17.3540703953, -17.8821426497, -17.8662820891,
            -18.2858160818, -19.3438637449, -20.2230459372, -21.0216043797, -21.0466744977,
            -20.7082177659, -18.6774646858, -16.9533281456, -16.0395335686, -16.5081842234,
            -17.1255010334, -16.8679604716, -17.2231290381, -17.7435079898, -17.4041181747,
            -17.0778874251, -17.4067359462, -16.8195779505, -16.0196198427, -15.0339787919,
            -13.0536672544, -10.1707152246,  -8.1562582107,  -7.3757292751,  -6.7026790028,
             -6.2790386674,  -7.9221880124,  -8.3669892151,  -8.2763802091,  -7.4707675399,
             -7.8588418458,  -7.1941652705,  -6.9306934062,  -7.0304090165,  -8.5339347354,
        ]

        let result = arima(y, p: 1, d: 1, q: 1)
        XCTAssertNotNil(result, "ARIMA(1,1,1) fit should not return nil")
        guard let r = result else { return }

        // Oracle: statsmodels 0.14.6, SARIMAX(order=(1,1,1), trend='n').fit() on 200 obs.
        // AR(1) = 0.6900, MA(1) = 0.2817.
        //
        // Pre-fix symptom: old AR-only OLS on the differenced series gives AR ≈ 0.787
        // (the lag-1 autocorrelation of the differenced ARIMA(1,1,1) process).
        // The fixed Hannan-Rissanen estimate gives AR ≈ 0.694 (δ ≈ 0.004 from oracle).
        // Tolerance ±0.05 is narrow enough to exclude 0.787 while accepting 0.694.
        let arOracle = 0.6900
        let maOracle = 0.2817
        let tol = 0.05

        // Discriminating guard: the pre-fix symptom is AR inflated toward 1.
        // Old biased estimate on this differenced series: AR ≈ 0.787. Fixed ≈ 0.694.
        XCTAssertLessThan(r.arParams[0], 0.75,
            "AR(1) must be below 0.75; old AR-only estimator gives ≈ 0.787 on this series, " +
            "indicating MA contribution is being ignored in the differenced series.")

        XCTAssertEqual(r.arParams[0], arOracle, accuracy: tol,
            "ARIMA(1,1,1) AR(1) should be near oracle \(arOracle) (±\(tol)); got \(r.arParams[0]).")
        XCTAssertEqual(r.maParams[0], maOracle, accuracy: tol,
            "ARIMA(1,1,1) MA(1) should be near oracle \(maOracle) (±\(tol)); got \(r.maParams[0]).")
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
