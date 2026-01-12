//
//  Regression.swift
//  NumericSwift
//
//  Regression modeling following statsmodels patterns.
//
//  Licensed under the MIT License.
//

import Foundation
import Accelerate

// MARK: - OLS Regression Result

/// Result of OLS/WLS regression.
public struct OLSResult {
    // Per-parameter metrics
    /// Estimated coefficients.
    public let params: [Double]
    /// Standard errors of coefficients.
    public let bse: [Double]
    /// t-statistics.
    public let tvalues: [Double]
    /// Two-tailed p-values.
    public let pvalues: [Double]
    /// 95% confidence intervals.
    public let confInt: [[Double]]

    // Robust standard errors (HC0-HC3)
    /// HC0 robust standard errors.
    public let bseHC0: [Double]
    /// HC1 robust standard errors.
    public let bseHC1: [Double]
    /// HC2 robust standard errors.
    public let bseHC2: [Double]
    /// HC3 robust standard errors.
    public let bseHC3: [Double]

    // Per-observation metrics
    /// Residuals.
    public let residuals: [Double]
    /// Fitted values.
    public let fittedValues: [Double]
    /// Hat matrix diagonal (leverage).
    public let hatDiag: [Double]
    /// Studentized residuals.
    public let studentizedResiduals: [Double]
    /// Cook's distance.
    public let cooksDistance: [Double]
    /// DFFITS.
    public let dffits: [Double]

    // Model-level metrics
    /// R-squared.
    public let rsquared: Double
    /// Adjusted R-squared.
    public let rsquaredAdj: Double
    /// F-statistic.
    public let fvalue: Double
    /// F-statistic p-value.
    public let fPvalue: Double
    /// Log-likelihood.
    public let llf: Double
    /// Akaike information criterion.
    public let aic: Double
    /// Bayesian information criterion.
    public let bic: Double
    /// Sum of squared residuals.
    public let ssr: Double
    /// Explained sum of squares.
    public let ess: Double
    /// Mean squared error.
    public let mse: Double
    /// Total sum of squares.
    public let tss: Double
    /// Number of observations.
    public let nobs: Int
    /// Degrees of freedom for model.
    public let dfModel: Int
    /// Degrees of freedom for residuals.
    public let dfResid: Int

    // Multicollinearity diagnostics
    /// Condition number of X'X.
    public let conditionNumber: Double
    /// Eigenvalues of X'X.
    public let eigenvalues: [Double]
}

// MARK: - OLS/WLS Regression

/// Fit Ordinary Least Squares regression.
///
/// - Parameters:
///   - y: Response variable
///   - X: Design matrix (each row is an observation)
///   - weights: Optional observation weights (for WLS)
/// - Returns: OLSResult or nil if fitting fails
public func ols(_ y: [Double], _ X: [[Double]], weights: [Double]? = nil) -> OLSResult? {
    let n = y.count
    guard n > 0, X.count == n else { return nil }

    let k = X.first?.count ?? 0
    guard k > 0, n > k else { return nil }

    // Apply weights if provided (WLS)
    var yWork = y
    var XWork = X
    if let w = weights, w.count == n {
        for i in 0..<n {
            let sqrtW = Darwin.sqrt(w[i])
            yWork[i] *= sqrtW
            for j in 0..<k {
                XWork[i][j] *= sqrtW
            }
        }
    }

    // Flatten X to column-major order for LAPACK
    var XFlat = [Double](repeating: 0.0, count: n * k)
    for j in 0..<k {
        for i in 0..<n {
            XFlat[j * n + i] = XWork[i][j]
        }
    }

    // Copy y to work array
    var yCopy = yWork
    if k > n {
        yCopy.append(contentsOf: [Double](repeating: 0.0, count: k - n))
    }

    // Call LAPACK dgels (least squares solver)
    var m = __CLPK_integer(n)
    var nCols = __CLPK_integer(k)
    var nrhs = __CLPK_integer(1)
    var lda = __CLPK_integer(n)
    var ldb = __CLPK_integer(max(n, k))
    var info: __CLPK_integer = 0

    // Query optimal workspace size
    var workQuery = [Double](repeating: 0.0, count: 1)
    var lwork: __CLPK_integer = -1
    dgels_(UnsafeMutablePointer(mutating: ("N" as NSString).utf8String),
           &m, &nCols, &nrhs, &XFlat, &lda, &yCopy, &ldb, &workQuery, &lwork, &info)

    lwork = __CLPK_integer(workQuery[0])
    var work = [Double](repeating: 0.0, count: Int(lwork))

    // Solve least squares
    dgels_(UnsafeMutablePointer(mutating: ("N" as NSString).utf8String),
           &m, &nCols, &nrhs, &XFlat, &lda, &yCopy, &ldb, &work, &lwork, &info)

    guard info == 0 else { return nil }

    // Extract parameters
    let params = Array(yCopy.prefix(k))

    // Compute fitted values and residuals (using original data)
    var fittedValues = [Double](repeating: 0.0, count: n)
    var residuals = [Double](repeating: 0.0, count: n)

    for i in 0..<n {
        var fitted = 0.0
        for j in 0..<k {
            fitted += X[i][j] * params[j]
        }
        fittedValues[i] = fitted
        residuals[i] = y[i] - fitted
    }

    // Compute statistics
    let yMean: Double = y.reduce(0.0) { $0 + $1 } / Double(n)

    var ssr = 0.0
    var ess = 0.0
    var tss = 0.0

    for i in 0..<n {
        ssr += residuals[i] * residuals[i]
        ess += (fittedValues[i] - yMean) * (fittedValues[i] - yMean)
        tss += (y[i] - yMean) * (y[i] - yMean)
    }

    let dfModel = k - 1
    let dfResid = n - k

    let rsquared = tss > 0 ? 1.0 - ssr / tss : 0.0
    let rsquaredAdj = dfResid > 0 ? 1.0 - (1.0 - rsquared) * Double(n - 1) / Double(dfResid) : 0.0

    let mse = dfResid > 0 ? ssr / Double(dfResid) : 0.0

    // Compute (X'X)^-1 for standard errors
    var XtX = [Double](repeating: 0.0, count: k * k)
    for i in 0..<k {
        for j in 0..<k {
            var sum = 0.0
            for obs in 0..<n {
                sum += X[obs][i] * X[obs][j]
            }
            XtX[j * k + i] = sum
        }
    }

    var kInt = __CLPK_integer(k)
    var ldaXtX = __CLPK_integer(k)
    var infoInv: __CLPK_integer = 0

    dpotrf_(UnsafeMutablePointer(mutating: ("U" as NSString).utf8String),
            &kInt, &XtX, &ldaXtX, &infoInv)

    var XtXInv = XtX
    if infoInv == 0 {
        dpotri_(UnsafeMutablePointer(mutating: ("U" as NSString).utf8String),
                &kInt, &XtXInv, &ldaXtX, &infoInv)

        for i in 0..<k {
            for j in i+1..<k {
                XtXInv[i * k + j] = XtXInv[j * k + i]
            }
        }
    }

    // Standard errors
    var bse = [Double](repeating: 0.0, count: k)
    if infoInv == 0 {
        for i in 0..<k {
            bse[i] = Darwin.sqrt(XtXInv[i * k + i] * mse)
        }
    }

    // t-values and p-values
    var tvalues = [Double](repeating: 0.0, count: k)
    var pvalues = [Double](repeating: 1.0, count: k)

    for i in 0..<k {
        if bse[i] > 0 {
            tvalues[i] = params[i] / bse[i]
            pvalues[i] = 2.0 * (1.0 - tCDF(abs(tvalues[i]), Double(dfResid)))
        }
    }

    // F-statistic
    var fvalue = 0.0
    var fPvalue = 1.0
    if dfModel > 0 && dfResid > 0 && ssr > 0 {
        fvalue = (ess / Double(dfModel)) / (ssr / Double(dfResid))
        fPvalue = 1.0 - fCDF(fvalue, Double(dfModel), Double(dfResid))
    }

    // Log-likelihood
    let llf = -Double(n) / 2.0 * (Darwin.log(2.0 * Double.pi) + Darwin.log(ssr / Double(n)) + 1.0)

    // Information criteria
    let aic = -2.0 * llf + 2.0 * Double(k)
    let bic = -2.0 * llf + Darwin.log(Double(n)) * Double(k)

    // Confidence intervals (95%)
    let tCrit = tPPF(0.975, Double(dfResid))
    var confInt: [[Double]] = []
    for i in 0..<k {
        confInt.append([params[i] - tCrit * bse[i], params[i] + tCrit * bse[i]])
    }

    // Hat matrix diagonal
    var hatDiag = [Double](repeating: 0.0, count: n)
    if infoInv == 0 {
        for i in 0..<n {
            var h_ii = 0.0
            for j in 0..<k {
                var sum = 0.0
                for l in 0..<k {
                    sum += XtXInv[l * k + j] * X[i][l]
                }
                h_ii += X[i][j] * sum
            }
            hatDiag[i] = h_ii
        }
    }

    // Studentized residuals
    var studentizedResiduals = [Double](repeating: 0.0, count: n)
    let rmse = Darwin.sqrt(mse)
    for i in 0..<n {
        let denom = rmse * Darwin.sqrt(max(1e-15, 1.0 - hatDiag[i]))
        studentizedResiduals[i] = denom > 1e-15 ? residuals[i] / denom : 0.0
    }

    // Cook's distance
    var cooksDistance = [Double](repeating: 0.0, count: n)
    if k > 0 && mse > 0 {
        for i in 0..<n {
            let h_ii = hatDiag[i]
            if h_ii < 1.0 - 1e-10 {
                let e2 = residuals[i] * residuals[i]
                cooksDistance[i] = (e2 / (Double(k) * mse)) * (h_ii / ((1.0 - h_ii) * (1.0 - h_ii)))
            }
        }
    }

    // DFFITS
    var dffits = [Double](repeating: 0.0, count: n)
    for i in 0..<n {
        let h_ii = hatDiag[i]
        if h_ii < 1.0 - 1e-10 {
            dffits[i] = studentizedResiduals[i] * Darwin.sqrt(h_ii / (1.0 - h_ii))
        }
    }

    // Condition number and eigenvalues
    var conditionNumber = Double.infinity
    var eigenvalues = [Double](repeating: 0.0, count: k)

    if k > 0 {
        var XtXCopy = [Double](repeating: 0.0, count: k * k)
        for i in 0..<k {
            for j in 0..<k {
                var sum = 0.0
                for obs in 0..<n {
                    sum += X[obs][i] * X[obs][j]
                }
                XtXCopy[j * k + i] = sum
            }
        }

        var kEig = __CLPK_integer(k)
        var ldaEig = __CLPK_integer(k)
        var eigenW = [Double](repeating: 0.0, count: k)
        var workEig = [Double](repeating: 0.0, count: 1)
        var lworkEig: __CLPK_integer = -1
        var infoEig: __CLPK_integer = 0

        dsyev_(UnsafeMutablePointer(mutating: ("V" as NSString).utf8String),
               UnsafeMutablePointer(mutating: ("U" as NSString).utf8String),
               &kEig, &XtXCopy, &ldaEig, &eigenW, &workEig, &lworkEig, &infoEig)

        lworkEig = __CLPK_integer(workEig[0])
        workEig = [Double](repeating: 0.0, count: Int(lworkEig))

        dsyev_(UnsafeMutablePointer(mutating: ("V" as NSString).utf8String),
               UnsafeMutablePointer(mutating: ("U" as NSString).utf8String),
               &kEig, &XtXCopy, &ldaEig, &eigenW, &workEig, &lworkEig, &infoEig)

        if infoEig == 0 {
            eigenvalues = eigenW.sorted(by: >)
            let maxEig = eigenvalues.first ?? 1.0
            let minEig = eigenvalues.last ?? 1.0
            if minEig > 1e-15 {
                conditionNumber = Darwin.sqrt(maxEig / minEig)
            }
        }
    }

    // Robust standard errors (HC0-HC3)
    var bseHC0 = [Double](repeating: 0.0, count: k)
    var bseHC1 = [Double](repeating: 0.0, count: k)
    var bseHC2 = [Double](repeating: 0.0, count: k)
    var bseHC3 = [Double](repeating: 0.0, count: k)

    if infoInv == 0 && n > k {
        func computeHCStdErrors(omega: [Double]) -> [Double] {
            var XtOmegaX = [Double](repeating: 0.0, count: k * k)
            for i in 0..<k {
                for j in 0..<k {
                    var sum = 0.0
                    for obs in 0..<n {
                        sum += X[obs][i] * omega[obs] * X[obs][j]
                    }
                    XtOmegaX[j * k + i] = sum
                }
            }

            var temp = [Double](repeating: 0.0, count: k * k)
            for i in 0..<k {
                for j in 0..<k {
                    var sum = 0.0
                    for l in 0..<k {
                        sum += XtOmegaX[l * k + i] * XtXInv[j * k + l]
                    }
                    temp[j * k + i] = sum
                }
            }

            var cov = [Double](repeating: 0.0, count: k * k)
            for i in 0..<k {
                for j in 0..<k {
                    var sum = 0.0
                    for l in 0..<k {
                        sum += XtXInv[l * k + i] * temp[j * k + l]
                    }
                    cov[j * k + i] = sum
                }
            }

            var se = [Double](repeating: 0.0, count: k)
            for i in 0..<k {
                se[i] = Darwin.sqrt(max(0.0, cov[i * k + i]))
            }
            return se
        }

        let omegaHC0 = residuals.map { $0 * $0 }
        bseHC0 = computeHCStdErrors(omega: omegaHC0)

        let hc1Factor = Double(n) / Double(n - k)
        bseHC1 = bseHC0.map { $0 * Darwin.sqrt(hc1Factor) }

        var omegaHC2 = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            let denom = max(1e-15, 1.0 - hatDiag[i])
            omegaHC2[i] = residuals[i] * residuals[i] / denom
        }
        bseHC2 = computeHCStdErrors(omega: omegaHC2)

        var omegaHC3 = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            let denom = max(1e-15, 1.0 - hatDiag[i])
            omegaHC3[i] = residuals[i] * residuals[i] / (denom * denom)
        }
        bseHC3 = computeHCStdErrors(omega: omegaHC3)
    }

    return OLSResult(
        params: params,
        bse: bse,
        tvalues: tvalues,
        pvalues: pvalues,
        confInt: confInt,
        bseHC0: bseHC0,
        bseHC1: bseHC1,
        bseHC2: bseHC2,
        bseHC3: bseHC3,
        residuals: residuals,
        fittedValues: fittedValues,
        hatDiag: hatDiag,
        studentizedResiduals: studentizedResiduals,
        cooksDistance: cooksDistance,
        dffits: dffits,
        rsquared: rsquared,
        rsquaredAdj: rsquaredAdj,
        fvalue: fvalue,
        fPvalue: fPvalue,
        llf: llf,
        aic: aic,
        bic: bic,
        ssr: ssr,
        ess: ess,
        mse: mse,
        tss: tss,
        nobs: n,
        dfModel: dfModel,
        dfResid: dfResid,
        conditionNumber: conditionNumber,
        eigenvalues: eigenvalues
    )
}

/// Fit Weighted Least Squares regression.
///
/// - Parameters:
///   - y: Response variable
///   - X: Design matrix
///   - weights: Observation weights
/// - Returns: OLSResult or nil if fitting fails
public func wls(_ y: [Double], _ X: [[Double]], weights: [Double]) -> OLSResult? {
    return ols(y, X, weights: weights)
}

// MARK: - GLM Result

/// GLM family types.
public enum GLMFamily: String {
    case gaussian
    case binomial
    case poisson
    case gamma
}

/// GLM link functions.
public enum GLMLink: String {
    case identity
    case logit
    case log
    case inverse
    case probit
}

/// Result of GLM fitting.
public struct GLMResult {
    /// Estimated coefficients.
    public let params: [Double]
    /// Standard errors.
    public let bse: [Double]
    /// z-statistics.
    public let zvalues: [Double]
    /// p-values.
    public let pvalues: [Double]
    /// Fitted mean values.
    public let mu: [Double]
    /// Linear predictor.
    public let eta: [Double]
    /// Response residuals.
    public let residResponse: [Double]
    /// Deviance.
    public let deviance: Double
    /// Null deviance.
    public let nullDeviance: Double
    /// Pearson chi-squared.
    public let pearsonChi2: Double
    /// Log-likelihood.
    public let llf: Double
    /// AIC.
    public let aic: Double
    /// BIC.
    public let bic: Double
    /// Number of observations.
    public let nobs: Int
    /// Degrees of freedom for model.
    public let dfModel: Int
    /// Degrees of freedom for residuals.
    public let dfResid: Int
    /// Whether IRLS converged.
    public let converged: Bool
    /// Number of iterations.
    public let iterations: Int
}

/// Fit Generalized Linear Model using IRLS.
///
/// - Parameters:
///   - y: Response variable
///   - X: Design matrix
///   - family: GLM family (default .gaussian)
///   - link: Link function (default: canonical for family)
///   - maxiter: Maximum iterations (default 100)
///   - tol: Convergence tolerance (default 1e-8)
/// - Returns: GLMResult or nil if fitting fails
public func glm(
    _ y: [Double],
    _ X: [[Double]],
    family: GLMFamily = .gaussian,
    link: GLMLink? = nil,
    maxiter: Int = 100,
    tol: Double = 1e-8
) -> GLMResult? {
    let n = y.count
    guard n > 0, X.count == n else { return nil }

    let k = X.first?.count ?? 0
    guard k > 0, n > k else { return nil }

    // Use canonical link if not specified
    let actualLink: GLMLink
    if let link = link {
        actualLink = link
    } else {
        switch family {
        case .gaussian: actualLink = .identity
        case .binomial: actualLink = .logit
        case .poisson: actualLink = .log
        case .gamma: actualLink = .inverse
        }
    }

    // Initialize mu
    var mu = [Double](repeating: 0.0, count: n)
    var eta = [Double](repeating: 0.0, count: n)

    for i in 0..<n {
        switch family {
        case .binomial:
            mu[i] = max(0.001, min(0.999, (y[i] + 0.5) / 2.0))
        case .poisson, .gamma:
            mu[i] = max(y[i], 0.1)
        case .gaussian:
            mu[i] = y[i]
        }
        eta[i] = applyLink(mu[i], link: actualLink)
    }

    var beta = [Double](repeating: 0.0, count: k)
    var converged = false
    var iterations = 0

    // IRLS main loop
    for iter in 0..<maxiter {
        iterations = iter + 1

        var weights = [Double](repeating: 0.0, count: n)
        var z = [Double](repeating: 0.0, count: n)

        for i in 0..<n {
            let dmuDeta = inverseLinkDerivative(eta[i], link: actualLink)
            let variance = varianceFunction(mu[i], family: family)

            if variance > 1e-10 && dmuDeta.isFinite {
                weights[i] = (dmuDeta * dmuDeta) / variance
            } else {
                weights[i] = 1e-10
            }

            if abs(dmuDeta) > 1e-10 {
                z[i] = eta[i] + (y[i] - mu[i]) / dmuDeta
            } else {
                z[i] = eta[i]
            }
        }

        guard let newBeta = solveWLS(X: X, y: z, weights: weights) else {
            return nil
        }

        var maxChange = 0.0
        for j in 0..<k {
            let change = abs(newBeta[j] - beta[j])
            if change > maxChange { maxChange = change }
        }

        beta = newBeta

        for i in 0..<n {
            var etaNew = 0.0
            for j in 0..<k {
                etaNew += X[i][j] * beta[j]
            }
            eta[i] = etaNew
            mu[i] = applyInverseLink(eta[i], link: actualLink, family: family)
        }

        if maxChange < tol {
            converged = true
            break
        }
    }

    // Residuals
    var residResponse = [Double](repeating: 0.0, count: n)
    for i in 0..<n {
        residResponse[i] = y[i] - mu[i]
    }

    // Deviance
    var deviance = 0.0
    var nullDeviance = 0.0
    let yMean: Double = y.reduce(0.0) { $0 + $1 } / Double(n)

    for i in 0..<n {
        deviance += unitDeviance(y: y[i], mu: mu[i], family: family)
        nullDeviance += unitDeviance(y: y[i], mu: yMean, family: family)
    }

    // Pearson chi-squared
    var pearsonChi2 = 0.0
    for i in 0..<n {
        let variance = varianceFunction(mu[i], family: family)
        if variance > 1e-10 {
            pearsonChi2 += (y[i] - mu[i]) * (y[i] - mu[i]) / variance
        }
    }

    // Standard errors
    let bse = computeGLMStandardErrors(X: X, mu: mu, family: family, link: actualLink)

    // z-values and p-values
    var zvalues = [Double](repeating: 0.0, count: k)
    var pvalues = [Double](repeating: 1.0, count: k)

    for j in 0..<k {
        if bse[j] > 1e-10 {
            zvalues[j] = beta[j] / bse[j]
            pvalues[j] = 2.0 * (1.0 - standardNormalCDF(abs(zvalues[j])))
        }
    }

    // Log-likelihood
    let llf = glmLogLikelihood(y: y, mu: mu, family: family)

    let dfModel = k - 1
    let dfResid = n - k
    let aic = -2.0 * llf + 2.0 * Double(k)
    let bic = -2.0 * llf + Double(k) * Darwin.log(Double(n))

    return GLMResult(
        params: beta,
        bse: bse,
        zvalues: zvalues,
        pvalues: pvalues,
        mu: mu,
        eta: eta,
        residResponse: residResponse,
        deviance: deviance,
        nullDeviance: nullDeviance,
        pearsonChi2: pearsonChi2,
        llf: llf,
        aic: aic,
        bic: bic,
        nobs: n,
        dfModel: dfModel,
        dfResid: dfResid,
        converged: converged,
        iterations: iterations
    )
}

// MARK: - Helper Functions

/// Solve weighted least squares.
private func solveWLS(X: [[Double]], y: [Double], weights: [Double]) -> [Double]? {
    let n = X.count
    let k = X.first?.count ?? 0
    guard n > 0, k > 0 else { return nil }

    var yW = [Double](repeating: 0.0, count: n)
    var XW = [[Double]](repeating: [Double](repeating: 0.0, count: k), count: n)

    for i in 0..<n {
        let sqrtW = Darwin.sqrt(weights[i])
        yW[i] = y[i] * sqrtW
        for j in 0..<k {
            XW[i][j] = X[i][j] * sqrtW
        }
    }

    var XFlat = [Double](repeating: 0.0, count: n * k)
    for j in 0..<k {
        for i in 0..<n {
            XFlat[j * n + i] = XW[i][j]
        }
    }

    var yCopy = yW
    if k > n {
        yCopy.append(contentsOf: [Double](repeating: 0.0, count: k - n))
    }

    var m = __CLPK_integer(n)
    var nCols = __CLPK_integer(k)
    var nrhs = __CLPK_integer(1)
    var lda = __CLPK_integer(n)
    var ldb = __CLPK_integer(max(n, k))
    var info: __CLPK_integer = 0

    var workQuery = [Double](repeating: 0.0, count: 1)
    var lwork: __CLPK_integer = -1
    dgels_(UnsafeMutablePointer(mutating: ("N" as NSString).utf8String),
           &m, &nCols, &nrhs, &XFlat, &lda, &yCopy, &ldb, &workQuery, &lwork, &info)

    lwork = __CLPK_integer(workQuery[0])
    var work = [Double](repeating: 0.0, count: Int(lwork))

    dgels_(UnsafeMutablePointer(mutating: ("N" as NSString).utf8String),
           &m, &nCols, &nrhs, &XFlat, &lda, &yCopy, &ldb, &work, &lwork, &info)

    guard info == 0 else { return nil }
    return Array(yCopy.prefix(k))
}

/// Apply link function.
private func applyLink(_ mu: Double, link: GLMLink) -> Double {
    switch link {
    case .identity: return mu
    case .logit:
        let p = max(1e-10, min(1.0 - 1e-10, mu))
        return Darwin.log(p / (1.0 - p))
    case .log: return Darwin.log(max(1e-10, mu))
    case .inverse: return 1.0 / max(1e-10, mu)
    case .probit:
        let p = max(1e-10, min(1.0 - 1e-10, mu))
        return erfinv(2.0 * p - 1.0) * Darwin.sqrt(2.0)
    }
}

/// Apply inverse link function.
private func applyInverseLink(_ eta: Double, link: GLMLink, family: GLMFamily) -> Double {
    var mu: Double
    switch link {
    case .identity: mu = eta
    case .logit:
        let expEta = Darwin.exp(min(700, max(-700, eta)))
        mu = expEta / (1.0 + expEta)
    case .log: mu = Darwin.exp(min(700, eta))
    case .inverse: mu = 1.0 / eta
    case .probit: mu = 0.5 * (1.0 + Darwin.erf(eta / Darwin.sqrt(2.0)))
    }

    switch family {
    case .binomial: mu = max(1e-10, min(1.0 - 1e-10, mu))
    case .poisson, .gamma: mu = max(1e-10, mu)
    case .gaussian: break
    }

    return mu
}

/// Derivative of inverse link.
private func inverseLinkDerivative(_ eta: Double, link: GLMLink) -> Double {
    switch link {
    case .identity: return 1.0
    case .logit:
        let expEta = Darwin.exp(min(700, max(-700, eta)))
        let p = expEta / (1.0 + expEta)
        return p * (1.0 - p)
    case .log: return Darwin.exp(min(700, eta))
    case .inverse: return -1.0 / (eta * eta)
    case .probit: return Darwin.exp(-eta * eta / 2.0) / Darwin.sqrt(2.0 * .pi)
    }
}

/// Variance function.
private func varianceFunction(_ mu: Double, family: GLMFamily) -> Double {
    switch family {
    case .gaussian: return 1.0
    case .binomial:
        let p = max(1e-10, min(1.0 - 1e-10, mu))
        return p * (1.0 - p)
    case .poisson: return max(1e-10, mu)
    case .gamma: return mu * mu
    }
}

/// Unit deviance.
private func unitDeviance(y: Double, mu: Double, family: GLMFamily) -> Double {
    let muBound = max(1e-10, mu)

    switch family {
    case .gaussian:
        return (y - mu) * (y - mu)
    case .binomial:
        let yBound = max(1e-10, min(1.0 - 1e-10, y))
        let muBoundBinom = max(1e-10, min(1.0 - 1e-10, mu))
        var dev = 0.0
        if y > 0 {
            dev += 2.0 * y * Darwin.log(yBound / muBoundBinom)
        }
        if y < 1 {
            dev += 2.0 * (1.0 - y) * Darwin.log((1.0 - yBound) / (1.0 - muBoundBinom))
        }
        return dev
    case .poisson:
        if y > 0 {
            return 2.0 * (y * Darwin.log(y / muBound) - (y - muBound))
        }
        return 2.0 * muBound
    case .gamma:
        return 2.0 * ((y - muBound) / muBound - Darwin.log(max(1e-10, y) / muBound))
    }
}

/// GLM log-likelihood.
private func glmLogLikelihood(y: [Double], mu: [Double], family: GLMFamily) -> Double {
    let n = y.count
    var llf = 0.0

    switch family {
    case .gaussian:
        var ssr = 0.0
        for i in 0..<n {
            ssr += (y[i] - mu[i]) * (y[i] - mu[i])
        }
        llf = -Double(n) / 2.0 * Darwin.log(2.0 * .pi) - Double(n) / 2.0 * Darwin.log(ssr / Double(n)) - Double(n) / 2.0
    case .binomial:
        for i in 0..<n {
            let p = max(1e-10, min(1.0 - 1e-10, mu[i]))
            if y[i] > 0 { llf += y[i] * Darwin.log(p) }
            if y[i] < 1 { llf += (1.0 - y[i]) * Darwin.log(1.0 - p) }
        }
    case .poisson:
        for i in 0..<n {
            let muBound = max(1e-10, mu[i])
            llf += y[i] * Darwin.log(muBound) - muBound - lgamma(y[i] + 1.0)
        }
    case .gamma:
        for i in 0..<n {
            let muBound = max(1e-10, mu[i])
            let yBound = max(1e-10, y[i])
            llf += -yBound / muBound - Darwin.log(muBound) - Darwin.log(yBound)
        }
    }

    return llf
}

/// Compute GLM standard errors.
private func computeGLMStandardErrors(X: [[Double]], mu: [Double], family: GLMFamily, link: GLMLink) -> [Double] {
    let n = X.count
    let k = X.first?.count ?? 0

    var weights = [Double](repeating: 0.0, count: n)
    for i in 0..<n {
        let eta = applyLink(mu[i], link: link)
        let dmuDeta = inverseLinkDerivative(eta, link: link)
        let variance = varianceFunction(mu[i], family: family)
        if variance > 1e-10 && dmuDeta.isFinite {
            weights[i] = (dmuDeta * dmuDeta) / variance
        } else {
            weights[i] = 1e-10
        }
    }

    // Compute X'WX
    var XtWX = [Double](repeating: 0.0, count: k * k)
    for i in 0..<k {
        for j in 0..<k {
            var sum = 0.0
            for obs in 0..<n {
                sum += X[obs][i] * weights[obs] * X[obs][j]
            }
            XtWX[j * k + i] = sum
        }
    }

    // Invert
    var kInt = __CLPK_integer(k)
    var lda = __CLPK_integer(k)
    var info: __CLPK_integer = 0

    dpotrf_(UnsafeMutablePointer(mutating: ("U" as NSString).utf8String),
            &kInt, &XtWX, &lda, &info)

    guard info == 0 else { return [Double](repeating: 0.0, count: k) }

    dpotri_(UnsafeMutablePointer(mutating: ("U" as NSString).utf8String),
            &kInt, &XtWX, &lda, &info)

    var bse = [Double](repeating: 0.0, count: k)
    for i in 0..<k {
        bse[i] = Darwin.sqrt(max(0, XtWX[i * k + i]))
    }
    return bse
}

// MARK: - Statistical Distribution Functions

/// Student's t-distribution CDF.
public func tCDF(_ t: Double, _ df: Double) -> Double {
    if t == 0 { return 0.5 }
    let x = df / (df + t * t)
    let p = 0.5 * betainc(df / 2.0, 0.5, x)
    return t > 0 ? 1.0 - p : p
}

/// Student's t-distribution PPF (inverse CDF).
public func tPPF(_ p: Double, _ df: Double) -> Double {
    guard p > 0 && p < 1 else {
        if p <= 0 { return -.infinity }
        if p >= 1 { return .infinity }
        return .nan
    }

    var x = Darwin.sqrt(2.0) * erfinv(2.0 * p - 1.0)

    for _ in 0..<50 {
        let cdfVal = tCDF(x, df)
        let pdfVal = tPDF(x, df)
        guard pdfVal > 1e-30 else { break }
        let dx = (cdfVal - p) / pdfVal
        x -= dx
        if abs(dx) < 1e-10 { break }
    }

    return x
}

/// Student's t-distribution PDF.
public func tPDF(_ x: Double, _ df: Double) -> Double {
    let coef = Darwin.exp(lgamma((df + 1) / 2.0) - lgamma(df / 2.0)) / Darwin.sqrt(df * .pi)
    return coef * Darwin.pow(1.0 + x * x / df, -(df + 1) / 2.0)
}

/// F-distribution CDF.
public func fCDF(_ x: Double, _ dfn: Double, _ dfd: Double) -> Double {
    guard x > 0 else { return 0.0 }
    let u = dfn * x / (dfn * x + dfd)
    return betainc(dfn / 2.0, dfd / 2.0, u)
}

/// Standard normal CDF.
public func standardNormalCDF(_ x: Double) -> Double {
    return 0.5 * (1.0 + Darwin.erf(x / Darwin.sqrt(2.0)))
}

/// Add constant column to design matrix.
///
/// - Parameter X: Design matrix or 1D array
/// - Returns: Matrix with column of ones prepended
public func addConstant(_ X: [[Double]]) -> [[Double]] {
    var result = [[Double]]()
    for row in X {
        result.append([1.0] + row)
    }
    return result
}

/// Add constant to 1D array.
public func addConstant(_ x: [Double]) -> [[Double]] {
    return x.map { [1.0, $0] }
}

// MARK: - ARIMA (AutoRegressive Integrated Moving Average)

/// Result of ARIMA model fitting.
public struct ARIMAResult {
    /// AR coefficients.
    public let arParams: [Double]
    /// MA coefficients.
    public let maParams: [Double]
    /// Standard errors for AR coefficients.
    public let arBse: [Double]
    /// Standard errors for MA coefficients.
    public let maBse: [Double]
    /// Residuals.
    public let residuals: [Double]
    /// Fitted values.
    public let fittedValues: [Double]
    /// Differenced series.
    public let yDiff: [Double]
    /// Estimated variance.
    public let sigma2: Double
    /// Log-likelihood.
    public let llf: Double
    /// AIC.
    public let aic: Double
    /// BIC.
    public let bic: Double
    /// Number of observations.
    public let nobs: Int
    /// Whether estimation converged.
    public let converged: Bool
    /// Number of iterations.
    public let iterations: Int
    /// Original series (for forecasting).
    public let original: [Double]
    /// Order (p, d, q).
    public let order: (p: Int, d: Int, q: Int)
}

/// Fit ARIMA(p, d, q) model using CSS (Conditional Sum of Squares).
///
/// - Parameters:
///   - y: Time series data
///   - p: AR order
///   - d: Differencing order
///   - q: MA order
///   - maxiter: Maximum iterations (default 100)
///   - tol: Convergence tolerance (default 1e-8)
/// - Returns: ARIMAResult or nil if fitting fails
public func arima(_ y: [Double], p: Int, d: Int, q: Int, maxiter: Int = 100, tol: Double = 1e-8) -> ARIMAResult? {
    let originalN = y.count
    guard originalN > 2 else { return nil }

    // Apply differencing
    var yDiff = y
    for _ in 0..<d {
        if yDiff.count < 2 { break }
        var diffed = [Double](repeating: 0.0, count: yDiff.count - 1)
        for i in 0..<(yDiff.count - 1) {
            diffed[i] = yDiff[i + 1] - yDiff[i]
        }
        yDiff = diffed
    }

    let n = yDiff.count
    guard n > max(p, q) else { return nil }

    // Initialize parameters
    var arParams = [Double](repeating: 0.0, count: p)
    var maParams = [Double](repeating: 0.0, count: q)

    // Initialize AR params with small values
    for i in 0..<p {
        arParams[i] = 0.1 / Double(i + 1)
    }

    // CSS estimation using iterative refinement
    var converged = false
    var iterations = 0

    for iter in 0..<maxiter {
        iterations = iter + 1

        // Compute residuals with current parameters
        var resid = computeARMAResiduals(y: yDiff, ar: arParams, ma: maParams)

        // Update AR parameters using least squares
        if p > 0 {
            if let newParams = estimateARParams(y: yDiff, p: p) {
                var maxChange = 0.0
                for i in 0..<p {
                    let change = abs(newParams[i] - arParams[i])
                    if change > maxChange { maxChange = change }
                }
                arParams = newParams

                // Recompute residuals with updated AR
                resid = computeARMAResiduals(y: yDiff, ar: arParams, ma: maParams)

                if maxChange < tol && q == 0 {
                    converged = true
                    break
                }
            }
        }

        // Update MA parameters using innovations algorithm approximation
        if q > 0 {
            if let newParams = estimateMAParams(resid: resid, q: q) {
                var maxChange = 0.0
                for j in 0..<q {
                    let change = abs(newParams[j] - maParams[j])
                    if change > maxChange { maxChange = change }
                }
                maParams = newParams

                if maxChange < tol {
                    converged = true
                    break
                }
            }
        }

        // Check overall convergence
        if p == 0 && q == 0 {
            converged = true
            break
        }
    }

    // Final residuals
    let resid = computeARMAResiduals(y: yDiff, ar: arParams, ma: maParams)

    // Compute fitted values
    var fitted = [Double](repeating: 0.0, count: n)
    for i in 0..<n {
        fitted[i] = yDiff[i] - resid[i]
    }

    // Compute sigma^2 (variance of residuals)
    let startIdx = max(p, q)
    var ssr = 0.0
    var validCount = 0
    for i in startIdx..<n {
        ssr += resid[i] * resid[i]
        validCount += 1
    }
    let sigma2 = validCount > 0 ? ssr / Double(validCount) : 0.0

    // Compute log-likelihood (Gaussian)
    let nEff = Double(validCount)
    var llf = -nEff / 2.0 * Darwin.log(2.0 * .pi)
    if sigma2 > 0 {
        llf -= nEff / 2.0 * Darwin.log(sigma2)
    }
    llf -= ssr / (2.0 * max(sigma2, 1e-10))

    // Number of parameters
    let numParams = p + q + 1  // AR + MA + sigma2

    // AIC and BIC
    let aic = -2.0 * llf + 2.0 * Double(numParams)
    let bic = -2.0 * llf + Double(numParams) * Darwin.log(nEff)

    // Approximate standard errors (using residual variance)
    var arBse = [Double](repeating: 0.0, count: p)
    var maBse = [Double](repeating: 0.0, count: q)

    // Simple approximation: SE ≈ sqrt(sigma2 / n)
    let baseSE = Darwin.sqrt(max(sigma2, 1e-10) / max(Double(n), 1.0))
    for i in 0..<p {
        arBse[i] = baseSE
    }
    for j in 0..<q {
        maBse[j] = baseSE
    }

    return ARIMAResult(
        arParams: arParams,
        maParams: maParams,
        arBse: arBse,
        maBse: maBse,
        residuals: resid,
        fittedValues: fitted,
        yDiff: yDiff,
        sigma2: sigma2,
        llf: llf,
        aic: aic,
        bic: bic,
        nobs: n,
        converged: converged,
        iterations: iterations,
        original: y,
        order: (p, d, q)
    )
}

/// Forecast future values from an ARIMA model.
///
/// - Parameters:
///   - result: ARIMA result
///   - steps: Number of steps ahead
/// - Returns: Array of forecasted values
public func arimaForecast(_ result: ARIMAResult, steps: Int) -> [Double] {
    guard steps > 0 else { return [] }

    let p = result.order.p
    let q = result.order.q
    let d = result.order.d

    // Start with the differenced series
    var yExt = result.yDiff
    var residExt = result.residuals

    // Forecast on differenced data
    var forecasts = [Double](repeating: 0.0, count: steps)

    for s in 0..<steps {
        var pred = 0.0

        // AR component
        for i in 0..<p {
            let idx = yExt.count - 1 - i
            if idx >= 0 {
                pred += result.arParams[i] * yExt[idx]
            }
        }

        // MA component (use 0 for future residuals)
        for j in 0..<q {
            let idx = residExt.count - 1 - j
            if idx >= 0 {
                pred += result.maParams[j] * residExt[idx]
            }
        }

        forecasts[s] = pred
        yExt.append(pred)
        residExt.append(0.0)
    }

    // Integrate back if differenced
    if d > 0 {
        var integrated = forecasts
        let original = result.original

        for _ in 0..<d {
            // Get the last value from original to integrate
            let lastVal = original.last ?? 0.0
            var cumsum = [Double](repeating: 0.0, count: integrated.count)
            var prev = lastVal
            for i in 0..<integrated.count {
                prev = prev + integrated[i]
                cumsum[i] = prev
            }
            integrated = cumsum
        }
        return integrated
    }

    return forecasts
}

// MARK: - ARIMA Helper Functions

/// Compute ARMA residuals given parameters.
private func computeARMAResiduals(y: [Double], ar: [Double], ma: [Double]) -> [Double] {
    let n = y.count
    let p = ar.count
    let q = ma.count
    var resid = [Double](repeating: 0.0, count: n)

    for t in 0..<n {
        var pred = 0.0

        // AR component: sum of phi_i * y_{t-i}
        for i in 0..<p {
            let idx = t - i - 1
            if idx >= 0 {
                pred += ar[i] * y[idx]
            }
        }

        // MA component: sum of theta_j * e_{t-j}
        for j in 0..<q {
            let idx = t - j - 1
            if idx >= 0 {
                pred += ma[j] * resid[idx]
            }
        }

        resid[t] = y[t] - pred
    }

    return resid
}

/// Estimate AR parameters using OLS on lagged values.
private func estimateARParams(y: [Double], p: Int) -> [Double]? {
    let n = y.count
    guard p > 0 && n > p else { return nil }

    // Build design matrix with lagged y values
    let nEff = n - p
    guard nEff > 0 else { return nil }

    var X = [[Double]](repeating: [Double](repeating: 0.0, count: p), count: nEff)
    var yTarget = [Double](repeating: 0.0, count: nEff)

    for t in 0..<nEff {
        let actualT = t + p
        yTarget[t] = y[actualT]
        for i in 0..<p {
            X[t][i] = y[actualT - i - 1]
        }
    }

    // Solve using OLS
    return solveSimpleOLS(X: X, y: yTarget)
}

/// Estimate MA parameters using autocorrelation matching.
private func estimateMAParams(resid: [Double], q: Int) -> [Double]? {
    let n = resid.count
    guard q > 0 && n > q else { return nil }

    var maParams = [Double](repeating: 0.0, count: q)

    // Compute autocorrelations of residuals
    var gamma = [Double](repeating: 0.0, count: q + 1)
    for lag in 0...q {
        var sum = 0.0
        var count = 0
        for t in lag..<n {
            sum += resid[t] * resid[t - lag]
            count += 1
        }
        gamma[lag] = count > 0 ? sum / Double(count) : 0.0
    }

    // Simple estimation: theta_j ≈ -gamma[j] / gamma[0]
    if abs(gamma[0]) > 1e-10 {
        for j in 0..<q {
            maParams[j] = -gamma[j + 1] / gamma[0]
            // Bound to ensure invertibility
            maParams[j] = max(-0.99, min(0.99, maParams[j]))
        }
    }

    return maParams
}

/// Simple OLS solver for ARIMA.
private func solveSimpleOLS(X: [[Double]], y: [Double]) -> [Double]? {
    let n = X.count
    let k = X.first?.count ?? 0
    guard n > k, k > 0 else { return nil }

    // Flatten X to column-major
    var XFlat = [Double](repeating: 0.0, count: n * k)
    for j in 0..<k {
        for i in 0..<n {
            XFlat[j * n + i] = X[i][j]
        }
    }

    var yCopy = y
    if k > n {
        yCopy.append(contentsOf: [Double](repeating: 0.0, count: k - n))
    }

    var m = __CLPK_integer(n)
    var nCols = __CLPK_integer(k)
    var nrhs = __CLPK_integer(1)
    var lda = __CLPK_integer(n)
    var ldb = __CLPK_integer(max(n, k))
    var info: __CLPK_integer = 0

    var workQuery = [Double](repeating: 0.0, count: 1)
    var lwork: __CLPK_integer = -1
    dgels_(UnsafeMutablePointer(mutating: ("N" as NSString).utf8String),
           &m, &nCols, &nrhs, &XFlat, &lda, &yCopy, &ldb, &workQuery, &lwork, &info)

    lwork = __CLPK_integer(workQuery[0])
    var work = [Double](repeating: 0.0, count: Int(lwork))

    dgels_(UnsafeMutablePointer(mutating: ("N" as NSString).utf8String),
           &m, &nCols, &nrhs, &XFlat, &lda, &yCopy, &ldb, &work, &lwork, &info)

    guard info == 0 else { return nil }
    return Array(yCopy.prefix(k))
}
