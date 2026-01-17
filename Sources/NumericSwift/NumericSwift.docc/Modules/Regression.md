# Regression

Linear and nonlinear regression, ARIMA time series.

## Overview

The Regression module provides regression analysis tools including ordinary least squares, weighted least squares, generalized linear models, and ARIMA time series modeling, inspired by statsmodels.

## Linear Regression

### Ordinary Least Squares

```swift
let X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
let y = [2.0, 4.0, 5.0, 4.0, 5.0]

// Add constant column for intercept
let XWithConst = addConstant(X)

if let result = ols(y, XWithConst) {
    print(result.params)        // Coefficients [intercept, slope]
    print(result.rSquared)      // R-squared
    print(result.residuals)     // Residuals
    print(result.bse)           // Standard errors
    print(result.tvalues)       // t-statistics
    print(result.pvalues)       // p-values
}
```

### Weighted Least Squares

```swift
let weights = [1.0, 2.0, 1.0, 2.0, 1.0]

if let result = wls(y, X, weights: weights) {
    print(result.params)
}
```

### OLS Diagnostics

```swift
if let result = ols(y, X) {
    // Confidence intervals
    let ci = result.confInt(alpha: 0.05)

    // Robust standard errors
    let hc0 = result.hc0_se   // White's estimator
    let hc3 = result.hc3_se   // HC3 (small sample bias corrected)

    // Influence diagnostics
    print(result.hatDiag)            // Leverage
    print(result.studentizedResid)   // Studentized residuals
    print(result.cooksDistance)      // Cook's distance
    print(result.dffits)             // DFFITS

    // Model fit
    print(result.aic)                // Akaike Information Criterion
    print(result.bic)                // Bayesian Information Criterion
    print(result.fstat)              // F-statistic
    print(result.fPvalue)            // F-statistic p-value
    print(result.conditionNumber)    // Condition number
}
```

## Generalized Linear Models

```swift
// Logistic regression
if let result = glm(y, X, family: .binomial, link: .logit) {
    print(result.params)
    print(result.deviance)
    print(result.pearsonChi2)
}

// Poisson regression
if let result = glm(y, X, family: .poisson, link: .log) {
    print(result.params)
}

// Gamma regression
if let result = glm(y, X, family: .gamma, link: .inverse) {
    print(result.params)
}
```

### GLM Families and Links

Families: `.gaussian`, `.binomial`, `.poisson`, `.gamma`

Links: `.identity`, `.logit`, `.log`, `.inverse`, `.probit`

## ARIMA Time Series

### Model Fitting

```swift
let data = [112.0, 118.0, 132.0, 129.0, 121.0, ...]

// Fit ARIMA(p, d, q) model
if let result = arima(data, p: 1, d: 1, q: 1) {
    print(result.arCoeffs)    // AR coefficients
    print(result.maCoeffs)    // MA coefficients
    print(result.sigma2)      // Residual variance
    print(result.aic)         // Akaike Information Criterion
    print(result.bic)         // Bayesian Information Criterion
    print(result.residuals)   // Model residuals
}
```

### Forecasting

```swift
if let result = arima(data, p: 1, d: 1, q: 1) {
    // Generate forecasts
    let forecasts = arimaForecast(result, steps: 12)
    print(forecasts)
}
```

## Utility Functions

```swift
// Add constant column (intercept) to design matrix
let XWithConst = addConstant(X)

// Or for a 1D array
let XWithConst = addConstant(x)
```

## Topics

### Linear Regression

- ``ols(_:_:weights:)``
- ``wls(_:_:weights:)``
- ``OLSResult``
- ``addConstant(_:)-([[Double]])``
- ``addConstant(_:)-([Double])``

### Generalized Linear Models

- ``glm(_:_:family:link:maxiter:tol:)``
- ``GLMResult``
- ``GLMFamily``
- ``GLMLink``

### ARIMA

- ``arima(_:p:d:q:maxiter:tol:)``
- ``arimaForecast(_:steps:)``
- ``ARIMAResult``

### Statistical Functions

- ``tCDF(_:_:)``
- ``tPPF(_:_:)``
- ``tPDF(_:_:)``
- ``fCDF(_:_:_:)``
- ``standardNormalCDF(_:)``
