# Regression

Linear and nonlinear regression, ARIMA time series.

## Overview

The Regression module provides regression analysis tools including ordinary least squares, polynomial regression, and ARIMA time series modeling, inspired by statsmodels.

## Linear Regression

### Ordinary Least Squares

```swift
let x = [[1.0], [2.0], [3.0], [4.0], [5.0]]
let y = [2.0, 4.0, 5.0, 4.0, 5.0]

let result = ols(y: y, x: x)

print(result.coefficients)  // [intercept, slope]
print(result.rSquared)      // R-squared
print(result.residuals)     // Residuals
print(result.standardErrors) // Standard errors
print(result.tValues)       // t-statistics
print(result.pValues)       // p-values
```

### Multiple Regression

```swift
let x = [
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
]
let y = [3.0, 5.0, 7.0, 9.0]

let result = ols(y: y, x: x, addConstant: true)
```

## Polynomial Regression

```swift
let x = [1.0, 2.0, 3.0, 4.0, 5.0]
let y = [1.0, 4.0, 9.0, 16.0, 25.0]

// Fit polynomial of degree 2
let coeffs = polyfit(x: x, y: y, degree: 2)
// coeffs ≈ [0, 0, 1] for y = x^2

// Evaluate fitted polynomial
let yPred = polyval(coeffs, at: 3.0)  // 9.0
```

## Nonlinear Regression

```swift
// Define model: y = a * exp(b * x)
func model(_ x: Double, _ params: [Double]) -> Double {
    return params[0] * exp(params[1] * x)
}

let result = nonlinearFit(
    model: model,
    x: xData,
    y: yData,
    initialGuess: [1.0, 0.1]
)

print(result.parameters)
print(result.residuals)
print(result.rSquared)
```

## ARIMA Time Series

### Model Fitting

```swift
let data = [112.0, 118.0, 132.0, 129.0, 121.0, ...]  // Time series

// Fit ARIMA(p, d, q) model
let model = ARIMA(data, order: (1, 1, 1))
let fitted = model.fit()

print(fitted.arCoeffs)    // AR coefficients
print(fitted.maCoeffs)    // MA coefficients
print(fitted.aic)         // Akaike Information Criterion
print(fitted.bic)         // Bayesian Information Criterion
```

### Forecasting

```swift
// Generate forecasts
let forecasts = fitted.forecast(steps: 12)

print(forecasts.values)         // Point forecasts
print(forecasts.confidenceInterval) // Confidence intervals
```

### Diagnostics

```swift
// Residual analysis
let residuals = fitted.residuals
let ljungBox = fitted.ljungBoxTest(lags: 10)

print(ljungBox.statistic)
print(ljungBox.pValue)
```

## Topics

### Linear Regression

- ``ols(y:x:addConstant:)``
- ``OLSResult``

### Polynomial Regression

- ``polyfit(x:y:degree:)``

### Nonlinear Regression

- ``nonlinearFit(model:x:y:initialGuess:)``
- ``NonlinearFitResult``

### ARIMA

- ``ARIMA``
- ``ARIMAFitted``
- ``ARIMAForecast``
