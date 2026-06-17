//
//  IntegrationStiff.swift
//  Sources/NumericSwift/
//
//  Stiff ODE solver: fixed-order BDF-1 (implicit Euler) with adaptive step size.
//  Called by solveIVP(method: .bdf).
//
//  Architecture
//  ─────────────
//    bdfSolveIVP       — driver; step control, history, output
//    bdfAttemptStep    — one BDF-1 step: Newton solve + Milne error estimate
//    bdfJacobian       — finite-difference Jacobian (per-variable step, O(√ε))
//    bdfNewtonSolve    — simplified Newton iteration for the implicit BDF system
//    interpolateOutput — tEval interpolation
//
//  Error estimate
//  ──────────────
//  The local truncation error (LTE) is estimated by the MILNE DEVICE:
//
//    err_i  ≈  C_1 · |y_{n+1,i} − ŷ_{n+1,i}|
//
//  where ŷ_{n+1} = y_n + h · f_n  (explicit Euler / AB-1 predictor) and
//  C_1 = 1/2 (Hairer & Wanner, §III.1).
//
//  Stiff behaviour: during the stiff transient |ŷ − yNew| = h²λ²/(1+hλ) → 0 as
//  hλ → ∞, keeping errNorm bounded and preventing step-size collapse.  After the
//  transient, |ŷ − yNew| = O(h²) and the PI controller grows h to O(√rtol).
//
//  Accuracy caveat: BDF-1 is a first-order method; adaptive LTE control produces
//  global error O(h) = O(√rtol) for non-stiff problems — not O(rtol).  Upgrading
//  to the Nordsieck divided-difference framework (as in MATLAB ODE15S / SciPy BDF)
//  would give variable-order BDF-2 with global error O(rtol^{2/3}) and properly
//  stiff-stable high-order error estimation; that is deferred to a future milestone.
//
//  References:
//    Shampine & Reichelt, "The MATLAB ODE Suite", SIAM J. Sci. Comput. 18(1) 1997.
//    Hairer & Wanner, "Solving ODEs II", Springer 1996, §III.1–§III.8.
//    Ascher & Petzold, "Computer Methods for ODEs and DAEs", SIAM 1998, §4.3.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - BDF formula coefficients

/// BDF-k formula: Σ_{j=0}^{k} α_j · y_{n+1−j} = h · f(y_{n+1}).
///
/// `bdfAlpha[k−1]` = [α_0, α_1, …, α_k].
///
/// Reference: Hairer & Wanner, Table III.1.1.
private let bdfAlpha: [[Double]] = [
  [1.0, -1.0],             // k=1 (implicit Euler / BDF-1)
  [3.0/2, -2.0, 1.0/2],   // k=2
]

/// Error constant for the Milne-device LTE estimate.
///
/// C_k = 1 / (2 · α_0(k)), calibrated at BDF-1 to recover the classical
/// Milne estimate, conservative at BDF-2.
private let bdfErrorConst: [Double] = [
  1.0 / (2.0 * 1.0),    // k=1: α_0=1   → 0.500
  1.0 / (2.0 * 1.5),    // k=2: α_0=3/2 → 0.333
]

// MARK: - Finite-difference Jacobian

/// J[i][j] = ∂f_i/∂y_j by forward finite differences.
///
/// Step size δ_j = √ε_machine · max(1, |y_j|); consistent with
/// `Optimization.swift` curveFit Jacobian.
private func bdfJacobian(
  _ f: ([Double], Double) -> [Double],
  y: [Double],
  t: Double,
  f0: [Double]
) -> [[Double]] {
  let n = y.count
  let sqrtEps = (2.22e-16 as Double).squareRoot()   // ≈ 1.49e-8
  var J = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
  for j in 0..<n {
    let delta = sqrtEps * max(1.0, abs(y[j]))
    var yPert = y
    yPert[j] += delta
    let fPert = f(yPert, t)
    for i in 0..<n { J[i][j] = (fPert[i] - f0[i]) / delta }
  }
  return J
}

// MARK: - Newton iteration

private struct NewtonResult {
  var y: [Double]
  var fNew: [Double]
  var nfev: Int
  var converged: Bool
}

/// Simplified Newton for G(y) = α_0 · y − h · f(y, t) + rhs0 = 0.
///
/// Frozen Jacobian J_G = α_0 · I − h · J_f; factorised once per step.
///
/// Reference: Hairer & Wanner §III.8.
private func bdfNewtonSolve(
  _ f: ([Double], Double) -> [Double],
  tNew: Double,
  yPred: [Double],
  rhs0: [Double],
  alpha0: Double,
  h: Double,
  J_f: [[Double]],
  rtol: Double,
  atol: Double,
  maxIter: Int = 12
) -> NewtonResult {
  let n = yPred.count
  var yNew = yPred
  var nfev = 0

  let jGFlat = (0..<n).flatMap { i -> [Double] in
    (0..<n).map { j in (i == j ? alpha0 : 0.0) - h * J_f[i][j] }
  }
  let jMat = LinAlg.Matrix(rows: n, cols: n, data: jGFlat)
  var lastFNew = [Double](repeating: 0, count: n)

  for _ in 0..<maxIter {
    let fEval = f(yNew, tNew)
    nfev += 1
    lastFNew = fEval

    var residual = [Double](repeating: 0, count: n)
    for i in 0..<n {
      residual[i] = alpha0 * yNew[i] - h * fEval[i] + rhs0[i]
    }

    var normRes: Double = 0
    for i in 0..<n {
      let sc = atol + rtol * abs(yNew[i])
      normRes += (residual[i] / sc) * (residual[i] / sc)
    }
    normRes = (normRes / Double(n)).squareRoot()
    if normRes < 0.1 {
      return NewtonResult(y: yNew, fNew: lastFNew, nfev: nfev, converged: true)
    }

    let bMat = LinAlg.Matrix(rows: n, cols: 1, data: residual)
    guard let delta = (try? LinAlg.solve(jMat, bMat)) ?? nil else {
      return NewtonResult(y: yNew, fNew: lastFNew, nfev: nfev, converged: false)
    }
    for i in 0..<n { yNew[i] -= delta[i, 0] }
  }

  let fFinal = f(yNew, tNew)
  nfev += 1
  var normFinal: Double = 0
  for i in 0..<n {
    let sc = atol + rtol * abs(yNew[i])
    let res = alpha0 * yNew[i] - h * fFinal[i] + rhs0[i]
    normFinal += (res / sc) * (res / sc)
  }
  normFinal = (normFinal / Double(n)).squareRoot()
  return NewtonResult(y: yNew, fNew: fFinal, nfev: nfev, converged: normFinal < 1.0)
}

// MARK: - BDF step attempt

private struct BDFStepOutcome {
  var yNew: [Double]
  var fNew: [Double]
  var errNorm: Double
  var nfev: Int
  var converged: Bool
}

/// Attempt one BDF-1 step from the current history.
///
/// - Jacobian: evaluated at explicit Euler prediction y_n + h·f_n.
/// - Error estimate: Milne device  err = C_1 · |yNew − (y_n + h·f_n)|.
///
/// Stiff behaviour: during stiff transient |ŷ − yNew| = h²λ²/(1+hλ) → 0 as
/// hλ → ∞, keeping errNorm bounded.  After the transient, |ŷ − yNew| = O(h²) and
/// the PI controller grows h to O(√rtol).
private func bdfAttemptStep(
  _ f: ([Double], Double) -> [Double],
  t: Double,
  tNew: Double,
  yHistory: [[Double]],
  tHistory: [Double],
  fHistory: [[Double]],
  h: Double,
  order k: Int,
  rtol: Double,
  atol: Double,
  jacobianFn: (([Double], Double) -> [[Double]])?
) -> BDFStepOutcome {
  let n = yHistory[0].count
  let alpha = bdfAlpha[k - 1]
  let yn = yHistory[0]
  let fn = fHistory[0]

  // BDF residual constant term: Σ_{j=1}^{k} α_j · y_{n+1−j}
  var rhs0 = [Double](repeating: 0, count: n)
  for j in 1...k {
    for i in 0..<n { rhs0[i] += alpha[j] * yHistory[j - 1][i] }
  }

  // Explicit Euler predictor (error estimate anchor — stiff-stable).
  var yPredEE = yn
  for i in 0..<n { yPredEE[i] += h * fn[i] }

  // Newton warm start: step-size–scaled linear extrapolation (k≥2),
  // explicit Euler otherwise.  Warm start affects only Newton convergence
  // speed, not the solution or error estimate.
  // (Currently k=1 always so EE branch is always taken.)
  var yPredNewton: [Double]
  if k >= 2 && tHistory.count >= 2 {
    let hPrev = abs(tHistory[0] - tHistory[1])
    if hPrev > 1e-20 {
      let ratio = abs(h) / hPrev
      yPredNewton = yn
      for i in 0..<n { yPredNewton[i] += ratio * (yn[i] - yHistory[1][i]) }
    } else {
      yPredNewton = yPredEE
    }
  } else {
    yPredNewton = yPredEE
  }

  // Jacobian at the EE predictor (stiff-stable evaluation point).
  let f0 = f(yPredEE, tNew)
  let J_f: [[Double]] = jacobianFn != nil
    ? jacobianFn!(yPredEE, tNew)
    : bdfJacobian(f, y: yPredEE, t: tNew, f0: f0)

  let newton = bdfNewtonSolve(
    f, tNew: tNew, yPred: yPredNewton,
    rhs0: rhs0, alpha0: alpha[0], h: h,
    J_f: J_f, rtol: rtol, atol: atol
  )
  let nfev = newton.nfev + 1

  guard newton.converged else {
    return BDFStepOutcome(yNew: yn, fNew: fn, errNorm: 10.0,
                          nfev: nfev, converged: false)
  }

  let yNew = newton.y
  let fNew = newton.fNew

  // Milne error estimate: C_1 · WRMS(y_{n+1} − ŷ_EE).
  //
  // |yNew − yPredEE| = O(h^2) for smooth problems; C_1 = 0.5 gives the
  // classical Milne bound (BDF-1 LTE = h^2/2 * y'').
  //
  // Stiff stability (BDF-1): for y'=λy, λ << 0:
  //   yPredEE = yn + h*λ*yn,  yNew = yn / (1 − h*λ)
  //   |yNew − yPredEE| = h²λ²*yn / (1 − hλ) → 0 as hλ → −∞
  // So errNorm stays bounded during the stiff transient and the PI controller
  // does NOT collapse.  (This argument does NOT extend to BDF-2 with the EE
  // predictor when history comes from variable-step or mixed-order steps.)
  //
  // Reference: Shampine & Reichelt (1997), §2; Ascher & Petzold §4.3.
  let ck = bdfErrorConst[k - 1]
  var errNorm: Double = 0
  for i in 0..<n {
    let errI = ck * (yNew[i] - yPredEE[i])
    let sc = atol + rtol * max(abs(yNew[i]), abs(yn[i]))
    errNorm += (errI / sc) * (errI / sc)
  }
  errNorm = (errNorm / Double(n)).squareRoot()

  return BDFStepOutcome(yNew: yNew, fNew: fNew, errNorm: errNorm,
                        nfev: nfev, converged: true)
}

// MARK: - BDF IVP driver

/// Internal BDF driver — called by `solveIVP(method: .bdf)`.
///
/// Fixed-order BDF-1 (implicit Euler) with the Milne stiff-stable error
/// estimate.  Variable-order BDF-2 and higher are omitted: the Milne
/// error estimate `C_k|y_{n+1}−ŷ_EE|` is NOT O(h^{k+1}) for BDF-k>1
/// when the history comes from mixed-order or variable-step history, because
/// the BDF-2 corrector at h→0 approaches a constant that differs from yPredEE
/// by O(y_n − y_{n-1}), making the estimate step-size-independent and causing
/// the step controller to collapse.  The Nordsieck divided-difference framework
/// (as in MATLAB ODE15S and SciPy BDF) is required for stiff-stable high-order
/// error estimation across variable step sizes; implementing it is deferred.
/// BDF-1 is A-stable, L-stable, and sufficient for all test cases in the suite.
func bdfSolveIVP(
  _ f: @escaping ([Double], Double) -> [Double],
  tSpan: (Double, Double),
  y0: [Double],
  tEval: [Double]?,
  maxStep: Double,
  rtol: Double,
  atol: Double,
  firstStep: Double,
  jacobian: (([Double], Double) -> [[Double]])?
) -> ODEResult {
  let t0 = tSpan.0
  let tf = tSpan.1
  let n = y0.count
  let direction: Double = tf >= t0 ? 1.0 : -1.0

  let f0 = f(y0, t0)
  var nfev = 1

  let maxOrder = 1
  var yHistory: [[Double]] = [y0]
  var fHistory: [[Double]] = [f0]
  var tHistory: [Double] = [t0]

  var tList: [Double] = [t0]
  var yList: [[Double]] = [y0]

  var t = t0
  var order = 1
  var h = direction * min(abs(firstStep), maxStep)

  let hMin = abs(tf - t0) * 1e-14
  if abs(h) < hMin { h = direction * hMin }

  let maxSteps = 500_000
  var stepCount = 0
  var successiveAccepts = 0

  while direction * (tf - t) > 1e-12 * abs(tf) && stepCount < maxSteps {
    stepCount += 1
    if direction * (t + h - tf) > 0 { h = tf - t }

    let tNew = t + h
    let outcome = bdfAttemptStep(
      f, t: t, tNew: tNew,
      yHistory: yHistory, tHistory: tHistory, fHistory: fHistory,
      h: h, order: order, rtol: rtol, atol: atol, jacobianFn: jacobian
    )
    nfev += outcome.nfev

    if outcome.converged && outcome.errNorm <= 1.0 {
      t = tNew
      yHistory.insert(outcome.yNew, at: 0)
      fHistory.insert(outcome.fNew, at: 0)
      tHistory.insert(tNew, at: 0)
      if yHistory.count > maxOrder + 1 {
        yHistory.removeLast(); fHistory.removeLast(); tHistory.removeLast()
      }
      tList.append(t); yList.append(outcome.yNew)
      successiveAccepts += 1

      // PI step controller: exponent 1/(k+1) reflects BDF-k convergence rate.
      // For BDF-1: exp = 0.5, consistent with EE Milne estimate O(h^2).
      let exp = 1.0 / Double(order + 1)
      let factor = min(5.0, 0.9 * pow(max(1e-10, 1.0 / outcome.errNorm), exp))
      h = direction * min(abs(h * factor), maxStep)

      // Order increase logic: inactive because maxOrder = 1.  Infrastructure
      // retained for a future Nordsieck variable-order upgrade.
      if successiveAccepts >= 2 && order < maxOrder && yHistory.count > order {
        order += 1
      }

    } else {
      successiveAccepts = 0
      if !outcome.converged {
        // Newton failure: halve step and optionally reduce order.
        h = direction * max(abs(h) * 0.5, hMin)
        if order > 1 && abs(h) <= hMin * 4 { order -= 1 }
      } else {
        // Error too large: shrink by PI formula.
        let exp = 1.0 / Double(order + 1)
        let factor = max(0.1, 0.9 * pow(max(1e-10, 1.0 / outcome.errNorm), exp))
        h = direction * min(abs(h * factor), maxStep)
      }
      if abs(h) <= hMin {
        return ODEResult(
          t: tList, y: yList, success: false,
          message: "BDF: step size h ≤ \(hMin). System may be singular or extremely stiff.",
          nfev: nfev
        )
      }
    }
  }

  let success = direction * (tf - t) <= 1e-12 * abs(tf)
  let (resultT, resultY) = interpolateOutput(tList: tList, yList: yList,
                                             tEval: tEval, n: n)
  return ODEResult(
    t: resultT, y: resultY, success: success,
    message: success ? "BDF integration successful"
                     : "BDF reached maximum step count (\(maxSteps))",
    nfev: nfev
  )
}

// MARK: - Output interpolation

/// Piecewise-linear interpolation at `tEval` times.
private func interpolateOutput(
  tList: [Double],
  yList: [[Double]],
  tEval: [Double]?,
  n: Int
) -> ([Double], [[Double]]) {
  guard let evalTimes = tEval else { return (tList, yList) }
  var resultY: [[Double]] = []
  for tE in evalTimes {
    var idx = max(0, tList.count - 2)
    for j in 0..<(tList.count - 1) {
      let lo = tList[j], hi = tList[j + 1]
      if (lo <= tE && tE <= hi) || (lo >= tE && tE >= hi) { idx = j; break }
    }
    let t1 = tList[idx], t2 = tList[min(idx + 1, tList.count - 1)]
    let frac = abs(t2 - t1) > 1e-15 ? (tE - t1) / (t2 - t1) : 0.0
    let y1 = yList[idx], y2 = yList[min(idx + 1, yList.count - 1)]
    var yI = [Double](repeating: 0, count: n)
    for i in 0..<n { yI[i] = y1[i] + frac * (y2[i] - y1[i]) }
    resultY.append(yI)
  }
  return (evalTimes, resultY)
}
