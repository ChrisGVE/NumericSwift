//
//  BFGS.swift
//  NumericSwift
//
//  BFGS quasi-Newton optimizer following scipy.optimize.minimize(method='BFGS').
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Result Type

/// Result of BFGS minimization.
public struct BFGSResult {
  /// Optimal parameter values.
  public let x: [Double]
  /// Function value at optimum.
  public let fun: Double
  /// Number of function evaluations.
  public let nfev: Int
  /// Number of gradient evaluations.
  public let njev: Int
  /// Number of iterations.
  public let nit: Int
  /// Whether optimization converged.
  public let success: Bool

  public init(
    x: [Double], fun: Double, nfev: Int, njev: Int, nit: Int, success: Bool
  ) {
    self.x = x
    self.fun = fun
    self.nfev = nfev
    self.njev = njev
    self.nit = nit
    self.success = success
  }
}

// MARK: - Public Entry Point

/// Minimize a function using the BFGS quasi-Newton method.
///
/// Uses an inverse-Hessian approximation updated via the Broyden-Fletcher-
/// Goldfarb-Shanno formula.  A backtracking Armijo line search is used to
/// select the step length at each iteration.
///
/// - Parameters:
///   - f: Objective function to minimize.
///   - x0: Initial guess.
///   - grad: Gradient function.  When `nil`, central finite differences
///     (h = 1e-8) are used.
///   - maxIter: Maximum number of iterations (default 200).
///   - gtol: Gradient ∞-norm tolerance for convergence (default 1e-5).
/// - Returns: ``BFGSResult`` containing the solution and diagnostics.
public func bfgs(
  _ f: @escaping ([Double]) -> Double,
  x0: [Double],
  grad: (([Double]) -> [Double])? = nil,
  maxIter: Int = 200,
  gtol: Double = 1e-5
) -> BFGSResult {
  let n = x0.count
  var x = x0
  var nfev = 0
  var njev = 0
  var nit = 0

  // Gradient oracle – central differences when no analytic grad is given.
  let gradientOf: ([Double]) -> [Double] =
    grad ?? { pt in
      bfgsCentralDiff(f, at: pt, nfev: &nfev)
    }

  // Start with the identity as the inverse-Hessian approximation.
  var H = bfgsIdentity(n)

  var g = gradientOf(x)
  njev += 1
  var fVal = f(x)
  nfev += 1

  for _ in 0..<maxIter {
    // Convergence check: ‖g‖∞ < gtol
    if bfgsInfNorm(g) < gtol {
      return BFGSResult(
        x: x, fun: fVal, nfev: nfev, njev: njev, nit: nit, success: true
      )
    }

    // Search direction: d = -H * g
    let d = bfgsNegate(bfgsMatVec(H, g))

    // Armijo backtracking line search
    let alpha = bfgsLineSearch(
      f: f, x: x, d: d, g: g, fVal: fVal, nfev: &nfev
    )

    // Parameter update
    let s = d.map { $0 * alpha }  // s = x_new - x
    let xNew = zip(x, s).map { $0 + $1 }
    let gNew = gradientOf(xNew)
    njev += 1
    fVal = f(xNew)
    nfev += 1
    nit += 1

    // BFGS inverse-Hessian update
    let y = zip(gNew, g).map { $0 - $1 }  // y = g_new - g
    let sy = bfgsDot(s, y)

    if sy > 1e-10 {
      H = bfgsHessianUpdate(H, s: s, y: y, sy: sy)
    }

    x = xNew
    g = gNew
  }

  return BFGSResult(
    x: x, fun: fVal, nfev: nfev, njev: njev, nit: nit, success: false
  )
}

// MARK: - Private Helpers

/// Central finite-difference gradient with h = 1e-8.
private func bfgsCentralDiff(
  _ f: ([Double]) -> Double,
  at x: [Double],
  nfev: inout Int
) -> [Double] {
  let h = 1e-8
  var g = [Double](repeating: 0, count: x.count)
  for i in 0..<x.count {
    var xp = x
    xp[i] += h
    var xm = x
    xm[i] -= h
    g[i] = (f(xp) - f(xm)) / (2 * h)
    nfev += 2
  }
  return g
}

/// Armijo backtracking line search.
///
/// Starts with `alpha = 1` and halves until the sufficient-decrease condition
/// `f(x + α·d) ≤ f(x) + c₁·α·(g·d)` is satisfied.
private func bfgsLineSearch(
  f: ([Double]) -> Double,
  x: [Double],
  d: [Double],
  g: [Double],
  fVal: Double,
  nfev: inout Int
) -> Double {
  let c1 = 1e-4
  let tau = 0.5
  let minAlpha = 1e-16
  var alpha = 1.0
  let slope = bfgsDot(g, d)  // should be negative

  while alpha > minAlpha {
    let xNew = zip(x, d).map { $0 + alpha * $1 }
    let fNew = f(xNew)
    nfev += 1
    if fNew <= fVal + c1 * alpha * slope {
      return alpha
    }
    alpha *= tau
  }
  return alpha
}

/// Symmetric rank-2 BFGS update of the inverse Hessian approximation.
///
/// H ← (I − ρ·s·yᵀ) · H · (I − ρ·y·sᵀ) + ρ·s·sᵀ,  ρ = 1/(yᵀs)
private func bfgsHessianUpdate(
  _ H: [[Double]], s: [Double], y: [Double], sy: Double
) -> [[Double]] {
  let n = s.count
  let rho = 1.0 / sy

  // A = I - rho * s * y'
  // H_new = A * H * A' + rho * s * s'

  // Step 1: W = H * (I - rho * y * s') = H - rho * (H * y) * s'
  let Hy = bfgsMatVec(H, y)
  var W = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
  for i in 0..<n {
    for j in 0..<n {
      W[i][j] = H[i][j] - rho * Hy[i] * s[j]
    }
  }

  // Step 2: H_new = (I - rho * s * y') * W + rho * s * s'
  //               = W - rho * s * (y' * W) + rho * s * s'
  let yW = bfgsRowMul(y, W)  // yᵀ · W  (row vector)
  var Hnew = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
  for i in 0..<n {
    for j in 0..<n {
      Hnew[i][j] = W[i][j] - rho * s[i] * yW[j] + rho * s[i] * s[j]
    }
  }
  return Hnew
}

// MARK: - Tiny Linear-Algebra Utilities

/// n × n identity matrix.
private func bfgsIdentity(_ n: Int) -> [[Double]] {
  var I = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
  for i in 0..<n { I[i][i] = 1 }
  return I
}

/// Matrix–vector product M · v.
private func bfgsMatVec(_ M: [[Double]], _ v: [Double]) -> [Double] {
  M.map { row in zip(row, v).reduce(0) { $0 + $1.0 * $1.1 } }
}

/// Row-vector × matrix product:  vᵀ · M.
private func bfgsRowMul(_ v: [Double], _ M: [[Double]]) -> [Double] {
  let n = v.count
  var result = [Double](repeating: 0, count: n)
  for j in 0..<n {
    result[j] = (0..<n).reduce(0) { $0 + v[$1] * M[$1][j] }
  }
  return result
}

/// Dot product.
private func bfgsDot(_ a: [Double], _ b: [Double]) -> Double {
  zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
}

/// Element-wise negation.
private func bfgsNegate(_ v: [Double]) -> [Double] { v.map { -$0 } }

/// Infinity norm (max absolute value).
private func bfgsInfNorm(_ v: [Double]) -> Double {
  v.reduce(0) { max($0, abs($1)) }
}
