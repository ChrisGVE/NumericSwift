//
//  Integration.swift
//  NumericSwift
//
//  Numerical integration and ODE solvers following scipy.integrate patterns.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Constants

/// Default absolute tolerance for adaptive quadrature
public let quadDefaultEpsAbs: Double = 1.49e-8

/// Default relative tolerance for adaptive quadrature
public let quadDefaultEpsRel: Double = 1.49e-8

/// Default maximum subdivisions for adaptive quadrature
public let quadDefaultLimit: Int = 50

// MARK: - Result Types

/// Result of a quadrature integration
public struct QuadResult {
  /// Computed integral value
  public let value: Double
  /// Estimated absolute error
  public let error: Double
  /// Number of function evaluations
  public let neval: Int

  public init(value: Double, error: Double, neval: Int) {
    self.value = value
    self.error = error
    self.neval = neval
  }
}

/// Result of an ODE integration
public struct ODEResult {
  /// Time points
  public let t: [Double]
  /// Solution values at each time point (y[i] is array of components at t[i])
  public let y: [[Double]]
  /// Whether integration was successful
  public let success: Bool
  /// Status message
  public let message: String
  /// Number of function evaluations
  public let nfev: Int

  public init(t: [Double], y: [[Double]], success: Bool, message: String, nfev: Int) {
    self.t = t
    self.y = y
    self.success = success
    self.message = message
    self.nfev = nfev
  }
}

// MARK: - Gauss-Kronrod Quadrature

/// Gauss-Kronrod 15-point abscissae (symmetric, only positive half stored)
private let gkAbscissae: [Double] = [
  0.991455371120813,
  0.949107912342759,
  0.864864423359769,
  0.741531185599394,
  0.586087235467691,
  0.405845151377397,
  0.207784955007898,
  0.0,
]

/// Weights for 15-point Kronrod rule
private let gkWeights: [Double] = [
  0.022935322010529,
  0.063092092629979,
  0.104790010322250,
  0.140653259715525,
  0.169004726639267,
  0.190350578064785,
  0.204432940075298,
  0.209482141084728,
]

/// Weights for 7-point Gauss rule (embedded in K15)
private let gWeights: [Double] = [
  0.129484966168870,
  0.279705391489277,
  0.381830050505119,
  0.417959183673469,
]

/// Single Gauss-Kronrod 15-point quadrature step
private func gk15(_ f: (Double) -> Double, _ a: Double, _ b: Double) -> (Double, Double) {
  let center = 0.5 * (a + b)
  let halfLength = 0.5 * (b - a)
  let fCenter = f(center)

  var resultKronrod = fCenter * gkWeights[7]
  var resultGauss = fCenter * gWeights[3]

  for i in 0..<7 {
    let x = halfLength * gkAbscissae[i]
    let fval1 = f(center - x)
    let fval2 = f(center + x)
    let fsum = fval1 + fval2

    resultKronrod += fsum * gkWeights[i]

    // Gauss points (even indices in 0-based correspond to odd in 1-based)
    if (i + 1) % 2 == 0 {
      resultGauss += fsum * gWeights[(i + 1) / 2 - 1]
    }
  }

  resultKronrod *= halfLength
  resultGauss *= halfLength

  let absError = abs(resultKronrod - resultGauss)
  return (resultKronrod, absError)
}

/// Adaptive quadrature using Gauss-Kronrod rule.
///
/// - Parameters:
///   - f: Function to integrate
///   - a: Lower limit (can be -∞)
///   - b: Upper limit (can be +∞)
///   - epsabs: Absolute tolerance
///   - epsrel: Relative tolerance
///   - limit: Maximum number of subdivisions
/// - Returns: QuadResult with value, error, and evaluation count
public func quad(
  _ f: @escaping (Double) -> Double,
  _ a: Double,
  _ b: Double,
  epsabs: Double = quadDefaultEpsAbs,
  epsrel: Double = quadDefaultEpsRel,
  limit: Int = quadDefaultLimit
) -> QuadResult {
  var actualA = a
  var actualB = b
  var fTransformed = f

  // Handle infinite limits with variable transformations
  if a == -.infinity && b == .infinity {
    // Transform: x = t / (1 - t²), t in (-1, 1)
    fTransformed = { t in
      if abs(t) >= 1 { return 0 }
      let x = t / (1.0 - t * t)
      let dxdt = (1.0 + t * t) / pow(1.0 - t * t, 2)
      return f(x) * dxdt
    }
    actualA = -1
    actualB = 1
  } else if a == -.infinity {
    // Transform: x = b - (1-t)/t, t in (0, 1)
    let bOrig = b
    fTransformed = { t in
      if t <= 0 { return 0 }
      let x = bOrig - (1.0 - t) / t
      let dxdt = 1.0 / (t * t)
      return f(x) * dxdt
    }
    actualA = 0
    actualB = 1
  } else if b == .infinity {
    // Transform: x = a + t/(1-t), t in (0, 1)
    let aOrig = a
    fTransformed = { t in
      if t >= 1 { return 0 }
      let x = aOrig + t / (1.0 - t)
      let dxdt = 1.0 / pow(1.0 - t, 2)
      return f(x) * dxdt
    }
    actualA = 0
    actualB = 1
  }

  // Stack-based adaptive integration
  var stack: [(Double, Double)] = [(actualA, actualB)]
  var totalResult: Double = 0
  var totalError: Double = 0
  var neval = 0
  var subdivisions = 0

  while !stack.isEmpty && subdivisions < limit {
    let (ia, ib) = stack.removeLast()
    let (result, absError) = gk15(fTransformed, ia, ib)
    neval += 15

    let tolerance = max(epsabs, epsrel * abs(result))

    if absError <= tolerance || (ib - ia) < 1e-15 {
      totalResult += result
      totalError += absError
    } else {
      subdivisions += 1
      let mid = 0.5 * (ia + ib)
      stack.append((ia, mid))
      stack.append((mid, ib))
    }
  }

  // Process remaining intervals if limit reached
  while !stack.isEmpty {
    let (ia, ib) = stack.removeLast()
    let (result, absError) = gk15(fTransformed, ia, ib)
    totalResult += result
    totalError += absError
    neval += 15
  }

  return QuadResult(value: totalResult, error: totalError, neval: neval)
}

// MARK: - Double Integration

/// Double integration over a rectangular or non-rectangular region.
///
/// - Parameters:
///   - f: Function f(y, x) to integrate
///   - xa: Lower x limit
///   - xb: Upper x limit
///   - ya: Lower y limit as function of x
///   - yb: Upper y limit as function of x
///   - epsabs: Absolute tolerance
///   - epsrel: Relative tolerance
/// - Returns: QuadResult with value and error
public func dblquad(
  _ f: @escaping (Double, Double) -> Double,
  xa: Double,
  xb: Double,
  ya: @escaping (Double) -> Double,
  yb: @escaping (Double) -> Double,
  epsabs: Double = quadDefaultEpsAbs,
  epsrel: Double = quadDefaultEpsRel
) -> QuadResult {
  let inner: (Double) -> Double = { x in
    let yLower = ya(x)
    let yUpper = yb(x)
    let result = quad({ y in f(y, x) }, yLower, yUpper, epsabs: epsabs, epsrel: epsrel)
    return result.value
  }
  return quad(inner, xa, xb, epsabs: epsabs, epsrel: epsrel)
}

/// Double integration over a rectangular region (constant limits).
public func dblquad(
  _ f: @escaping (Double, Double) -> Double,
  xa: Double,
  xb: Double,
  ya: Double,
  yb: Double,
  epsabs: Double = quadDefaultEpsAbs,
  epsrel: Double = quadDefaultEpsRel
) -> QuadResult {
  return dblquad(
    f, xa: xa, xb: xb, ya: { _ in ya }, yb: { _ in yb }, epsabs: epsabs, epsrel: epsrel)
}

// MARK: - Triple Integration

/// Triple integration.
///
/// - Parameters:
///   - f: Function f(z, y, x) to integrate
///   - xa: Lower x limit
///   - xb: Upper x limit
///   - ya: Lower y limit as function of x
///   - yb: Upper y limit as function of x
///   - za: Lower z limit as function of x, y
///   - zb: Upper z limit as function of x, y
///   - epsabs: Absolute error tolerance
///   - epsrel: Relative error tolerance
public func tplquad(
  _ f: @escaping (Double, Double, Double) -> Double,
  xa: Double,
  xb: Double,
  ya: @escaping (Double) -> Double,
  yb: @escaping (Double) -> Double,
  za: @escaping (Double, Double) -> Double,
  zb: @escaping (Double, Double) -> Double,
  epsabs: Double = quadDefaultEpsAbs,
  epsrel: Double = quadDefaultEpsRel
) -> QuadResult {
  let innerXY: (Double, Double) -> Double = { y, x in
    let zLower = za(x, y)
    let zUpper = zb(x, y)
    let result = quad({ z in f(z, y, x) }, zLower, zUpper, epsabs: epsabs, epsrel: epsrel)
    return result.value
  }
  return dblquad(innerXY, xa: xa, xb: xb, ya: ya, yb: yb, epsabs: epsabs, epsrel: epsrel)
}

/// Triple integration over rectangular region (constant limits).
public func tplquad(
  _ f: @escaping (Double, Double, Double) -> Double,
  xa: Double, xb: Double,
  ya: Double, yb: Double,
  za: Double, zb: Double,
  epsabs: Double = quadDefaultEpsAbs,
  epsrel: Double = quadDefaultEpsRel
) -> QuadResult {
  return tplquad(
    f, xa: xa, xb: xb,
    ya: { _ in ya }, yb: { _ in yb },
    za: { _, _ in za }, zb: { _, _ in zb },
    epsabs: epsabs, epsrel: epsrel)
}

// MARK: - Fixed-Order Gaussian Quadrature

/// Gauss-Legendre quadrature points and weights for n=1 to 10
private let gaussLegendre: [Int: [(x: Double, w: Double)]] = [
  1: [(0, 2)],
  2: [(-0.5773502691896257, 1), (0.5773502691896257, 1)],
  3: [
    (-0.7745966692414834, 0.5555555555555556),
    (0, 0.8888888888888888),
    (0.7745966692414834, 0.5555555555555556),
  ],
  4: [
    (-0.8611363115940526, 0.3478548451374538),
    (-0.3399810435848563, 0.6521451548625461),
    (0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.3478548451374538),
  ],
  5: [
    (-0.9061798459386640, 0.2369268850561891),
    (-0.5384693101056831, 0.4786286704993665),
    (0, 0.5688888888888889),
    (0.5384693101056831, 0.4786286704993665),
    (0.9061798459386640, 0.2369268850561891),
  ],
  6: [
    (-0.9324695142031521, 0.1713244923791704),
    (-0.6612093864662645, 0.3607615730481386),
    (-0.2386191860831969, 0.4679139345726910),
    (0.2386191860831969, 0.4679139345726910),
    (0.6612093864662645, 0.3607615730481386),
    (0.9324695142031521, 0.1713244923791704),
  ],
  7: [
    (-0.9491079123427585, 0.1294849661688697),
    (-0.7415311855993945, 0.2797053914892766),
    (-0.4058451513773972, 0.3818300505051189),
    (0, 0.4179591836734694),
    (0.4058451513773972, 0.3818300505051189),
    (0.7415311855993945, 0.2797053914892766),
    (0.9491079123427585, 0.1294849661688697),
  ],
  8: [
    (-0.9602898564975363, 0.1012285362903763),
    (-0.7966664774136267, 0.2223810344533745),
    (-0.5255324099163290, 0.3137066458778873),
    (-0.1834346424956498, 0.3626837833783620),
    (0.1834346424956498, 0.3626837833783620),
    (0.5255324099163290, 0.3137066458778873),
    (0.7966664774136267, 0.2223810344533745),
    (0.9602898564975363, 0.1012285362903763),
  ],
  9: [
    (-0.9681602395076261, 0.0812743883615744),
    (-0.8360311073266358, 0.1806481606948574),
    (-0.6133714327005904, 0.2606106964029354),
    (-0.3242534234038089, 0.3123470770400029),
    (0, 0.3302393550012598),
    (0.3242534234038089, 0.3123470770400029),
    (0.6133714327005904, 0.2606106964029354),
    (0.8360311073266358, 0.1806481606948574),
    (0.9681602395076261, 0.0812743883615744),
  ],
  10: [
    (-0.9739065285171717, 0.0666713443086881),
    (-0.8650633666889845, 0.1494513491505806),
    (-0.6794095682990244, 0.2190863625159820),
    (-0.4333953941292472, 0.2692667193099963),
    (-0.1488743389816312, 0.2955242247147529),
    (0.1488743389816312, 0.2955242247147529),
    (0.4333953941292472, 0.2692667193099963),
    (0.6794095682990244, 0.2190863625159820),
    (0.8650633666889845, 0.1494513491505806),
    (0.9739065285171717, 0.0666713443086881),
  ],
]

/// Fixed-order Gauss-Legendre quadrature.
///
/// - Parameters:
///   - f: Function to integrate
///   - a: Lower limit
///   - b: Upper limit
///   - n: Number of points (1-10)
/// - Returns: Integral approximation
public func fixedQuad(_ f: (Double) -> Double, _ a: Double, _ b: Double, n: Int = 5) -> Double {
  let order = min(max(n, 1), 10)
  guard let points = gaussLegendre[order] else { return 0 }

  let center = 0.5 * (a + b)
  let halfLength = 0.5 * (b - a)
  var result: Double = 0

  for point in points {
    let x = center + halfLength * point.x
    result += point.w * f(x)
  }

  return result * halfLength
}

// MARK: - Romberg Integration

/// Romberg integration using Richardson extrapolation.
///
/// - Parameters:
///   - f: Function to integrate
///   - a: Lower limit
///   - b: Upper limit
///   - tol: Tolerance for convergence
///   - divmax: Maximum number of extrapolation steps
/// - Returns: QuadResult
public func romberg(
  _ f: (Double) -> Double,
  _ a: Double,
  _ b: Double,
  tol: Double = 1e-8,
  divmax: Int = 10
) -> QuadResult {
  var R: [[Double]] = Array(repeating: [], count: divmax + 2)
  var h = b - a

  // R[0][0] = trapezoidal rule
  R[0] = [0.5 * h * (f(a) + f(b))]

  for i in 1...divmax {
    h /= 2
    R[i] = []

    // Composite trapezoidal rule
    var sum: Double = 0
    let n = 1 << (i - 1)  // 2^(i-1)
    for k in 1...n {
      sum += f(a + Double(2 * k - 1) * h)
    }
    R[i].append(0.5 * R[i - 1][0] + h * sum)

    // Richardson extrapolation
    for j in 1...i {
      let factor = pow(4.0, Double(j))
      let value = (factor * R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1.0)
      R[i].append(value)
    }

    // Check convergence
    let error = abs(R[i][i] - R[i - 1][i - 1])
    if error < tol {
      return QuadResult(value: R[i][i], error: error, neval: (1 << (i + 1)) - 1)
    }
  }

  let finalError = abs(R[divmax][divmax] - R[divmax - 1][divmax - 1])
  return QuadResult(value: R[divmax][divmax], error: finalError, neval: (1 << (divmax + 1)) - 1)
}

// MARK: - Simpson's Rule

/// Simpson's rule integration from array of values.
///
/// - Parameters:
///   - y: Array of function values
///   - dx: Step size (default 1)
/// - Returns: Integral approximation
public func simps(_ y: [Double], dx: Double = 1) -> Double {
  let n = y.count
  guard n >= 3 else { return 0 }

  var result = y[0] + y[n - 1]

  if n % 2 == 1 {
    // Odd number of points (even number of intervals)
    for i in 1..<(n - 1) {
      result += (i % 2 == 1 ? 4.0 : 2.0) * y[i]
    }
    return result * dx / 3.0
  } else {
    // Even number of points - use Simpson's for n-1 points, trapezoid for last
    result = (y[0] + 4.0 * y[1] + y[2]) * dx / 3.0
    for i in stride(from: 2, to: n - 2, by: 2) {
      result += (y[i] + 4.0 * y[i + 1] + y[i + 2]) * dx / 3.0
    }
    // Add last interval with trapezoid if needed
    if (n - 1) % 2 == 1 {
      result += dx * (y[n - 2] + y[n - 1]) / 2.0
    }
    return result
  }
}

/// Simpson's rule integration with non-uniform sample points.
///
/// Mirrors `scipy.integrate.simpson` (Cartwright composite Simpson for
/// irregularly-spaced data): each consecutive *pair* of intervals is
/// integrated with the general unequal-spacing 3-point rule, and when the
/// number of intervals is odd the final interval is added with the Cartwright
/// correction. The previous implementation averaged the spacing and was
/// silently wrong on non-uniform grids.
///
/// Exact for polynomials of degree ≤ 2 on an arbitrary grid (degree ≤ 3 on a
/// uniform grid). With two points it reduces to a single trapezoid; with fewer
/// than two it returns 0 — both matching SciPy.
///
/// - Parameters:
///   - y: Array of function values.
///   - x: Sample points (same length as `y`); need not be uniformly spaced.
/// - Returns: Integral approximation.
///
/// Reference: Cartwright, K. V., "Simpson's Rule Cumulative Integration with
/// MS Excel and Irregularly-spaced Data", J. Math. Sci. & Math. Educ. 12(2).
public func simps(_ y: [Double], x: [Double]) -> Double {
  let n = y.count
  guard n == x.count else { return 0 }
  guard n >= 3 else {
    if n == 2 { return 0.5 * (x[1] - x[0]) * (y[0] + y[1]) }
    return 0
  }

  // Guarded division mirroring SciPy's `true_divide(..., where: den != 0)`,
  // so degenerate (coincident) sample points zero the affected term instead
  // of producing inf/NaN.
  func safeDiv(_ a: Double, _ b: Double) -> Double { b == 0 ? 0 : a / b }

  // Composite Simpson over consecutive interval pairs (i, i+1, i+2),
  // advancing two intervals at a time up to (but not including) `stop`.
  func basicSimpson(upTo stop: Int) -> Double {
    var sum = 0.0
    var i = 0
    while i < stop {
      let h0 = x[i + 1] - x[i]
      let h1 = x[i + 2] - x[i + 1]
      let hSum = h0 + h1
      let hProd = h0 * h1
      let h0DivH1 = safeDiv(h0, h1)
      sum += hSum / 6.0
        * (y[i] * (2.0 - safeDiv(1.0, h0DivH1))
          + y[i + 1] * (hSum * safeDiv(hSum, hProd))
          + y[i + 2] * (2.0 - h0DivH1))
      i += 2
    }
    return sum
  }

  let intervals = n - 1
  if intervals % 2 == 0 {
    // Even number of intervals → pure composite Simpson over every pair.
    return basicSimpson(upTo: n - 2)
  }
  // Odd number of intervals → Simpson on all but the last interval, then the
  // Cartwright correction for the final interval using its last three points.
  var result = basicSimpson(upTo: n - 3)
  let h0 = x[n - 2] - x[n - 3]  // second-to-last spacing
  let h1 = x[n - 1] - x[n - 2]  // last spacing
  let alpha = safeDiv(2.0 * h1 * h1 + 3.0 * h0 * h1, 6.0 * (h1 + h0))
  let beta = safeDiv(h1 * h1 + 3.0 * h0 * h1, 6.0 * h0)
  let eta = safeDiv(h1 * h1 * h1, 6.0 * h0 * (h0 + h1))
  result += alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3]
  return result
}

// MARK: - Trapezoidal Rule

/// Trapezoidal rule integration from array of values.
public func trapz(_ y: [Double], dx: Double = 1) -> Double {
  guard y.count >= 2 else { return 0 }
  var result: Double = 0
  for i in 0..<(y.count - 1) {
    result += 0.5 * (y[i] + y[i + 1]) * dx
  }
  return result
}

/// Trapezoidal rule with non-uniform spacing.
public func trapz(_ y: [Double], x: [Double]) -> Double {
  guard y.count == x.count && y.count >= 2 else { return 0 }
  var result: Double = 0
  for i in 0..<(y.count - 1) {
    result += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
  }
  return result
}

// MARK: - Cumulative Integration

/// Cumulative trapezoidal integration with uniform spacing.
///
/// Returns an array of length `y.count - 1` where element `i` is the
/// integral from `y[0]` to `y[i+1]` using the trapezoidal rule.
/// Matches `scipy.integrate.cumulative_trapezoid`.
///
/// - Parameters:
///   - y: Function values at evenly spaced points.
///   - dx: Spacing between points (default 1).
/// - Returns: Cumulative integral values, or empty array if fewer than 2 points.
public func cumulativeTrapezoid(_ y: [Double], dx: Double = 1) -> [Double] {
  guard y.count >= 2 else { return [] }
  var result = [Double](repeating: 0.0, count: y.count - 1)
  var cumulative = 0.0
  for i in 0..<(y.count - 1) {
    cumulative += 0.5 * (y[i] + y[i + 1]) * dx
    result[i] = cumulative
  }
  return result
}

/// Cumulative trapezoidal integration with non-uniform spacing.
///
/// - Parameters:
///   - y: Function values.
///   - x: Corresponding x positions (must have same count as y).
/// - Returns: Cumulative integral values, or empty array if fewer than 2 points.
public func cumulativeTrapezoid(_ y: [Double], x: [Double]) -> [Double] {
  guard y.count == x.count && y.count >= 2 else { return [] }
  var result = [Double](repeating: 0.0, count: y.count - 1)
  var cumulative = 0.0
  for i in 0..<(y.count - 1) {
    cumulative += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    result[i] = cumulative
  }
  return result
}

/// Cumulative Simpson's rule integration with uniform spacing.
///
/// Uses Simpson's rule on each pair of subintervals (requires odd number of points,
/// i.e. even number of intervals). For even number of points, the last interval
/// uses the trapezoidal rule.
/// Returns an array of length `y.count - 1`.
///
/// - Parameters:
///   - y: Function values at evenly spaced points.
///   - dx: Spacing between points (default 1).
/// - Returns: Cumulative integral values, or empty array if fewer than 2 points.
public func cumulativeSimpson(_ y: [Double], dx: Double = 1) -> [Double] {
  guard y.count >= 2 else { return [] }
  var result = [Double](repeating: 0.0, count: y.count - 1)

  // First interval: trapezoidal
  result[0] = 0.5 * (y[0] + y[1]) * dx

  // Subsequent pairs: use Simpson's 1/3 rule on each pair of intervals
  var i = 1
  while i < y.count - 1 {
    if i + 1 < y.count - 1 {
      // Simpson's rule over two intervals: (dx/3) * (y[i-1] + 4*y[i] + y[i+1])
      // But we need cumulative, so we add Simpson increment for [x_{i}, x_{i+2}]
      let simpsonIncrement = dx / 3.0 * (y[i] + 4.0 * y[i + 1] + y[i + 2])
      result[i] = result[i - 1] + 0.5 * (y[i] + y[i + 1]) * dx
      result[i + 1] = result[i - 1] + simpsonIncrement
      i += 2
    } else {
      // Odd remaining interval: use trapezoidal
      result[i] = result[i - 1] + 0.5 * (y[i] + y[i + 1]) * dx
      i += 1
    }
  }
  return result
}

// MARK: - ODE Solvers

/// Dormand-Prince A matrix (lower triangular part)
private let dpA: [[Double]] = [
  [],
  [1.0 / 5.0],
  [3.0 / 40.0, 9.0 / 40.0],
  [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
  [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
  [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
  [35.0 / 384.0, 0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
]

/// Dormand-Prince C coefficients (nodes)
private let dpC: [Double] = [0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1, 1]

/// Dormand-Prince B coefficients (5th order weights)
private let dpB: [Double] = [
  35.0 / 384.0, 0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0,
]

/// Dormand-Prince E coefficients (error estimate: 5th - 4th order)
private let dpE: [Double] = [
  71.0 / 57600.0, 0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0,
]

/// Bogacki-Shampine A matrix
private let bsA: [[Double]] = [
  [],
  [1.0 / 2.0],
  [0, 3.0 / 4.0],
  [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
]

private let bsC: [Double] = [0, 1.0 / 2.0, 3.0 / 4.0, 1]
private let bsB: [Double] = [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0]
private let bsE: [Double] = [-5.0 / 72.0, 1.0 / 12.0, 1.0 / 9.0, -1.0 / 8.0]

/// Single RK4 step (classical 4th order Runge-Kutta)
private func rk4Step(
  _ f: ([Double], Double) -> [Double],
  _ t: Double,
  _ y: [Double],
  _ h: Double
) -> [Double] {
  let n = y.count
  let k1 = f(y, t)

  var y2 = [Double](repeating: 0, count: n)
  for i in 0..<n { y2[i] = y[i] + 0.5 * h * k1[i] }
  let k2 = f(y2, t + 0.5 * h)

  var y3 = [Double](repeating: 0, count: n)
  for i in 0..<n { y3[i] = y[i] + 0.5 * h * k2[i] }
  let k3 = f(y3, t + 0.5 * h)

  var y4 = [Double](repeating: 0, count: n)
  for i in 0..<n { y4[i] = y[i] + h * k3[i] }
  let k4 = f(y4, t + h)

  var yNew = [Double](repeating: 0, count: n)
  for i in 0..<n {
    yNew[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
  }
  return yNew
}

/// Single Dormand-Prince RK45 step
///
/// Returns the 5th-order solution and error estimate.  The stage derivatives
/// `k[0..6]` are also returned so that the caller can build the dense-output
/// quartic interpolant without additional function evaluations.
///
/// Reference: Dormand & Prince, "A family of embedded Runge-Kutta formulae",
/// J. Comput. Appl. Math. 6(1), 1980.
private func rk45Step(
  _ f: ([Double], Double) -> [Double],
  _ t: Double,
  _ y: [Double],
  _ h: Double
) -> (yNew: [Double], error: [Double], stages: [[Double]]) {
  let n = y.count
  var k: [[Double]] = []

  k.append(f(y, t))

  for stage in 1..<7 {
    var yStage = [Double](repeating: 0, count: n)
    for i in 0..<n {
      var sum = y[i]
      for j in 0..<stage {
        if j < dpA[stage].count {
          sum += h * dpA[stage][j] * k[j][i]
        }
      }
      yStage[i] = sum
    }
    k.append(f(yStage, t + dpC[stage] * h))
  }

  // Compute 5th order solution
  var yNew = [Double](repeating: 0, count: n)
  for i in 0..<n {
    var sum = y[i]
    for j in 0..<7 {
      sum += h * dpB[j] * k[j][i]
    }
    yNew[i] = sum
  }

  // Compute error estimate
  var err = [Double](repeating: 0, count: n)
  for i in 0..<n {
    var sum: Double = 0
    for j in 0..<7 {
      sum += h * dpE[j] * k[j][i]
    }
    err[i] = sum
  }

  return (yNew, err, k)
}

/// Single Bogacki-Shampine RK23 step
///
/// Returns the 3rd-order solution and error estimate.  The stage derivatives
/// `k[0..3]` are also returned so that the caller can build the dense-output
/// cubic Hermite interpolant without additional function evaluations.
///
/// Reference: Bogacki & Shampine, "A 3(2) pair of Runge-Kutta formulas",
/// Appl. Math. Lett. 2(4), 1989.
private func rk23Step(
  _ f: ([Double], Double) -> [Double],
  _ t: Double,
  _ y: [Double],
  _ h: Double
) -> (yNew: [Double], error: [Double], stages: [[Double]]) {
  let n = y.count
  var k: [[Double]] = []

  k.append(f(y, t))

  for stage in 1..<4 {
    var yStage = [Double](repeating: 0, count: n)
    for i in 0..<n {
      var sum = y[i]
      for j in 0..<stage {
        if j < bsA[stage].count {
          sum += h * bsA[stage][j] * k[j][i]
        }
      }
      yStage[i] = sum
    }
    k.append(f(yStage, t + bsC[stage] * h))
  }

  // 3rd order solution
  var yNew = [Double](repeating: 0, count: n)
  for i in 0..<n {
    var sum = y[i]
    for j in 0..<4 {
      sum += h * bsB[j] * k[j][i]
    }
    yNew[i] = sum
  }

  // Error estimate
  var err = [Double](repeating: 0, count: n)
  for i in 0..<n {
    var sum: Double = 0
    for j in 0..<4 {
      sum += h * bsE[j] * k[j][i]
    }
    err[i] = sum
  }

  return (yNew, err, k)
}

// MARK: - Dense-output interpolants

/// One accepted solver step recorded for dense-output evaluation.
///
/// `tStart` and `tEnd` bracket the step, `yStart` is the state at `tStart`,
/// `stages` are the per-component stage derivatives used by the interpolant,
/// and `h` is the (signed) step size `tEnd − tStart`.
private struct DenseStep {
  let tStart: Double
  let tEnd: Double
  let yStart: [Double]
  let yEnd: [Double]
  let stages: [[Double]]  // k[0..6] for RK45 or k[0..3] for RK23
  let h: Double
}

/// Dormand-Prince RK45 quartic (4th-order) dense-output interpolant.
///
/// The continuous extension evaluates the 4th-order polynomial
///
///   y(t) = y₀ + h · (Kᵀ · P) · [s, s², s³, s⁴]
///
/// where s = (t − t₀) / h and P is the coefficient matrix from SciPy's
/// `RK45.dense_output` implementation (scipy/integrate/_ivp/rk.py, constant
/// `P`).  Using the quartic gives O(h⁵) error — the same order as the
/// 5th-order solver — far better than the O(h²) of linear interpolation.
///
/// P matrix columns (one per power of s, rows = stages k[0]..k[6]):
/// ```
/// P = [
///   [ 1,  -8048581381/2820520608,   8663915743/2820520608, -12715105075/11282082432],
///   [ 0,   0,                        0,                       0                   ],
///   [ 0,   131558114200/32700410799, -68118460800/10900136933, 87487479700/32700410799],
///   [ 0,  -1754552775/470086768,     14199869525/1410260304, -10690763975/1880347072 ],
///   [ 0,   127303824393/49829197408, -318862633509/49829197408, 701980252875/199316789632],
///   [ 0,  -282668133/205662961,      2019193451/616988883,   -1453857185/822651844  ],
///   [ 0,   40617522/29380423,       -110615467/29380423,      69997945/29380423     ]
/// ]
/// ```
///
/// Reference: Dormand & Prince (1980); SciPy scipy.integrate._ivp.rk.RK45.
private func rk45DenseOutput(step: DenseStep, at t: Double) -> [Double] {
  // Normalised parameter s ∈ [0, 1] across the step
  let s = (t - step.tStart) / step.h

  // Horner-form evaluation of the 4th-degree polynomial in s:
  //   c₀·s + c₁·s² + c₂·s³ + c₃·s⁴
  // where cⱼ = dot(P[:,j], K[:,i]) for each state component i.
  // P column 0 = [1,0,...,0] so c₀ = k[0][i] directly (no dot product needed).
  // P row 1 = [0,0,0,0] so stage k[1] never contributes; the j-loop skips it.

  // P column 1 (coefficient of s²):
  let p1: [Double] = [
    -8048581381.0 / 2820520608.0,
    0,
    131558114200.0 / 32700410799.0,
    -1754552775.0 / 470086768.0,
    127303824393.0 / 49829197408.0,
    -282668133.0 / 205662961.0,
    40617522.0 / 29380423.0,
  ]
  // P column 2 (coefficient of s³):
  let p2: [Double] = [
    8663915743.0 / 2820520608.0,
    0,
    -68118460800.0 / 10900136933.0,
    14199869525.0 / 1410260304.0,
    -318862633509.0 / 49829197408.0,
    2019193451.0 / 616988883.0,
    -110615467.0 / 29380423.0,
  ]
  // P column 3 (coefficient of s⁴):
  let p3: [Double] = [
    -12715105075.0 / 11282082432.0,
    0,
    87487479700.0 / 32700410799.0,
    -10690763975.0 / 1880347072.0,
    701980252875.0 / 199316789632.0,
    -1453857185.0 / 822651844.0,
    69997945.0 / 29380423.0,
  ]

  let n = step.yStart.count
  var y = [Double](repeating: 0, count: n)

  for i in 0..<n {
    // c0 = dot(P[:,0], K[:,i]).  P[:,0] = [1,0,0,0,0,0,0] so c0 = k[0][i].
    let c0 = step.stages[0][i]

    // c1..c3 = dot products of K column i with P columns 1..3.
    // Start from j=2 because P[1,:] = 0 for all columns (row for stage k[1]).
    var c1 = 0.0, c2 = 0.0, c3 = 0.0
    for j in [0, 2, 3, 4, 5, 6] {
      let kji = step.stages[j][i]
      c1 += p1[j] * kji
      c2 += p2[j] * kji
      c3 += p3[j] * kji
    }

    // Horner evaluation: s·(c0 + s·(c1 + s·(c2 + s·c3)))
    let poly = s * (c0 + s * (c1 + s * (c2 + s * c3)))
    y[i] = step.yStart[i] + step.h * poly
  }

  return y
}

/// Bogacki-Shampine RK23 cubic Hermite dense-output interpolant.
///
/// Uses the standard four-term cubic Hermite polynomial whose basis
/// functions are defined by matching values and first derivatives at
/// the two step endpoints.  Let s = (t − t₀)/h ∈ [0, 1]:
///
///   h₀₀(s) = 2s³ − 3s² + 1        (value blending at tStart)
///   h₁₀(s) = s³ − 2s² + s = s(s−1)²  (slope at tStart)
///   h₀₁(s) = −2s³ + 3s²           (value blending at tEnd)
///   h₁₁(s) = s³ − s² = s²(s−1)    (slope at tEnd)
///
///   y(t) = h₀₀(s)·y₀ + h₁₀(s)·h·f₀ + h₀₁(s)·y₁ + h₁₁(s)·h·f₁
///
/// where f₀ = k[0] = f(y₀, t₀) and f₁ = k[3] = f(y₁, t₁).
/// Boundary conditions are exact: y(t₀)=y₀ and y(t₁)=y₁.
/// This gives O(h⁴) interpolation accuracy inside the step.
///
/// Reference: De Boor, C., "A Practical Guide to Splines", Springer (1978),
/// Ch. IV §1 (Hermite interpolation).  Applied to RK23 dense output:
/// Bogacki & Shampine (1989); SciPy scipy.integrate._ivp.rk.RK23.
private func rk23DenseOutput(step: DenseStep, at t: Double) -> [Double] {
  let s = (t - step.tStart) / step.h
  let s2 = s * s
  let s3 = s2 * s

  // Standard cubic Hermite basis polynomials
  let h00 = 2.0 * s3 - 3.0 * s2 + 1.0   // value at tStart
  let h10 = s3 - 2.0 * s2 + s            // slope at tStart  (= s(s-1)²)
  let h01 = -2.0 * s3 + 3.0 * s2         // value at tEnd
  let h11 = s3 - s2                       // slope at tEnd    (= s²(s-1))

  let n = step.yStart.count
  var y = [Double](repeating: 0, count: n)

  for i in 0..<n {
    let f0 = step.stages[0][i]  // k[0] = f(y₀, t₀)
    let f1 = step.stages[3][i]  // k[3] = f(y₁, t₁), available via FSAL property
    y[i] = h00 * step.yStart[i]
          + h10 * step.h * f0
          + h01 * step.yEnd[i]
          + h11 * step.h * f1
  }

  return y
}

/// ODE solver method
public enum ODEMethod: String {
  case rk45 = "RK45"
  case rk23 = "RK23"
  case rk4 = "RK4"
}

/// Solve initial value problem for ODE system.
///
/// When `tEval` is supplied, the output is computed using a higher-order
/// continuous-extension dense-output interpolant — the quartic polynomial for
/// RK45 (Dormand-Prince) and the cubic Hermite for RK23 (Bogacki-Shampine).
/// These give O(h⁵) and O(h⁴) interpolation accuracy respectively, far
/// better than the O(h²) of simple linear interpolation.
///
/// - Parameters:
///   - fun: Function(y, t) returning dy/dt
///   - tSpan: (t0, tf) initial and final time
///   - y0: Initial state
///   - method: ODE method (RK45, RK23, RK4)
///   - tEval: Optional specific times for output; dense-output interpolation
///     is used for adaptive methods so accuracy is independent of step size
///   - maxStep: Maximum step size
///   - rtol: Relative tolerance
///   - atol: Absolute tolerance
///   - firstStep: Initial step size (nil for auto)
/// - Returns: ODEResult
public func solveIVP(
  _ fun: @escaping ([Double], Double) -> [Double],
  tSpan: (Double, Double),
  y0: [Double],
  method: ODEMethod = .rk45,
  tEval: [Double]? = nil,
  maxStep: Double = .infinity,
  rtol: Double = 1e-3,
  atol: Double = 1e-6,
  firstStep: Double? = nil
) -> ODEResult {
  let t0 = tSpan.0
  let tf = tSpan.1
  let direction: Double = tf >= t0 ? 1 : -1
  let n = y0.count

  var t = t0
  var y = y0
  var tList = [t0]
  var yList = [y0]
  // Dense steps are recorded for every accepted adaptive step so tEval can
  // use higher-order interpolation between solver step boundaries.
  var denseSteps: [DenseStep] = []
  var nfev = 0

  // Initial step size estimation
  var h: Double
  if let first = firstStep {
    h = first
  } else {
    let f0 = fun(y0, t0)
    nfev += 1
    let d0 = max(y0.map { abs($0) }.max() ?? 1, 1e-5)
    let d1 = max(f0.map { abs($0) }.max() ?? 1, 1e-5)
    h = 0.01 * d0 / d1
    h = min(h, abs(tf - t0) / 10)
  }
  h = direction * min(abs(h), maxStep)

  let maxIter = 10000
  var iter = 0

  while direction * (tf - t) > 1e-12 * abs(tf) && iter < maxIter {
    iter += 1

    // Don't overshoot tf
    if direction * (t + h - tf) > 0 {
      h = tf - t
    }

    if method == .rk4 {
      let tOld = t
      let yOld = y
      y = rk4Step(fun, t, y, h)
      nfev += 4
      t += h
      tList.append(t)
      yList.append(y)
      // RK4 provides no higher-order dense output; the tEval path uses
      // linear interpolation between the step's start and end values.
      // No extra function evaluations are needed.
      denseSteps.append(DenseStep(
        tStart: tOld, tEnd: t,
        yStart: yOld, yEnd: y,
        stages: [],   // Empty: RK4 branch uses linear interpolation only
        h: h
      ))
    } else {
      let stepResult =
        method == .rk23
        ? rk23Step(fun, t, y, h)
        : rk45Step(fun, t, y, h)
      let yNew = stepResult.yNew
      let err = stepResult.error
      let stages = stepResult.stages
      nfev += method == .rk23 ? 4 : 7

      // Error control
      var errNorm: Double = 0
      for i in 0..<n {
        let scale = atol + rtol * max(abs(y[i]), abs(yNew[i]))
        errNorm += pow(err[i] / scale, 2)
      }
      errNorm = sqrt(errNorm / Double(n))

      if errNorm <= 1 {
        let tOld = t
        let yOld = y
        t += h
        y = yNew
        tList.append(t)
        yList.append(y)

        // Record the accepted step for dense-output evaluation
        denseSteps.append(DenseStep(
          tStart: tOld, tEnd: t,
          yStart: yOld, yEnd: y,
          stages: stages,
          h: h
        ))

        // Increase step size
        if errNorm > 0 {
          let factor = min(5, 0.9 * pow(1 / errNorm, 0.2))
          h = direction * min(abs(h * factor), maxStep)
        } else {
          h = direction * min(abs(h * 5), maxStep)
        }
      } else {
        // Reject step, decrease step size
        let factor = max(0.1, 0.9 * pow(1 / errNorm, 0.25))
        h *= factor
      }
    }
  }

  // Build output at requested tEval points using dense-output interpolation,
  // or return all solver steps when tEval is nil.
  var resultT: [Double]
  var resultY: [[Double]]

  if let evalTimes = tEval {
    resultT = evalTimes
    resultY = []

    for tE in evalTimes {
      // Find the dense step whose interval brackets tE.
      // A point at exactly t0 maps to index 0 even though no dense step
      // covers it; handle that boundary first.
      if abs(tE - t0) < 1e-15 * max(1, abs(t0)) {
        resultY.append(y0)
        continue
      }

      // Search for the step containing tE (handles forward and backward integration).
      var bestStep: DenseStep? = nil
      for step in denseSteps {
        let inInterval =
          direction > 0
          ? (step.tStart <= tE + 1e-14 && tE <= step.tEnd + 1e-14)
          : (step.tEnd - 1e-14 <= tE && tE <= step.tStart + 1e-14)
        if inInterval {
          bestStep = step
          break
        }
      }

      guard let step = bestStep else {
        // tE is outside the integration range — clamp to nearest endpoint.
        let yFallback = direction > 0
          ? (tE <= t0 ? y0 : yList.last ?? y0)
          : (tE >= t0 ? y0 : yList.last ?? y0)
        resultY.append(yFallback)
        continue
      }

      // Snap to exact endpoints to avoid floating-point drift at boundaries.
      // The interpolant at s=0 gives yStart and at s=1 gives yEnd exactly
      // in exact arithmetic, but floating-point rounding can introduce tiny
      // errors; returning the stored values directly is both faster and exact.
      let eps = 1e-12 * max(1.0, abs(step.h))
      if abs(tE - step.tStart) < eps {
        resultY.append(step.yStart)
        continue
      }
      if abs(tE - step.tEnd) < eps {
        resultY.append(step.yEnd)
        continue
      }

      // Choose the interpolant appropriate for the current method.
      let yInterp: [Double]
      switch method {
      case .rk45:
        yInterp = rk45DenseOutput(step: step, at: tE)
      case .rk23:
        yInterp = rk23DenseOutput(step: step, at: tE)
      case .rk4:
        // RK4 records only two stages (endpoint derivatives); use linear
        // interpolation between the solver's output values for this step.
        let frac = abs(step.h) > 1e-15 ? (tE - step.tStart) / step.h : 0
        yInterp = (0..<n).map { i in
          step.yStart[i] + frac * (step.yEnd[i] - step.yStart[i])
        }
      }
      resultY.append(yInterp)
    }
  } else {
    resultT = tList
    resultY = yList
  }

  let success = direction * (tf - t) <= 1e-12 * abs(tf)

  return ODEResult(
    t: resultT,
    y: resultY,
    success: success,
    message: success ? "Integration successful" : "Max iterations reached",
    nfev: nfev
  )
}

/// odeint-style ODE integration (scipy.integrate.odeint compatible).
///
/// Integrates the ODE in a single continuous pass from `t[0]` to `t[last]`
/// and evaluates the dense-output interpolant at each requested time point.
/// This avoids per-interval solver restarts and the accuracy loss and overhead
/// they introduce: the solver step-size controller never has to re-estimate an
/// initial step, and the FSAL (first-same-as-last) property of RK45 is
/// preserved across the full interval.
///
/// - Parameters:
///   - func_: Function(y, t) returning dy/dt (note: y first, then t)
///   - y0: Initial state
///   - t: Monotone array of times at which to evaluate the solution; must
///     have at least 1 element (the initial condition is returned for a
///     single-element array)
///   - rtol: Relative tolerance
///   - atol: Absolute tolerance
/// - Returns: Array of solutions y[time_index][component_index]
public func odeint(
  _ func_: @escaping ([Double], Double) -> [Double],
  y0: [Double],
  t: [Double],
  rtol: Double = 1.49e-8,
  atol: Double = 1.49e-8
) -> [[Double]] {
  guard t.count >= 2 else { return [y0] }

  // Single solver run over the full time span with dense-output tEval.
  // solveIVP evaluates the continuous-extension interpolant at each
  // requested time, so no per-interval restart occurs.
  let sol = solveIVP(
    { y, tVal in func_(y, tVal) },
    tSpan: (t.first!, t.last!),
    y0: y0,
    method: .rk45,
    tEval: t,
    rtol: rtol,
    atol: atol
  )

  // sol.y is already indexed by t; return it directly.
  // If the solve failed we still return whatever was computed so the caller
  // gets partial results rather than an empty array.
  return sol.y.isEmpty ? [y0] : sol.y
}
