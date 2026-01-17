//
//  Integration.swift
//  NumericSwift
//
//  Numerical integration and ODE solvers following scipy.integrate patterns.
//
//  Licensed under the MIT License.
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
    0.0
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
    0.209482141084728
]

/// Weights for 7-point Gauss rule (embedded in K15)
private let gWeights: [Double] = [
    0.129484966168870,
    0.279705391489277,
    0.381830050505119,
    0.417959183673469
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
    return dblquad(f, xa: xa, xb: xb, ya: { _ in ya }, yb: { _ in yb }, epsabs: epsabs, epsrel: epsrel)
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
    return tplquad(f, xa: xa, xb: xb,
                   ya: { _ in ya }, yb: { _ in yb },
                   za: { _, _ in za }, zb: { _, _ in zb },
                   epsabs: epsabs, epsrel: epsrel)
}

// MARK: - Fixed-Order Gaussian Quadrature

/// Gauss-Legendre quadrature points and weights for n=1 to 10
private let gaussLegendre: [Int: [(x: Double, w: Double)]] = [
    1: [(0, 2)],
    2: [(-0.5773502691896257, 1), (0.5773502691896257, 1)],
    3: [(-0.7745966692414834, 0.5555555555555556),
        (0, 0.8888888888888888),
        (0.7745966692414834, 0.5555555555555556)],
    4: [(-0.8611363115940526, 0.3478548451374538),
        (-0.3399810435848563, 0.6521451548625461),
        (0.3399810435848563, 0.6521451548625461),
        (0.8611363115940526, 0.3478548451374538)],
    5: [(-0.9061798459386640, 0.2369268850561891),
        (-0.5384693101056831, 0.4786286704993665),
        (0, 0.5688888888888889),
        (0.5384693101056831, 0.4786286704993665),
        (0.9061798459386640, 0.2369268850561891)],
    6: [(-0.9324695142031521, 0.1713244923791704),
        (-0.6612093864662645, 0.3607615730481386),
        (-0.2386191860831969, 0.4679139345726910),
        (0.2386191860831969, 0.4679139345726910),
        (0.6612093864662645, 0.3607615730481386),
        (0.9324695142031521, 0.1713244923791704)],
    7: [(-0.9491079123427585, 0.1294849661688697),
        (-0.7415311855993945, 0.2797053914892766),
        (-0.4058451513773972, 0.3818300505051189),
        (0, 0.4179591836734694),
        (0.4058451513773972, 0.3818300505051189),
        (0.7415311855993945, 0.2797053914892766),
        (0.9491079123427585, 0.1294849661688697)],
    8: [(-0.9602898564975363, 0.1012285362903763),
        (-0.7966664774136267, 0.2223810344533745),
        (-0.5255324099163290, 0.3137066458778873),
        (-0.1834346424956498, 0.3626837833783620),
        (0.1834346424956498, 0.3626837833783620),
        (0.5255324099163290, 0.3137066458778873),
        (0.7966664774136267, 0.2223810344533745),
        (0.9602898564975363, 0.1012285362903763)],
    9: [(-0.9681602395076261, 0.0812743883615744),
        (-0.8360311073266358, 0.1806481606948574),
        (-0.6133714327005904, 0.2606106964029354),
        (-0.3242534234038089, 0.3123470770400029),
        (0, 0.3302393550012598),
        (0.3242534234038089, 0.3123470770400029),
        (0.6133714327005904, 0.2606106964029354),
        (0.8360311073266358, 0.1806481606948574),
        (0.9681602395076261, 0.0812743883615744)],
    10: [(-0.9739065285171717, 0.0666713443086881),
         (-0.8650633666889845, 0.1494513491505806),
         (-0.6794095682990244, 0.2190863625159820),
         (-0.4333953941292472, 0.2692667193099963),
         (-0.1488743389816312, 0.2955242247147529),
         (0.1488743389816312, 0.2955242247147529),
         (0.4333953941292472, 0.2692667193099963),
         (0.6794095682990244, 0.2190863625159820),
         (0.8650633666889845, 0.1494513491505806),
         (0.9739065285171717, 0.0666713443086881)]
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
        R[i].append(0.5 * R[i-1][0] + h * sum)

        // Richardson extrapolation
        for j in 1...i {
            let factor = pow(4.0, Double(j))
            let value = (factor * R[i][j-1] - R[i-1][j-1]) / (factor - 1.0)
            R[i].append(value)
        }

        // Check convergence
        let error = abs(R[i][i] - R[i-1][i-1])
        if error < tol {
            return QuadResult(value: R[i][i], error: error, neval: (1 << (i + 1)) - 1)
        }
    }

    let finalError = abs(R[divmax][divmax] - R[divmax-1][divmax-1])
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

/// Simpson's rule integration with x values.
public func simps(_ y: [Double], x: [Double]) -> Double {
    guard y.count == x.count && y.count >= 3 else { return 0 }
    let dx = (x[x.count - 1] - x[0]) / Double(x.count - 1)
    return simps(y, dx: dx)
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

// MARK: - ODE Solvers

/// Dormand-Prince A matrix (lower triangular part)
private let dpA: [[Double]] = [
    [],
    [1.0/5.0],
    [3.0/40.0, 9.0/40.0],
    [44.0/45.0, -56.0/15.0, 32.0/9.0],
    [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0],
    [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0],
    [35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0]
]

/// Dormand-Prince C coefficients (nodes)
private let dpC: [Double] = [0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1, 1]

/// Dormand-Prince B coefficients (5th order weights)
private let dpB: [Double] = [35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0]

/// Dormand-Prince E coefficients (error estimate: 5th - 4th order)
private let dpE: [Double] = [71.0/57600.0, 0, -71.0/16695.0, 71.0/1920.0, -17253.0/339200.0, 22.0/525.0, -1.0/40.0]

/// Bogacki-Shampine A matrix
private let bsA: [[Double]] = [
    [],
    [1.0/2.0],
    [0, 3.0/4.0],
    [2.0/9.0, 1.0/3.0, 4.0/9.0]
]

private let bsC: [Double] = [0, 1.0/2.0, 3.0/4.0, 1]
private let bsB: [Double] = [2.0/9.0, 1.0/3.0, 4.0/9.0, 0]
private let bsE: [Double] = [-5.0/72.0, 1.0/12.0, 1.0/9.0, -1.0/8.0]

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
        yNew[i] = y[i] + (h / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
    }
    return yNew
}

/// Single Dormand-Prince RK45 step
private func rk45Step(
    _ f: ([Double], Double) -> [Double],
    _ t: Double,
    _ y: [Double],
    _ h: Double
) -> (yNew: [Double], error: [Double]) {
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

    return (yNew, err)
}

/// Single Bogacki-Shampine RK23 step
private func rk23Step(
    _ f: ([Double], Double) -> [Double],
    _ t: Double,
    _ y: [Double],
    _ h: Double
) -> (yNew: [Double], error: [Double]) {
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

    return (yNew, err)
}

/// ODE solver method
public enum ODEMethod: String {
    case rk45 = "RK45"
    case rk23 = "RK23"
    case rk4 = "RK4"
}

/// Solve initial value problem for ODE system.
///
/// - Parameters:
///   - fun: Function(y, t) returning dy/dt
///   - tSpan: (t0, tf) initial and final time
///   - y0: Initial state
///   - method: ODE method (RK45, RK23, RK4)
///   - tEval: Optional specific times for output
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
            y = rk4Step(fun, t, y, h)
            nfev += 4
            t += h
            tList.append(t)
            yList.append(y)
        } else {
            let (yNew, err) = method == .rk23
                ? rk23Step(fun, t, y, h)
                : rk45Step(fun, t, y, h)
            nfev += method == .rk23 ? 4 : 7

            // Error control
            var errNorm: Double = 0
            for i in 0..<n {
                let scale = atol + rtol * max(abs(y[i]), abs(yNew[i]))
                errNorm += pow(err[i] / scale, 2)
            }
            errNorm = sqrt(errNorm / Double(n))

            if errNorm <= 1 {
                t += h
                y = yNew
                tList.append(t)
                yList.append(y)

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

    // Interpolate to tEval if specified
    var resultT: [Double]
    var resultY: [[Double]]

    if let evalTimes = tEval {
        resultT = evalTimes
        resultY = []
        for tE in evalTimes {
            // Find bracketing interval
            var idx = 0
            for j in 0..<(tList.count - 1) {
                if (tList[j] <= tE && tE <= tList[j + 1]) ||
                   (tList[j] >= tE && tE >= tList[j + 1]) {
                    idx = j
                    break
                }
            }

            // Linear interpolation
            let t1 = tList[idx]
            let t2 = tList[min(idx + 1, tList.count - 1)]
            let frac = abs(t2 - t1) > 1e-15 ? (tE - t1) / (t2 - t1) : 0

            var yInterp = [Double](repeating: 0, count: n)
            for i in 0..<n {
                yInterp[i] = yList[idx][i] + frac * (yList[min(idx + 1, yList.count - 1)][i] - yList[idx][i])
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
/// - Parameters:
///   - func_: Function(y, t) returning dy/dt (note: y first, then t)
///   - y0: Initial state
///   - t: Array of times at which to compute solution
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

    var result: [[Double]] = [y0]
    var currentY = y0

    for j in 0..<(t.count - 1) {
        let sol = solveIVP(
            { y, tVal in func_(y, tVal) },
            tSpan: (t[j], t[j + 1]),
            y0: currentY,
            method: .rk45,
            rtol: rtol,
            atol: atol
        )
        currentY = sol.y.last ?? currentY
        result.append(currentY)
    }

    return result
}
