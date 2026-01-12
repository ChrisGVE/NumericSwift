//
//  Optimization.swift
//  NumericSwift
//
//  Numerical optimization algorithms following scipy.optimize patterns.
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Constants

/// Default tolerance for convergence in x
public let optimDefaultXTol: Double = 1e-8

/// Default tolerance for convergence in f(x)
public let optimDefaultFTol: Double = 1e-8

/// Default maximum iterations
public let optimDefaultMaxIter: Int = 500

/// Golden ratio
private let phi: Double = (1 + sqrt(5)) / 2
private let resphi: Double = 2 - phi  // ≈ 0.382

// MARK: - Result Types

/// Result from scalar minimization
public struct MinimizeScalarResult {
    public let x: Double
    public let fun: Double
    public let nfev: Int
    public let nit: Int
    public let success: Bool
    public let message: String

    public init(x: Double, fun: Double, nfev: Int, nit: Int, success: Bool, message: String) {
        self.x = x
        self.fun = fun
        self.nfev = nfev
        self.nit = nit
        self.success = success
        self.message = message
    }
}

/// Result from scalar root finding
public struct RootScalarResult {
    public let root: Double
    public let iterations: Int
    public let functionCalls: Int
    public let converged: Bool
    public let flag: String

    public init(root: Double, iterations: Int, functionCalls: Int, converged: Bool, flag: String) {
        self.root = root
        self.iterations = iterations
        self.functionCalls = functionCalls
        self.converged = converged
        self.flag = flag
    }
}

/// Result from multivariate minimization
public struct MinimizeResult {
    public let x: [Double]
    public let fun: Double
    public let nfev: Int
    public let nit: Int
    public let success: Bool
    public let message: String

    public init(x: [Double], fun: Double, nfev: Int, nit: Int, success: Bool, message: String) {
        self.x = x
        self.fun = fun
        self.nfev = nfev
        self.nit = nit
        self.success = success
        self.message = message
    }
}

/// Result from multivariate root finding
public struct RootResult {
    public let x: [Double]
    public let fun: [Double]
    public let success: Bool
    public let message: String
    public let nfev: Int
    public let nit: Int

    public init(x: [Double], fun: [Double], success: Bool, message: String, nfev: Int, nit: Int) {
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.nfev = nfev
        self.nit = nit
    }
}

/// Result from least squares optimization
public struct LeastSquaresResult {
    public let x: [Double]
    public let cost: Double
    public let fun: [Double]
    public let nfev: Int
    public let njev: Int
    public let success: Bool
    public let message: String

    public init(x: [Double], cost: Double, fun: [Double], nfev: Int, njev: Int, success: Bool, message: String) {
        self.x = x
        self.cost = cost
        self.fun = fun
        self.nfev = nfev
        self.njev = njev
        self.success = success
        self.message = message
    }
}

// MARK: - Scalar Minimization

/// Golden section search for scalar minimization.
///
/// Finds the minimum of a unimodal function in the interval [a, b].
///
/// - Parameters:
///   - f: Function to minimize
///   - a: Left bound
///   - b: Right bound
///   - xtol: Tolerance
///   - maxiter: Maximum iterations
/// - Returns: Optimization result
public func goldenSection(
    _ f: (Double) -> Double,
    a: Double,
    b: Double,
    xtol: Double = optimDefaultXTol,
    maxiter: Int = optimDefaultMaxIter
) -> MinimizeScalarResult {
    var a = a, b = b
    var nfev = 0
    var nit = 0

    // Internal points using golden ratio
    var c = a + resphi * (b - a)  // left internal point
    var d = b - resphi * (b - a)  // right internal point
    var fc = f(c); nfev += 1
    var fd = f(d); nfev += 1

    while abs(b - a) > xtol && nit < maxiter {
        nit += 1
        if fc < fd {
            // Minimum is in [a, d], narrow to [a, d]
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = f(c); nfev += 1
        } else {
            // Minimum is in [c, b], narrow to [c, b]
            a = c
            c = d
            fc = fd
            d = b - resphi * (b - a)
            fd = f(d); nfev += 1
        }
    }

    let x = (a + b) / 2.0
    let fval = f(x); nfev += 1

    return MinimizeScalarResult(
        x: x,
        fun: fval,
        nfev: nfev,
        nit: nit,
        success: abs(b - a) <= xtol,
        message: abs(b - a) <= xtol ? "Optimization terminated successfully." : "Maximum iterations reached."
    )
}

/// Brent's method for scalar minimization.
///
/// Combines parabolic interpolation with golden section for faster convergence.
///
/// - Parameters:
///   - f: Function to minimize
///   - a: Left bound
///   - b: Right bound
///   - xtol: Tolerance
///   - maxiter: Maximum iterations
/// - Returns: Optimization result
public func brent(
    _ f: (Double) -> Double,
    a: Double,
    b: Double,
    xtol: Double = optimDefaultXTol,
    maxiter: Int = optimDefaultMaxIter
) -> MinimizeScalarResult {
    let goldenMean: Double = 0.5 * (3.0 - sqrt(5.0))
    let sqrtEps = sqrt(Double.ulpOfOne)

    var a = a, b = b
    if a > b { swap(&a, &b) }

    var nfev = 0
    var nit = 0

    var x = a + goldenMean * (b - a)
    var w = x, v = x
    var fx = f(x); nfev += 1
    var fw = fx, fv = fx

    var d: Double = 0, e: Double = 0

    while nit < maxiter {
        nit += 1
        let midpoint = 0.5 * (a + b)
        let tol1 = sqrtEps * abs(x) + xtol / 3.0
        let tol2 = 2.0 * tol1

        // Check for convergence
        if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)) {
            return MinimizeScalarResult(
                x: x, fun: fx, nfev: nfev, nit: nit,
                success: true, message: "Optimization terminated successfully."
            )
        }

        var useParabolic = false
        var p: Double = 0, q: Double = 0, r: Double = 0

        if abs(e) > tol1 {
            // Fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0 { p = -p } else { q = -q }
            r = e
            e = d

            if abs(p) < abs(0.5 * q * r) && p > q * (a - x) && p < q * (b - x) {
                // Parabolic step accepted
                d = p / q
                let u = x + d
                // Don't evaluate too close to bounds
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = x < midpoint ? tol1 : -tol1
                }
                useParabolic = true
            }
        }

        if !useParabolic {
            // Golden section step
            e = (x < midpoint) ? (b - x) : (a - x)
            d = goldenMean * e
        }

        // Don't evaluate too close to current point
        let u: Double
        if abs(d) >= tol1 {
            u = x + d
        } else {
            u = x + (d >= 0 ? tol1 : -tol1)
        }

        let fu = f(u); nfev += 1

        // Update interval
        if fu <= fx {
            if u < x { b = x } else { a = x }
            v = w; fv = fw
            w = x; fw = fx
            x = u; fx = fu
        } else {
            if u < x { a = u } else { b = u }
            if fu <= fw || w == x {
                v = w; fv = fw
                w = u; fw = fu
            } else if fu <= fv || v == x || v == w {
                v = u; fv = fu
            }
        }
    }

    return MinimizeScalarResult(
        x: x, fun: fx, nfev: nfev, nit: nit,
        success: false, message: "Maximum iterations reached."
    )
}

// MARK: - Scalar Root Finding

/// Bisection method for root finding.
///
/// - Parameters:
///   - f: Function to find root of
///   - a: Left bound
///   - b: Right bound
///   - xtol: Tolerance
///   - maxiter: Maximum iterations
/// - Returns: Root finding result
public func bisect(
    _ f: (Double) -> Double,
    a: Double,
    b: Double,
    xtol: Double = optimDefaultXTol,
    maxiter: Int = optimDefaultMaxIter
) -> RootScalarResult {
    var a = a, b = b
    var nfev = 0
    var nit = 0

    var fa = f(a); nfev += 1
    var fb = f(b); nfev += 1

    // Check for sign change
    if fa * fb > 0 {
        return RootScalarResult(
            root: .nan, iterations: 0, functionCalls: nfev,
            converged: false, flag: "f(a) and f(b) must have different signs"
        )
    }

    // Handle exact roots at boundaries
    if fa == 0 {
        return RootScalarResult(root: a, iterations: 0, functionCalls: nfev, converged: true, flag: "converged")
    }
    if fb == 0 {
        return RootScalarResult(root: b, iterations: 0, functionCalls: nfev, converged: true, flag: "converged")
    }

    while abs(b - a) > xtol && nit < maxiter {
        nit += 1
        let c = (a + b) / 2.0
        let fc = f(c); nfev += 1

        if fc == 0 {
            return RootScalarResult(root: c, iterations: nit, functionCalls: nfev, converged: true, flag: "converged")
        }

        if fa * fc < 0 {
            b = c
            fb = fc
        } else {
            a = c
            fa = fc
        }
    }

    let root = (a + b) / 2.0
    return RootScalarResult(
        root: root, iterations: nit, functionCalls: nfev,
        converged: abs(b - a) <= xtol, flag: abs(b - a) <= xtol ? "converged" : "maxiter reached"
    )
}

/// Newton's method for scalar root finding.
///
/// - Parameters:
///   - f: Function to find root of
///   - fprime: Derivative of f (optional, uses numerical diff if nil)
///   - x0: Initial guess
///   - xtol: Tolerance
///   - maxiter: Maximum iterations
/// - Returns: Root finding result
public func newton(
    _ f: (Double) -> Double,
    fprime: ((Double) -> Double)? = nil,
    x0: Double,
    xtol: Double = optimDefaultXTol,
    maxiter: Int = optimDefaultMaxIter
) -> RootScalarResult {
    var x = x0
    var nfev = 0
    var nit = 0

    for _ in 0..<maxiter {
        nit += 1
        let fx = f(x); nfev += 1

        // Calculate derivative (use provided or numerical)
        let fp: Double
        if let fprime = fprime {
            fp = fprime(x)
        } else {
            let h = sqrt(Double.ulpOfOne) * max(abs(x), 1.0)
            let fplus = f(x + h); nfev += 1
            let fminus = f(x - h); nfev += 1
            fp = (fplus - fminus) / (2 * h)
        }

        if abs(fp) < 1e-14 {
            return RootScalarResult(
                root: x, iterations: nit, functionCalls: nfev,
                converged: false, flag: "derivative is zero"
            )
        }

        let dx = fx / fp
        x = x - dx

        if abs(dx) < xtol || abs(fx) < xtol {
            return RootScalarResult(
                root: x, iterations: nit, functionCalls: nfev,
                converged: true, flag: "converged"
            )
        }
    }

    return RootScalarResult(
        root: x, iterations: nit, functionCalls: nfev,
        converged: false, flag: "maxiter reached"
    )
}

/// Secant method for scalar root finding.
///
/// - Parameters:
///   - f: Function to find root of
///   - x0: First initial guess
///   - x1: Second initial guess (optional)
///   - xtol: Tolerance
///   - maxiter: Maximum iterations
/// - Returns: Root finding result
public func secant(
    _ f: (Double) -> Double,
    x0: Double,
    x1: Double? = nil,
    xtol: Double = optimDefaultXTol,
    maxiter: Int = optimDefaultMaxIter
) -> RootScalarResult {
    var x0 = x0
    var x1 = x1 ?? (x0 + 0.001 * max(abs(x0), 1.0))
    var nfev = 0
    var nit = 0

    var f0 = f(x0); nfev += 1
    var f1 = f(x1); nfev += 1

    for _ in 0..<maxiter {
        nit += 1

        if abs(f1 - f0) < 1e-14 {
            return RootScalarResult(
                root: x1, iterations: nit, functionCalls: nfev,
                converged: false, flag: "denominator too small"
            )
        }

        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0 = x1; f0 = f1
        x1 = x2
        f1 = f(x1); nfev += 1

        if abs(x1 - x0) < xtol || abs(f1) < xtol {
            return RootScalarResult(
                root: x1, iterations: nit, functionCalls: nfev,
                converged: true, flag: "converged"
            )
        }
    }

    return RootScalarResult(
        root: x1, iterations: nit, functionCalls: nfev,
        converged: false, flag: "maxiter reached"
    )
}

// MARK: - Multivariate Minimization

/// Nelder-Mead simplex method for multivariate minimization.
///
/// - Parameters:
///   - f: Function to minimize
///   - x0: Initial guess
///   - xtol: Tolerance in x
///   - ftol: Tolerance in f(x)
///   - maxiter: Maximum iterations
/// - Returns: Optimization result
public func nelderMead(
    _ f: @escaping ([Double]) -> Double,
    x0: [Double],
    xtol: Double = optimDefaultXTol,
    ftol: Double = optimDefaultFTol,
    maxiter: Int? = nil
) -> MinimizeResult {
    let n = x0.count
    let maxIterations = maxiter ?? 200 * n
    var nfev = 0
    var nit = 0

    // Nelder-Mead coefficients
    let alpha: Double = 1.0   // Reflection
    let gamma: Double = 2.0   // Expansion
    let rho: Double = 0.5     // Contraction
    let sigma: Double = 0.5   // Shrink

    // Initialize simplex
    var simplex: [[Double]] = [x0]
    var fvalues: [Double] = [f(x0)]; nfev += 1

    for i in 0..<n {
        var point = x0
        let delta = abs(x0[i]) > 0.00025 ? 0.05 : 0.00025
        point[i] += delta
        simplex.append(point)
        fvalues.append(f(point)); nfev += 1
    }

    while nit < maxIterations {
        nit += 1

        // Sort simplex by function values
        let sorted = zip(simplex.indices, fvalues).sorted { $0.1 < $1.1 }
        simplex = sorted.map { simplex[$0.0] }
        fvalues = sorted.map { $0.1 }

        // Check convergence
        let frange = fvalues.last! - fvalues.first!
        var xrange: Double = 0
        for i in 1...n {
            for j in 0..<n {
                xrange = max(xrange, abs(simplex[i][j] - simplex[0][j]))
            }
        }

        if frange < ftol && xrange < xtol {
            return MinimizeResult(
                x: simplex[0], fun: fvalues[0], nfev: nfev, nit: nit,
                success: true, message: "Optimization terminated successfully."
            )
        }

        // Calculate centroid (excluding worst point)
        var centroid = [Double](repeating: 0, count: n)
        for i in 0..<n {
            for j in 0..<n {
                centroid[j] += simplex[i][j]
            }
        }
        centroid = centroid.map { $0 / Double(n) }

        // Reflection
        var reflected = [Double](repeating: 0, count: n)
        for j in 0..<n {
            reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[n][j])
        }
        let fReflected = f(reflected); nfev += 1

        if fReflected < fvalues[0] {
            // Expansion
            var expanded = [Double](repeating: 0, count: n)
            for j in 0..<n {
                expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j])
            }
            let fExpanded = f(expanded); nfev += 1

            if fExpanded < fReflected {
                simplex[n] = expanded
                fvalues[n] = fExpanded
            } else {
                simplex[n] = reflected
                fvalues[n] = fReflected
            }
        } else if fReflected < fvalues[n-1] {
            simplex[n] = reflected
            fvalues[n] = fReflected
        } else {
            // Contraction
            let useOutside = fReflected < fvalues[n]
            var contracted = [Double](repeating: 0, count: n)
            for j in 0..<n {
                if useOutside {
                    contracted[j] = centroid[j] + rho * (reflected[j] - centroid[j])
                } else {
                    contracted[j] = centroid[j] + rho * (simplex[n][j] - centroid[j])
                }
            }
            let fContracted = f(contracted); nfev += 1

            if fContracted < (useOutside ? fReflected : fvalues[n]) {
                simplex[n] = contracted
                fvalues[n] = fContracted
            } else {
                // Shrink
                for i in 1...n {
                    for j in 0..<n {
                        simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j])
                    }
                    fvalues[i] = f(simplex[i]); nfev += 1
                }
            }
        }
    }

    // Sort one final time
    let sorted = zip(simplex.indices, fvalues).sorted { $0.1 < $1.1 }
    let bestX = simplex[sorted[0].0]
    let bestF = sorted[0].1

    return MinimizeResult(
        x: bestX, fun: bestF, nfev: nfev, nit: nit,
        success: false, message: "Maximum iterations reached."
    )
}

// MARK: - Multivariate Root Finding

/// Newton's method for systems of equations.
///
/// - Parameters:
///   - f: Function returning vector of residuals
///   - x0: Initial guess
///   - tol: Tolerance
///   - maxiter: Maximum iterations
/// - Returns: Root finding result
public func newtonMulti(
    _ f: @escaping ([Double]) -> [Double],
    x0: [Double],
    tol: Double = optimDefaultXTol,
    maxiter: Int = optimDefaultMaxIter
) -> RootResult {
    let n = x0.count
    var x = x0
    var nfev = 0
    var nit = 0

    for _ in 0..<maxiter {
        nit += 1
        let fx = f(x); nfev += 1

        // Check convergence
        let norm = sqrt(fx.reduce(0) { $0 + $1 * $1 })
        if norm < tol {
            return RootResult(
                x: x, fun: fx, success: true,
                message: "Root found.", nfev: nfev, nit: nit
            )
        }

        // Compute Jacobian numerically
        var jacobian = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        let h = sqrt(Double.ulpOfOne)
        for j in 0..<n {
            var xp = x
            xp[j] += h
            let fxp = f(xp); nfev += 1
            for i in 0..<n {
                jacobian[i][j] = (fxp[i] - fx[i]) / h
            }
        }

        // Solve J * dx = -f using Gaussian elimination
        var A = jacobian
        var b = fx.map { -$0 }

        for col in 0..<n {
            // Find pivot
            var maxRow = col
            for row in (col+1)..<n {
                if abs(A[row][col]) > abs(A[maxRow][col]) {
                    maxRow = row
                }
            }
            A.swapAt(col, maxRow)
            b.swapAt(col, maxRow)

            if abs(A[col][col]) < 1e-14 {
                return RootResult(
                    x: x, fun: fx, success: false,
                    message: "Singular Jacobian.", nfev: nfev, nit: nit
                )
            }

            // Eliminate
            for row in (col+1)..<n {
                let factor = A[row][col] / A[col][col]
                for k in col..<n {
                    A[row][k] -= factor * A[col][k]
                }
                b[row] -= factor * b[col]
            }
        }

        // Back substitution
        var dx = [Double](repeating: 0, count: n)
        for i in stride(from: n-1, through: 0, by: -1) {
            var sum = b[i]
            for j in (i+1)..<n {
                sum -= A[i][j] * dx[j]
            }
            dx[i] = sum / A[i][i]
        }

        // Update x
        for i in 0..<n {
            x[i] += dx[i]
        }

        // Check for small step
        let dxNorm = sqrt(dx.reduce(0) { $0 + $1 * $1 })
        if dxNorm < tol {
            return RootResult(
                x: x, fun: f(x), success: true,
                message: "Root found.", nfev: nfev + 1, nit: nit
            )
        }
    }

    return RootResult(
        x: x, fun: f(x), success: false,
        message: "Maximum iterations reached.", nfev: nfev + 1, nit: nit
    )
}

// MARK: - Least Squares

/// Levenberg-Marquardt algorithm for nonlinear least squares.
///
/// Minimizes sum(residuals(x)^2) using the Levenberg-Marquardt method.
///
/// - Parameters:
///   - residuals: Function returning residuals vector
///   - x0: Initial guess
///   - ftol: Relative tolerance for cost function
///   - xtol: Relative tolerance for parameters
///   - maxiter: Maximum iterations
/// - Returns: Least squares result
public func leastSquares(
    _ residuals: @escaping ([Double]) -> [Double],
    x0: [Double],
    ftol: Double = 1e-8,
    xtol: Double = 1e-8,
    maxiter: Int = 100
) -> LeastSquaresResult {
    let n = x0.count
    var x = x0
    var nfev = 0
    var njev = 0

    // Initial residual evaluation
    var r = residuals(x); nfev += 1
    let m = r.count

    // Compute initial cost
    var cost = 0.5 * r.reduce(0) { $0 + $1 * $1 }

    // Levenberg-Marquardt parameters
    var lambda = 0.001
    let lambdaUp = 10.0
    let lambdaDown = 0.1

    for _ in 0..<maxiter {
        // Compute Jacobian numerically (m x n matrix)
        var J = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)
        let h = sqrt(Double.ulpOfOne) * max(1.0, x.map { abs($0) }.max() ?? 1.0)
        for j in 0..<n {
            var xp = x
            xp[j] += h
            let rp = residuals(xp); nfev += 1
            for i in 0..<m {
                J[i][j] = (rp[i] - r[i]) / h
            }
        }
        njev += 1

        // Compute J^T * J (n x n)
        var JTJ = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in 0..<n {
                var sum = 0.0
                for k in 0..<m {
                    sum += J[k][i] * J[k][j]
                }
                JTJ[i][j] = sum
            }
        }

        // Compute J^T * r (n x 1)
        var JTr = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var sum = 0.0
            for k in 0..<m {
                sum += J[k][i] * r[k]
            }
            JTr[i] = sum
        }

        // Solve (J^T*J + lambda*diag(J^T*J)) * dx = -J^T*r
        var A = JTJ
        for i in 0..<n {
            A[i][i] += lambda * max(JTJ[i][i], 1e-10)
        }
        var b = JTr.map { -$0 }

        // Gaussian elimination with partial pivoting
        for col in 0..<n {
            var maxRow = col
            for row in (col+1)..<n {
                if abs(A[row][col]) > abs(A[maxRow][col]) {
                    maxRow = row
                }
            }
            A.swapAt(col, maxRow)
            b.swapAt(col, maxRow)

            guard abs(A[col][col]) > 1e-14 else {
                return LeastSquaresResult(
                    x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                    success: false, message: "Singular matrix in LM step."
                )
            }

            for row in (col+1)..<n {
                let factor = A[row][col] / A[col][col]
                for k in col..<n {
                    A[row][k] -= factor * A[col][k]
                }
                b[row] -= factor * b[col]
            }
        }

        // Back substitution
        var dx = [Double](repeating: 0, count: n)
        for i in stride(from: n-1, through: 0, by: -1) {
            var sum = b[i]
            for j in (i+1)..<n {
                sum -= A[i][j] * dx[j]
            }
            dx[i] = sum / A[i][i]
        }

        // Trial step
        var xNew = x
        for i in 0..<n {
            xNew[i] += dx[i]
        }

        let rNew = residuals(xNew); nfev += 1
        let costNew = 0.5 * rNew.reduce(0) { $0 + $1 * $1 }

        // Check if step is accepted
        if costNew < cost {
            // Accept step
            let costRatio = abs(cost - costNew) / max(cost, 1e-14)
            let xNorm = sqrt(dx.reduce(0) { $0 + $1 * $1 })
            let paramNorm = sqrt(x.reduce(0) { $0 + $1 * $1 })

            x = xNew
            r = rNew
            cost = costNew
            lambda *= lambdaDown

            // Check convergence
            if costRatio < ftol {
                return LeastSquaresResult(
                    x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                    success: true, message: "Both `ftol` and `xtol` termination conditions are satisfied."
                )
            }
            if xNorm < xtol * (1 + paramNorm) {
                return LeastSquaresResult(
                    x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                    success: true, message: "Both `ftol` and `xtol` termination conditions are satisfied."
                )
            }
        } else {
            // Reject step, increase damping
            lambda *= lambdaUp
        }
    }

    return LeastSquaresResult(
        x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
        success: false, message: "Maximum iterations reached."
    )
}

/// Curve fitting using nonlinear least squares.
///
/// Fits a model function to data points using Levenberg-Marquardt.
///
/// - Parameters:
///   - f: Model function (params, x) -> y
///   - xdata: X data points
///   - ydata: Y data points
///   - p0: Initial parameter guess
///   - ftol: Relative tolerance for cost function
///   - xtol: Relative tolerance for parameters
///   - maxiter: Maximum iterations
/// - Returns: Tuple of (optimal params, covariance matrix, info)
public func curveFit(
    _ f: @escaping ([Double], Double) -> Double,
    xdata: [Double],
    ydata: [Double],
    p0: [Double],
    ftol: Double = 1e-8,
    xtol: Double = 1e-8,
    maxiter: Int = 100
) -> (popt: [Double], pcov: [[Double]], info: LeastSquaresResult) {
    let n = p0.count
    let m = xdata.count

    // Create residuals function
    let residuals: ([Double]) -> [Double] = { params in
        var r = [Double](repeating: 0, count: m)
        for i in 0..<m {
            r[i] = ydata[i] - f(params, xdata[i])
        }
        return r
    }

    // Run least squares
    let result = leastSquares(residuals, x0: p0, ftol: ftol, xtol: xtol, maxiter: maxiter)

    // Estimate covariance matrix
    let h = sqrt(Double.ulpOfOne) * max(1.0, result.x.map { abs($0) }.max() ?? 1.0)
    var J = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)

    for j in 0..<n {
        var pp = result.x
        var pm = result.x
        pp[j] += h
        pm[j] -= h

        for i in 0..<m {
            let fp = f(pp, xdata[i])
            let fm = f(pm, xdata[i])
            J[i][j] = (fp - fm) / (2 * h)
        }
    }

    // Compute J^T * J
    var JTJ = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
    for i in 0..<n {
        for j in 0..<n {
            var sum = 0.0
            for k in 0..<m {
                sum += J[k][i] * J[k][j]
            }
            JTJ[i][j] = sum
        }
    }

    // Invert JTJ to get pcov
    var pcov = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
    var A = JTJ
    // Initialize identity
    for i in 0..<n {
        pcov[i][i] = 1.0
    }

    // Forward elimination with full pivot
    for col in 0..<n {
        var maxRow = col
        for row in (col+1)..<n {
            if abs(A[row][col]) > abs(A[maxRow][col]) {
                maxRow = row
            }
        }
        A.swapAt(col, maxRow)
        pcov.swapAt(col, maxRow)

        let pivot = A[col][col]
        if abs(pivot) > 1e-14 {
            for j in 0..<n {
                A[col][j] /= pivot
                pcov[col][j] /= pivot
            }
            for row in 0..<n where row != col {
                let factor = A[row][col]
                for j in 0..<n {
                    A[row][j] -= factor * A[col][j]
                    pcov[row][j] -= factor * pcov[col][j]
                }
            }
        }
    }

    // Scale by variance estimate
    let dof = max(1, m - n)
    let s2 = result.cost * 2 / Double(dof)
    for i in 0..<n {
        for j in 0..<n {
            pcov[i][j] *= s2
        }
    }

    return (result.x, pcov, result)
}
