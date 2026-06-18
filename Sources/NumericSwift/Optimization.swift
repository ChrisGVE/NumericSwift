//
//  Optimization.swift
//  NumericSwift
//
//  Numerical optimization algorithms following scipy.optimize patterns.
//
//  Licensed under the Apache License, Version 2.0.
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

/// Relative step scale for central finite-difference Jacobians.
///
/// Each column step h_j = finiteDiffStepScale * max(1, |x_j|) follows the
/// scipy.optimize.approx_derivative formula, giving O(h²) accuracy while
/// keeping h above the floating-point ULP of x_j for any variable magnitude.
///
/// Reference: scipy.optimize._numdiff.approx_derivative (scipy.org)
private let finiteDiffStepScale: Double = sqrt(Double.ulpOfOne)

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

/// Result from scalar root finding.
///
/// The `diagnostics` field is the recoverable self-awareness channel
/// (``NumericDiagnostic``). The bracketing methods (``bisect(_:a:b:xtol:maxiter:)``,
/// ``brentq(_:a:b:xtol:rtol:maxiter:)``) append an
/// ``NumericDiagnostic/outsideEnvelope(method:reason:)`` when the supplied bracket
/// has no sign change (`f(a)·f(b) > 0`) — bracketing root finders are mathematically
/// invalid there. The open methods (``newton(_:fprime:x0:xtol:maxiter:)``,
/// ``secant(_:x0:x1:xtol:maxiter:)``) append one when the derivative (or secant
/// slope) collapses toward zero or the iteration diverges / exhausts its budget —
/// regimes where the returned iterate is not a trustworthy root. A self-aware
/// caller inspects `diagnostics` (mirroring SciPy's `ValueError`/`RuntimeWarning`).
public struct RootScalarResult: Sendable {
    public let root: Double
    public let iterations: Int
    public let functionCalls: Int
    public let converged: Bool
    public let flag: String

    /// Recoverable diagnostics emitted during root finding.
    ///
    /// Empty for a clean convergence inside the method's valid envelope. A
    /// non-empty list with an ``NumericDiagnostic/outsideEnvelope(method:reason:)``
    /// entry means the result may be unreliable (invalid bracket, vanishing
    /// derivative, or divergence). See the type-level discussion above.
    public let diagnostics: [NumericDiagnostic]

    public init(
        root: Double,
        iterations: Int,
        functionCalls: Int,
        converged: Bool,
        flag: String,
        diagnostics: [NumericDiagnostic] = []
    ) {
        self.root = root
        self.iterations = iterations
        self.functionCalls = functionCalls
        self.converged = converged
        self.flag = flag
        self.diagnostics = diagnostics
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

    // A NaN endpoint evaluation defeats the sign-change test (`NaN * x > 0` is
    // false for any x), so it would otherwise slip past the guard below and the
    // iteration would propagate NaN while reporting `converged`. Detect it first.
    if fa.isNaN || fb.isNaN {
        return RootScalarResult(
            root: .nan, iterations: 0, functionCalls: nfev,
            converged: false, flag: "f(a) or f(b) is NaN",
            diagnostics: [.outsideEnvelope(
                method: "bisect",
                reason: "f evaluated to NaN at a bracket endpoint — the function is "
                    + "undefined on [\(a), \(b)] and no root can be bracketed"
            )]
        )
    }

    // Check for sign change. A bracketing method is mathematically invalid when
    // f(a) and f(b) share a sign — there is no guaranteed root in [a, b]. SciPy
    // raises a ValueError here; we surface a recoverable `outsideEnvelope`
    // diagnostic so a self-aware caller can detect the misuse (workbench §5).
    if fa * fb > 0 {
        return RootScalarResult(
            root: .nan, iterations: 0, functionCalls: nfev,
            converged: false, flag: "f(a) and f(b) must have different signs",
            diagnostics: [.outsideEnvelope(
                method: "bisect",
                reason: "f(a)·f(b) > 0 — bracket has no sign change; bisection is "
                    + "invalid and cannot guarantee a root in [\(a), \(b)]"
            )]
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

        // A vanishing derivative makes the Newton step f(x)/f'(x) blow up — the
        // method is outside its valid envelope (it needs f'(x) ≠ 0 near the root).
        if abs(fp) < 1e-14 {
            return RootScalarResult(
                root: x, iterations: nit, functionCalls: nfev,
                converged: false, flag: "derivative is zero",
                diagnostics: [.outsideEnvelope(
                    method: "newton",
                    reason: "|f'(x)| < 1e-14 at x=\(x) — near-zero derivative; the "
                        + "Newton step is ill-defined and the result is unreliable"
                )]
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

    // Budget exhausted without meeting tolerance — Newton diverged or stalled
    // (e.g. f'(x) carries the iterate away from the root). The last iterate is
    // not a trustworthy root, so flag it outside the convergence envelope.
    return RootScalarResult(
        root: x, iterations: nit, functionCalls: nfev,
        converged: false, flag: "maxiter reached",
        diagnostics: [.outsideEnvelope(
            method: "newton",
            reason: "exceeded maxiter=\(maxiter) without converging — the iteration "
                + "diverged or stalled; the returned iterate may not be a root"
        )]
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

        // A vanishing secant slope f(x1)−f(x0) makes the update ill-defined — the
        // iterates have stalled on a near-flat region, outside secant's envelope.
        if abs(f1 - f0) < 1e-14 {
            return RootScalarResult(
                root: x1, iterations: nit, functionCalls: nfev,
                converged: false, flag: "denominator too small",
                diagnostics: [.outsideEnvelope(
                    method: "secant",
                    reason: "|f(x1)−f(x0)| < 1e-14 — secant slope collapsed near a "
                        + "flat region; the update is ill-defined and unreliable"
                )]
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

    // Budget exhausted without meeting tolerance — the secant iteration diverged
    // or stalled; the last iterate is not a trustworthy root.
    return RootScalarResult(
        root: x1, iterations: nit, functionCalls: nfev,
        converged: false, flag: "maxiter reached",
        diagnostics: [.outsideEnvelope(
            method: "secant",
            reason: "exceeded maxiter=\(maxiter) without converging — the iteration "
                + "diverged or stalled; the returned iterate may not be a root"
        )]
    )
}

// MARK: - Brent's method (bracketing root finder)

/// Brent's method for scalar root finding (`scipy.optimize.brentq` analogue).
///
/// Combines the guaranteed convergence of bisection with the speed of inverse
/// quadratic interpolation and the secant method. Requires a **bracket** `[a, b]`
/// across which `f` changes sign (`f(a)·f(b) < 0`); within that bracket Brent's
/// method always converges to a root.
///
/// Distinct from ``brent(_:a:b:xtol:maxiter:)``, which **minimizes** a scalar
/// function. This routine finds a **root** (a zero), matching SciPy's `brentq`.
///
/// ## Limitation envelope
///
/// Like every bracketing method, `brentq` is mathematically invalid when the
/// supplied endpoints do not straddle a root (`f(a)·f(b) > 0`). In that regime it
/// returns `.nan` and appends an ``NumericDiagnostic/outsideEnvelope(method:reason:)``
/// diagnostic (SciPy raises a `ValueError`). A self-aware caller inspects
/// ``RootScalarResult/diagnostics``.
///
/// - Parameters:
///   - f: Function whose root is sought.
///   - a: Left bracket endpoint.
///   - b: Right bracket endpoint.
///   - xtol: Absolute tolerance on the root location.
///   - rtol: Relative tolerance on the root location.
///   - maxiter: Maximum iterations.
/// - Returns: Root finding result, with diagnostics for out-of-envelope brackets.
///
/// Reference: R. P. Brent, *Algorithms for Minimization without Derivatives* (1973);
/// scipy.optimize.brentq.
public func brentq(
    _ f: (Double) -> Double,
    a: Double,
    b: Double,
    xtol: Double = optimDefaultXTol,
    rtol: Double = 4 * Double.ulpOfOne,
    maxiter: Int = optimDefaultMaxIter
) -> RootScalarResult {
    var xa = a, xb = b
    var nfev = 0

    var fa = f(xa); nfev += 1
    var fb = f(xb); nfev += 1

    // A NaN endpoint evaluation defeats the sign-change test below (`NaN * x > 0`
    // is false), so guard it first — otherwise Brent's iteration would propagate
    // NaN iterates while falsely reporting `converged`.
    if fa.isNaN || fb.isNaN {
        return RootScalarResult(
            root: .nan, iterations: 0, functionCalls: nfev,
            converged: false, flag: "f(a) or f(b) is NaN",
            diagnostics: [.outsideEnvelope(
                method: "brentq",
                reason: "f evaluated to NaN at a bracket endpoint — the function is "
                    + "undefined on [\(a), \(b)] and no root can be bracketed"
            )]
        )
    }

    // Exact roots at the endpoints.
    if fa == 0 {
        return RootScalarResult(root: xa, iterations: 0, functionCalls: nfev, converged: true, flag: "converged")
    }
    if fb == 0 {
        return RootScalarResult(root: xb, iterations: 0, functionCalls: nfev, converged: true, flag: "converged")
    }

    // No sign change → invalid bracket → outside the envelope.
    if fa * fb > 0 {
        return RootScalarResult(
            root: .nan, iterations: 0, functionCalls: nfev,
            converged: false, flag: "f(a) and f(b) must have different signs",
            diagnostics: [.outsideEnvelope(
                method: "brentq",
                reason: "f(a)·f(b) > 0 — bracket has no sign change; Brent's method "
                    + "is invalid and cannot guarantee a root in [\(a), \(b)]"
            )]
        )
    }

    // Brent's method state. `xb` holds the best estimate; `xc` the contrapoint.
    var xc = xa, fc = fa
    var d = xb - xa, e = d
    var nit = 0

    while nit < maxiter {
        nit += 1

        // Ensure |f(b)| <= |f(c)| so b is the better root estimate.
        if abs(fc) < abs(fb) {
            xa = xb; xb = xc; xc = xa
            fa = fb; fb = fc; fc = fa
        }

        let tol = 2 * rtol * abs(xb) + xtol / 2
        let m = 0.5 * (xc - xb)

        if abs(m) <= tol || fb == 0 {
            return RootScalarResult(
                root: xb, iterations: nit, functionCalls: nfev,
                converged: true, flag: "converged"
            )
        }

        if abs(e) < tol || abs(fa) <= abs(fb) {
            // Bisection step.
            d = m; e = m
        } else {
            // Attempt inverse quadratic interpolation (or secant if a == c).
            let s = fb / fa
            var p: Double, q: Double
            if xa == xc {
                p = 2 * m * s
                q = 1 - s
            } else {
                let qq = fa / fc
                let r = fb / fc
                p = s * (2 * m * qq * (qq - r) - (xb - xa) * (r - 1))
                q = (qq - 1) * (r - 1) * (s - 1)
            }
            if p > 0 { q = -q } else { p = -p }
            // Accept interpolation only if it stays within bounds and shrinks.
            if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q)) {
                e = d
                d = p / q
            } else {
                d = m; e = m
            }
        }

        xa = xb; fa = fb
        // Step at least `tol` toward the root.
        if abs(d) > tol {
            xb += d
        } else {
            xb += (m > 0 ? tol : -tol)
        }
        fb = f(xb); nfev += 1

        // Maintain the bracket: c becomes the endpoint with opposite sign to b.
        if (fb > 0) == (fc > 0) {
            xc = xa; fc = fa
            d = xb - xa; e = d
        }
    }

    return RootScalarResult(
        root: xb, iterations: nit, functionCalls: nfev,
        converged: false, flag: "maxiter reached",
        diagnostics: [.outsideEnvelope(
            method: "brentq",
            reason: "exceeded maxiter=\(maxiter) without converging"
        )]
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

        // Compute Jacobian numerically using per-variable relative step sizing.
        //
        // Step formula: h_j = finiteDiffStepScale * max(1, |x_j|)
        // This matches scipy.optimize.approx_fprime and prevents underflow
        // for large-magnitude variables (where a global h would be below the
        // floating-point ULP of x_j, making x_j + h == x_j).
        //
        // Reference: scipy.optimize._numdiff.approx_derivative (scipy.org)
        //
        // Central differences — accuracy O(h²) vs. O(h) for forward differences —
        // are preferred here because the solver calls f once per iteration anyway
        // and the 2n evaluations amortise well over Newton steps.
        var jacobian = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for j in 0..<n {
            let h_j = finiteDiffStepScale * max(1.0, abs(x[j]))
            var xp = x; xp[j] += h_j
            var xm = x; xm[j] -= h_j
            let fxp = f(xp); nfev += 1
            let fxm = f(xm); nfev += 1
            for i in 0..<n {
                jacobian[i][j] = (fxp[i] - fxm[i]) / (2.0 * h_j)
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

/// Levenberg-Marquardt algorithm for nonlinear least squares with optional bounds.
///
/// Minimizes sum(residuals(x)^2) using the Levenberg-Marquardt method.
/// When bounds are provided, uses Projected Levenberg-Marquardt which
/// projects parameters onto the feasible region after each step.
///
/// - Parameters:
///   - residuals: Function returning residuals vector
///   - x0: Initial guess
///   - bounds: Optional tuple of (lower, upper) bounds arrays
///   - ftol: Relative tolerance for cost function
///   - xtol: Relative tolerance for parameters
///   - gtol: Tolerance for projected gradient (for bounded problems)
///   - maxiter: Maximum iterations
/// - Returns: Least squares result
public func leastSquares(
    _ residuals: @escaping ([Double]) -> [Double],
    x0: [Double],
    bounds: (lower: [Double], upper: [Double])? = nil,
    ftol: Double = 1e-8,
    xtol: Double = 1e-8,
    gtol: Double = 1e-8,
    maxiter: Int = 100
) -> LeastSquaresResult {
    // If no bounds, use standard LM
    guard let bounds = bounds else {
        return leastSquaresUnbounded(residuals, x0: x0, ftol: ftol, xtol: xtol, maxiter: maxiter)
    }

    let lower = bounds.lower
    let upper = bounds.upper
    let n = x0.count

    // Validate bounds
    guard lower.count == n, upper.count == n else {
        return LeastSquaresResult(
            x: x0, cost: .infinity, fun: [], nfev: 0, njev: 0,
            success: false, message: "Bounds arrays must have same length as x0"
        )
    }

    // Project function: clamp x to [lower, upper]
    func project(_ x: [Double]) -> [Double] {
        return zip(zip(x, lower), upper).map { arg in
            let ((xi, lo), hi) = arg
            return min(max(xi, lo), hi)
        }
    }

    // Compute projected gradient norm for KKT optimality check
    // At a bound-constrained optimum:
    // - If x_i is at lower bound: gradient should be >= 0 (can't go lower)
    // - If x_i is at upper bound: gradient should be <= 0 (can't go higher)
    // - If x_i is interior: gradient should be 0
    func projectedGradientNorm(_ x: [Double], _ grad: [Double]) -> Double {
        var normSq = 0.0
        let eps = 1e-10
        for i in 0..<n {
            let g = grad[i]
            let atLower = abs(x[i] - lower[i]) < eps
            let atUpper = abs(x[i] - upper[i]) < eps

            // Projected gradient component
            let projG: Double
            if atLower && g > 0 {
                // At lower bound with positive gradient (pushing up) - optimal
                projG = 0
            } else if atUpper && g < 0 {
                // At upper bound with negative gradient (pushing down) - optimal
                projG = 0
            } else {
                projG = g
            }
            normSq += projG * projG
        }
        return sqrt(normSq)
    }

    // Start with projected x0
    var x = project(x0)
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

    var stagnantCount = 0
    let maxStagnant = 10

    for _ in 0..<maxiter {
        // Compute Jacobian numerically using per-variable relative step sizing.
        //
        // Step formula: h_j = finiteDiffStepScale * max(1, |x_j|)
        // See: scipy.optimize.approx_derivative (scipy.org)
        //
        // Central differences give O(h²) accuracy without a current-residual
        // baseline dependency, which is safer near bounds where r may be stale.
        // Don't project the perturbed point — we need the unconstrained derivative.
        var J = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)
        for j in 0..<n {
            let h_j = finiteDiffStepScale * max(1.0, abs(x[j]))
            var xp = x; xp[j] += h_j
            var xm = x; xm[j] -= h_j
            let rp = residuals(xp); nfev += 1
            let rm = residuals(xm); nfev += 1
            for i in 0..<m {
                J[i][j] = (rp[i] - rm[i]) / (2.0 * h_j)
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

        // Compute gradient = J^T * r (n x 1)
        var grad = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var sum = 0.0
            for k in 0..<m {
                sum += J[k][i] * r[k]
            }
            grad[i] = sum
        }

        // Check KKT optimality condition using projected gradient
        let projGradNorm = projectedGradientNorm(x, grad)
        if projGradNorm < gtol {
            return LeastSquaresResult(
                x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                success: true, message: "Convergence: projected gradient tolerance satisfied."
            )
        }

        // Solve (J^T*J + lambda*diag(J^T*J)) * dx = -J^T*r
        var A = JTJ
        for i in 0..<n {
            A[i][i] += lambda * max(JTJ[i][i], 1e-10)
        }
        var b = grad.map { -$0 }

        // Gaussian elimination with partial pivoting
        var singular = false
        for col in 0..<n {
            var maxRow = col
            for row in (col+1)..<n {
                if abs(A[row][col]) > abs(A[maxRow][col]) {
                    maxRow = row
                }
            }
            A.swapAt(col, maxRow)
            b.swapAt(col, maxRow)

            if abs(A[col][col]) <= 1e-14 {
                singular = true
                break
            }

            for row in (col+1)..<n {
                let factor = A[row][col] / A[col][col]
                for k in col..<n {
                    A[row][k] -= factor * A[col][k]
                }
                b[row] -= factor * b[col]
            }
        }

        if singular {
            // Increase lambda and retry
            lambda *= lambdaUp
            stagnantCount += 1
            if stagnantCount > maxStagnant {
                return LeastSquaresResult(
                    x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                    success: false, message: "Singular matrix in LM step."
                )
            }
            continue
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

        // Trial step with projection
        var xNew = x
        for i in 0..<n {
            xNew[i] += dx[i]
        }
        xNew = project(xNew)  // Project onto bounds

        let rNew = residuals(xNew); nfev += 1
        let costNew = 0.5 * rNew.reduce(0) { $0 + $1 * $1 }

        // Compute actual step after projection
        let actualDx = zip(xNew, x).map { $0 - $1 }
        let xNorm = sqrt(actualDx.reduce(0) { $0 + $1 * $1 })

        // Check if step is accepted
        if costNew < cost || xNorm < 1e-14 {
            let costRatio = abs(cost - costNew) / max(cost, 1e-14)
            let paramNorm = sqrt(x.reduce(0) { $0 + $1 * $1 })

            // If we're not making any progress (projected to same point)
            if xNorm < 1e-14 {
                // Check if we're at a constrained optimum
                let projGradNormFinal = projectedGradientNorm(x, grad)
                if projGradNormFinal < gtol * 100 {  // More relaxed for stagnant case
                    return LeastSquaresResult(
                        x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                        success: true, message: "Convergence: at boundary optimum."
                    )
                }
                // Increase lambda to try a different direction
                lambda *= lambdaUp
                stagnantCount += 1
                if stagnantCount > maxStagnant {
                    // Still declare success if projected gradient is reasonably small
                    if projGradNormFinal < gtol * 1000 {
                        return LeastSquaresResult(
                            x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                            success: true, message: "Convergence: boundary solution found."
                        )
                    }
                    return LeastSquaresResult(
                        x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                        success: false, message: "Stagnation at boundary."
                    )
                }
                continue
            }

            stagnantCount = 0
            x = xNew
            r = rNew
            cost = costNew
            lambda *= lambdaDown

            // Check convergence
            if costRatio < ftol {
                return LeastSquaresResult(
                    x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                    success: true, message: "Convergence: cost function tolerance satisfied."
                )
            }
            if xNorm < xtol * (1 + paramNorm) {
                return LeastSquaresResult(
                    x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                    success: true, message: "Convergence: parameter tolerance satisfied."
                )
            }
        } else {
            // Reject step, increase damping
            lambda *= lambdaUp
            stagnantCount += 1
            if stagnantCount > maxStagnant {
                // Check if current point is actually optimal
                let projGradNormFinal = projectedGradientNorm(x, grad)
                if projGradNormFinal < gtol * 1000 {
                    return LeastSquaresResult(
                        x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
                        success: true, message: "Convergence: boundary solution found."
                    )
                }
            }
        }
    }

    return LeastSquaresResult(
        x: x, cost: cost, fun: r, nfev: nfev, njev: njev,
        success: false, message: "Maximum iterations reached."
    )
}

/// Unbounded Levenberg-Marquardt algorithm for nonlinear least squares.
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
private func leastSquaresUnbounded(
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
        // Compute Jacobian numerically using per-variable relative step sizing.
        //
        // Step formula: h_j = finiteDiffStepScale * max(1, |x_j|)
        // See: scipy.optimize.approx_derivative (scipy.org)
        //
        // Central differences give O(h²) accuracy; preferred over forward
        // differences which require a fresh baseline r evaluation each step.
        var J = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)
        for j in 0..<n {
            let h_j = finiteDiffStepScale * max(1.0, abs(x[j]))
            var xp = x; xp[j] += h_j
            var xm = x; xm[j] -= h_j
            let rp = residuals(xp); nfev += 1
            let rm = residuals(xm); nfev += 1
            for i in 0..<m {
                J[i][j] = (rp[i] - rm[i]) / (2.0 * h_j)
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

    // The residual closure indexes ydata[i] for i in 0..<m: mismatched x/y lengths
    // (or empty data / no parameters) are caller contract violations that would
    // otherwise trap with an opaque out-of-range error deep in the solver.
    precondition(
        ydata.count == m && m > 0 && n > 0,
        "curveFit requires xdata.count == ydata.count > 0 and a non-empty p0 "
            + "(got xdata.count=\(m), ydata.count=\(ydata.count), p0.count=\(n))")

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

    // Estimate covariance matrix via a central-difference Jacobian at the solution.
    //
    // Per-variable step formula: h_j = finiteDiffStepScale * max(1, |p_j|)
    // See: scipy.optimize.approx_derivative (scipy.org)
    var J = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)

    for j in 0..<n {
        let h_j = finiteDiffStepScale * max(1.0, abs(result.x[j]))
        var pp = result.x; pp[j] += h_j
        var pm = result.x; pm[j] -= h_j

        for i in 0..<m {
            let fp = f(pp, xdata[i])
            let fm = f(pm, xdata[i])
            J[i][j] = (fp - fm) / (2.0 * h_j)
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

    // Forward elimination with full pivot. A pivot at or below the
    // rank-deficiency threshold means J^T J is singular and the covariance
    // cannot be estimated.
    var singular = false
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
        guard abs(pivot) > 1e-14 else {
            singular = true
            break
        }
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

    // Rank-deficient Jacobian: return a covariance matrix filled with
    // infinity, matching `scipy.optimize.curve_fit`, which signals an
    // unestimable covariance that way rather than returning garbage.
    if singular {
        let infCov = [[Double]](repeating: [Double](repeating: .infinity, count: n), count: n)
        return (result.x, infCov, result)
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
