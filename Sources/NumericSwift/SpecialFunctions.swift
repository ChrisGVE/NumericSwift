//
//  SpecialFunctions.swift
//  NumericSwift
//
//  Special mathematical functions following scipy.special patterns.
//  Includes error functions, Bessel functions, gamma functions,
//  elliptic integrals, and more.
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Error Functions

/// Error function erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
public func erf(_ x: Double) -> Double {
    Darwin.erf(x)
}

/// Complementary error function erfc(x) = 1 - erf(x)
public func erfc(_ x: Double) -> Double {
    Darwin.erfc(x)
}

/// Inverse error function
/// Returns y such that erf(y) = x. Domain: x ∈ (-1, 1)
public func erfinv(_ x: Double) -> Double {
    if x == 0 { return 0 }
    guard x > -1 && x < 1 else { return .nan }

    let a = abs(x)

    // For |x| <= 0.7, use central approximation
    if a <= 0.7 {
        let x2 = x * x
        let r = x * ((((-0.140543331 * x2 + 0.914624893) * x2 - 1.645349621) * x2 + 0.886226899))
        let s = (((0.012229801 * x2 - 0.329097515) * x2 + 1.442710462) * x2 - 2.118377725) * x2 + 1.0
        return r / s
    }

    // For |x| > 0.7, use tail approximation
    let y = Darwin.sqrt(-Darwin.log((1.0 - a) / 2.0))

    let r: Double
    if y <= 5.0 {
        let t = y - 1.6
        r = ((((((0.00077454501427834 * t + 0.0227238449892691) * t + 0.24178072517745) * t +
               1.27045825245237) * t + 3.64784832476320) * t + 5.76949722146069) * t + 4.63033784615655) /
            ((((((0.00080529518738563 * t + 0.02287663117085) * t + 0.23601290952344) * t +
               1.21357729517684) * t + 3.34305755540406) * t + 4.77629303102970) * t + 1.0)
    } else {
        let t = y - 5.0
        r = ((((((0.0000100950558 * t + 0.000280756651) * t + 0.00326196717) * t +
               0.0206706341) * t + 0.0783478783) * t + 0.169827922) * t + 0.161895932) /
            ((((((0.0000100950558 * t + 0.000280756651) * t + 0.00326196717) * t +
               0.0206706341) * t + 0.0783478783) * t + 0.169827922) * t + 1.0)
    }

    return x >= 0 ? r : -r
}

/// Inverse complementary error function
/// Returns y such that erfc(y) = x. Domain: x ∈ (0, 2)
public func erfcinv(_ x: Double) -> Double {
    guard x > 0 && x < 2 else { return .nan }
    return erfinv(1.0 - x)
}

// MARK: - Beta Functions

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
public func beta(_ a: Double, _ b: Double) -> Double {
    Darwin.exp(lgamma(a) + lgamma(b) - lgamma(a + b))
}

/// Regularized incomplete beta function I_x(a,b)
/// I_x(a,b) = B(x; a,b) / B(a,b)
public func betainc(_ a: Double, _ b: Double, _ x: Double) -> Double {
    guard x >= 0 && x <= 1 else { return .nan }
    if x <= 0 { return 0 }
    if x >= 1 { return 1 }
    return regularizedIncompleteBeta(a: a, b: b, x: x)
}

/// Internal implementation of regularized incomplete beta
private func regularizedIncompleteBeta(a: Double, b: Double, x: Double) -> Double {
    // For x > (a+1)/(a+b+2), use the symmetry relation
    let symmetryPoint = (a + 1) / (a + b + 2)

    if x > symmetryPoint {
        return 1.0 - regularizedIncompleteBeta(a: b, b: a, x: 1.0 - x)
    }

    let bt: Double
    if x == 0 || x == 1 {
        bt = 0
    } else {
        bt = Darwin.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Darwin.log(x) + b * Darwin.log(1 - x))
    }

    // Continued fraction using Lentz's method
    let eps = 1.0e-15
    let maxIterations = 200

    var c = 1.0
    var d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1.0e-30 { d = 1.0e-30 }
    d = 1.0 / d
    var h = d

    for m in 1...maxIterations {
        let m2 = 2 * m
        let dm = Double(m)
        let dm2 = Double(m2)

        // Even step
        var aa = dm * (b - dm) * x / ((a + dm2 - 1) * (a + dm2))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30 { d = 1.0e-30 }
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30 { c = 1.0e-30 }
        d = 1.0 / d
        h *= d * c

        // Odd step
        aa = -(a + dm) * (a + b + dm) * x / ((a + dm2) * (a + dm2 + 1))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30 { d = 1.0e-30 }
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30 { c = 1.0e-30 }
        d = 1.0 / d
        let del = d * c
        h *= del

        if abs(del - 1.0) < eps {
            break
        }
    }

    return bt * h / a
}

// MARK: - Bessel Functions (First Kind)

/// Bessel function of the first kind, order 0: J₀(x)
public func j0(_ x: Double) -> Double {
    Darwin.j0(x)
}

/// Bessel function of the first kind, order 1: J₁(x)
public func j1(_ x: Double) -> Double {
    Darwin.j1(x)
}

/// Bessel function of the first kind, order n: Jₙ(x)
public func jn(_ n: Int, _ x: Double) -> Double {
    Darwin.jn(Int32(n), x)
}

// MARK: - Bessel Functions (Second Kind)

/// Bessel function of the second kind, order 0: Y₀(x)
/// Note: Y₀(x) is undefined for x ≤ 0
public func y0(_ x: Double) -> Double {
    guard x > 0 else { return -.infinity }
    return Darwin.y0(x)
}

/// Bessel function of the second kind, order 1: Y₁(x)
/// Note: Y₁(x) is undefined for x ≤ 0
public func y1(_ x: Double) -> Double {
    guard x > 0 else { return -.infinity }
    return Darwin.y1(x)
}

/// Bessel function of the second kind, order n: Yₙ(x)
/// Note: Yₙ(x) is undefined for x ≤ 0
public func yn(_ n: Int, _ x: Double) -> Double {
    guard x > 0 else { return -.infinity }
    return Darwin.yn(Int32(n), x)
}

// MARK: - Modified Bessel Functions

/// Modified Bessel function of the first kind: Iₙ(x)
public func besseli(_ n: Int, _ x: Double) -> Double {
    let absN = abs(n)

    if x == 0 {
        return absN == 0 ? 1.0 : 0.0
    }

    let absX = abs(x)

    if absX <= 20.0 + Double(absN) {
        return besseliSeries(absN, absX)
    } else {
        return besseliAsymptotic(absN, absX)
    }
}

private func besseliSeries(_ n: Int, _ x: Double) -> Double {
    let halfX = x / 2.0
    let quarterX2 = halfX * halfX
    let eps = 1.0e-15
    let maxIterations = 200

    var term = Darwin.pow(halfX, Double(n)) / tgamma(Double(n) + 1.0)
    var sum = term

    for k in 1...maxIterations {
        term *= quarterX2 / (Double(k) * Double(n + k))
        sum += term
        if abs(term) < abs(sum) * eps {
            break
        }
    }

    return sum
}

private func besseliAsymptotic(_ n: Int, _ x: Double) -> Double {
    let mu = 4.0 * Double(n * n)
    let x8 = 8.0 * x

    var sum = 1.0
    var term = 1.0
    let eps = 1.0e-15

    for k in 1...10 {
        let k2m1 = Double(2 * k - 1)
        term *= -(mu - k2m1 * k2m1) / (Double(k) * x8)
        let newSum = sum + term
        if abs(term) < abs(sum) * eps {
            break
        }
        sum = newSum
    }

    return Darwin.exp(x) / Darwin.sqrt(2.0 * .pi * x) * sum
}

/// Modified Bessel function of the second kind: Kₙ(x)
/// Note: Kₙ(x) is undefined for x ≤ 0
public func besselk(_ n: Int, _ x: Double) -> Double {
    let absN = abs(n)

    guard x > 0 else { return .infinity }

    if x <= 2.0 {
        return besselkSmall(absN, x)
    } else {
        return besselkAsymptotic(absN, x)
    }
}

private func besselkSmall(_ n: Int, _ x: Double) -> Double {
    let k0 = besselk0Small(x)
    if n == 0 { return k0 }

    let k1 = besselk1Small(x)
    if n == 1 { return k1 }

    var kPrev = k0
    var kCurr = k1
    for m in 1..<n {
        let kNext = kPrev + (2.0 * Double(m) / x) * kCurr
        kPrev = kCurr
        kCurr = kNext
    }

    return kCurr
}

private func besselk0Small(_ x: Double) -> Double {
    let halfX = x / 2.0
    let quarterX2 = halfX * halfX
    let gamma = MathConstants.eulerGamma

    let i0 = besseliSeries(0, x)

    var sum = 0.0
    var term = 1.0
    var psi = -gamma

    sum += term * psi

    for k in 1...50 {
        psi += 1.0 / Double(k)
        term *= quarterX2 / Double(k * k)
        sum += term * psi
        if abs(term) < 1.0e-15 * abs(sum) {
            break
        }
    }

    return -Darwin.log(halfX) * i0 + sum
}

private func besselk1Small(_ x: Double) -> Double {
    let halfX = x / 2.0
    let quarterX2 = halfX * halfX
    let gamma = MathConstants.eulerGamma

    let i1 = besseliSeries(1, x)

    var sum = 0.0
    var term = halfX
    var psiK = -gamma
    var psiK1 = 1.0 - gamma

    sum += term * (psiK + psiK1) / 2.0

    for k in 1...50 {
        psiK += 1.0 / Double(k)
        psiK1 += 1.0 / Double(k + 1)
        term *= quarterX2 / (Double(k) * Double(k + 1))
        sum += term * (psiK + psiK1) / 2.0
        if abs(term) < 1.0e-15 * abs(sum) {
            break
        }
    }

    return Darwin.log(halfX) * i1 + 1.0 / x - sum
}

private func besselkAsymptotic(_ n: Int, _ x: Double) -> Double {
    let mu = 4.0 * Double(n * n)
    let x8 = 8.0 * x

    var sum = 1.0
    var term = 1.0
    let eps = 1.0e-15

    for k in 1...20 {
        let k2m1 = Double(2 * k - 1)
        term *= (mu - k2m1 * k2m1) / (Double(k) * x8)
        let newSum = sum + term
        if abs(term) < abs(sum) * eps {
            break
        }
        sum = newSum
    }

    return Darwin.sqrt(.pi / (2.0 * x)) * Darwin.exp(-x) * sum
}

// MARK: - Gamma Functions

/// Digamma function ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
public func digamma(_ x: Double) -> Double {
    var z = x
    var result = 0.0

    // Reflection formula for x < 0.5
    if z < 0.5 {
        return digamma(1.0 - z) - Double.pi / Darwin.tan(Double.pi * z)
    }

    // Recurrence relation to shift to larger value
    while z < 6.0 {
        result -= 1.0 / z
        z += 1.0
    }

    // Asymptotic expansion
    result += Darwin.log(z) - 0.5 / z

    let z2 = 1.0 / (z * z)
    result -= z2 * (1.0/12.0 - z2 * (1.0/120.0 - z2 * (1.0/252.0 - z2 * 1.0/240.0)))

    return result
}

/// Alias for digamma function
public func psi(_ x: Double) -> Double {
    digamma(x)
}

/// Lower regularized incomplete gamma function P(a, x)
/// P(a,x) = γ(a,x) / Γ(a)
public func gammainc(_ a: Double, _ x: Double) -> Double {
    guard a > 0 else { return .nan }
    guard x >= 0 else { return .nan }
    if x == 0 { return 0 }

    if x < a + 1 {
        return gammaincSeries(a, x)
    } else {
        return 1.0 - gammaincCF(a, x)
    }
}

/// Upper regularized incomplete gamma function Q(a, x) = 1 - P(a, x)
public func gammaincc(_ a: Double, _ x: Double) -> Double {
    1.0 - gammainc(a, x)
}

private func gammaincSeries(_ a: Double, _ x: Double) -> Double {
    let eps = 1.0e-15
    let maxIterations = 200

    var sum = 1.0 / a
    var term = 1.0 / a

    for n in 1...maxIterations {
        term *= x / (a + Double(n))
        sum += term
        if abs(term) < abs(sum) * eps {
            break
        }
    }

    return sum * Darwin.exp(-x + a * Darwin.log(x) - lgamma(a))
}

private func gammaincCF(_ a: Double, _ x: Double) -> Double {
    let eps = 1.0e-15
    let maxIterations = 200

    var b = x + 1.0 - a
    var c = 1.0 / 1.0e-30
    var d = 1.0 / b
    var h = d

    for i in 1...maxIterations {
        let an = -Double(i) * (Double(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1.0e-30 { d = 1.0e-30 }
        c = b + an / c
        if abs(c) < 1.0e-30 { c = 1.0e-30 }
        d = 1.0 / d
        let del = d * c
        h *= del
        if abs(del - 1.0) < eps {
            break
        }
    }

    return Darwin.exp(-x + a * Darwin.log(x) - lgamma(a)) * h
}

// MARK: - Elliptic Integrals

/// Complete elliptic integral of the first kind K(m)
/// K(m) = ∫₀^(π/2) 1/√(1 - m·sin²θ) dθ
/// Domain: 0 ≤ m < 1
public func ellipk(_ m: Double) -> Double {
    guard m >= 0 && m < 1 else { return .nan }

    if m == 0 {
        return .pi / 2
    }

    // AGM iteration
    var a = 1.0
    var g = Darwin.sqrt(1.0 - m)
    let eps = 1.0e-15

    while abs(a - g) > eps * abs(a) {
        let aNew = (a + g) / 2.0
        g = Darwin.sqrt(a * g)
        a = aNew
    }

    return .pi / (2.0 * a)
}

/// Complete elliptic integral of the second kind E(m)
/// E(m) = ∫₀^(π/2) √(1 - m·sin²θ) dθ
/// Domain: 0 ≤ m ≤ 1
public func ellipe(_ m: Double) -> Double {
    guard m >= 0 && m <= 1 else { return .nan }

    if m == 0 {
        return .pi / 2
    }
    if m == 1 {
        return 1.0
    }

    var a = 1.0
    var g = Darwin.sqrt(1.0 - m)
    var c = Darwin.sqrt(m)
    var sum = c * c
    var power = 1.0
    let eps = 1.0e-15

    while abs(c) > eps {
        let aNew = (a + g) / 2.0
        c = (a - g) / 2.0
        g = Darwin.sqrt(a * g)
        a = aNew
        power *= 2
        sum += power * c * c
    }

    let k = .pi / (2 * a)
    return k * (1.0 - sum / 2)
}

// MARK: - Riemann Zeta Function

/// Riemann zeta function ζ(s) = Σ_{n=1}^∞ 1/n^s for s > 1
public func zeta(_ s: Double) -> Double {
    if s == 1 {
        return .infinity
    }
    if s == 0 {
        return -0.5
    }

    if s < 0 {
        if s.truncatingRemainder(dividingBy: 2) == 0 {
            return 0
        }
        let t = 1.0 - s
        return Darwin.pow(2, s) * Darwin.pow(.pi, s - 1) * Darwin.sin(.pi * s / 2) * tgamma(t) * zeta(t)
    }

    if s < 1 {
        let eta = zetaEta(s)
        let factor = 1.0 - Darwin.pow(2, 1 - s)
        return eta / factor
    }

    if s < 10 {
        return zetaDirichlet(s)
    } else {
        var sum = 1.0
        for n in 2...100 {
            let term = Darwin.pow(Double(n), -s)
            sum += term
            if term < 1e-15 * sum {
                break
            }
        }
        return sum
    }
}

private func zetaEta(_ s: Double) -> Double {
    var sum = 0.0
    var sign = 1.0
    for k in 1...200 {
        let term = sign / Darwin.pow(Double(k), s)
        sum += term
        sign = -sign
        if abs(term) < 1e-15 * abs(sum) && k > 10 {
            break
        }
    }
    return sum
}

private func zetaDirichlet(_ s: Double) -> Double {
    let N = 100
    var sum = 0.0

    for n in 1...N {
        sum += Darwin.pow(Double(n), -s)
    }

    let Ns = Darwin.pow(Double(N), s)
    let Ns1 = Darwin.pow(Double(N), s - 1)

    sum += 1.0 / ((s - 1) * Ns1)
    sum += 0.5 / Ns

    let bernoulli: [Double] = [
        1.0/6, -1.0/30, 1.0/42, -1.0/30, 5.0/66,
        -691.0/2730, 7.0/6, -3617.0/510, 43867.0/798, -174611.0/330
    ]

    var Npow = Ns * Double(N)

    for (i, b2k) in bernoulli.enumerated() {
        let k2 = 2 * (i + 1)

        var factorial: Double = 1
        for j in 1...k2 {
            factorial *= Double(j)
        }

        var rising: Double = 1
        for j in 0..<(k2 - 1) {
            rising *= (s + Double(j))
        }

        let term = b2k * rising / (factorial * Npow)
        sum += term

        if abs(term) < 1e-16 * abs(sum) {
            break
        }

        Npow *= Double(N) * Double(N)
    }

    return sum
}

// MARK: - Lambert W Function

/// Lambert W function W(x) - principal branch
/// W(x) is the solution to w·e^w = x
/// Domain: x ≥ -1/e ≈ -0.36788
public func lambertw(_ x: Double) -> Double {
    let minVal = -1.0 / Darwin.M_E

    guard x >= minVal else { return .nan }

    if x == 0 {
        return 0
    }
    if x == Darwin.M_E {
        return 1
    }
    if abs(x - minVal) < 1e-15 {
        return -1
    }

    var w: Double
    if x < -0.25 {
        let p = Darwin.sqrt(2.0 * (Darwin.M_E * x + 1.0))
        w = -1.0 + p - p * p / 3.0 + 11.0 * p * p * p / 72.0
    } else if x < 3 {
        w = 0.5 * Darwin.log(1 + x)
        if w < 0 { w = 0 }
    } else {
        let lnx = Darwin.log(x)
        let lnlnx = Darwin.log(lnx)
        w = lnx - lnlnx + lnlnx / lnx
    }

    let eps = 1e-15
    let maxIterations = 50

    for _ in 0..<maxIterations {
        let ew = Darwin.exp(w)
        let wew = w * ew
        let f = wew - x
        let fp = ew * (w + 1)

        let correction = f * fp / (fp * fp - f * ew * (w + 2) / 2)
        w -= correction

        if abs(correction) < eps * (1 + abs(w)) {
            break
        }
    }

    return w
}

// MARK: - Complex Gamma Function

/// Lanczos coefficients for g=7, n=9
private let lanczosCoeffs: [Double] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
]

/// Complex gamma function Γ(z)
public func cgamma(_ z: Complex) -> Complex {
    if z.im == 0 && z.re > 0 {
        return Complex(Darwin.tgamma(z.re))
    }

    // Reflection formula for Re(z) < 0.5
    if z.re < 0.5 {
        // sin(π·z)
        let sinZ = (Complex.i * .pi * z).exp - (-Complex.i * .pi * z).exp
        let sinPiZ = sinZ / (2 * Complex.i)

        // Γ(1-z)
        let g1z = cgammaPositive(Complex.one - z)

        // π / (sin(πz) · Γ(1-z))
        return .pi / (sinPiZ * g1z)
    }

    return cgammaPositive(z)
}

private func cgammaPositive(_ z: Complex) -> Complex {
    let g = 7.0
    let zShifted = z - 1

    // Sum of Lanczos series
    var sum = Complex(lanczosCoeffs[0])

    for i in 1..<lanczosCoeffs.count {
        sum = sum + lanczosCoeffs[i] / (zShifted + Double(i))
    }

    // t = z + g + 0.5
    let t = zShifted + g + 0.5

    // sqrt(2π) · t^(z+0.5) · e^(-t) · sum
    let sqrt2pi = Darwin.sqrt(2 * .pi)

    let power = t.pow(zShifted + 0.5)
    let expNegT = (-t).exp

    return sqrt2pi * power * expNegT * sum
}

/// Complex log-gamma function ln(Γ(z))
public func clgamma(_ z: Complex) -> Complex {
    if z.re < 0.5 {
        // log(Γ(z)) = log(π) - log(sin(πz)) - log(Γ(1-z))
        let sinPiZ = (Complex.i * Double.pi * z).exp - (-Complex.i * Double.pi * z).exp
        let sinZ = sinPiZ / (2.0 * Complex.i)

        let lgZ = clgammaPositive(Complex.one - z)

        return Complex(Darwin.log(Double.pi)) - sinZ.log - lgZ
    }

    return clgammaPositive(z)
}

private func clgammaPositive(_ z: Complex) -> Complex {
    let g = 7.0
    let zShifted = z - 1

    var sum = Complex(lanczosCoeffs[0])

    for i in 1..<lanczosCoeffs.count {
        sum = sum + lanczosCoeffs[i] / (zShifted + Double(i))
    }

    let t = zShifted + g + 0.5

    let halfLog2pi = 0.5 * Darwin.log(2 * .pi)

    // (z + 0.5) · log(t) - t + log(sum)
    let term = (zShifted + 0.5) * t.log

    return halfLog2pi + term - t + sum.log
}

// MARK: - Complex Zeta Function

/// Complex Riemann zeta function ζ(s)
public func czeta(_ s: Complex) -> Complex {
    // Special case: pole at s = 1
    if s.isReal(tolerance: 1e-15) && abs(s.re - 1) < 1e-15 {
        return Complex(re: .infinity, im: 0)
    }

    // Pure real case
    if s.isReal(tolerance: 1e-15) {
        return Complex(zeta(s.re))
    }

    // For Re(s) < 0.5, use reflection formula
    if s.re < 0.5 {
        return czetaReflection(s)
    }

    return czetaDirichlet(s)
}

private func czetaReflection(_ s: Complex) -> Complex {
    // ζ(s) = 2^s · π^(s-1) · sin(πs/2) · Γ(1-s) · ζ(1-s)

    let twoS = Complex(2).pow(s)
    let piS1 = Complex(.pi).pow(s - 1)

    // sin(πs/2)
    let halfPiS = (Double.pi / 2.0) * s
    let sinHalfPiS = halfPiS.sin

    // Γ(1-s)
    let gamma1s = cgamma(Complex.one - s)

    // ζ(1-s)
    let zeta1s = czetaDirichlet(Complex.one - s)

    return twoS * piS1 * sinHalfPiS * gamma1s * zeta1s
}

private func czetaDirichlet(_ s: Complex) -> Complex {
    // Use eta function approach
    let eta = czetaEta(s)

    // 2^(1-s)
    let twoFactor = Complex(2).pow(Complex.one - s)

    // 1 - 2^(1-s)
    let denom = Complex.one - twoFactor

    if denom.abs < 1e-30 {
        return Complex(re: .infinity, im: 0)
    }

    return eta / denom
}

private func czetaEta(_ s: Complex) -> Complex {
    var sum = Complex.zero
    var sign = 1.0

    let maxTerms = 500

    for n in 1...maxTerms {
        let nDouble = Double(n)

        // n^(-s)
        let term = sign * Complex(nDouble).pow(-s)

        sum = sum + term

        let termMag = term.abs
        let sumMag = sum.abs
        if termMag < 1e-15 * sumMag && n > 50 {
            break
        }

        sign = -sign
    }

    return sum
}
