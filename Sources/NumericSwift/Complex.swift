//
//  Complex.swift
//  NumericSwift
//
//  Complex number type with comprehensive arithmetic operations.
//  Follows numpy/scipy conventions for complex number handling.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Complex Number Type

/// A complex number with double-precision real and imaginary parts.
///
/// Complex numbers are represented as `a + bi` where `a` is the real part
/// and `b` is the imaginary part. This type provides comprehensive arithmetic
/// operations and mathematical functions.
///
/// ## Usage
///
/// ```swift
/// let z1 = Complex(re: 3, im: 4)
/// let z2 = Complex(re: 1, im: 2)
///
/// let sum = z1 + z2           // 4 + 6i
/// let product = z1 * z2       // -5 + 10i
/// let magnitude = z1.abs      // 5.0
/// let conjugate = z1.conj     // 3 - 4i
/// ```
public struct Complex: Equatable, Hashable, Sendable {
    /// Real part
    public var re: Double

    /// Imaginary part
    public var im: Double

    // MARK: - Initializers

    /// Create a complex number from real and imaginary parts.
    public init(re: Double, im: Double) {
        self.re = re
        self.im = im
    }

    /// Create a complex number from just a real part (imaginary = 0).
    public init(_ real: Double) {
        self.re = real
        self.im = 0
    }

    /// Create a complex number from polar coordinates.
    /// - Parameters:
    ///   - r: The magnitude (radius)
    ///   - theta: The angle in radians
    public static func polar(r: Double, theta: Double) -> Complex {
        Complex(re: r * Darwin.cos(theta), im: r * Darwin.sin(theta))
    }

    // MARK: - Static Constants

    /// The imaginary unit i
    public static let i = Complex(re: 0, im: 1)

    /// Zero
    public static let zero = Complex(re: 0, im: 0)

    /// One
    public static let one = Complex(re: 1, im: 0)

    // MARK: - Properties

    /// Magnitude (absolute value): |z| = sqrt(re² + im²)
    public var abs: Double {
        hypot(re, im)
    }

    /// Squared magnitude: |z|² = re² + im²
    public var abs2: Double {
        re * re + im * im
    }

    /// Phase angle (argument) in radians: arg(z) = atan2(im, re)
    public var arg: Double {
        atan2(im, re)
    }

    /// Complex conjugate: conj(a + bi) = a - bi
    public var conj: Complex {
        Complex(re: re, im: -im)
    }

    /// Whether this is a real number (imaginary part is zero or negligible)
    public func isReal(tolerance: Double = 1e-15) -> Bool {
        Swift.abs(im) < tolerance
    }

    /// Whether this is a pure imaginary number (real part is zero or negligible)
    public func isImaginary(tolerance: Double = 1e-15) -> Bool {
        Swift.abs(re) < tolerance
    }

    /// Whether this is zero
    public var isZero: Bool {
        re == 0 && im == 0
    }

    /// Whether this is finite (not inf or nan)
    public var isFinite: Bool {
        re.isFinite && im.isFinite
    }

    /// Whether this contains NaN
    public var isNaN: Bool {
        re.isNaN || im.isNaN
    }

    // MARK: - Basic Arithmetic Operators

    /// Negation
    public static prefix func - (z: Complex) -> Complex {
        Complex(re: -z.re, im: -z.im)
    }

    /// Addition
    public static func + (lhs: Complex, rhs: Complex) -> Complex {
        Complex(re: lhs.re + rhs.re, im: lhs.im + rhs.im)
    }

    /// Subtraction
    public static func - (lhs: Complex, rhs: Complex) -> Complex {
        Complex(re: lhs.re - rhs.re, im: lhs.im - rhs.im)
    }

    /// Multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    public static func * (lhs: Complex, rhs: Complex) -> Complex {
        Complex(
            re: lhs.re * rhs.re - lhs.im * rhs.im,
            im: lhs.re * rhs.im + lhs.im * rhs.re
        )
    }

    /// Division using Smith's algorithm for numerical stability.
    ///
    /// The naive formula `(ac+bd)/(c²+d²)` overflows to NaN when `c` or `d`
    /// is near `Double.greatestFiniteMagnitude`, because `c²+d²` exceeds the
    /// representable range. Smith's algorithm avoids squaring by branching on
    /// the larger component and forming a scaled ratio.
    ///
    /// Algorithm (R.L. Smith 1962, "Algorithm 116: Complex division",
    /// CACM 5(8):435; also C99 Annex G §G.5.1):
    ///
    ///     if |c| ≥ |d|: r = d/c,  den = c + d·r,  re = (a + b·r)/den,  im = (b − a·r)/den
    ///     else:         r = c/d,  den = c·r + d,   re = (a·r + b)/den,  im = (b·r − a)/den
    ///
    /// Special case: denominator is exact zero and numerator is finite — C99
    /// Annex G §G.5.1 requires the result to be ±inf (not NaN). We detect this
    /// before entering Smith's branches so that `0/0 = NaN` inside Smith does
    /// not silently win over the mathematically correct ±inf.
    public static func / (lhs: Complex, rhs: Complex) -> Complex {
        let a = lhs.re, b = lhs.im
        let c = rhs.re, d = rhs.im

        // C99 Annex G §G.5.1: finite / ±0 → ±∞.
        // Smith's r=d/c=0/0 path gives NaN; we catch exact zero denominators first.
        if c == 0.0 && d == 0.0 {
            // Multiply by ±∞ preserving NaN for NaN numerators.
            let scale = Double.infinity
            return Complex(re: a * scale, im: b * scale)
        }

        if Swift.abs(c) >= Swift.abs(d) {
            let r   = d / c
            let den = c + d * r
            return Complex(re: (a + b * r) / den, im: (b - a * r) / den)
        } else {
            let r   = c / d
            let den = c * r + d
            return Complex(re: (a * r + b) / den, im: (b * r - a) / den)
        }
    }

    // MARK: - Scalar Arithmetic

    public static func + (lhs: Complex, rhs: Double) -> Complex {
        Complex(re: lhs.re + rhs, im: lhs.im)
    }

    public static func + (lhs: Double, rhs: Complex) -> Complex {
        Complex(re: lhs + rhs.re, im: rhs.im)
    }

    public static func - (lhs: Complex, rhs: Double) -> Complex {
        Complex(re: lhs.re - rhs, im: lhs.im)
    }

    public static func - (lhs: Double, rhs: Complex) -> Complex {
        Complex(re: lhs - rhs.re, im: -rhs.im)
    }

    public static func * (lhs: Complex, rhs: Double) -> Complex {
        Complex(re: lhs.re * rhs, im: lhs.im * rhs)
    }

    public static func * (lhs: Double, rhs: Complex) -> Complex {
        Complex(re: lhs * rhs.re, im: lhs * rhs.im)
    }

    public static func / (lhs: Complex, rhs: Double) -> Complex {
        Complex(re: lhs.re / rhs, im: lhs.im / rhs)
    }

    /// Scalar-divided-by-complex: `a / (c + di)` using Smith's algorithm.
    ///
    /// Equivalent to `Complex(re: lhs, im: 0) / rhs` but avoids the zero imaginary
    /// part carrying through the general path. Smith's branch on |c| vs |d| applies
    /// identically; the imaginary numerator term vanishes (b = 0):
    ///
    ///     if |c| ≥ |d|: r = d/c, den = c + d·r, re = a/den, im = −a·r/den
    ///     else:         r = c/d, den = c·r + d,  re = a·r/den, im = −a/den
    public static func / (lhs: Double, rhs: Complex) -> Complex {
        let a = lhs
        let c = rhs.re, d = rhs.im
        if Swift.abs(c) >= Swift.abs(d) {
            let r   = d / c
            let den = c + d * r
            return Complex(re: a / den, im: -(a * r) / den)
        } else {
            let r   = c / d
            let den = c * r + d
            return Complex(re: (a * r) / den, im: -a / den)
        }
    }

    // MARK: - Compound Assignment

    public static func += (lhs: inout Complex, rhs: Complex) {
        lhs = lhs + rhs
    }

    public static func -= (lhs: inout Complex, rhs: Complex) {
        lhs = lhs - rhs
    }

    public static func *= (lhs: inout Complex, rhs: Complex) {
        lhs = lhs * rhs
    }

    public static func /= (lhs: inout Complex, rhs: Complex) {
        lhs = lhs / rhs
    }
}

// MARK: - Mathematical Functions

extension Complex {

    /// Square root using rectangular special-casing then polar form (principal square root).
    ///
    /// Follows C99 Annex G.6.4.2 and IEEE-754-2008 for all exceptional operands:
    ///
    /// | Input (re, im) | Output (re, im) |
    /// |---|---|
    /// | (+∞, NaN) | (+∞, NaN) |
    /// | (-∞, NaN) | (+∞, NaN) |
    /// | (+∞, y) for finite y | (+∞, 0) |
    /// | (-∞, y) for finite y | (0, +∞) |
    /// | (x, ±∞) for any x including ±∞ | (+∞, +∞) |
    /// | (NaN, …) or (finite, NaN) | (NaN, NaN) |
    ///
    /// For the general finite case the standard polar formula is used.
    /// Oracle: numpy.complex128 principal square root, C99 cmath sqrtf.
    public var sqrt: Complex {
        let x = re, y = im

        // C99 Annex G.6.4.2 — handle exceptional operands first.
        // Any infinite imaginary part → (+∞, +∞) regardless of real part (including NaN real).
        if y.isInfinite {
            return Complex(re: .infinity, im: .infinity)
        }
        // +∞ real, finite im → (+∞, 0).  +∞ real, NaN im → (+∞, NaN).
        if x == .infinity {
            return Complex(re: .infinity, im: y.isNaN ? .nan : 0.0)
        }
        // -∞ real, NaN im → (+∞, NaN) per C99 Annex G.6.4.2.
        // -∞ real, finite im → (0, +∞).
        if x == -.infinity {
            return y.isNaN ? Complex(re: .infinity, im: .nan)
                           : Complex(re: 0.0, im: .infinity)
        }
        // NaN in either component — propagate as (NaN, NaN).
        if x.isNaN || y.isNaN {
            return Complex(re: .nan, im: .nan)
        }

        // General finite case: polar form for the principal square root.
        let r = self.abs
        let theta = self.arg
        let sqrtR = Darwin.sqrt(r)
        return Complex(re: sqrtR * Darwin.cos(theta / 2),
                       im: sqrtR * Darwin.sin(theta / 2))
    }

    /// Natural logarithm: log(z) = log|z| + i*arg(z)
    public var log: Complex {
        Complex(re: Darwin.log(self.abs), im: self.arg)
    }

    /// Common logarithm (base 10)
    public var log10: Complex {
        self.log / Darwin.log(10)
    }

    /// Exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
    public var exp: Complex {
        let expA = Darwin.exp(re)
        return Complex(re: expA * Darwin.cos(im), im: expA * Darwin.sin(im))
    }

    /// Power with real exponent using De Moivre's formula.
    /// z^n = r^n * (cos(n*theta) + i*sin(n*theta))
    public func pow(_ n: Double) -> Complex {
        let r = self.abs
        let theta = self.arg
        let rn = Darwin.pow(r, n)
        let ntheta = n * theta
        return Complex(re: rn * Darwin.cos(ntheta), im: rn * Darwin.sin(ntheta))
    }

    /// Power with complex exponent: z^w = exp(w * log(z))
    public func pow(_ w: Complex) -> Complex {
        (w * self.log).exp
    }

    /// Square
    public var squared: Complex {
        self * self
    }

    /// Cube
    public var cubed: Complex {
        self * self * self
    }

    /// Reciprocal: 1/z
    ///
    /// Delegates to the Smith's-algorithm division operator so large-magnitude
    /// values stay finite (same overflow protection as `/`).
    public var reciprocal: Complex {
        1.0 / self
    }
}

// MARK: - Trigonometric Functions

extension Complex {

    /// Sine: sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
    public var sin: Complex {
        Complex(
            re: Darwin.sin(re) * Darwin.cosh(im),
            im: Darwin.cos(re) * Darwin.sinh(im)
        )
    }

    /// Cosine: cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
    public var cos: Complex {
        Complex(
            re: Darwin.cos(re) * Darwin.cosh(im),
            im: -Darwin.sin(re) * Darwin.sinh(im)
        )
    }

    /// Tangent: tan(z) = sin(z) / cos(z)
    public var tan: Complex {
        self.sin / self.cos
    }

    /// Inverse sine (arcsine)
    /// asin(z) = -i * log(iz + sqrt(1 - z²))
    public var asin: Complex {
        let iz = Complex.i * self
        let sqrt1mz2 = (Complex.one - self.squared).sqrt
        return -Complex.i * (iz + sqrt1mz2).log
    }

    /// Inverse cosine (arccosine)
    /// acos(z) = -i * log(z + sqrt(z² - 1))
    public var acos: Complex {
        let sqrtz2m1 = (self.squared - Complex.one).sqrt
        return -Complex.i * (self + sqrtz2m1).log
    }

    /// Inverse tangent (arctangent)
    /// atan(z) = (i/2) * log((1-iz)/(1+iz))
    public var atan: Complex {
        let iz = Complex.i * self
        let ratio = (Complex.one - iz) / (Complex.one + iz)
        return Complex(re: 0, im: 0.5) * ratio.log
    }
}

// MARK: - Hyperbolic Functions

extension Complex {

    /// Hyperbolic sine: sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
    public var sinh: Complex {
        Complex(
            re: Darwin.sinh(re) * Darwin.cos(im),
            im: Darwin.cosh(re) * Darwin.sin(im)
        )
    }

    /// Hyperbolic cosine: cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
    public var cosh: Complex {
        Complex(
            re: Darwin.cosh(re) * Darwin.cos(im),
            im: Darwin.sinh(re) * Darwin.sin(im)
        )
    }

    /// Hyperbolic tangent: tanh(z) = sinh(z) / cosh(z)
    public var tanh: Complex {
        self.sinh / self.cosh
    }

    /// Inverse hyperbolic sine: asinh(z) = log(z + sqrt(z² + 1))
    public var asinh: Complex {
        let sqrtz2p1 = (self.squared + Complex.one).sqrt
        return (self + sqrtz2p1).log
    }

    /// Inverse hyperbolic cosine: acosh(z) = log(z + sqrt(z² - 1))
    public var acosh: Complex {
        let sqrtz2m1 = (self.squared - Complex.one).sqrt
        return (self + sqrtz2m1).log
    }

    /// Inverse hyperbolic tangent: atanh(z) = 0.5 * log((1+z)/(1-z))
    public var atanh: Complex {
        let ratio = (Complex.one + self) / (Complex.one - self)
        return 0.5 * ratio.log
    }
}

// MARK: - CustomStringConvertible

extension Complex: CustomStringConvertible {
    public var description: String {
        if im >= 0 {
            return "\(re)+\(im)i"
        } else {
            return "\(re)\(im)i"
        }
    }
}

// MARK: - ExpressibleByFloatLiteral

extension Complex: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.re = value
        self.im = 0
    }
}

// MARK: - ExpressibleByIntegerLiteral

extension Complex: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.re = Double(value)
        self.im = 0
    }
}

// MARK: - Free Functions (scipy-style API)

/// Complex square root
public func csqrt(_ z: Complex) -> Complex {
    z.sqrt
}

/// Complex exponential
public func cexp(_ z: Complex) -> Complex {
    z.exp
}

/// Complex natural logarithm
public func clog(_ z: Complex) -> Complex {
    z.log
}

/// Complex power
public func cpow(_ z: Complex, _ w: Complex) -> Complex {
    z.pow(w)
}

/// Complex power with real exponent
public func cpow(_ z: Complex, _ n: Double) -> Complex {
    z.pow(n)
}

/// Complex sine
public func csin(_ z: Complex) -> Complex {
    z.sin
}

/// Complex cosine
public func ccos(_ z: Complex) -> Complex {
    z.cos
}

/// Complex tangent
public func ctan(_ z: Complex) -> Complex {
    z.tan
}

/// Complex hyperbolic sine
public func csinh(_ z: Complex) -> Complex {
    z.sinh
}

/// Complex hyperbolic cosine
public func ccosh(_ z: Complex) -> Complex {
    z.cosh
}

/// Complex hyperbolic tangent
public func ctanh(_ z: Complex) -> Complex {
    z.tanh
}

/// Complex arcsine
public func casin(_ z: Complex) -> Complex {
    z.asin
}

/// Complex arccosine
public func cacos(_ z: Complex) -> Complex {
    z.acos
}

/// Complex arctangent
public func catan(_ z: Complex) -> Complex {
    z.atan
}

/// Complex inverse hyperbolic sine
public func casinh(_ z: Complex) -> Complex {
    z.asinh
}

/// Complex inverse hyperbolic cosine
public func cacosh(_ z: Complex) -> Complex {
    z.acosh
}

/// Complex inverse hyperbolic tangent
public func catanh(_ z: Complex) -> Complex {
    z.atanh
}

/// Complex magnitude
public func cabs(_ z: Complex) -> Double {
    z.abs
}

/// Complex phase/argument
public func carg(_ z: Complex) -> Double {
    z.arg
}

/// Complex conjugate
public func conj(_ z: Complex) -> Complex {
    z.conj
}
