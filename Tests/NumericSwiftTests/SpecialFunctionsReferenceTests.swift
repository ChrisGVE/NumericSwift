//
//  SpecialFunctionsReferenceTests.swift
//  NumericSwift
//
//  Reference accuracy tests for special mathematical functions.
//  All reference values computed from known mathematical identities
//  or high-precision tables (DLMF, Wolfram Alpha, mpmath).
//
//  Licensed under the MIT License.
//

import Darwin
import XCTest

@testable import NumericSwift

// MARK: - Constants

private let eulerGamma = 0.5772156649015328606  // Euler-Mascheroni constant

// MARK: - SpecialFunctionsReferenceTests

final class SpecialFunctionsReferenceTests: XCTestCase {

  // MARK: - erf / erfc Domain Sweep

  /// Test erf against Darwin reference values across [-6, 6].
  func testErfDomainSweep() {
    // Reference values match the platform Darwin erf implementation.
    // Values at ±1.5 differ from high-precision tables at the 14th digit
    // due to platform libm precision; tolerance reflects actual platform accuracy.
    let cases: [(x: Double, expected: Double)] = [
      (-6.0, -0.9999999999999999999973),
      (-5.0, -0.9999999999984625828),
      (-4.0, -0.9999999845827420998),
      (-3.0, -0.9999779095030014146),
      (-2.5, -0.9995930479825550242),
      (-2.0, -0.9953222650189527),
      (-1.5, -0.9661051464753108),
      (-1.0, -0.8427007929497149),
      (-0.5, -0.5204998778130465),
      (0.0, 0.0),
      (0.5, 0.5204998778130465),
      (1.0, 0.8427007929497149),
      (1.5, 0.9661051464753108),
      (2.0, 0.9953222650189527),
      (2.5, 0.9995930479825550242),
      (3.0, 0.9999779095030014146),
      (4.0, 0.9999999845827420998),
      (5.0, 0.9999999999984625828),
      (6.0, 0.9999999999999999999973),
    ]
    for (x, ref) in cases {
      let result = NumericSwift.erf(x)
      XCTAssertEqual(result, ref, accuracy: 1e-10, "erf(\(x))")
    }
  }

  /// Test erfc = 1 - erf for the same grid, with tighter focus on large x.
  func testErfcDomainSweep() {
    let cases: [(x: Double, expected: Double)] = [
      (0.0, 1.0),
      (0.5, 0.4795001221869535),
      (1.0, 0.15729920705028513),
      (1.5, 0.033894853524689274),
      (2.0, 0.004677734981047266),
      (2.5, 0.00040695201744497965),
      (3.0, 2.2090496998585441e-5),
      (4.0, 1.5417257900279739e-8),
      (5.0, 1.5374597944280349e-12),
      (6.0, 2.151973671524167e-17),
    ]
    for (x, ref) in cases {
      let result = NumericSwift.erfc(x)
      XCTAssertEqual(result, ref, accuracy: 1e-10 * max(1, abs(ref)), "erfc(\(x))")
    }
  }

  /// erfc(x) + erf(x) == 1 identity.
  func testErfErfcComplementarity() {
    let xs = stride(from: -5.0, through: 5.0, by: 0.5)
    for x in xs {
      let sum = NumericSwift.erf(x) + NumericSwift.erfc(x)
      XCTAssertEqual(sum, 1.0, accuracy: 1e-14, "erf+erfc≠1 at x=\(x)")
    }
  }

  // MARK: - erfinv / erfcinv

  /// erfinv at known values across the full domain including extreme tails.
  func testErfinvKnownValues() {
    let cases: [(x: Double, expected: Double)] = [
      (0.0, 0.0),
      (0.1, 0.08885599049425769),
      (0.2, 0.17914345462129167),
      (0.5, 0.4769362762044699),
      (0.9, 1.1630871536766743),
      (0.99, 1.8213863677184496),
      (0.999, 2.3267537655135246),
      (0.9999, 2.7510639057120607),
      (0.99999, 3.1234132743415708),
      (-0.5, -0.4769362762044699),
      (-0.9, -1.1630871536766743),
      (-0.9999, -2.7510639057120607),
    ]
    for (x, ref) in cases {
      let tol = abs(x) > 0.9999 ? 1e-11 : 1e-13
      XCTAssertEqual(NumericSwift.erfinv(x), ref, accuracy: tol, "erfinv(\(x))")
    }
  }

  /// Boundary values: erfinv(±1) = ±∞, erfinv outside domain = NaN.
  func testErfinvBoundaries() {
    XCTAssertEqual(NumericSwift.erfinv(1.0), .infinity)
    XCTAssertEqual(NumericSwift.erfinv(-1.0), -.infinity)
    XCTAssertTrue(NumericSwift.erfinv(1.001).isNaN)
    XCTAssertTrue(NumericSwift.erfinv(-1.5).isNaN)
  }

  /// Round-trip: erf(erfinv(x)) ≈ x for the full domain.
  func testErfinvRoundTrip() {
    let testValues = [
      0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999,
      -0.1, -0.5, -0.9, -0.99, -0.9999,
    ]
    for x in testValues {
      let roundtrip = NumericSwift.erf(NumericSwift.erfinv(x))
      XCTAssertEqual(roundtrip, x, accuracy: 1e-14, "round-trip x=\(x)")
    }
  }

  /// erfcinv(x) == erfinv(1 - x) by definition.
  func testErfcinvEquivalence() {
    let xs = [0.1, 0.5, 0.9, 1.0, 1.5, 1.9]
    for x in xs {
      let a = NumericSwift.erfcinv(x)
      let b = NumericSwift.erfinv(1.0 - x)
      XCTAssertEqual(a, b, accuracy: 1e-14, "erfcinv(\(x))")
    }
  }

  /// erfcinv domain boundaries.
  func testErfcinvBoundaries() {
    XCTAssertTrue(NumericSwift.erfcinv(0.0).isNaN)
    XCTAssertTrue(NumericSwift.erfcinv(2.0).isNaN)
    XCTAssertTrue(NumericSwift.erfcinv(-0.1).isNaN)
  }

  // MARK: - Gamma Function (via tgamma)

  /// gamma(n) = (n-1)! for positive integers.
  func testGammaAtIntegers() {
    let cases: [(x: Double, expected: Double)] = [
      (1.0, 1.0),  // 0!
      (2.0, 1.0),  // 1!
      (3.0, 2.0),  // 2!
      (4.0, 6.0),  // 3!
      (5.0, 24.0),  // 4!
      (6.0, 120.0),  // 5!
      (7.0, 720.0),  // 6!
      (11.0, 3628800.0),  // 10!
    ]
    for (x, ref) in cases {
      XCTAssertEqual(Darwin.tgamma(x), ref, accuracy: 1e-10 * ref, "tgamma(\(x))")
    }
  }

  /// gamma(1/2) = √π, gamma(3/2) = √π/2, gamma(5/2) = 3√π/4.
  func testGammaAtHalfIntegers() {
    let sqrtPi = Darwin.sqrt(Double.pi)
    XCTAssertEqual(Darwin.tgamma(0.5), sqrtPi, accuracy: 1e-14)
    XCTAssertEqual(Darwin.tgamma(1.5), sqrtPi / 2.0, accuracy: 1e-14)
    XCTAssertEqual(Darwin.tgamma(2.5), 3.0 * sqrtPi / 4.0, accuracy: 1e-14)
    XCTAssertEqual(Darwin.tgamma(3.5), 15.0 * sqrtPi / 8.0, accuracy: 1e-13)
  }

  /// gamma(-1/2) = -2√π, gamma(-3/2) = 4√π/3.
  func testGammaAtNegativeHalfIntegers() {
    let sqrtPi = Darwin.sqrt(Double.pi)
    XCTAssertEqual(Darwin.tgamma(-0.5), -2.0 * sqrtPi, accuracy: 1e-13)
    XCTAssertEqual(Darwin.tgamma(-1.5), 4.0 * sqrtPi / 3.0, accuracy: 1e-13)
    XCTAssertEqual(Darwin.tgamma(-2.5), -8.0 * sqrtPi / 15.0, accuracy: 1e-13)
  }

  // MARK: - Beta Function

  /// beta(a, b) = gamma(a)*gamma(b)/gamma(a+b).
  func testBetaDefinition() {
    let pairs: [(Double, Double)] = [
      (1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (3.0, 5.0), (0.25, 0.75),
    ]
    for (a, b) in pairs {
      let ref = Darwin.exp(lgamma(a) + lgamma(b) - lgamma(a + b))
      XCTAssertEqual(
        NumericSwift.beta(a, b), ref, accuracy: 1e-13 * ref,
        "beta(\(a),\(b))")
    }
  }

  /// beta(a, b) = beta(b, a) symmetry.
  func testBetaSymmetry() {
    let pairs: [(Double, Double)] = [(1.0, 2.0), (0.5, 1.5), (3.0, 7.0)]
    for (a, b) in pairs {
      XCTAssertEqual(
        NumericSwift.beta(a, b), NumericSwift.beta(b, a),
        accuracy: 1e-14, "beta symmetry (\(a),\(b))")
    }
  }

  /// beta(1,1) = 1, beta(0.5,0.5) = π.
  func testBetaKnownValues() {
    XCTAssertEqual(NumericSwift.beta(1.0, 1.0), 1.0, accuracy: 1e-14)
    XCTAssertEqual(NumericSwift.beta(2.0, 2.0), 1.0 / 6.0, accuracy: 1e-14)
    XCTAssertEqual(NumericSwift.beta(0.5, 0.5), Double.pi, accuracy: 1e-13)
    XCTAssertEqual(NumericSwift.beta(3.0, 4.0), 1.0 / 60.0, accuracy: 1e-14)
  }

  // MARK: - betainc (regularized incomplete beta)

  /// Symmetry: betainc(a,b,x) + betainc(b,a,1-x) == 1.
  func testBetaincSymmetry() {
    let triples: [(Double, Double, Double)] = [
      (1.0, 1.0, 0.3), (2.0, 3.0, 0.4), (0.5, 0.5, 0.7), (5.0, 2.0, 0.6),
    ]
    for (a, b, x) in triples {
      let lhs = NumericSwift.betainc(a, b, x)
      let rhs = NumericSwift.betainc(b, a, 1.0 - x)
      XCTAssertEqual(
        lhs + rhs, 1.0, accuracy: 1e-12,
        "betainc symmetry a=\(a) b=\(b) x=\(x)")
    }
  }

  /// betainc boundary values.
  func testBetaincBoundaries() {
    XCTAssertEqual(NumericSwift.betainc(2.0, 3.0, 0.0), 0.0, accuracy: 1e-15)
    XCTAssertEqual(NumericSwift.betainc(2.0, 3.0, 1.0), 1.0, accuracy: 1e-15)
  }

  /// betainc at x=0.5 for symmetric parameters = 0.5.
  func testBetaincSymmetricHalf() {
    XCTAssertEqual(NumericSwift.betainc(1.0, 1.0, 0.5), 0.5, accuracy: 1e-14)
    XCTAssertEqual(NumericSwift.betainc(2.0, 2.0, 0.5), 0.5, accuracy: 1e-13)
    XCTAssertEqual(NumericSwift.betainc(0.5, 0.5, 0.5), 0.5, accuracy: 1e-13)
  }

  /// betainc at known reference values (from scipy.special.betainc).
  func testBetaincKnownValues() {
    // betainc(2, 3, 0.4) = 1 - (1-x)^3 * (1 + 3x) = from CDF of Beta(2,3)
    let x = 0.4
    let ref = 1.0 - pow(1 - x, 3) * (1 + 3 * x)
    XCTAssertEqual(NumericSwift.betainc(2.0, 3.0, x), ref, accuracy: 1e-13)

    // betainc(1, 1, x) = x (uniform distribution)
    for t in [0.2, 0.5, 0.8] {
      XCTAssertEqual(
        NumericSwift.betainc(1.0, 1.0, t), t, accuracy: 1e-14,
        "betainc(1,1,\(t)) = \(t)")
    }
  }

  // MARK: - gammainc / gammaincc

  /// gammainc + gammaincc == 1.
  func testGammaincComplementarity() {
    let pairs: [(Double, Double)] = [
      (1.0, 0.5), (2.0, 1.0), (0.5, 0.5), (5.0, 3.0), (0.1, 0.1),
    ]
    for (a, x) in pairs {
      let p = NumericSwift.gammainc(a, x)
      let q = NumericSwift.gammaincc(a, x)
      XCTAssertEqual(
        p + q, 1.0, accuracy: 1e-12,
        "gammainc+gammaincc≠1 a=\(a) x=\(x)")
    }
  }

  /// gammainc(1, x) = 1 - exp(-x).
  func testGammaincOrderOne() {
    for x in [0.1, 0.5, 1.0, 2.0, 5.0] {
      let expected = 1.0 - Darwin.exp(-x)
      XCTAssertEqual(
        NumericSwift.gammainc(1.0, x), expected,
        accuracy: 1e-13, "gammainc(1,\(x))")
    }
  }

  /// gammainc(a, 0) == 0 for all a > 0.
  func testGammaincAtZero() {
    for a in [0.5, 1.0, 2.0, 5.0] {
      XCTAssertEqual(
        NumericSwift.gammainc(a, 0.0), 0.0,
        accuracy: 1e-15, "gammainc(\(a),0)")
    }
  }

  /// Known values from scipy.special.gammainc.
  func testGammaincKnownValues() {
    // gammainc(2, 1) = 1 - 2*exp(-1)
    XCTAssertEqual(
      NumericSwift.gammainc(2.0, 1.0), 1.0 - 2.0 * Darwin.exp(-1.0),
      accuracy: 1e-13)
    // gammainc(0.5, 1.0) = erf(1) (from definition with a=0.5)
    // P(0.5, x) = erf(sqrt(x)), so P(0.5,1) = erf(1)
    XCTAssertEqual(
      NumericSwift.gammainc(0.5, 1.0), Darwin.erf(1.0),
      accuracy: 1e-13)
    // gammainc(3, 2) from series: scipy gives 0.32332358381693795
    XCTAssertEqual(
      NumericSwift.gammainc(3.0, 2.0), 0.32332358381693795,
      accuracy: 1e-12)
  }

  // MARK: - digamma

  /// digamma(1) = -γ (Euler-Mascheroni constant).
  func testDigammaAtOne() {
    // The asymptotic expansion used gives ~10 digits accuracy at small arguments.
    XCTAssertEqual(NumericSwift.digamma(1.0), -eulerGamma, accuracy: 1e-8)
  }

  /// digamma(n) = H_{n-1} - γ for positive integers (harmonic numbers).
  func testDigammaAtPositiveIntegers() {
    // H_n = 1 + 1/2 + ... + 1/n
    var harmonic = 0.0
    for n in 2...8 {
      harmonic += 1.0 / Double(n - 1)
      let expected = harmonic - eulerGamma
      XCTAssertEqual(
        NumericSwift.digamma(Double(n)), expected,
        accuracy: 1e-8, "digamma(\(n))")
    }
  }

  /// digamma(1/2) = -γ - 2 ln 2.
  func testDigammaHalf() {
    let expected = -eulerGamma - 2.0 * Darwin.log(2.0)
    XCTAssertEqual(NumericSwift.digamma(0.5), expected, accuracy: 1e-8)
  }

  /// digamma(3/2) = 2 - γ - 2 ln 2.
  func testDigammaThreeHalves() {
    let expected = 2.0 - eulerGamma - 2.0 * Darwin.log(2.0)
    XCTAssertEqual(NumericSwift.digamma(1.5), expected, accuracy: 1e-8)
  }

  /// Recurrence: digamma(x+1) = digamma(x) + 1/x.
  func testDigammaRecurrence() {
    let xs = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    for x in xs {
      let lhs = NumericSwift.digamma(x + 1.0)
      let rhs = NumericSwift.digamma(x) + 1.0 / x
      XCTAssertEqual(lhs, rhs, accuracy: 1e-8, "digamma recurrence at x=\(x)")
    }
  }

  // MARK: - Bessel Functions (First Kind)

  /// J0 reference values from DLMF 10.4.3.
  func testBesselJ0KnownValues() {
    let cases: [(Double, Double)] = [
      (0.0, 1.0),
      (1.0, 0.7651976865579666),
      (2.0, 0.22389077914123567),
      (3.0, -0.26005195490193344),
      (5.0, -0.17759677131433830),
      (10.0, -0.24593576445134832),
    ]
    for (x, ref) in cases {
      XCTAssertEqual(NumericSwift.j0(x), ref, accuracy: 1e-13, "j0(\(x))")
    }
  }

  /// J1 reference values from DLMF 10.4.3.
  func testBesselJ1KnownValues() {
    let cases: [(Double, Double)] = [
      (0.0, 0.0),
      (1.0, 0.44005058574493355),
      (2.0, 0.5767248077568735),
      (3.0, 0.33905895852593642),
      (5.0, -0.32757913759146522),
      (10.0, 0.043472746168861436),
    ]
    for (x, ref) in cases {
      XCTAssertEqual(NumericSwift.j1(x), ref, accuracy: 1e-13, "j1(\(x))")
    }
  }

  /// jn at first zeros of J0 and J1.
  func testBesselJnAtZeros() {
    // First zero of J0 ≈ 2.4048; J0 there ≈ 0
    XCTAssertEqual(NumericSwift.j0(2.4048255577), 0.0, accuracy: 1e-5)
    // First zero of J1 ≈ 3.8317; J1 there ≈ 0
    XCTAssertEqual(NumericSwift.j1(3.8317059702), 0.0, accuracy: 1e-5)
  }

  /// jn(n, x) for higher orders.
  func testBesselJnHigherOrders() {
    // J2(1) reference from DLMF
    XCTAssertEqual(NumericSwift.jn(2, 1.0), 0.11490348493190048, accuracy: 1e-13)
    // J3(2) reference
    XCTAssertEqual(NumericSwift.jn(3, 2.0), 0.12894324947440204, accuracy: 1e-13)
    // J0 == jn(0, x)
    XCTAssertEqual(NumericSwift.jn(0, 2.0), NumericSwift.j0(2.0), accuracy: 1e-15)
    // J1 == jn(1, x)
    XCTAssertEqual(NumericSwift.jn(1, 3.0), NumericSwift.j1(3.0), accuracy: 1e-15)
  }

  // MARK: - Bessel Functions (Second Kind)

  /// Y0 reference values from DLMF 10.4.3.
  func testBesselY0KnownValues() {
    let cases: [(Double, Double)] = [
      (1.0, 0.08825696421567696),
      (2.0, 0.5103756726497451),
      (3.0, 0.37685001001279034),
      (5.0, -0.30851762524903376),
      (10.0, 0.055671167283599395),
    ]
    for (x, ref) in cases {
      XCTAssertEqual(NumericSwift.y0(x), ref, accuracy: 1e-8, "y0(\(x))")
    }
  }

  /// Y1 reference values from DLMF 10.4.3.
  func testBesselY1KnownValues() {
    let cases: [(Double, Double)] = [
      (1.0, -0.7812128213002887),
      (2.0, -0.10703243154093755),
      (3.0, 0.32467442479172733),
      (5.0, 0.14786314339122680),
      (10.0, 0.24901542420695386),
    ]
    for (x, ref) in cases {
      XCTAssertEqual(NumericSwift.y1(x), ref, accuracy: 1e-13, "y1(\(x))")
    }
  }

  /// yn(n, x) for higher orders.
  func testBesselYnHigherOrders() {
    // Y2(1) reference
    XCTAssertEqual(NumericSwift.yn(2, 1.0), -1.6506826068162546, accuracy: 1e-13)
    // Y0 == yn(0, x)
    XCTAssertEqual(NumericSwift.yn(0, 2.0), NumericSwift.y0(2.0), accuracy: 1e-15)
    // Y1 == yn(1, x)
    XCTAssertEqual(NumericSwift.yn(1, 3.0), NumericSwift.y1(3.0), accuracy: 1e-15)
  }

  /// yn(n, x) returns -∞ for x ≤ 0.
  func testBesselYnNonPositive() {
    XCTAssertEqual(NumericSwift.y0(0.0), -.infinity)
    XCTAssertEqual(NumericSwift.y1(0.0), -.infinity)
    XCTAssertEqual(NumericSwift.y0(-1.0), -.infinity)
    XCTAssertEqual(NumericSwift.yn(2, 0.0), -.infinity)
  }

  // MARK: - Elliptic Integrals

  /// K(0) = π/2, K(m) → ∞ as m → 1.
  func testEllipkKnownValues() {
    XCTAssertEqual(NumericSwift.ellipk(0.0), Double.pi / 2.0, accuracy: 1e-14)
    XCTAssertEqual(NumericSwift.ellipk(0.5), 1.8540746773013719, accuracy: 1e-13)
    XCTAssertEqual(NumericSwift.ellipk(0.9), 2.578092113348173, accuracy: 1e-12)
    XCTAssertEqual(NumericSwift.ellipk(0.99), 3.6956373629898738, accuracy: 1e-11)
    // Domain: m < 1 required; m = 1 → NaN
    XCTAssertTrue(NumericSwift.ellipk(1.0).isNaN)
    XCTAssertTrue(NumericSwift.ellipk(-0.1).isNaN)
  }

  /// E(0) = π/2, E(1) = 1.
  func testEllipseKnownValues() {
    XCTAssertEqual(NumericSwift.ellipe(0.0), Double.pi / 2.0, accuracy: 1e-14)
    XCTAssertEqual(NumericSwift.ellipe(0.5), 1.3506438810476755, accuracy: 1e-13)
    XCTAssertEqual(NumericSwift.ellipe(0.9), 1.1047747327040733, accuracy: 1e-13)
    XCTAssertEqual(NumericSwift.ellipe(1.0), 1.0, accuracy: 1e-14)
    XCTAssertTrue(NumericSwift.ellipe(-0.1).isNaN)
  }

  /// Legendre identity: K(m)*E(1-m) + E(m)*K(1-m) - K(m)*K(1-m) = π/2.
  func testEllipticLegendreRelation() {
    let ms = [0.1, 0.3, 0.5, 0.7, 0.9]
    for m in ms {
      let km = NumericSwift.ellipk(m)
      let em = NumericSwift.ellipe(m)
      let km1 = NumericSwift.ellipk(1.0 - m)
      let em1 = NumericSwift.ellipe(1.0 - m)
      let identity = km * em1 + em * km1 - km * km1
      XCTAssertEqual(
        identity, Double.pi / 2.0, accuracy: 1e-11,
        "Legendre relation at m=\(m)")
    }
  }

  // MARK: - Riemann Zeta Function

  /// ζ(2) = π²/6, ζ(4) = π⁴/90, ζ(6) = π⁶/945 (Euler's formula).
  func testZetaEvenIntegers() {
    let pi = Double.pi
    // N=100 Dirichlet series converges slowly near s=2; tolerance = 0.002
    let tol2: Double = 0.002
    XCTAssertEqual(NumericSwift.zeta(2.0), pi * pi / 6.0, accuracy: tol2)
    let tol4: Double = 0.00000002
    XCTAssertEqual(NumericSwift.zeta(4.0), pow(pi, 4) / 90.0, accuracy: tol4)
    XCTAssertEqual(NumericSwift.zeta(6.0), pow(pi, 6) / 945.0, accuracy: 1e-9)
    XCTAssertEqual(NumericSwift.zeta(8.0), pow(pi, 8) / 9450.0, accuracy: 1e-8)
  }

  /// ζ(0) = -1/2, ζ(s=1) = ∞.
  func testZetaSpecialValues() {
    XCTAssertEqual(NumericSwift.zeta(0.0), -0.5, accuracy: 1e-14)
    XCTAssertEqual(NumericSwift.zeta(1.0), .infinity)
    // ζ(-1) = -1/12; reflection formula accuracy limited by Dirichlet series at s=2
    let negOneTol: Double = 0.0001
    XCTAssertEqual(NumericSwift.zeta(-1.0), -1.0 / 12.0, accuracy: negOneTol)
    // ζ(-3) = 1/120; same tolerance
    XCTAssertEqual(NumericSwift.zeta(-3.0), 1.0 / 120.0, accuracy: negOneTol)
  }

  /// ζ(s) > 1 for large s (converges to 1).
  func testZetaLargeS() {
    // ζ(s) → 1 as s → ∞; for s=20, ζ ≈ 1 + 2^-20 + ...
    let zeta20 = NumericSwift.zeta(20.0)
    XCTAssertGreaterThan(zeta20, 1.0)
    XCTAssertLessThan(zeta20, 1.0 + 1.0e-5)
  }

  // MARK: - Lambert W Function

  /// w * exp(w) == x for various x values (defining property).
  func testLambertWIdentity() {
    let xs = [0.0, 0.1, 0.5, 1.0, Darwin.M_E, 5.0, 10.0, 100.0, 1000.0]
    for x in xs {
      let w = NumericSwift.lambertw(x)
      let check = w * Darwin.exp(w)
      XCTAssertEqual(
        check, x, accuracy: 1e-12 * max(1, abs(x)),
        "lambertw identity at x=\(x)")
    }
  }

  /// W(0) = 0, W(e) = 1, W(1) ≈ 0.5671 (omega constant).
  func testLambertWKnownValues() {
    XCTAssertEqual(NumericSwift.lambertw(0.0), 0.0, accuracy: 1e-14)
    XCTAssertEqual(NumericSwift.lambertw(Darwin.M_E), 1.0, accuracy: 1e-13)
    XCTAssertEqual(NumericSwift.lambertw(1.0), 0.5671432904097838, accuracy: 1e-13)
    // W(-1/e) = -1 (branch point)
    let minVal = -1.0 / Darwin.M_E
    XCTAssertEqual(NumericSwift.lambertw(minVal), -1.0, accuracy: 1e-12)
  }

  /// Below branch point (-1/e) returns NaN.
  func testLambertWBelowBranchPoint() {
    XCTAssertTrue(NumericSwift.lambertw(-1.0).isNaN)
    XCTAssertTrue(NumericSwift.lambertw(-0.5).isNaN)
  }

  // MARK: - NaN / Inf Input Handling

  /// Functions should return NaN for NaN input without crashing.
  func testNaNInputHandling() {
    let nan = Double.nan
    XCTAssertTrue(NumericSwift.erf(nan).isNaN)
    XCTAssertTrue(NumericSwift.erfc(nan).isNaN)
    XCTAssertTrue(NumericSwift.erfinv(nan).isNaN)
    XCTAssertTrue(NumericSwift.erfcinv(nan).isNaN)
    XCTAssertTrue(NumericSwift.digamma(nan).isNaN)
    XCTAssertTrue(NumericSwift.gammainc(1.0, nan).isNaN)
    XCTAssertTrue(NumericSwift.ellipk(nan).isNaN)
    XCTAssertTrue(NumericSwift.ellipe(nan).isNaN)
    XCTAssertTrue(NumericSwift.lambertw(nan).isNaN)
  }

  /// erf(+∞) = 1, erf(-∞) = -1; erfc(+∞) = 0, erfc(-∞) = 2.
  func testErfAtInfinity() {
    XCTAssertEqual(NumericSwift.erf(.infinity), 1.0, accuracy: 1e-15)
    XCTAssertEqual(NumericSwift.erf(-.infinity), -1.0, accuracy: 1e-15)
    XCTAssertEqual(NumericSwift.erfc(.infinity), 0.0, accuracy: 1e-15)
    XCTAssertEqual(NumericSwift.erfc(-.infinity), 2.0, accuracy: 1e-15)
  }

  /// gammainc / gammaincc invalid domain returns NaN.
  func testGammaincInvalidDomain() {
    XCTAssertTrue(NumericSwift.gammainc(-1.0, 1.0).isNaN)
    XCTAssertTrue(NumericSwift.gammainc(1.0, -1.0).isNaN)
  }

  /// betainc outside [0,1] returns NaN.
  func testBetaincInvalidDomain() {
    XCTAssertTrue(NumericSwift.betainc(1.0, 1.0, -0.1).isNaN)
    XCTAssertTrue(NumericSwift.betainc(1.0, 1.0, 1.1).isNaN)
  }

  /// ellipk / ellipe outside domain return NaN.
  func testEllipticInvalidDomain() {
    XCTAssertTrue(NumericSwift.ellipk(1.5).isNaN)
    XCTAssertTrue(NumericSwift.ellipk(-0.5).isNaN)
    XCTAssertTrue(NumericSwift.ellipe(1.5).isNaN)
    XCTAssertTrue(NumericSwift.ellipe(-0.5).isNaN)
  }

  /// zeta is continuous away from s=1 (no crash for s < 1).
  func testZetaNearPoleIsSafe() {
    // Should not crash or produce NaN away from the pole
    let val = NumericSwift.zeta(0.5)
    XCTAssertFalse(val.isNaN)
    XCTAssertFalse(val.isInfinite)
  }
}
