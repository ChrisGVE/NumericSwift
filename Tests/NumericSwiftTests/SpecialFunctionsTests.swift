//
//  SpecialFunctionsTests.swift
//  NumericSwift
//
//  Tests for special mathematical functions.
//
//  Licensed under the MIT License.
//

import XCTest
@testable import NumericSwift

final class SpecialFunctionsTests: XCTestCase {

    // MARK: - erfinv Tests

    // Test central region (|x| <= 0.7)
    func testErfinvCentralRegion() {
        // Values where erfinv is well-behaved
        XCTAssertEqual(NumericSwift.erfinv(0.0), 0.0, accuracy: 1e-15)
        XCTAssertEqual(NumericSwift.erfinv(0.1), 0.08885599049425769, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.2), 0.17914345462129167, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.3), 0.27246271472675443, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.4), 0.37080715859355795, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.5), 0.4769362762044699, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.6), 0.5951160814499948, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.7), 0.7328690779592168, accuracy: 1e-14)

        // Negative values (odd function)
        XCTAssertEqual(NumericSwift.erfinv(-0.5), -0.4769362762044699, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(-0.3), -0.27246271472675443, accuracy: 1e-14)
    }

    // Test near-tail region (0.7 < |x| < 0.99)
    func testErfinvNearTailRegion() {
        XCTAssertEqual(NumericSwift.erfinv(0.8), 0.9061938024368232, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.9), 1.1630871536766743, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.95), 1.3859038243496775, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(0.98), 1.6449763571331872, accuracy: 1e-14)

        // Negative values
        XCTAssertEqual(NumericSwift.erfinv(-0.9), -1.1630871536766743, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfinv(-0.95), -1.3859038243496775, accuracy: 1e-14)
    }

    // Test extreme tail region (|x| >= 0.99) - previously buggy
    func testErfinvExtremeTailRegion() {
        // These values previously returned ~1.2 instead of correct values
        XCTAssertEqual(NumericSwift.erfinv(0.99), 1.8213863677184496, accuracy: 1e-13)
        XCTAssertEqual(NumericSwift.erfinv(0.999), 2.3267537655135246, accuracy: 1e-13)
        XCTAssertEqual(NumericSwift.erfinv(0.9999), 2.7510639057120607, accuracy: 1e-13)
        XCTAssertEqual(NumericSwift.erfinv(0.99999), 3.1234132743415708, accuracy: 1e-12)
        XCTAssertEqual(NumericSwift.erfinv(0.999999), 3.458910737279028, accuracy: 1e-11)
        XCTAssertEqual(NumericSwift.erfinv(0.9999999), 3.7665625815708778, accuracy: 1e-10)

        // Negative extreme values
        XCTAssertEqual(NumericSwift.erfinv(-0.99), -1.8213863677184496, accuracy: 1e-13)
        XCTAssertEqual(NumericSwift.erfinv(-0.9999), -2.7510639057120607, accuracy: 1e-13)
        XCTAssertEqual(NumericSwift.erfinv(-0.99999), -3.1234132743415708, accuracy: 1e-12)
    }

    // Test edge cases
    func testErfinvEdgeCases() {
        // Zero
        XCTAssertEqual(NumericSwift.erfinv(0.0), 0.0, accuracy: 1e-15)

        // Boundary values return infinity
        XCTAssertEqual(NumericSwift.erfinv(1.0), .infinity)
        XCTAssertEqual(NumericSwift.erfinv(-1.0), -.infinity)

        // Out of domain returns NaN
        XCTAssert(NumericSwift.erfinv(1.5).isNaN)
        XCTAssert(NumericSwift.erfinv(-1.5).isNaN)
        XCTAssert(NumericSwift.erfinv(2.0).isNaN)
        XCTAssert(NumericSwift.erfinv(-100.0).isNaN)
    }

    // Test roundtrip: erf(erfinv(x)) == x
    func testErfinvRoundtrip() {
        let testValues = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                          0.95, 0.99, 0.999, 0.9999, 0.99999,
                          -0.1, -0.5, -0.9, -0.99, -0.9999]

        for x in testValues {
            let y = NumericSwift.erfinv(x)
            let roundtrip = NumericSwift.erf(y)
            XCTAssertEqual(roundtrip, x, accuracy: 1e-14, "Roundtrip failed for x=\(x)")
        }
    }

    // Test symmetry: erfinv(-x) == -erfinv(x)
    func testErfinvSymmetry() {
        let testValues = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999]

        for x in testValues {
            let pos = NumericSwift.erfinv(x)
            let neg = NumericSwift.erfinv(-x)
            XCTAssertEqual(neg, -pos, accuracy: 1e-15, "Symmetry failed for x=\(x)")
        }
    }

    // MARK: - erf Tests

    func testErf() {
        XCTAssertEqual(NumericSwift.erf(0.0), 0.0, accuracy: 1e-15)
        XCTAssertEqual(NumericSwift.erf(1.0), 0.8427007929497149, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erf(2.0), 0.9953222650189527, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erf(-1.0), -0.8427007929497149, accuracy: 1e-14)
    }

    func testErfc() {
        XCTAssertEqual(NumericSwift.erfc(0.0), 1.0, accuracy: 1e-15)
        XCTAssertEqual(NumericSwift.erfc(1.0), 0.15729920705028513, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfc(2.0), 0.004677734981047266, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.erfc(-1.0), 1.8427007929497148, accuracy: 1e-14)
    }

    // MARK: - Beta Function Tests

    func testBeta() {
        // beta(a, b) = gamma(a) * gamma(b) / gamma(a+b)
        XCTAssertEqual(NumericSwift.beta(1.0, 1.0), 1.0, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.beta(2.0, 2.0), 1.0/6.0, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.beta(0.5, 0.5), .pi, accuracy: 1e-13)
    }

    // MARK: - Bessel Function Tests

    func testBesselJ0() {
        XCTAssertEqual(NumericSwift.j0(0.0), 1.0, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.j0(1.0), 0.7651976865579666, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.j0(2.0), 0.22389077914123567, accuracy: 1e-14)
    }

    func testBesselJ1() {
        XCTAssertEqual(NumericSwift.j1(0.0), 0.0, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.j1(1.0), 0.44005058574493355, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.j1(2.0), 0.5767248077568735, accuracy: 1e-14)
    }

    func testBesselY0() {
        XCTAssertEqual(NumericSwift.y0(1.0), 0.08825696421567696, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.y0(2.0), 0.5103756726497451, accuracy: 1e-14)
    }

    func testBesselY1() {
        XCTAssertEqual(NumericSwift.y1(1.0), -0.7812128213002887, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.y1(2.0), -0.10703243154093755, accuracy: 1e-14)
    }

    // MARK: - Elliptic Integral Tests

    func testEllipticK() {
        XCTAssertEqual(NumericSwift.ellipk(0.0), .pi / 2, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.ellipk(0.5), 1.8540746773013719, accuracy: 1e-13)
    }

    func testEllipticE() {
        XCTAssertEqual(NumericSwift.ellipe(0.0), .pi / 2, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.ellipe(0.5), 1.3506438810476755, accuracy: 1e-13)
    }

    // MARK: - Incomplete Gamma Tests

    func testGammaInc() {
        // gammainc(a, 0) = 0
        XCTAssertEqual(NumericSwift.gammainc(1.0, 0.0), 0.0, accuracy: 1e-14)
        // gammainc(1, x) = 1 - exp(-x)
        XCTAssertEqual(NumericSwift.gammainc(1.0, 1.0), 1.0 - exp(-1.0), accuracy: 1e-13)
        XCTAssertEqual(NumericSwift.gammainc(2.0, 1.0), 0.26424111765711533, accuracy: 1e-13)
    }

    func testGammaIncC() {
        // gammaincC(a, x) = 1 - gammainc(a, x)
        XCTAssertEqual(NumericSwift.gammaincc(1.0, 1.0), exp(-1.0), accuracy: 1e-13)
    }

    // MARK: - Incomplete Beta Tests

    func testBetaInc() {
        XCTAssertEqual(NumericSwift.betainc(1.0, 1.0, 0.5), 0.5, accuracy: 1e-14)
        XCTAssertEqual(NumericSwift.betainc(2.0, 2.0, 0.5), 0.5, accuracy: 1e-13)
        XCTAssertEqual(NumericSwift.betainc(0.5, 0.5, 0.5), 0.5, accuracy: 1e-13)
    }

    // MARK: - Zeta Function Tests

    func testZeta() {
        // zeta(2) = pi^2/6
        XCTAssertEqual(NumericSwift.zeta(2.0), .pi * .pi / 6.0, accuracy: 1e-4)
        // zeta(4) = pi^4/90
        XCTAssertEqual(NumericSwift.zeta(4.0), pow(.pi, 4) / 90.0, accuracy: 1e-4)
    }

    // MARK: - Lambert W Tests

    func testLambertW() {
        // W(0) = 0
        XCTAssertEqual(NumericSwift.lambertw(0.0), 0.0, accuracy: 1e-14)
        // W(e) = 1
        XCTAssertEqual(NumericSwift.lambertw(M_E), 1.0, accuracy: 1e-13)
        // W(1) ~ 0.5671
        XCTAssertEqual(NumericSwift.lambertw(1.0), 0.5671432904097838, accuracy: 1e-13)
    }

    // MARK: - Digamma Tests

    func testDigamma() {
        // psi(1) = -gamma (Euler-Mascheroni constant)
        XCTAssertEqual(NumericSwift.digamma(1.0), -0.5772156649015329, accuracy: 1e-9)
        // psi(2) = 1 - gamma
        XCTAssertEqual(NumericSwift.digamma(2.0), 0.42278433509846714, accuracy: 1e-9)
    }
}
