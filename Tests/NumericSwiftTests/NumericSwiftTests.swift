//
//  NumericSwiftTests.swift
//  NumericSwift
//
//  Licensed under the MIT License.
//

import XCTest
@testable import NumericSwift

final class NumericSwiftTests: XCTestCase {

    // MARK: - Complex Number Tests

    func testComplexCreation() {
        let z = Complex(re: 3, im: 4)
        XCTAssertEqual(z.re, 3)
        XCTAssertEqual(z.im, 4)
    }

    func testComplexMagnitude() {
        let z = Complex(re: 3, im: 4)
        XCTAssertEqual(z.abs, 5, accuracy: 1e-10)
    }

    func testComplexAddition() {
        let z1 = Complex(re: 1, im: 2)
        let z2 = Complex(re: 3, im: 4)
        let sum = z1 + z2
        XCTAssertEqual(sum.re, 4, accuracy: 1e-10)
        XCTAssertEqual(sum.im, 6, accuracy: 1e-10)
    }

    func testComplexMultiplication() {
        let z1 = Complex(re: 3, im: 4)
        let z2 = Complex(re: 1, im: 2)
        let product = z1 * z2
        // (3+4i)(1+2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
        XCTAssertEqual(product.re, -5, accuracy: 1e-10)
        XCTAssertEqual(product.im, 10, accuracy: 1e-10)
    }

    func testComplexConjugate() {
        let z = Complex(re: 3, im: 4)
        let conj = z.conj
        XCTAssertEqual(conj.re, 3, accuracy: 1e-10)
        XCTAssertEqual(conj.im, -4, accuracy: 1e-10)
    }

    func testComplexExp() {
        // e^(i*pi) = -1
        let z = Complex(re: 0, im: .pi)
        let result = z.exp
        XCTAssertEqual(result.re, -1, accuracy: 1e-10)
        XCTAssertEqual(result.im, 0, accuracy: 1e-10)
    }

    // MARK: - Constants Tests

    func testMathConstants() {
        XCTAssertEqual(MathConstants.pi, .pi, accuracy: 1e-15)
        XCTAssertEqual(MathConstants.e, 2.718281828459045, accuracy: 1e-10)
        XCTAssertEqual(MathConstants.sqrt2, sqrt(2), accuracy: 1e-10)
    }

    func testPhysicalConstants() {
        XCTAssertEqual(PhysicalConstants.c, 299792458)
        XCTAssertEqual(PhysicalConstants.h, 6.62607015e-34, accuracy: 1e-44)
    }

    // MARK: - Statistics Tests

    func testMean() {
        XCTAssertEqual(mean([1, 2, 3, 4, 5]), 3, accuracy: 1e-10)
        XCTAssertEqual(mean([10]), 10, accuracy: 1e-10)
        XCTAssert(mean([]).isNaN)
    }

    func testMedian() {
        XCTAssertEqual(median([1, 2, 3, 4, 5]), 3, accuracy: 1e-10)
        XCTAssertEqual(median([1, 2, 3, 4]), 2.5, accuracy: 1e-10)
        XCTAssert(median([]).isNaN)
    }

    func testVariance() {
        // Population variance
        XCTAssertEqual(variance([2, 4, 4, 4, 5, 5, 7, 9], ddof: 0), 4, accuracy: 1e-10)
        // Sample variance
        XCTAssertEqual(variance([2, 4, 4, 4, 5, 5, 7, 9], ddof: 1), 4.571428571428571, accuracy: 1e-10)
    }

    func testStddev() {
        XCTAssertEqual(stddev([2, 4, 4, 4, 5, 5, 7, 9], ddof: 0), 2, accuracy: 1e-10)
    }

    func testPercentile() {
        XCTAssertEqual(percentile([1, 2, 3, 4, 5], 50), 3, accuracy: 1e-10)
        XCTAssertEqual(percentile([1, 2, 3, 4, 5], 0), 1, accuracy: 1e-10)
        XCTAssertEqual(percentile([1, 2, 3, 4, 5], 100), 5, accuracy: 1e-10)
    }

    func testGeometricMean() {
        XCTAssertEqual(gmean([1, 2, 4, 8]), 2.8284271247461903, accuracy: 1e-10)
        XCTAssert(gmean([1, 0, 2]).isNaN) // Non-positive values
    }

    func testHarmonicMean() {
        XCTAssertEqual(hmean([1, 2, 4]), 1.7142857142857142, accuracy: 1e-10)
    }

    func testMode() {
        XCTAssertEqual(mode([1, 2, 2, 3, 3, 3]), 3, accuracy: 1e-10)
        XCTAssertEqual(mode([1, 1, 2, 2]), 1, accuracy: 1e-10) // Returns smallest on ties
    }

    // MARK: - Cumulative Functions Tests

    func testCumsum() {
        XCTAssertEqual(cumsum([1, 2, 3, 4]), [1, 3, 6, 10])
    }

    func testCumprod() {
        XCTAssertEqual(cumprod([1, 2, 3, 4]), [1, 2, 6, 24])
    }

    func testDiff() {
        XCTAssertEqual(diff([1, 3, 6, 10]), [2, 3, 4])
    }

    // MARK: - Combinatorics Tests

    func testFactorial() {
        XCTAssertEqual(factorial(0), 1, accuracy: 1e-10)
        XCTAssertEqual(factorial(5), 120, accuracy: 1e-10)
        XCTAssertEqual(factorial(10), 3628800, accuracy: 1e-10)
    }

    func testPermutations() {
        XCTAssertEqual(perm(5, 2), 20, accuracy: 1e-10)
        XCTAssertEqual(perm(5, 0), 1, accuracy: 1e-10)
    }

    func testCombinations() {
        XCTAssertEqual(comb(5, 2), 10, accuracy: 1e-10)
        XCTAssertEqual(comb(10, 3), 120, accuracy: 1e-10)
        XCTAssertEqual(binomial(5, 2), 10, accuracy: 1e-10) // Alias
    }

    // MARK: - Coordinate Conversion Tests

    func testPolarToCartesian() {
        let (x, y) = polarToCart(r: 1, theta: .pi / 2)
        XCTAssertEqual(x, 0, accuracy: 1e-10)
        XCTAssertEqual(y, 1, accuracy: 1e-10)
    }

    func testCartesianToPolar() {
        let (r, theta) = cartToPolar(x: 1, y: 1)
        XCTAssertEqual(r, sqrt(2), accuracy: 1e-10)
        XCTAssertEqual(theta, .pi / 4, accuracy: 1e-10)
    }

    func testDegRadConversions() {
        XCTAssertEqual(deg2rad(180), .pi, accuracy: 1e-10)
        XCTAssertEqual(rad2deg(.pi), 180, accuracy: 1e-10)
    }

    func testClip() {
        XCTAssertEqual(clip(5, min: 0, max: 10), 5)
        XCTAssertEqual(clip(-5, min: 0, max: 10), 0)
        XCTAssertEqual(clip(15, min: 0, max: 10), 10)
        XCTAssertEqual(clip([1, 5, 10, 15], min: 3, max: 12), [3, 5, 10, 12])
    }

    // MARK: - Number Theory Tests

    func testIsPrime() {
        XCTAssertFalse(isPrime(0))
        XCTAssertFalse(isPrime(1))
        XCTAssertTrue(isPrime(2))
        XCTAssertTrue(isPrime(3))
        XCTAssertFalse(isPrime(4))
        XCTAssertTrue(isPrime(5))
        XCTAssertTrue(isPrime(97))
        XCTAssertFalse(isPrime(100))
    }

    func testPrimeFactors() {
        XCTAssertEqual(primeFactors(1).count, 0)
        XCTAssertEqual(primeFactors(12).map { $0.prime }, [2, 3])
        XCTAssertEqual(primeFactors(12).map { $0.exponent }, [2, 1])
        XCTAssertEqual(primeFactors(100).map { $0.prime }, [2, 5])
        XCTAssertEqual(primeFactors(100).map { $0.exponent }, [2, 2])
    }

    func testPrimesUpTo() {
        XCTAssertEqual(primesUpTo(10), [2, 3, 5, 7])
        XCTAssertEqual(primesUpTo(20), [2, 3, 5, 7, 11, 13, 17, 19])
        XCTAssertEqual(primesUpTo(1), [])
    }

    func testGCD() {
        XCTAssertEqual(gcd(12, 18), 6)
        XCTAssertEqual(gcd(17, 13), 1)
        XCTAssertEqual(gcd(0, 5), 5)
        XCTAssertEqual(gcd(-12, 18), 6)
    }

    func testLCM() {
        XCTAssertEqual(lcm(4, 6), 12)
        XCTAssertEqual(lcm(3, 5), 15)
        XCTAssertEqual(lcm(0, 5), 0)
    }

    func testEulerPhi() {
        XCTAssertEqual(eulerPhi(1), 1)
        XCTAssertEqual(eulerPhi(10), 4)  // 1, 3, 7, 9 are coprime to 10
        XCTAssertEqual(eulerPhi(12), 4)  // 1, 5, 7, 11 are coprime to 12
        XCTAssertEqual(eulerPhi(7), 6)   // All 1-6 coprime (7 is prime)
    }

    func testDivisorSigma() {
        // σ_0(12) = 6 (divisors: 1, 2, 3, 4, 6, 12)
        XCTAssertEqual(divisorSigma(12, k: 0)!, 6, accuracy: 1e-10)
        // σ_1(12) = 28 (sum: 1+2+3+4+6+12)
        XCTAssertEqual(divisorSigma(12, k: 1)!, 28, accuracy: 1e-10)
    }

    func testMobius() {
        XCTAssertEqual(mobius(1), 1)
        XCTAssertEqual(mobius(2), -1)    // Prime
        XCTAssertEqual(mobius(6), 1)     // 2*3, two distinct primes
        XCTAssertEqual(mobius(30), -1)   // 2*3*5, three distinct primes
        XCTAssertEqual(mobius(4), 0)     // 2^2, has squared factor
        XCTAssertEqual(mobius(12), 0)    // 2^2*3, has squared factor
    }

    func testLiouville() {
        XCTAssertEqual(liouville(1), 1)
        XCTAssertEqual(liouville(2), -1)   // Ω(2) = 1
        XCTAssertEqual(liouville(4), 1)    // Ω(4) = 2
        XCTAssertEqual(liouville(8), -1)   // Ω(8) = 3
        XCTAssertEqual(liouville(12), -1)  // 12 = 2^2 * 3, Ω = 2 + 1 = 3, (-1)^3 = -1
    }

    func testVonMangoldt() {
        XCTAssertEqual(vonMangoldt(1), 0, accuracy: 1e-10)
        XCTAssertEqual(vonMangoldt(2), log(2), accuracy: 1e-10)
        XCTAssertEqual(vonMangoldt(4), log(2), accuracy: 1e-10)  // 2^2
        XCTAssertEqual(vonMangoldt(6), 0, accuracy: 1e-10)       // Not a prime power
    }

    func testPrimePi() {
        XCTAssertEqual(primePi(10), 4)    // 2, 3, 5, 7
        XCTAssertEqual(primePi(100), 25)
        XCTAssertEqual(primePi(1), 0)
    }

    func testModPow() {
        XCTAssertEqual(modPow(2, 10, 1000), 24)  // 1024 mod 1000
        XCTAssertEqual(modPow(3, 5, 7), 5)       // 243 mod 7
        XCTAssertEqual(modPow(2, 0, 5), 1)
    }

    func testExtendedGcd() {
        let (g, x, y) = extendedGcd(35, 15)
        XCTAssertEqual(g, 5)
        XCTAssertEqual(35 * x + 15 * y, 5)
    }

    func testModInverse() {
        XCTAssertEqual(modInverse(3, 7), 5)  // 3*5 = 15 ≡ 1 (mod 7)
        XCTAssertNil(modInverse(2, 4))        // gcd(2,4) = 2 ≠ 1
    }

    func testDigitSum() {
        XCTAssertEqual(digitSum(123), 6)
        XCTAssertEqual(digitSum(9999), 36)
        XCTAssertEqual(digitSum(15, base: 2), 4)  // 15 = 1111 in binary
    }

    func testDigitalRoot() {
        XCTAssertEqual(digitalRoot(123), 6)   // 1+2+3 = 6
        XCTAssertEqual(digitalRoot(9999), 9)  // 36 -> 9
    }

    // MARK: - Series Tests

    func testPolyval() {
        // p(x) = 1 + 2x + 3x² evaluated at x=2: 1 + 4 + 12 = 17
        XCTAssertEqual(polyval([1, 2, 3], at: 2), 17, accuracy: 1e-10)
        // Empty polynomial
        XCTAssertEqual(polyval([], at: 5), 0)
        // Constant
        XCTAssertEqual(polyval([7], at: 100), 7, accuracy: 1e-10)
    }

    func testPolyvalCentered() {
        // p(x) = 1 + 2(x-1) + 3(x-1)² at x=2: 1 + 2 + 3 = 6
        XCTAssertEqual(polyval([1, 2, 3], at: 2, center: 1), 6, accuracy: 1e-10)
    }

    func testPolyadd() {
        // (1 + 2x) + (3 + 4x + 5x²) = 4 + 6x + 5x²
        XCTAssertEqual(polyadd([1, 2], [3, 4, 5]), [4, 6, 5])
    }

    func testPolymul() {
        // (1 + x) * (1 + x) = 1 + 2x + x²
        XCTAssertEqual(polymul([1, 1], [1, 1]), [1, 2, 1])
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
        XCTAssertEqual(polymul([1, 2], [3, 4]), [3, 10, 8])
    }

    func testPolyder() {
        // d/dx (1 + 2x + 3x²) = 2 + 6x
        XCTAssertEqual(polyder([1, 2, 3]), [2, 6])
    }

    func testPolyint() {
        // ∫(2 + 6x)dx = 2x + 3x² (constant = 0)
        XCTAssertEqual(polyint([2, 6]), [0, 2, 3])
    }

    func testTaylorCoefficients() {
        // sin(x) = x - x³/6 + x⁵/120 - ...
        let sinCoeffs = taylorCoefficients(for: "sin", terms: 6)!
        XCTAssertEqual(sinCoeffs[0], 0, accuracy: 1e-10)
        XCTAssertEqual(sinCoeffs[1], 1, accuracy: 1e-10)
        XCTAssertEqual(sinCoeffs[2], 0, accuracy: 1e-10)
        XCTAssertEqual(sinCoeffs[3], -1.0/6.0, accuracy: 1e-10)

        // exp(x) = 1 + x + x²/2 + x³/6 + ...
        let expCoeffs = taylorCoefficients(for: "exp", terms: 4)!
        XCTAssertEqual(expCoeffs[0], 1, accuracy: 1e-10)
        XCTAssertEqual(expCoeffs[1], 1, accuracy: 1e-10)
        XCTAssertEqual(expCoeffs[2], 0.5, accuracy: 1e-10)
        XCTAssertEqual(expCoeffs[3], 1.0/6.0, accuracy: 1e-10)
    }

    func testTaylorEval() {
        // Taylor approximation of sin(0.5) should be close to actual
        let approx = taylorEval("sin", at: 0.5, terms: 10)!
        let actual = Darwin.sin(0.5)
        XCTAssertEqual(approx, actual, accuracy: 1e-10)

        // Taylor approximation of exp(1) ≈ e
        let expApprox = taylorEval("exp", at: 1, terms: 20)!
        XCTAssertEqual(expApprox, Darwin.exp(1), accuracy: 1e-10)
    }

    func testSeriesSum() {
        // Sum of 1/2^n from n=0 to 10
        let (sum, converged, _) = seriesSum(from: 0, to: 10) { n in
            Darwin.pow(0.5, Double(n))
        }
        XCTAssertTrue(converged)
        // Geometric series: (1 - 0.5^11) / (1 - 0.5) = 2 * (1 - 1/2048)
        XCTAssertEqual(sum, 2.0 - Darwin.pow(0.5, 10), accuracy: 1e-10)
    }

    func testSeriesSumConvergence() {
        // Sum of 1/n² converges to π²/6 (slowly - need many terms)
        let (sum, _, _) = seriesSum(from: 1, tolerance: 1e-10, maxIterations: 100000) { n in
            1.0 / Double(n * n)
        }
        // 1/n² converges slowly, within 0.001 of limit with 100k terms
        XCTAssertEqual(sum, baselSum, accuracy: 0.001)
    }

    func testSeriesProduct() {
        // Product of (1 - 1/n²) from n=2 to 10: equals 1/2 * (n+1)/n for Wallis-like product
        let (product, _, _) = seriesProduct(from: 2, to: 10) { n in
            1.0 - 1.0 / Double(n * n)
        }
        XCTAssertTrue(product > 0)
        XCTAssertTrue(product < 1)
    }

    func testPartialSums() {
        let sums = partialSums(from: 1, count: 5) { n in Double(n) }
        XCTAssertEqual(sums, [1, 3, 6, 10, 15])  // Triangular numbers
    }

    func testChebyshevPoints() {
        let points = chebyshevPoints(center: 0, scale: 1, count: 5)
        XCTAssertEqual(points.count, 5)
        XCTAssertEqual(points[0], 1, accuracy: 1e-10)  // cos(0) = 1
        XCTAssertEqual(points[4], -1, accuracy: 1e-10)  // cos(π) = -1
    }

    func testWellKnownConstants() {
        XCTAssertEqual(baselSum, Double.pi * Double.pi / 6, accuracy: 1e-10)
        XCTAssertTrue(eulerMascheroni > 0.577 && eulerMascheroni < 0.578)
        XCTAssertTrue(aperyConstant > 1.202 && aperyConstant < 1.203)
    }
}
