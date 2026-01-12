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

    // MARK: - Interpolation Tests

    func testFindInterval() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0]
        XCTAssertEqual(findInterval(x, 0.5), 0)
        XCTAssertEqual(findInterval(x, 1.5), 1)
        XCTAssertEqual(findInterval(x, 3.5), 3)
    }

    func testLinearInterp1d() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0]
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]

        // Linear interpolation at midpoints
        XCTAssertEqual(interp1d(x: x, y: y, xNew: 0.5, kind: .linear), 0.5, accuracy: 1e-10)
        XCTAssertEqual(interp1d(x: x, y: y, xNew: 1.5, kind: .linear), 2.5, accuracy: 1e-10)

        // Boundary values
        XCTAssertEqual(interp1d(x: x, y: y, xNew: 0, kind: .linear), 0, accuracy: 1e-10)
        XCTAssertEqual(interp1d(x: x, y: y, xNew: 4, kind: .linear), 16, accuracy: 1e-10)
    }

    func testNearestInterp1d() {
        let x = [0.0, 1.0, 2.0, 3.0]
        let y = [0.0, 10.0, 20.0, 30.0]

        XCTAssertEqual(interp1d(x: x, y: y, xNew: 0.3, kind: .nearest), 0, accuracy: 1e-10)
        XCTAssertEqual(interp1d(x: x, y: y, xNew: 0.6, kind: .nearest), 10, accuracy: 1e-10)
    }

    func testCubicSplineBasic() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0]
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]  // y = x²

        let coeffs = computeSplineCoeffs(x: x, y: y, bc: .natural)
        XCTAssertEqual(coeffs.count, 4)  // n-1 segments

        // Evaluate at data points (should match exactly)
        XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 0), 0, accuracy: 1e-10)
        XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 2), 4, accuracy: 1e-10)
        XCTAssertEqual(evalCubicSpline(x: x, coeffs: coeffs, xNew: 4), 16, accuracy: 1e-10)

        // Evaluate at intermediate point
        let mid = evalCubicSpline(x: x, coeffs: coeffs, xNew: 1.5)
        XCTAssertTrue(mid > 2 && mid < 2.5)  // Should be close to 1.5² = 2.25
    }

    func testCubicSplineDerivative() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0]
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]  // y = x²

        let coeffs = computeSplineCoeffs(x: x, y: y, bc: .natural)

        // Derivative of x² is 2x
        // At x=2, derivative should be approximately 4
        let deriv = evalCubicSplineDerivative(x: x, coeffs: coeffs, xNew: 2, order: 1)
        XCTAssertEqual(deriv, 4, accuracy: 0.5)  // Some error expected due to spline approximation
    }

    func testPchipInterpolation() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0]
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]

        let d = computePchipDerivatives(x: x, y: y)
        XCTAssertEqual(d.count, 5)

        // Should interpolate through data points
        XCTAssertEqual(evalPchip(x: x, y: y, d: d, xNew: 0), 0, accuracy: 1e-10)
        XCTAssertEqual(evalPchip(x: x, y: y, d: d, xNew: 2), 4, accuracy: 1e-10)
        XCTAssertEqual(evalPchip(x: x, y: y, d: d, xNew: 4), 16, accuracy: 1e-10)
    }

    func testAkimaInterpolation() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0]
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]

        let coeffs = computeAkimaCoeffs(x: x, y: y)
        XCTAssertEqual(coeffs.count, 4)

        // Should interpolate through data points
        XCTAssertEqual(evalAkima(x: x, coeffs: coeffs, xNew: 0), 0, accuracy: 1e-10)
        XCTAssertEqual(evalAkima(x: x, coeffs: coeffs, xNew: 2), 4, accuracy: 1e-10)
    }

    func testLagrangeInterpolation() {
        let x = [0.0, 1.0, 2.0]
        let y = [0.0, 1.0, 4.0]  // y = x²

        // Lagrange through 3 points of x² should be exact
        XCTAssertEqual(evalLagrange(x: x, y: y, xNew: 0), 0, accuracy: 1e-10)
        XCTAssertEqual(evalLagrange(x: x, y: y, xNew: 1), 1, accuracy: 1e-10)
        XCTAssertEqual(evalLagrange(x: x, y: y, xNew: 0.5), 0.25, accuracy: 1e-10)
        XCTAssertEqual(evalLagrange(x: x, y: y, xNew: 1.5), 2.25, accuracy: 1e-10)
    }

    func testBarycentricInterpolation() {
        let x = [0.0, 1.0, 2.0]
        let y = [0.0, 1.0, 4.0]  // y = x²

        let w = computeBarycentricWeights(x: x)
        XCTAssertEqual(w.count, 3)

        // Should match Lagrange results
        XCTAssertEqual(evalBarycentric(x: x, y: y, w: w, xNew: 0), 0, accuracy: 1e-10)
        XCTAssertEqual(evalBarycentric(x: x, y: y, w: w, xNew: 1), 1, accuracy: 1e-10)
        XCTAssertEqual(evalBarycentric(x: x, y: y, w: w, xNew: 0.5), 0.25, accuracy: 1e-10)
    }

    func testTridiagonalSolver() {
        // Solve: [2 1 0] [x0]   [5]
        //        [1 2 1] [x1] = [6]
        //        [0 1 2] [x2]   [5]
        // Solution: x = [1, 2, 1.5] (approximately)
        let diag = [2.0, 2.0, 2.0]
        let offDiag = [1.0, 1.0]
        let rhs = [5.0, 6.0, 5.0]

        let result = solveTridiagonal(diag: diag, offDiag: offDiag, rhs: rhs)
        XCTAssertEqual(result.count, 3)

        // Verify solution satisfies equations
        XCTAssertEqual(2*result[0] + result[1], 5, accuracy: 1e-10)
        XCTAssertEqual(result[0] + 2*result[1] + result[2], 6, accuracy: 1e-10)
        XCTAssertEqual(result[1] + 2*result[2], 5, accuracy: 1e-10)
    }

    // MARK: - Integration Tests

    func testQuadBasic() {
        // ∫₀¹ x² dx = 1/3
        let result = quad({ x in x * x }, 0, 1)
        XCTAssertEqual(result.value, 1.0/3.0, accuracy: 1e-10)
        XCTAssertTrue(result.error < 1e-10)
    }

    func testQuadSin() {
        // ∫₀^π sin(x) dx = 2
        let result = quad({ x in Darwin.sin(x) }, 0, .pi)
        XCTAssertEqual(result.value, 2, accuracy: 1e-10)
    }

    func testQuadGaussian() {
        // ∫_{-∞}^{∞} e^(-x²) dx = √π
        let result = quad({ x in Darwin.exp(-x * x) }, -.infinity, .infinity, limit: 100)
        XCTAssertEqual(result.value, sqrt(.pi), accuracy: 1e-6)
    }

    func testDblquadRectangle() {
        // ∫₀¹ ∫₀¹ xy dydx = 0.25
        let result = dblquad({ y, x in x * y }, xa: 0, xb: 1, ya: 0, yb: 1)
        XCTAssertEqual(result.value, 0.25, accuracy: 1e-8)
    }

    func testTplquadCube() {
        // ∫₀¹ ∫₀¹ ∫₀¹ xyz dzdydx = 0.125
        let result = tplquad({ z, y, x in x * y * z }, xa: 0, xb: 1, ya: 0, yb: 1, za: 0, zb: 1)
        XCTAssertEqual(result.value, 0.125, accuracy: 1e-6)
    }

    func testFixedQuad() {
        // ∫₀¹ x² dx = 1/3
        let result = fixedQuad({ x in x * x }, 0, 1, n: 5)
        XCTAssertEqual(result, 1.0/3.0, accuracy: 1e-10)
    }

    func testRomberg() {
        // ∫₀^π sin(x) dx = 2
        let result = romberg({ x in Darwin.sin(x) }, 0, .pi)
        XCTAssertEqual(result.value, 2, accuracy: 1e-8)
    }

    func testSimps() {
        // Simpson's rule for x² from 0 to 4 with h=1
        // y = [0, 1, 4, 9, 16]
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]
        let result = simps(y, dx: 1)
        // Exact integral of x² from 0 to 4 is 64/3 ≈ 21.333
        XCTAssertEqual(result, 64.0/3.0, accuracy: 0.1)
    }

    func testTrapz() {
        // Trapezoidal rule for x² from 0 to 4
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]
        let result = trapz(y, dx: 1)
        // Trapezoidal will overestimate for convex functions
        XCTAssertTrue(result > 21)
    }

    func testTrapzNonUniform() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0]
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]
        let result = trapz(y, x: x)
        XCTAssertTrue(result > 21)
    }

    func testSolveIVPExponential() {
        // dy/dt = y with y(0) = 1 => y(t) = e^t
        let result = solveIVP(
            { y, t in [y[0]] },  // dy/dt = y
            tSpan: (0, 1),
            y0: [1.0],
            method: .rk45
        )

        XCTAssertTrue(result.success)
        XCTAssertEqual(result.y.last![0], Darwin.exp(1), accuracy: 0.01)
    }

    func testSolveIVPOscillator() {
        // Simple harmonic oscillator: y'' = -y
        // Let y[0] = position, y[1] = velocity
        // dy[0]/dt = y[1], dy[1]/dt = -y[0]
        // y(0) = 1, y'(0) = 0 => y(t) = cos(t)
        let result = solveIVP(
            { y, t in [y[1], -y[0]] },
            tSpan: (0, .pi),
            y0: [1.0, 0.0],
            method: .rk45
        )

        XCTAssertTrue(result.success)
        // At t=π, cos(π) = -1
        XCTAssertEqual(result.y.last![0], -1, accuracy: 0.01)
    }

    func testSolveIVPWithTEval() {
        // dy/dt = 1 with y(0) = 0 => y(t) = t
        let result = solveIVP(
            { y, t in [1.0] },
            tSpan: (0, 5),
            y0: [0.0],
            tEval: [0, 1, 2, 3, 4, 5]
        )

        XCTAssertEqual(result.t.count, 6)
        for (i, tVal) in result.t.enumerated() {
            XCTAssertEqual(result.y[i][0], tVal, accuracy: 0.01)
        }
    }

    func testOdeint() {
        // dy/dt = -y with y(0) = 1 => y(t) = e^(-t)
        let t = [0.0, 0.5, 1.0, 1.5, 2.0]
        let result = odeint(
            { y, t in [-y[0]] },
            y0: [1.0],
            t: t
        )

        XCTAssertEqual(result.count, 5)
        for (i, tVal) in t.enumerated() {
            XCTAssertEqual(result[i][0], Darwin.exp(-tVal), accuracy: 0.01)
        }
    }

    func testRK4Method() {
        // Test RK4 specifically
        let result = solveIVP(
            { y, t in [y[0]] },
            tSpan: (0, 1),
            y0: [1.0],
            method: .rk4
        )

        XCTAssertTrue(result.success)
        XCTAssertEqual(result.y.last![0], Darwin.exp(1), accuracy: 0.01)
    }

    func testRK23Method() {
        // Test RK23 specifically
        let result = solveIVP(
            { y, t in [y[0]] },
            tSpan: (0, 1),
            y0: [1.0],
            method: .rk23
        )

        XCTAssertTrue(result.success)
        XCTAssertEqual(result.y.last![0], Darwin.exp(1), accuracy: 0.05)  // Lower order, less accurate
    }

    // MARK: - Optimization Tests

    func testGoldenSection() {
        // Minimize (x-2)²
        let result = goldenSection({ x in (x - 2) * (x - 2) }, a: 0, b: 5)
        XCTAssertTrue(result.success)
        XCTAssertEqual(result.x, 2, accuracy: 1e-6)
        XCTAssertEqual(result.fun, 0, accuracy: 1e-10)
    }

    func testBrentMinimization() {
        // Minimize (x-3)²
        let result = brent({ x in (x - 3) * (x - 3) }, a: 0, b: 10)
        XCTAssertTrue(result.success)
        XCTAssertEqual(result.x, 3, accuracy: 1e-6)
        XCTAssertEqual(result.fun, 0, accuracy: 1e-10)
    }

    func testBisect() {
        // Find root of x² - 4 = 0 in [0, 5]
        let result = bisect({ x in x * x - 4 }, a: 0, b: 5)
        XCTAssertTrue(result.converged)
        XCTAssertEqual(result.root, 2, accuracy: 1e-6)
    }

    func testNewtonScalar() {
        // Find root of x³ - x - 2 = 0 starting at x=1.5
        let result = newton({ x in x*x*x - x - 2 }, x0: 1.5)
        XCTAssertTrue(result.converged)
        // Verify it's actually a root
        let residual = result.root * result.root * result.root - result.root - 2
        XCTAssertEqual(residual, 0, accuracy: 1e-8)
    }

    func testSecant() {
        // Find root of x² - 2 = 0
        let result = secant({ x in x * x - 2 }, x0: 1)
        XCTAssertTrue(result.converged)
        XCTAssertEqual(result.root, sqrt(2), accuracy: 1e-6)
    }

    func testNelderMead() {
        // Minimize Rosenbrock function: (1-x)² + 100(y-x²)²
        // Minimum is at (1, 1)
        let result = nelderMead({ x in
            let a = 1 - x[0]
            let b = x[1] - x[0] * x[0]
            return a * a + 100 * b * b
        }, x0: [0.0, 0.0])

        XCTAssertTrue(result.success)
        XCTAssertEqual(result.x[0], 1, accuracy: 0.01)
        XCTAssertEqual(result.x[1], 1, accuracy: 0.01)
    }

    func testNewtonMulti() {
        // Solve x² + y² = 1, x = y
        // Solution is (±1/√2, ±1/√2)
        let result = newtonMulti({ x in
            [x[0] * x[0] + x[1] * x[1] - 1, x[0] - x[1]]
        }, x0: [0.5, 0.5])

        XCTAssertTrue(result.success)
        let target = 1.0 / sqrt(2.0)
        XCTAssertEqual(result.x[0], target, accuracy: 1e-6)
        XCTAssertEqual(result.x[1], target, accuracy: 1e-6)
    }

    func testLeastSquares() {
        // Fit y = a*x + b to data
        let xdata = [0.0, 1.0, 2.0, 3.0, 4.0]
        let ydata = [1.0, 3.0, 5.0, 7.0, 9.0]  // y = 2x + 1

        let residuals: ([Double]) -> [Double] = { params in
            var r = [Double]()
            for i in 0..<xdata.count {
                r.append(ydata[i] - (params[0] * xdata[i] + params[1]))
            }
            return r
        }

        let result = leastSquares(residuals, x0: [1.0, 0.0])
        XCTAssertTrue(result.success)
        XCTAssertEqual(result.x[0], 2, accuracy: 1e-6)  // Slope
        XCTAssertEqual(result.x[1], 1, accuracy: 1e-6)  // Intercept
    }

    func testCurveFit() {
        // Fit y = a*exp(-b*x) to data
        let xdata = [0.0, 1.0, 2.0, 3.0, 4.0]
        let ydata = xdata.map { 2.0 * exp(-0.5 * $0) }  // y = 2*exp(-0.5*x)

        let (popt, pcov, info) = curveFit(
            { params, x in params[0] * exp(-params[1] * x) },
            xdata: xdata,
            ydata: ydata,
            p0: [1.0, 1.0]
        )

        XCTAssertTrue(info.success)
        XCTAssertEqual(popt[0], 2, accuracy: 0.01)    // Amplitude
        XCTAssertEqual(popt[1], 0.5, accuracy: 0.01) // Decay rate
        XCTAssertEqual(pcov.count, 2)  // 2x2 covariance matrix
    }

    func testMinimizeQuadratic() {
        // Minimize x² + y² + z²
        let result = nelderMead({ x in
            x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
        }, x0: [1.0, 2.0, 3.0])

        XCTAssertTrue(result.success)
        XCTAssertEqual(result.fun, 0, accuracy: 0.001)
        for val in result.x {
            XCTAssertEqual(val, 0, accuracy: 0.001)
        }
    }
}
