//
//  ConstantsTests.swift
//  Tests/NumericSwiftTests/
//
//  Coverage tests for Constants.swift: MathConstants, PhysicalConstants (CODATA 2018),
//  and all unit-conversion enums including round-trip verification.
//
//  Oracle: scipy.constants, CODATA 2018 (https://physics.nist.gov/cuu/Constants/)
//  Branch: fix/issue-19-coverage  Refs #19
//

import XCTest
@testable import NumericSwift

final class ConstantsTests: XCTestCase {

    private let relTol = 1e-9  // relative tolerance for CODATA values

    private func assertRelClose(_ a: Double, _ b: Double, tol: Double = 1e-9,
                                _ msg: String = "", file: StaticString = #file, line: UInt = #line) {
        let rel = Swift.abs(a - b) / Swift.max(Swift.abs(b), 1e-300)
        XCTAssertLessThanOrEqual(rel, tol, "\(msg): |a-b|/|b| = \(rel) > \(tol) (a=\(a), b=\(b))",
                                 file: file, line: line)
    }

    private func assertClose(_ a: Double, _ b: Double, tol: Double = 1e-12,
                             _ msg: String = "", file: StaticString = #file, line: UInt = #line) {
        XCTAssertEqual(a, b, accuracy: tol, msg, file: file, line: line)
    }

    // MARK: - MathConstants

    func testPi_matchesSwiftBuiltin() {
        XCTAssertEqual(MathConstants.pi, Double.pi)
    }

    func testTau_isTwoPi() {
        assertClose(MathConstants.tau, 2 * Double.pi)
    }

    func testE_knownValue() {
        // scipy.constants.e = 2.718281828459045
        assertRelClose(MathConstants.e, 2.718281828459045, tol: 1e-15)
    }

    func testPhi_goldenRatioFormula() {
        // φ = (1 + √5)/2
        let expected = (1.0 + 5.0.squareRoot()) / 2.0
        assertClose(MathConstants.phi, expected)
    }

    func testPhi_goldenRatioProperty() {
        // φ² = φ + 1
        let phi = MathConstants.phi
        assertClose(phi * phi, phi + 1, tol: 1e-10)
    }

    func testEulerGamma_knownValue() {
        // OEIS A001620: 0.5772156649015328...
        assertRelClose(MathConstants.eulerGamma, 0.5772156649015329, tol: 1e-14)
    }

    func testSqrt2_squared() {
        // sqrt2^2 must be exactly 2 within floating-point precision
        assertClose(MathConstants.sqrt2 * MathConstants.sqrt2, 2.0, tol: 1e-14)
    }

    func testSqrt3_squared() {
        assertClose(MathConstants.sqrt3 * MathConstants.sqrt3, 3.0, tol: 1e-14)
    }

    func testLn2_expIdentity() {
        // exp(ln2) = 2
        assertClose(Foundation.exp(MathConstants.ln2), 2.0, tol: 1e-14)
    }

    func testLn10_expIdentity() {
        // exp(ln10) = 10
        assertClose(Foundation.exp(MathConstants.ln10), 10.0, tol: 1e-12)
    }

    func testInf_isInfinity() {
        XCTAssertTrue(MathConstants.inf.isInfinite)
        XCTAssertTrue(MathConstants.inf > 0)
    }

    func testNan_isNaN() {
        XCTAssertTrue(MathConstants.nan.isNaN)
    }

    // MARK: - Math alias

    func testMathAlias_sameAsConstants() {
        XCTAssertEqual(Math.pi, MathConstants.pi)
        XCTAssertEqual(Math.e, MathConstants.e)
        XCTAssertEqual(Math.phi, MathConstants.phi)
    }

    // MARK: - PhysicalConstants (CODATA 2018)

    func testSpeedOfLight_exactValue() {
        // c is exact by SI definition: 299792458 m/s
        XCTAssertEqual(PhysicalConstants.c, 299792458.0)
    }

    func testPlanckConstant_codata2018() {
        // h is exact by SI definition: 6.62607015e-34 J·s
        XCTAssertEqual(PhysicalConstants.h, 6.62607015e-34)
    }

    func testHbar_relationToH() {
        // ℏ = h/(2π) — verify to 6 significant figures (CODATA defines both independently as exact)
        let expected = PhysicalConstants.h / (2 * Double.pi)
        assertRelClose(PhysicalConstants.hbar, expected, tol: 1e-6)
    }

    func testElementaryCharge_exactValue() {
        // e is exact by SI definition: 1.602176634e-19 C
        XCTAssertEqual(PhysicalConstants.elementaryCharge, 1.602176634e-19)
    }

    func testBoltzmannConstant_exactValue() {
        // k_B is exact by SI definition: 1.380649e-23 J/K
        XCTAssertEqual(PhysicalConstants.k, 1.380649e-23)
    }

    func testAvogadroConstant_exactValue() {
        // N_A is exact by SI definition: 6.02214076e23 mol^-1
        XCTAssertEqual(PhysicalConstants.N_A, 6.02214076e23)
    }

    func testMolarGasConstant_derivedFromKAndNA() {
        // R = k_B * N_A
        let derived = PhysicalConstants.k * PhysicalConstants.N_A
        assertRelClose(PhysicalConstants.R, derived, tol: 1e-6)
    }

    func testGravitationalConstant_magnitude() {
        // G ≈ 6.674e-11 m³/(kg·s²)
        assertRelClose(PhysicalConstants.G, 6.67430e-11, tol: 1e-5)
    }

    func testStandardGravity_knownValue() {
        // g = 9.80665 m/s² (exact by ISO 80000-3)
        XCTAssertEqual(PhysicalConstants.g, 9.80665)
    }

    func testElectronMass_codata2018() {
        // m_e = 9.1093837015e-31 kg
        assertRelClose(PhysicalConstants.electronMass, 9.1093837015e-31, tol: 1e-10)
    }

    func testProtonMass_codata2018() {
        assertRelClose(PhysicalConstants.protonMass, 1.67262192369e-27, tol: 1e-10)
    }

    func testProtonToElectronMassRatio_knownValue() {
        // m_p/m_e ≈ 1836.15267343 (NIST CODATA 2018)
        let ratio = PhysicalConstants.protonMass / PhysicalConstants.electronMass
        assertRelClose(ratio, 1836.15267343, tol: 1e-6)
    }

    func testFineStructureConstant_codata2018() {
        // α ≈ 7.2973525693e-3
        assertRelClose(PhysicalConstants.alpha, 7.2973525693e-3, tol: 1e-9)
    }

    func testStefanBoltzmann_codata2018() {
        // σ ≈ 5.670374419e-8 W/(m²·K⁴)
        assertRelClose(PhysicalConstants.sigma, 5.670374419e-8, tol: 1e-9)
    }

    func testEpsilon0_codata2018() {
        // ε₀ ≈ 8.8541878128e-12 F/m
        assertRelClose(PhysicalConstants.epsilon0, 8.8541878128e-12, tol: 1e-10)
    }

    func testMu0_codata2018() {
        // μ₀ ≈ 1.25663706212e-6 H/m
        assertRelClose(PhysicalConstants.mu0, 1.25663706212e-6, tol: 1e-10)
    }

    func testPhysicsAlias_sameAsConstants() {
        XCTAssertEqual(Physics.c, PhysicalConstants.c)
        XCTAssertEqual(Physics.h, PhysicalConstants.h)
    }

    // MARK: - AngleConversions

    func testAngleDegree_radiansPerDegree() {
        // 1° = π/180 rad
        assertClose(AngleConversions.degree, Double.pi / 180)
    }

    func testAngleArcmin_radiansPerArcmin() {
        assertClose(AngleConversions.arcmin, Double.pi / 10800)
    }

    func testAngleArcsec_radiansPerArcsec() {
        assertClose(AngleConversions.arcsec, Double.pi / 648000)
    }

    func testAngleToRadians_90degrees() {
        assertClose(AngleConversions.toRadians(90), Double.pi / 2)
    }

    func testAngleToDegrees_halfPi() {
        assertClose(AngleConversions.toDegrees(Double.pi / 2), 90.0, tol: 1e-12)
    }

    func testAngleRoundTrip_randomAngle() {
        let deg = 123.456
        assertClose(AngleConversions.toDegrees(AngleConversions.toRadians(deg)), deg, tol: 1e-10)
    }

    func testAngleRoundTrip_zero() {
        assertClose(AngleConversions.toDegrees(AngleConversions.toRadians(0)), 0.0)
    }

    func testAngleRoundTrip_360degrees() {
        assertClose(AngleConversions.toDegrees(AngleConversions.toRadians(360)), 360.0, tol: 1e-10)
    }

    // MARK: - LengthConversions

    func testLengthInch_toMeters() {
        // 1 inch = 0.0254 m (exact)
        XCTAssertEqual(LengthConversions.inch, 0.0254)
    }

    func testLengthFoot_toMeters() {
        // 1 ft = 12 in
        assertClose(LengthConversions.foot, 12 * LengthConversions.inch)
    }

    func testLengthYard_toMeters() {
        // 1 yd = 3 ft
        assertClose(LengthConversions.yard, 3 * LengthConversions.foot)
    }

    func testLengthMile_toMeters() {
        // 1 mi = 1609.344 m (exact)
        XCTAssertEqual(LengthConversions.mile, 1609.344)
    }

    func testLengthNauticalMile_toMeters() {
        XCTAssertEqual(LengthConversions.nauticalMile, 1852.0)
    }

    func testLengthAngstrom_toMeters() {
        XCTAssertEqual(LengthConversions.angstrom, 1e-10)
    }

    func testLengthMicron_toMeters() {
        XCTAssertEqual(LengthConversions.micron, 1e-6)
    }

    func testLengthAU_knownValue() {
        // 1 AU = 149597870700 m (exact IAU 2012)
        XCTAssertEqual(LengthConversions.au, 149597870700.0)
    }

    // MARK: - MassConversions

    func testMassGram_toKilograms() {
        XCTAssertEqual(MassConversions.gram, 0.001)
    }

    func testMassPound_toKilograms() {
        // 1 lb (avoirdupois) = 0.45359237 kg (exact)
        XCTAssertEqual(MassConversions.pound, 0.45359237)
    }

    func testMassOunce_toKilograms() {
        // 1 oz = 1/16 lb
        assertClose(MassConversions.ounce, MassConversions.pound / 16, tol: 1e-12)
    }

    func testMassTonne_toKilograms() {
        XCTAssertEqual(MassConversions.tonne, 1000.0)
    }

    // MARK: - TimeConversions

    func testTimeMinute_toSeconds() {
        XCTAssertEqual(TimeConversions.minute, 60.0)
    }

    func testTimeHour_toSeconds() {
        XCTAssertEqual(TimeConversions.hour, 3600.0)
    }

    func testTimeDay_toSeconds() {
        XCTAssertEqual(TimeConversions.day, 86400.0)
    }

    func testTimeWeek_toSeconds() {
        // 1 week = 7 days
        assertClose(TimeConversions.week, 7 * TimeConversions.day)
    }

    func testTimeYear_julianYear() {
        // 1 Julian year = 365.25 days
        assertClose(TimeConversions.year, 365.25 * TimeConversions.day)
    }

    // MARK: - TemperatureConversions

    func testTemperatureAbsoluteZero() {
        XCTAssertEqual(TemperatureConversions.zeroCelsius, 273.15)
    }

    func testCelsiusToKelvin_freezingPoint() {
        // 0°C = 273.15 K
        assertClose(TemperatureConversions.celsiusToKelvin(0), 273.15)
    }

    func testCelsiusToKelvin_boilingPoint() {
        // 100°C = 373.15 K
        assertClose(TemperatureConversions.celsiusToKelvin(100), 373.15)
    }

    func testKelvinToCelsius_absoluteZero() {
        // 0 K = -273.15°C
        assertClose(TemperatureConversions.kelvinToCelsius(0), -273.15)
    }

    func testCelsiusKelvinRoundTrip() {
        let celsius = 25.0
        assertClose(
            TemperatureConversions.kelvinToCelsius(TemperatureConversions.celsiusToKelvin(celsius)),
            celsius
        )
    }

    func testFahrenheitToKelvin_freezingPoint() {
        // 32°F = 273.15 K
        assertClose(TemperatureConversions.fahrenheitToKelvin(32), 273.15)
    }

    func testFahrenheitToKelvin_boilingPoint() {
        // 212°F = 373.15 K
        assertClose(TemperatureConversions.fahrenheitToKelvin(212), 373.15)
    }

    func testKelvinToFahrenheit_boilingPoint() {
        // 373.15 K = 212°F
        assertClose(TemperatureConversions.kelvinToFahrenheit(373.15), 212.0, tol: 1e-10)
    }

    func testFahrenheitToKelvinRoundTrip() {
        let fahrenheit = 98.6
        assertClose(
            TemperatureConversions.kelvinToFahrenheit(TemperatureConversions.fahrenheitToKelvin(fahrenheit)),
            fahrenheit,
            tol: 1e-10
        )
    }

    func testFahrenheitToCelsius_freezingPoint() {
        // 32°F = 0°C
        assertClose(TemperatureConversions.fahrenheitToCelsius(32), 0.0, tol: 1e-13)
    }

    func testFahrenheitToCelsius_boilingPoint() {
        // 212°F = 100°C
        assertClose(TemperatureConversions.fahrenheitToCelsius(212), 100.0, tol: 1e-13)
    }

    func testCelsiusToFahrenheit_freezingPoint() {
        // 0°C = 32°F
        assertClose(TemperatureConversions.celsiusToFahrenheit(0), 32.0)
    }

    func testCelsiusToFahrenheit_bodyTemp() {
        // 37°C = 98.6°F
        assertClose(TemperatureConversions.celsiusToFahrenheit(37), 98.6, tol: 1e-10)
    }

    func testCelsiusFahrenheitRoundTrip() {
        let celsius = -40.0  // special: -40°C = -40°F
        assertClose(TemperatureConversions.fahrenheitToCelsius(
            TemperatureConversions.celsiusToFahrenheit(celsius)
        ), celsius, tol: 1e-12)
    }

    // MARK: - PressureConversions

    func testPressureAtm_toPascals() {
        // 1 atm = 101325 Pa (exact)
        XCTAssertEqual(PressureConversions.atm, 101325.0)
    }

    func testPressureBar_toPascals() {
        XCTAssertEqual(PressureConversions.bar, 1e5)
    }

    func testPressureTorr_toPascals() {
        // 1 torr = 101325/760 Pa ≈ 133.322368...
        assertRelClose(PressureConversions.torr, 101325.0 / 760.0, tol: 1e-10)
    }

    // MARK: - EnergyConversions

    func testEnergyEV_toJoules() {
        // 1 eV = elementary charge in Joules (exact)
        XCTAssertEqual(EnergyConversions.eV, PhysicalConstants.elementaryCharge)
    }

    func testEnergyCalorie_toJoules() {
        // 1 cal (thermochemical) = 4.184 J (exact)
        XCTAssertEqual(EnergyConversions.calorie, 4.184)
    }

    func testEnergyErg_toJoules() {
        XCTAssertEqual(EnergyConversions.erg, 1e-7)
    }

    func testEnergykWh_toJoules() {
        // 1 kWh = 3.6e6 J
        XCTAssertEqual(EnergyConversions.kWh, 3.6e6)
    }

    // MARK: - PowerConversions

    func testPowerHorsepower_toWatts() {
        // 1 hp (mechanical) ≈ 745.69987158... W
        assertRelClose(PowerConversions.horsepower, 745.69987158227022, tol: 1e-10)
    }
}
