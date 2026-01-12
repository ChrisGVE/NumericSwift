//
//  Constants.swift
//  NumericSwift
//
//  Mathematical and physical constants following scipy.constants patterns.
//
//  Licensed under the MIT License.
//

import Foundation

// MARK: - Mathematical Constants

/// Mathematical constants
public enum MathConstants {
    /// Pi (π) - ratio of circle circumference to diameter
    public static let pi: Double = .pi

    /// Tau (τ) - 2π, the ratio of circle circumference to radius
    public static let tau: Double = 2 * .pi

    /// Euler's number (e) - base of natural logarithm
    public static let e: Double = 2.718281828459045235360287471352662497757

    /// Golden ratio (φ) - (1 + √5) / 2
    public static let phi: Double = 1.618033988749894848204586834365638117720

    /// Euler-Mascheroni constant (γ)
    public static let eulerGamma: Double = 0.577215664901532860606512090082402431042

    /// Square root of 2
    public static let sqrt2: Double = 1.414213562373095048801688724209698078570

    /// Square root of 3
    public static let sqrt3: Double = 1.732050807568877293527446341505872366943

    /// Natural logarithm of 2
    public static let ln2: Double = 0.693147180559945309417232121458176568076

    /// Natural logarithm of 10
    public static let ln10: Double = 2.302585092994045684017991454684364207601

    /// Infinity
    public static let inf: Double = .infinity

    /// Not a number
    public static let nan: Double = .nan
}

// MARK: - Physical Constants (CODATA 2018)

/// Physical constants in SI units (CODATA 2018 recommended values)
public enum PhysicalConstants {
    /// Speed of light in vacuum (m/s)
    public static let c: Double = 299792458

    /// Planck constant (J⋅s)
    public static let h: Double = 6.62607015e-34

    /// Reduced Planck constant ℏ = h/(2π) (J⋅s)
    public static let hbar: Double = 1.054571817e-34

    /// Newtonian gravitational constant (m³/(kg⋅s²))
    public static let G: Double = 6.67430e-11

    /// Standard acceleration of gravity (m/s²)
    public static let g: Double = 9.80665

    /// Elementary charge (C)
    public static let elementaryCharge: Double = 1.602176634e-19

    /// Electron mass (kg)
    public static let electronMass: Double = 9.1093837015e-31

    /// Proton mass (kg)
    public static let protonMass: Double = 1.67262192369e-27

    /// Neutron mass (kg)
    public static let neutronMass: Double = 1.67492749804e-27

    /// Atomic mass unit (kg)
    public static let atomicMass: Double = 1.66053906660e-27

    /// Boltzmann constant (J/K)
    public static let k: Double = 1.380649e-23

    /// Avogadro constant (1/mol)
    public static let N_A: Double = 6.02214076e23

    /// Molar gas constant (J/(mol⋅K))
    public static let R: Double = 8.314462618

    /// Vacuum electric permittivity (F/m)
    public static let epsilon0: Double = 8.8541878128e-12

    /// Vacuum magnetic permeability (H/m)
    public static let mu0: Double = 1.25663706212e-6

    /// Stefan-Boltzmann constant (W/(m²⋅K⁴))
    public static let sigma: Double = 5.670374419e-8

    /// Fine-structure constant (dimensionless)
    public static let alpha: Double = 7.2973525693e-3

    /// Rydberg constant (1/m)
    public static let Rydberg: Double = 10973731.568160

    /// Bohr radius (m)
    public static let bohrRadius: Double = 5.29177210903e-11

    /// Classical electron radius (m)
    public static let electronRadius: Double = 2.8179403262e-15

    /// Compton wavelength (m)
    public static let comptonWavelength: Double = 2.42631023867e-12

    /// Magnetic flux quantum (Wb)
    public static let fluxQuantum: Double = 2.067833848e-15

    /// Conductance quantum (S)
    public static let conductanceQuantum: Double = 7.748091729e-5

    /// Josephson constant (Hz/V)
    public static let josephsonConstant: Double = 483597.8484e9

    /// Von Klitzing constant (Ω)
    public static let vonKlitzingConstant: Double = 25812.80745
}

// MARK: - Unit Conversions

/// Angle conversion factors (to radians)
public enum AngleConversions {
    /// Radians per degree
    public static let degree: Double = .pi / 180

    /// Radians per arcminute
    public static let arcmin: Double = .pi / 10800

    /// Radians per arcsecond
    public static let arcsec: Double = .pi / 648000

    /// Convert degrees to radians
    public static func toRadians(_ degrees: Double) -> Double {
        degrees * degree
    }

    /// Convert radians to degrees
    public static func toDegrees(_ radians: Double) -> Double {
        radians / degree
    }
}

/// Length conversion factors (to meters)
public enum LengthConversions {
    /// Meters per inch
    public static let inch: Double = 0.0254

    /// Meters per foot
    public static let foot: Double = 0.3048

    /// Meters per yard
    public static let yard: Double = 0.9144

    /// Meters per mile
    public static let mile: Double = 1609.344

    /// Meters per nautical mile
    public static let nauticalMile: Double = 1852

    /// Meters per astronomical unit
    public static let au: Double = 149597870700

    /// Meters per light-year
    public static let lightYear: Double = 9.4607304725808e15

    /// Meters per parsec
    public static let parsec: Double = 3.0856775814913673e16

    /// Meters per angstrom
    public static let angstrom: Double = 1e-10

    /// Meters per micron
    public static let micron: Double = 1e-6
}

/// Mass conversion factors (to kilograms)
public enum MassConversions {
    /// Kilograms per gram
    public static let gram: Double = 0.001

    /// Kilograms per metric ton (tonne)
    public static let tonne: Double = 1000

    /// Kilograms per pound (avoirdupois)
    public static let pound: Double = 0.45359237

    /// Kilograms per ounce (avoirdupois)
    public static let ounce: Double = 0.028349523125

    /// Kilograms per stone
    public static let stone: Double = 6.35029318

    /// Kilograms per short ton (US)
    public static let shortTon: Double = 907.18474

    /// Kilograms per long ton (UK)
    public static let longTon: Double = 1016.0469088
}

/// Time conversion factors (to seconds)
public enum TimeConversions {
    /// Seconds per minute
    public static let minute: Double = 60

    /// Seconds per hour
    public static let hour: Double = 3600

    /// Seconds per day
    public static let day: Double = 86400

    /// Seconds per week
    public static let week: Double = 604800

    /// Seconds per Julian year (365.25 days)
    public static let year: Double = 31557600

    /// Seconds per sidereal year
    public static let siderealYear: Double = 31558149.504
}

/// Temperature conversion functions
public enum TemperatureConversions {
    /// Absolute zero in Celsius
    public static let zeroCelsius: Double = 273.15

    /// Convert Celsius to Kelvin
    public static func celsiusToKelvin(_ celsius: Double) -> Double {
        celsius + zeroCelsius
    }

    /// Convert Kelvin to Celsius
    public static func kelvinToCelsius(_ kelvin: Double) -> Double {
        kelvin - zeroCelsius
    }

    /// Convert Fahrenheit to Kelvin
    public static func fahrenheitToKelvin(_ fahrenheit: Double) -> Double {
        (fahrenheit + 459.67) * 5 / 9
    }

    /// Convert Kelvin to Fahrenheit
    public static func kelvinToFahrenheit(_ kelvin: Double) -> Double {
        kelvin * 9 / 5 - 459.67
    }

    /// Convert Fahrenheit to Celsius
    public static func fahrenheitToCelsius(_ fahrenheit: Double) -> Double {
        (fahrenheit - 32) * 5 / 9
    }

    /// Convert Celsius to Fahrenheit
    public static func celsiusToFahrenheit(_ celsius: Double) -> Double {
        celsius * 9 / 5 + 32
    }
}

/// Pressure conversion factors (to Pascal)
public enum PressureConversions {
    /// Pascals per atmosphere
    public static let atm: Double = 101325

    /// Pascals per bar
    public static let bar: Double = 1e5

    /// Pascals per torr (mmHg)
    public static let torr: Double = 133.32236842105263

    /// Pascals per psi
    public static let psi: Double = 6894.757293168361
}

/// Energy conversion factors (to Joules)
public enum EnergyConversions {
    /// Joules per electron volt
    public static let eV: Double = 1.602176634e-19

    /// Joules per calorie (thermochemical)
    public static let calorie: Double = 4.184

    /// Joules per erg
    public static let erg: Double = 1e-7

    /// Joules per British thermal unit
    public static let btu: Double = 1055.05585262

    /// Joules per kilowatt-hour
    public static let kWh: Double = 3.6e6
}

/// Power conversion factors (to Watts)
public enum PowerConversions {
    /// Watts per horsepower (mechanical)
    public static let horsepower: Double = 745.69987158227022
}

// MARK: - Convenience Type Aliases

/// Shorthand access to mathematical constants
public typealias Math = MathConstants

/// Shorthand access to physical constants
public typealias Physics = PhysicalConstants
