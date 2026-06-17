//
//  Constants.swift  (Domains/)
//  NumericSwiftWorkbenchKit
//
//  Per-domain workbench suite: Constants (mathematical / physical constants and
//  unit conversions).
//
//  This is a **single-strategy-per-constant correctness** domain (WORKBENCH.md
//  §4, "Single-strategy domains"): each constant or conversion IS a strategy id,
//  the comparison scalar is that constant's value (for a conversion function, the
//  converted value of a representative input carried in `inputs.x`), and the
//  oracle is the matching scipy.constants / CODATA-2018 / math reference
//  (Tools/workbench_oracles/constants.py).
//
//  ## Self-awareness
//
//  Constants are EXACT values and conversions are exact compositions, so EVERY
//  fixture case is in-envelope. There are ZERO out-of-envelope cases: the gate is
//  a pure correctness-vs-reference check (non-vacuous — the oracle is
//  scipy.constants / CODATA / math, never NumericSwift; FP1 / FP3). Accordingly,
//  every strategy closure returns a ``StrategyResult`` with EMPTY diagnostics —
//  the `*Constants` symbols have no documented limitation envelope to surface
//  here, and the harness never fabricates a diagnostic.
//
//  ## CODATA vintage (FP1)
//
//  NumericSwift's `PhysicalConstants` declares CODATA 2018. The oracle uses
//  scipy's historical CODATA-2018 table, so the comparison is vintage-aligned.
//  Seven *derived* physical constants (`hbar`, `R`, `sigma`, `fluxQuantum`,
//  `conductanceQuantum`, `josephsonConstant`, `vonKlitzingConstant`) carry a
//  per-case `tol` set to the documented agreement (NumericSwift stores the
//  published ~10-sig-fig CODATA-2018 literal; the full-precision table value
//  differs at the rounding boundary). See `constants.py` for the table and the
//  FP1 rationale. These cases remain in-envelope.
//
//  Strategy ids ↔ NumericSwift API (Sources/NumericSwift/Constants.swift):
//
//    math.<name>    → MathConstants.<name>
//    phys.<name>    → PhysicalConstants.<name>
//    conv.<factor>  → <Foo>Conversions.<factor>     (a stored factor)
//    convfn.<fn>    → <Foo>Conversions.<fn>(inputs.x) (a conversion function)
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import NumericSwift

extension Workbench {

    /// The Constants (math / physical constants + unit conversions) domain suite.
    public static let constantsSuite = DomainSuite(
        name: "constants",
        registerStrategies: registerConstantsStrategies,
        makeEnvelopeRegistry: makeConstantsEnvelopeRegistry
    )
}

// MARK: - Constant resolvers

/// The scalar value of the mathematical constant named by a `math.<name>` id.
@Sendable
private func mathConstant(_ name: String) -> Double? {
    switch name {
    case "math.pi":         return MathConstants.pi
    case "math.tau":        return MathConstants.tau
    case "math.e":          return MathConstants.e
    case "math.phi":        return MathConstants.phi
    case "math.eulerGamma": return MathConstants.eulerGamma
    case "math.sqrt2":      return MathConstants.sqrt2
    case "math.sqrt3":      return MathConstants.sqrt3
    case "math.ln2":        return MathConstants.ln2
    case "math.ln10":       return MathConstants.ln10
    case "math.inf":        return MathConstants.inf
    case "math.nan":        return MathConstants.nan
    default:                return nil
    }
}

/// The scalar value of the physical constant named by a `phys.<name>` id.
@Sendable
private func physicalConstant(_ name: String) -> Double? {
    switch name {
    case "phys.c":                   return PhysicalConstants.c
    case "phys.h":                   return PhysicalConstants.h
    case "phys.hbar":                return PhysicalConstants.hbar
    case "phys.G":                   return PhysicalConstants.G
    case "phys.g":                   return PhysicalConstants.g
    case "phys.elementaryCharge":    return PhysicalConstants.elementaryCharge
    case "phys.electronMass":        return PhysicalConstants.electronMass
    case "phys.protonMass":          return PhysicalConstants.protonMass
    case "phys.neutronMass":         return PhysicalConstants.neutronMass
    case "phys.atomicMass":          return PhysicalConstants.atomicMass
    case "phys.k":                   return PhysicalConstants.k
    case "phys.N_A":                 return PhysicalConstants.N_A
    case "phys.R":                   return PhysicalConstants.R
    case "phys.epsilon0":            return PhysicalConstants.epsilon0
    case "phys.mu0":                 return PhysicalConstants.mu0
    case "phys.sigma":               return PhysicalConstants.sigma
    case "phys.alpha":               return PhysicalConstants.alpha
    case "phys.Rydberg":             return PhysicalConstants.Rydberg
    case "phys.bohrRadius":          return PhysicalConstants.bohrRadius
    case "phys.electronRadius":      return PhysicalConstants.electronRadius
    case "phys.comptonWavelength":   return PhysicalConstants.comptonWavelength
    case "phys.fluxQuantum":         return PhysicalConstants.fluxQuantum
    case "phys.conductanceQuantum":  return PhysicalConstants.conductanceQuantum
    case "phys.josephsonConstant":   return PhysicalConstants.josephsonConstant
    case "phys.vonKlitzingConstant": return PhysicalConstants.vonKlitzingConstant
    default:                         return nil
    }
}

/// The scalar value of the unit-conversion factor named by a `conv.<factor>` id.
@Sendable
private func conversionFactor(_ name: String) -> Double? {
    switch name {
    // Angle (radians per unit)
    case "conv.degree":       return AngleConversions.degree
    case "conv.arcmin":       return AngleConversions.arcmin
    case "conv.arcsec":       return AngleConversions.arcsec
    // Length (meters per unit)
    case "conv.inch":         return LengthConversions.inch
    case "conv.foot":         return LengthConversions.foot
    case "conv.yard":         return LengthConversions.yard
    case "conv.mile":         return LengthConversions.mile
    case "conv.nauticalMile": return LengthConversions.nauticalMile
    case "conv.au":           return LengthConversions.au
    case "conv.lightYear":    return LengthConversions.lightYear
    case "conv.parsec":       return LengthConversions.parsec
    case "conv.angstrom":     return LengthConversions.angstrom
    case "conv.micron":       return LengthConversions.micron
    // Mass (kilograms per unit)
    case "conv.gram":         return MassConversions.gram
    case "conv.tonne":        return MassConversions.tonne
    case "conv.pound":        return MassConversions.pound
    case "conv.ounce":        return MassConversions.ounce
    case "conv.stone":        return MassConversions.stone
    case "conv.shortTon":     return MassConversions.shortTon
    case "conv.longTon":      return MassConversions.longTon
    // Time (seconds per unit)
    case "conv.minute":       return TimeConversions.minute
    case "conv.hour":         return TimeConversions.hour
    case "conv.day":          return TimeConversions.day
    case "conv.week":         return TimeConversions.week
    case "conv.year":         return TimeConversions.year
    // Temperature offset
    case "conv.zeroCelsius":  return TemperatureConversions.zeroCelsius
    // Pressure (pascals per unit)
    case "conv.atm":          return PressureConversions.atm
    case "conv.bar":          return PressureConversions.bar
    case "conv.torr":         return PressureConversions.torr
    case "conv.psi":          return PressureConversions.psi
    // Energy (joules per unit)
    case "conv.eV":           return EnergyConversions.eV
    case "conv.calorie":      return EnergyConversions.calorie
    case "conv.erg":          return EnergyConversions.erg
    case "conv.btu":          return EnergyConversions.btu
    case "conv.kWh":          return EnergyConversions.kWh
    // Power (watts per unit)
    case "conv.horsepower":   return PowerConversions.horsepower
    default:                  return nil
    }
}

/// Apply the conversion function named by a `convfn.<fn>` id to argument `x`.
@Sendable
private func conversionFunction(_ name: String, _ x: Double) -> Double? {
    switch name {
    case "convfn.angleToRadians":      return AngleConversions.toRadians(x)
    case "convfn.angleToDegrees":      return AngleConversions.toDegrees(x)
    case "convfn.celsiusToKelvin":     return TemperatureConversions.celsiusToKelvin(x)
    case "convfn.kelvinToCelsius":     return TemperatureConversions.kelvinToCelsius(x)
    case "convfn.fahrenheitToKelvin":  return TemperatureConversions.fahrenheitToKelvin(x)
    case "convfn.kelvinToFahrenheit":  return TemperatureConversions.kelvinToFahrenheit(x)
    case "convfn.fahrenheitToCelsius": return TemperatureConversions.fahrenheitToCelsius(x)
    case "convfn.celsiusToFahrenheit": return TemperatureConversions.celsiusToFahrenheit(x)
    default:                           return nil
    }
}

// MARK: - Strategy registrations

/// Populate `registry` with the Constants strategies (one per constant /
/// conversion factor / conversion function).
///
/// Every strategy id from the fixture resolves the matching NumericSwift symbol.
/// The `inputs.name` field carries the strategy id (so the closure knows which
/// constant to look up); conversion functions additionally read `inputs.x`. All
/// cases are in-envelope, hence empty diagnostics throughout (no `*Constants`
/// symbol has a documented limitation envelope).
@Sendable
public func registerConstantsStrategies(into registry: inout StrategyRegistry) {

    // Mathematical constants.
    for id in ["math.pi", "math.tau", "math.e", "math.phi", "math.eulerGamma",
               "math.sqrt2", "math.sqrt3", "math.ln2", "math.ln10",
               "math.inf", "math.nan"] {
        registry.register(id: id) { _ in
            mathConstant(id).map { StrategyResult(value: $0) }
        }
    }

    // Physical constants (CODATA 2018).
    for id in ["phys.c", "phys.h", "phys.hbar", "phys.G", "phys.g",
               "phys.elementaryCharge", "phys.electronMass", "phys.protonMass",
               "phys.neutronMass", "phys.atomicMass", "phys.k", "phys.N_A",
               "phys.R", "phys.epsilon0", "phys.mu0", "phys.sigma", "phys.alpha",
               "phys.Rydberg", "phys.bohrRadius", "phys.electronRadius",
               "phys.comptonWavelength", "phys.fluxQuantum",
               "phys.conductanceQuantum", "phys.josephsonConstant",
               "phys.vonKlitzingConstant"] {
        registry.register(id: id) { _ in
            physicalConstant(id).map { StrategyResult(value: $0) }
        }
    }

    // Unit-conversion factors.
    for id in ["conv.degree", "conv.arcmin", "conv.arcsec",
               "conv.inch", "conv.foot", "conv.yard", "conv.mile",
               "conv.nauticalMile", "conv.au", "conv.lightYear", "conv.parsec",
               "conv.angstrom", "conv.micron",
               "conv.gram", "conv.tonne", "conv.pound", "conv.ounce",
               "conv.stone", "conv.shortTon", "conv.longTon",
               "conv.minute", "conv.hour", "conv.day", "conv.week", "conv.year",
               "conv.zeroCelsius",
               "conv.atm", "conv.bar", "conv.torr", "conv.psi",
               "conv.eV", "conv.calorie", "conv.erg", "conv.btu", "conv.kWh",
               "conv.horsepower"] {
        registry.register(id: id) { _ in
            conversionFactor(id).map { StrategyResult(value: $0) }
        }
    }

    // Unit-conversion functions: read the scalar argument from `inputs.x`.
    for id in ["convfn.angleToRadians", "convfn.angleToDegrees",
               "convfn.celsiusToKelvin", "convfn.kelvinToCelsius",
               "convfn.fahrenheitToKelvin", "convfn.kelvinToFahrenheit",
               "convfn.fahrenheitToCelsius", "convfn.celsiusToFahrenheit"] {
        registry.register(id: id) { inputs in
            guard let x = inputs["x"]?.doubleValue,
                  let v = conversionFunction(id, x)
            else { return nil }
            return StrategyResult(value: v)
        }
    }
}

// MARK: - Envelope registry

/// Per-(strategy, tier) accuracy envelopes for the Constants domain.
///
/// Constants are exact and conversions are exact compositions, so the registry
/// is only a fallback: the per-case `tol` in the fixture is authoritative
/// (WORKBENCH.md §5). The seven derived physical constants carry a documented
/// per-case agreement tol there (FP1); everything else matches the reference to
/// ~1e-12 relative. A uniform 1e-9 *absolute* fallback here is never the binding
/// bound for any shipped case, but keeps the registry well-formed for strategies
/// the fixture might exercise without an explicit tol. No strategy has an
/// out-of-envelope regime, so no `outsideEnvelope` diagnostic is ever expected.
@Sendable
public func makeConstantsEnvelopeRegistry() -> EnvelopeRegistry {
    var reg = EnvelopeRegistry()
    let strategies =
        ["math.pi", "math.tau", "math.e", "math.phi", "math.eulerGamma",
         "math.sqrt2", "math.sqrt3", "math.ln2", "math.ln10", "math.inf", "math.nan",
         "phys.c", "phys.h", "phys.hbar", "phys.G", "phys.g",
         "phys.elementaryCharge", "phys.electronMass", "phys.protonMass",
         "phys.neutronMass", "phys.atomicMass", "phys.k", "phys.N_A", "phys.R",
         "phys.epsilon0", "phys.mu0", "phys.sigma", "phys.alpha", "phys.Rydberg",
         "phys.bohrRadius", "phys.electronRadius", "phys.comptonWavelength",
         "phys.fluxQuantum", "phys.conductanceQuantum", "phys.josephsonConstant",
         "phys.vonKlitzingConstant",
         "conv.degree", "conv.arcmin", "conv.arcsec", "conv.inch", "conv.foot",
         "conv.yard", "conv.mile", "conv.nauticalMile", "conv.au", "conv.lightYear",
         "conv.parsec", "conv.angstrom", "conv.micron", "conv.gram", "conv.tonne",
         "conv.pound", "conv.ounce", "conv.stone", "conv.shortTon", "conv.longTon",
         "conv.minute", "conv.hour", "conv.day", "conv.week", "conv.year",
         "conv.zeroCelsius", "conv.atm", "conv.bar", "conv.torr", "conv.psi",
         "conv.eV", "conv.calorie", "conv.erg", "conv.btu", "conv.kWh",
         "conv.horsepower",
         "convfn.angleToRadians", "convfn.angleToDegrees", "convfn.celsiusToKelvin",
         "convfn.kelvinToCelsius", "convfn.fahrenheitToKelvin",
         "convfn.kelvinToFahrenheit", "convfn.fahrenheitToCelsius",
         "convfn.celsiusToFahrenheit"]
    for strategy in strategies {
        for tier: CaseTier in [.trivial, .hard, .edge] {
            reg.register(EnvelopeEntry(
                strategy: strategy,
                tier: tier,
                maxAbsError: 1e-9,
                description: "\(strategy) — exact vs scipy.constants/CODATA-2018/math (\(tier.rawValue) cases)"))
        }
    }
    return reg
}
