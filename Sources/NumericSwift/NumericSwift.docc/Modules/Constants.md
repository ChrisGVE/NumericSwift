# Constants

Mathematical and physical constants.

## Overview

The Constants module provides access to mathematical constants (pi, e, etc.) and physical constants from CODATA 2018 recommendations.

## Mathematical Constants

```swift
MathConstants.pi          // 3.14159...
MathConstants.tau         // 2*pi
MathConstants.e           // 2.71828... (Euler's number)
MathConstants.phi         // 1.61803... (golden ratio)
MathConstants.eulerGamma  // 0.57721... (Euler-Mascheroni)
MathConstants.sqrt2       // sqrt(2)
MathConstants.sqrt3       // sqrt(3)
MathConstants.ln2         // ln(2)
MathConstants.ln10        // ln(10)
```

## Physical Constants (SI Units)

```swift
// Fundamental constants
PhysicalConstants.c       // Speed of light (m/s)
PhysicalConstants.h       // Planck constant (J*s)
PhysicalConstants.hbar    // Reduced Planck constant
PhysicalConstants.G       // Gravitational constant
PhysicalConstants.e       // Elementary charge (C)
PhysicalConstants.m_e     // Electron mass (kg)
PhysicalConstants.m_p     // Proton mass (kg)

// Thermodynamic constants
PhysicalConstants.k       // Boltzmann constant (J/K)
PhysicalConstants.N_A     // Avogadro constant (1/mol)
PhysicalConstants.R       // Gas constant (J/(mol*K))

// Electromagnetic constants
PhysicalConstants.epsilon_0  // Vacuum permittivity
PhysicalConstants.mu_0       // Vacuum permeability
```

## Unit Conversions

### Angles

```swift
// Degrees to radians
let rad = AngleConversions.toRadians(90)  // pi/2

// Radians to degrees
let deg = AngleConversions.toDegrees(.pi) // 180
```

### Length

```swift
LengthConversions.inch     // 0.0254 m
LengthConversions.foot     // 0.3048 m
LengthConversions.yard     // 0.9144 m
LengthConversions.mile     // 1609.344 m
LengthConversions.nauticalMile  // 1852 m
```

### Mass

```swift
MassConversions.pound      // 0.45359237 kg
MassConversions.ounce      // 0.028349523125 kg
MassConversions.ton        // 1000 kg
```

### Time

```swift
TimeConversions.minute     // 60 s
TimeConversions.hour       // 3600 s
TimeConversions.day        // 86400 s
TimeConversions.year       // 31557600 s (Julian year)
```

### Temperature

```swift
// Convert between temperature scales
let kelvin = TemperatureConversions.celsiusToKelvin(25)     // 298.15
let celsius = TemperatureConversions.kelvinToCelsius(300)   // 26.85
let fahrenheit = TemperatureConversions.celsiusToFahrenheit(0) // 32
```

## Topics

### Mathematical Constants

- ``MathConstants``

### Physical Constants

- ``PhysicalConstants``

### Unit Conversions

- ``AngleConversions``
- ``LengthConversions``
- ``MassConversions``
- ``TimeConversions``
- ``TemperatureConversions``
