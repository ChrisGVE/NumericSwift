#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Constants domain.

Computes bit-exact reference values for NumericSwift's mathematical constants,
physical constants, and unit-conversion factors/functions, freezing them as the
JSON fixture `Tests/NumericSwiftTests/Fixtures/workbench/constants.json`.

This is a **single-strategy-per-constant correctness** domain (WORKBENCH.md §4,
"Single-strategy domains"): each constant or conversion IS a strategy id, the
comparison scalar is that constant's value (for a conversion, the converted value
of a representative input), and the oracle is the matching scipy.constants / math
reference. Constants are EXACT values and conversions are exact compositions, so
EVERY case is in-envelope — there are ZERO out-of-envelope cases and the gate is a
pure correctness-vs-reference check (non-vacuous: the oracle is scipy.constants /
CODATA / math, never NumericSwift; FP1 / FP3).

## CODATA vintage (FP1 — cite the source, never loosen silently)

NumericSwift's `PhysicalConstants` declares **CODATA 2018** (Constants.swift §50).
scipy 1.17.1's *current* table is CODATA 2022, but scipy also ships the historical
`scipy.constants._codata._physical_constants_2018` table — the authoritative
CODATA-2018 set, the exact vintage NumericSwift cites. We oracle the physical
constants against THAT 2018 table, so the comparison is vintage-aligned.

Against CODATA 2018, 17 of the 24 physical constants match NumericSwift bit-for-bit
(`tol = 1e-12` relative → absolute). Seven are **derived** constants where
NumericSwift hard-codes the CODATA-2018 *published rounded literal* (≈10 sig figs)
while the 2018 table stores the full-precision computed value:

    constant            NS literal        rel. disagreement   reason
    ------------------  ----------------  ------------------  --------------------------
    hbar                1.054571817e-34   ~6.1e-10            h/(2π), NS rounds to 10 sf
    R                   8.314462618        ~1.8e-11            N_A·k, NS rounds to 10 sf
    sigma               5.670374419e-8     ~3.3e-11            π²k⁴/(60ℏ³c²), 10 sf
    fluxQuantum         2.067833848e-15    ~2.2e-10            h/(2e), 10 sf
    conductanceQuantum  7.748091729e-5     ~1.1e-10            2e²/h, 10 sf
    josephsonConstant   483597.8484e9      ~3.5e-11            2e/h, 10 sf
    vonKlitzingConstant 25812.80745        ~3.6e-10            h/e², 10 sf

For each of those seven the per-case `tol` is set to the **documented agreement**
(the actual absolute disagreement rounded up one significant figure) — NOT loosened
to hide a real mismatch (FP1). The `source` field cites CODATA 2018 and the digit
count. These cases stay in-envelope: NumericSwift's published 10-sf literal is a
correct CODATA-2018 value to its stated precision; the disagreement is the rounding
NumericSwift itself documents, not an error.

Strategy ids ↔ NumericSwift API (Sources/NumericSwift/Constants.swift):

  math.<name>        → MathConstants.<name>        (math / scipy oracle)
  phys.<name>        → PhysicalConstants.<name>    (CODATA-2018 / exact-defined oracle)
  conv.<factor>      → <FooConversions>.<factor>   (scipy.constants factor)
  convfn.<fn>@<x>    → <FooConversions>.<fn>(x)    (exact composition oracle)

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/constants.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import scipy
from scipy import constants as C
from scipy.constants import _codata as _cd

NUMPY_SCIPY = f"scipy {scipy.__version__}"
CODATA2018 = "CODATA 2018 (scipy.constants._codata._physical_constants_2018)"

# The authoritative CODATA-2018 physical-constants table NumericSwift cites.
P2018 = _cd._physical_constants_2018


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def codata2018(name: str) -> float:
    """Full-precision CODATA-2018 value for a physical-constant table key."""
    return float(P2018[name][0])


def case(cid, tier, strategy, value, source, *, tol):
    """Build one fixture case: a single constant/conversion vs its reference.

    `value` is the oracle scalar (from scipy / math / CODATA-2018 — never from
    NumericSwift). The Swift suite recomputes the comparison scalar from the
    matching NumericSwift symbol and checks |swift - oracle| <= tol.

    Every case is in-envelope (constants are exact, conversions are exact
    compositions), so `inEnvelope` is omitted (defaults to true in the decoder).
    The `inputs` bag carries the strategy id under `name` plus, for conversion
    functions, the scalar argument under `x`.
    """
    inputs = {"name": strategy}
    return {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": value, "bits": bits_hex(value)},
        "source": source,
        "strategies": [strategy],
        "tol": {strategy: tol},
    }


def convfn_case(cid, tier, strategy, value, source, *, x, tol):
    """Conversion-function case: applies the function to scalar `x`."""
    entry = case(cid, tier, strategy, value, source, tol=tol)
    entry["inputs"]["x"] = x
    return entry


# Documented per-constant disagreement for the seven derived physical constants
# (FP1: tol = actual absolute disagreement rounded up, never looser). Computed
# below from the NS literal vs the CODATA-2018 full-precision value.
NS_DERIVED_LITERALS = {
    "hbar": 1.054571817e-34,
    "R": 8.314462618,
    "sigma": 5.670374419e-8,
    "fluxQuantum": 2.067833848e-15,
    "conductanceQuantum": 7.748091729e-5,
    "josephsonConstant": 483597.8484e9,
    "vonKlitzingConstant": 25812.80745,
}


def documented_tol(ns_literal: float, oracle: float) -> float:
    """Absolute tolerance = the documented disagreement, rounded UP one sig-fig.

    This is the published-literal-vs-full-precision agreement, NOT a loosened
    bound to hide a mismatch (FP1). E.g. a disagreement of 6.5e-44 → 1e-43.
    """
    diff = abs(ns_literal - oracle)
    if diff == 0.0:
        return 1e-300
    exp = math.floor(math.log10(diff))
    return 10.0 ** (exp + 1)


# ── Mathematical constants ↔ MathConstants ──────────────────────────────────
# Oracle from math / scipy. `inf` / `nan` are exact IEEE-754 sentinels.
MATH_CONSTS = [
    (
        "math.pi",
        math.pi,
        f"math.pi (Python {sys.version_info.major}.{sys.version_info.minor})",
    ),
    ("math.tau", math.tau, "math.tau (2π)"),
    ("math.e", math.e, "math.e"),
    (
        "math.phi",
        (1 + math.sqrt(5)) / 2,
        "golden ratio (1+√5)/2 == scipy.constants.golden",
    ),
    (
        "math.eulerGamma",
        0.577215664901532860606512090082402431042,
        "Euler-Mascheroni γ (DLMF / mpmath reference literal)",
    ),
    ("math.sqrt2", math.sqrt(2.0), "math.sqrt(2)"),
    ("math.sqrt3", math.sqrt(3.0), "math.sqrt(3)"),
    ("math.ln2", math.log(2.0), "math.log(2)"),
    ("math.ln10", math.log(10.0), "math.log(10)"),
]

# ── Physical constants ↔ PhysicalConstants (CODATA 2018) ────────────────────
# (strategy id, CODATA-2018 table key, NS literal, is_derived)
PHYS_CONSTS = [
    ("phys.c", "speed of light in vacuum", 299792458.0, False),
    ("phys.h", "Planck constant", 6.62607015e-34, False),
    ("phys.hbar", "reduced Planck constant", 1.054571817e-34, True),
    ("phys.G", "Newtonian constant of gravitation", 6.67430e-11, False),
    ("phys.g", "standard acceleration of gravity", 9.80665, False),
    ("phys.elementaryCharge", "elementary charge", 1.602176634e-19, False),
    ("phys.electronMass", "electron mass", 9.1093837015e-31, False),
    ("phys.protonMass", "proton mass", 1.67262192369e-27, False),
    ("phys.neutronMass", "neutron mass", 1.67492749804e-27, False),
    ("phys.atomicMass", "atomic mass constant", 1.66053906660e-27, False),
    ("phys.k", "Boltzmann constant", 1.380649e-23, False),
    ("phys.N_A", "Avogadro constant", 6.02214076e23, False),
    ("phys.R", "molar gas constant", 8.314462618, True),
    ("phys.epsilon0", "vacuum electric permittivity", 8.8541878128e-12, False),
    ("phys.mu0", "vacuum mag. permeability", 1.25663706212e-6, False),
    ("phys.sigma", "Stefan-Boltzmann constant", 5.670374419e-8, True),
    ("phys.alpha", "fine-structure constant", 7.2973525693e-3, False),
    ("phys.Rydberg", "Rydberg constant", 10973731.568160, False),
    ("phys.bohrRadius", "Bohr radius", 5.29177210903e-11, False),
    ("phys.electronRadius", "classical electron radius", 2.8179403262e-15, False),
    ("phys.comptonWavelength", "Compton wavelength", 2.42631023867e-12, False),
    ("phys.fluxQuantum", "mag. flux quantum", 2.067833848e-15, True),
    ("phys.conductanceQuantum", "conductance quantum", 7.748091729e-5, True),
    ("phys.josephsonConstant", "Josephson constant", 483597.8484e9, True),
    ("phys.vonKlitzingConstant", "von Klitzing constant", 25812.80745, True),
]

# ── Unit-conversion factors ↔ <Foo>Conversions ─────────────────────────────
# Oracle from scipy.constants. NS literals match scipy to ≤ a few ULP (1e-15
# relative) → absolute tol scaled to the magnitude.
CONV_FACTORS = [
    # Angle (rad per unit)
    ("conv.degree", C.degree, "scipy.constants.degree"),
    ("conv.arcmin", C.arcmin, "scipy.constants.arcmin"),
    ("conv.arcsec", C.arcsec, "scipy.constants.arcsec"),
    # Length (m per unit)
    ("conv.inch", C.inch, "scipy.constants.inch"),
    ("conv.foot", C.foot, "scipy.constants.foot"),
    ("conv.yard", C.yard, "scipy.constants.yard"),
    ("conv.mile", C.mile, "scipy.constants.mile"),
    ("conv.nauticalMile", C.nautical_mile, "scipy.constants.nautical_mile"),
    ("conv.au", C.au, "scipy.constants.au"),
    ("conv.lightYear", C.light_year, "scipy.constants.light_year"),
    ("conv.parsec", C.parsec, "scipy.constants.parsec"),
    ("conv.angstrom", C.angstrom, "scipy.constants.angstrom"),
    ("conv.micron", C.micron, "scipy.constants.micron"),
    # Mass (kg per unit)
    ("conv.gram", C.gram, "scipy.constants.gram"),
    ("conv.tonne", C.metric_ton, "scipy.constants.metric_ton"),
    ("conv.pound", C.lb, "scipy.constants.lb"),
    ("conv.ounce", C.oz, "scipy.constants.oz"),
    ("conv.stone", C.stone, "scipy.constants.stone"),
    ("conv.shortTon", C.short_ton, "scipy.constants.short_ton"),
    ("conv.longTon", C.long_ton, "scipy.constants.long_ton"),
    # Time (s per unit)
    ("conv.minute", C.minute, "scipy.constants.minute"),
    ("conv.hour", C.hour, "scipy.constants.hour"),
    ("conv.day", C.day, "scipy.constants.day"),
    ("conv.week", C.week, "scipy.constants.week"),
    ("conv.year", C.Julian_year, "scipy.constants.Julian_year"),
    # Temperature offset
    ("conv.zeroCelsius", C.zero_Celsius, "scipy.constants.zero_Celsius"),
    # Pressure (Pa per unit)
    ("conv.atm", C.atm, "scipy.constants.atm"),
    ("conv.bar", C.bar, "scipy.constants.bar"),
    ("conv.torr", C.torr, "scipy.constants.torr"),
    ("conv.psi", C.psi, "scipy.constants.psi"),
    # Energy (J per unit)
    ("conv.eV", C.eV, "scipy.constants.eV"),
    ("conv.calorie", C.calorie, "scipy.constants.calorie"),
    ("conv.erg", C.erg, "scipy.constants.erg"),
    ("conv.btu", C.Btu, "scipy.constants.Btu"),
    ("conv.kWh", 3.6e6, "1 kWh = 3.6e6 J (exact, SI)"),
    # Power (W per unit)
    ("conv.horsepower", C.hp, "scipy.constants.hp"),
]


# Per-factor absolute tol = 1e-12 * |value| (≥ a few-ULP slack at any magnitude),
# floored at a tiny absolute so values near zero still get a sane bound.
def factor_tol(value: float) -> float:
    return max(abs(value) * 1e-12, 1e-300)


# ── Conversion functions ↔ <Foo>Conversions.<fn>(x) ─────────────────────────
# Exact compositions; oracle computed directly. Strategy id encodes the call as
# convfn.<fn>; the `x` input carries the argument.
def build_convfn_cases():
    out = []
    # AngleConversions.toRadians / toDegrees
    for x in (0.0, 90.0, 180.0, 360.0, 45.0):
        v = x * (math.pi / 180.0)
        out.append(
            (
                "convfn.angleToRadians",
                x,
                v,
                "degrees·(π/180) — exact composition vs scipy.constants.degree",
            )
        )
    for x in (0.0, math.pi / 2, math.pi, 2 * math.pi):
        v = x / (math.pi / 180.0)
        out.append(
            ("convfn.angleToDegrees", x, v, "radians/(π/180) — exact composition")
        )
    # TemperatureConversions
    for x in (0.0, 100.0, -40.0, 37.0):
        out.append(
            ("convfn.celsiusToKelvin", x, x + 273.15, "°C + 273.15 — exact composition")
        )
    for x in (273.15, 373.15, 0.0):
        out.append(
            ("convfn.kelvinToCelsius", x, x - 273.15, "K − 273.15 — exact composition")
        )
    for x in (32.0, 212.0, -40.0):
        out.append(
            (
                "convfn.fahrenheitToKelvin",
                x,
                (x + 459.67) * 5.0 / 9.0,
                "(°F + 459.67)·5/9 — exact composition",
            )
        )
    for x in (273.15, 373.15):
        out.append(
            (
                "convfn.kelvinToFahrenheit",
                x,
                x * 9.0 / 5.0 - 459.67,
                "K·9/5 − 459.67 — exact composition",
            )
        )
    for x in (32.0, 212.0, 98.6):
        out.append(
            (
                "convfn.fahrenheitToCelsius",
                x,
                (x - 32.0) * 5.0 / 9.0,
                "(°F − 32)·5/9 — exact composition",
            )
        )
    for x in (0.0, 100.0, 37.0):
        out.append(
            (
                "convfn.celsiusToFahrenheit",
                x,
                x * 9.0 / 5.0 + 32.0,
                "°C·9/5 + 32 — exact composition",
            )
        )
    return out


CONVFN = build_convfn_cases()


def build():
    cases = []

    # ── Trivial (~10): the core, hand-checkable constants ─────────────────────
    # Math constants + exact-defined SI physical constants (post-2019 SI, exact).
    trivial = [
        ("math.pi", math.pi, "math.pi"),
        ("math.e", math.e, "math.e"),
        ("math.tau", math.tau, "math.tau"),
        ("math.phi", (1 + math.sqrt(5)) / 2, "golden ratio (1+√5)/2"),
        ("math.sqrt2", math.sqrt(2.0), "math.sqrt(2)"),
        ("phys.c", 299792458.0, "CODATA 2018 (exact, defined SI)"),
        ("phys.h", 6.62607015e-34, "CODATA 2018 (exact, defined SI)"),
        ("phys.k", 1.380649e-23, "CODATA 2018 (exact, defined SI)"),
        ("phys.N_A", 6.02214076e23, "CODATA 2018 (exact, defined SI)"),
        ("phys.elementaryCharge", 1.602176634e-19, "CODATA 2018 (exact, defined SI)"),
    ]
    for strat, val, src in trivial:
        tier_id = strat.split(".", 1)[1]
        cases.append(
            case(f"constants.trivial.{tier_id}", "trivial", strat, val, src, tol=1e-12)
        )

    # ── Hard (~80): every constant + every conversion factor + functions ──────
    # All math constants (9).
    for strat, val, src in MATH_CONSTS:
        if strat in ("math.inf", "math.nan"):
            continue
        tier_id = strat.split(".", 1)[1]
        cases.append(
            case(f"constants.hard.{tier_id}", "hard", strat, val, src, tol=1e-12)
        )

    # All NON-derived physical constants (17) — bit-for-bit vs CODATA 2018.
    for strat, key, _ns_literal, derived in PHYS_CONSTS:
        if derived:
            continue
        tier_id = strat.split(".", 1)[1]
        val = codata2018(key)
        tol = max(abs(val) * 1e-12, 1e-300)
        cases.append(
            case(f"constants.hard.{tier_id}", "hard", strat, val, CODATA2018, tol=tol)
        )

    # All conversion factors (36).
    for strat, val, src in CONV_FACTORS:
        tier_id = strat.split(".", 1)[1]
        cases.append(
            case(
                f"constants.hard.{tier_id}",
                "hard",
                strat,
                float(val),
                f"{src} ({NUMPY_SCIPY})",
                tol=factor_tol(val),
            )
        )

    # A representative subset of conversion functions (keeps hard ≈ 80).
    for i, (strat, x, val, src) in enumerate(CONVFN):
        fn = strat.split(".", 1)[1]
        cases.append(
            convfn_case(
                f"constants.hard.{fn}_{i}",
                "hard",
                strat,
                val,
                src,
                x=x,
                tol=max(abs(val) * 1e-12, 1e-12),
            )
        )

    # Trim hard to ~80 (proportion enforcement, ±a few).
    hard_only = [c for c in cases if c["tier"] == "hard"]
    if len(hard_only) > 80:
        keep = hard_only[:80]
        cases = [c for c in cases if c["tier"] != "hard"] + keep

    # ── Edge (~10): the 7 derived constants (documented agreement) + extremes ─
    # Derived physical constants: NS rounds the CODATA-2018 derived literal to
    # ~10 sf; the per-case tol is the DOCUMENTED disagreement (FP1, not loosened
    # to hide). Still in-envelope: a correct CODATA-2018 value to stated precision.
    for strat, key, ns_literal, derived in PHYS_CONSTS:
        if not derived:
            continue
        tier_id = strat.split(".", 1)[1]
        val = codata2018(key)
        tol = documented_tol(ns_literal, val)
        rel = abs(ns_literal - val) / abs(val)
        src = (
            f"{CODATA2018}; NS uses the published ~10-sig-fig literal "
            f"(rel. disagreement {rel:.2e}; tol = documented agreement, FP1)"
        )
        cases.append(
            case(f"constants.edge.{tier_id}", "edge", strat, val, src, tol=tol)
        )

    # Extreme conversion-function inputs (still exact compositions, in-envelope).
    extremes = [
        (
            "convfn.celsiusToKelvin",
            -273.15,
            0.0,
            "absolute zero: −273.15 °C → 0 K (exact)",
        ),
        (
            "convfn.fahrenheitToCelsius",
            -459.67,
            (-459.67 - 32.0) * 5.0 / 9.0,
            "absolute zero in °F → °C (exact composition)",
        ),
        (
            "convfn.angleToRadians",
            1e6,
            1e6 * (math.pi / 180.0),
            "large angle 1e6° → rad (exact composition)",
        ),
    ]
    for i, (strat, x, val, src) in enumerate(extremes):
        fn = strat.split(".", 1)[1]
        cases.append(
            convfn_case(
                f"constants.edge.{fn}_ext_{i}",
                "edge",
                strat,
                val,
                src,
                x=x,
                tol=max(abs(val) * 1e-12, 1e-12),
            )
        )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"constants: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']} "
        f"(0 out-of-envelope)",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "constants.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
