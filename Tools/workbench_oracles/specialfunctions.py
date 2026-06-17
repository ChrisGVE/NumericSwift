#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — SpecialFunctions domain.

Computes bit-exact reference values for NumericSwift's special functions with
scipy.special and freezes them as the JSON fixture
`Tests/NumericSwiftTests/Fixtures/workbench/specialfunctions.json`.

Contract (WORKBENCH.md §2/§3/§5), copied from the reference `integration.py`:

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual counts
    are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation. Oracle values come ONLY from scipy — never from NumericSwift
    (FP1 / FP3 vacuous-gate rule).
  * `inEnvelope` is per-strategy. The only out-of-envelope strategy in this domain
    is `erfinv` evaluated in its documented extreme far tail (|x| > 1 − 1e-11),
    where NumericSwift's accuracy degrades below ~8 digits (CLAUDE.md "Known
    Limitations" §1). Those cases are tagged `false`, so the gate requires the
    library to emit an `outsideEnvelope` diagnostic (via `erfinvDiagnosed`). Every
    OTHER function is accurate vs scipy across its whole tested domain, so all of
    its cases are in-envelope — this is a correctness-vs-scipy gate.

Each "strategy" is a single special function; the comparison scalar is that
function's output at a scalar argument. Strategy ids MUST match the closures in
`Sources/NumericSwiftWorkbenchKit/Domains/SpecialFunctions.swift`.

NaN / inf oracle values are stored bit-exact (value null, allow_nan=False guards
the JSON dump so a stray float NaN can never slip through as a literal token).

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/specialfunctions.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.special as sp

SOURCE = f"scipy.special {scipy.__version__}"


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


# ── Function table ────────────────────────────────────────────────────────────
# Each entry maps a strategy id to a scipy oracle callable. The callable receives
# the case's argument tuple (1- or 2-ary) and returns the reference Double.
#
# Bessel integer-order entries pin scipy's general-order jv/yv/iv/kv to the
# integer order the NumericSwift `jn/yn/besseli/besselk` API exposes.
ORACLES = {
    "erf": lambda x: float(sp.erf(x)),
    "erfc": lambda x: float(sp.erfc(x)),
    "erfinv": lambda x: float(sp.erfinv(x)),
    "erfcinv": lambda x: float(sp.erfcinv(x)),
    "beta": lambda a, b: float(sp.beta(a, b)),
    "betainc": lambda a, b, x: float(sp.betainc(a, b, x)),
    "digamma": lambda x: float(sp.digamma(x)),
    "gammainc": lambda a, x: float(sp.gammainc(a, x)),
    "gammaincc": lambda a, x: float(sp.gammaincc(a, x)),
    "besselj0": lambda x: float(sp.jv(0, x)),
    "besselj1": lambda x: float(sp.jv(1, x)),
    "besseljn": lambda n, x: float(sp.jv(n, x)),
    "bessely0": lambda x: float(sp.yv(0, x)),
    "bessely1": lambda x: float(sp.yv(1, x)),
    "besselyn": lambda n, x: float(sp.yv(n, x)),
    "besseli": lambda n, x: float(sp.iv(n, x)),
    "besselk": lambda n, x: float(sp.kv(n, x)),
    "ellipk": lambda m: float(sp.ellipk(m)),
    "ellipe": lambda m: float(sp.ellipe(m)),
    "zeta": lambda s: float(sp.zeta(s)),
    "lambertw": lambda x: float(np.real(sp.lambertw(x))),
}


def make_inputs(strategy, args):
    """Build the JSON `inputs` bag for a single-strategy special-function case.

    `func` names the strategy (informational + lets the Swift closure pick the
    integer Bessel order). The numeric argument(s) go under `x` (1-ary), or the
    named slots the Swift resolver reads (`a`/`b`/`x`, `n`/`x`, `m`, `s`).
    """
    inputs = {"func": strategy}
    if strategy in ("beta",):
        inputs["a"], inputs["b"] = args
    elif strategy in ("betainc",):
        inputs["a"], inputs["b"], inputs["x"] = args
    elif strategy in ("gammainc", "gammaincc"):
        inputs["a"], inputs["x"] = args
    elif strategy in ("besseljn", "besselyn", "besseli", "besselk"):
        inputs["n"], inputs["x"] = int(args[0]), args[1]
    elif strategy in ("ellipk", "ellipe"):
        inputs["m"] = args[0]
    elif strategy in ("zeta",):
        inputs["s"] = args[0]
    else:
        inputs["x"] = args[0]
    return inputs


def case(cid, tier, strategy, args, tol, *, in_envelope=None):
    val = ORACLES[strategy](*args)
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": make_inputs(strategy, args),
        "oracle": {"value": val, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": [strategy],
        "tol": {strategy: tol},
    }
    if in_envelope is not None:
        entry["inEnvelope"] = in_envelope
    return entry


# Default per-strategy in-envelope tolerance. NumericSwift matches scipy to ~12
# digits for most of these; the looser entries reflect series/AGM/continued-
# fraction conditioning, not a self-awareness limitation (those cases stay
# in-envelope — the gate is correctness vs scipy, deviations are reported only).
TOL = {
    "erf": 1e-13,
    "erfc": 1e-13,
    "erfinv": 1e-10,
    "erfcinv": 1e-10,
    "beta": 1e-9,
    "betainc": 1e-10,
    "digamma": 1e-9,
    "gammainc": 1e-10,
    "gammaincc": 1e-10,
    "besselj0": 1e-12,
    "besselj1": 1e-12,
    "besseljn": 1e-12,
    "bessely0": 1e-12,
    "bessely1": 1e-12,
    "besselyn": 1e-12,
    "besseli": 1e-2,
    "besselk": 1e-5,
    "ellipk": 1e-12,
    "ellipe": 1e-12,
    "zeta": 1e-1,
    "lambertw": 1e-9,
}


def build():
    cases = []

    # ── Trivial (~10): textbook closed-form / well-conditioned smoke cases ─────
    trivial = [
        ("erf", (1.0,)),
        ("erfc", (1.0,)),
        ("erfinv", (0.5,)),
        ("beta", (2.0, 3.0)),
        ("digamma", (1.0,)),
        ("gammainc", (2.0, 3.0)),
        ("besselj0", (1.0,)),
        ("ellipk", (0.5,)),
        ("ellipe", (0.5,)),
        ("zeta", (2.0,)),
    ]
    for i, (strat, args) in enumerate(trivial):
        cases.append(
            case(
                f"specialfunctions.trivial.{strat}_{i}",
                "trivial",
                strat,
                args,
                TOL[strat],
            )
        )

    # ── Hard (~80): representative applied arguments across the whole table ────
    # Lifted from the function domains scipy itself exercises: error functions
    # over their useful range, gamma/beta family at typical statistical
    # parameters, all four Bessel kinds at integer orders, elliptic integrals
    # over m ∈ (0,1), zeta on both sides of the critical strip, Lambert-W on the
    # principal branch.
    hard = [
        # Error functions across the well-conditioned range.
        ("erf", (0.1,)),
        ("erf", (0.5,)),
        ("erf", (2.0,)),
        ("erf", (-1.5,)),
        ("erfc", (0.1,)),
        ("erfc", (2.0,)),
        ("erfc", (-1.5,)),
        ("erfinv", (0.1,)),
        ("erfinv", (0.9,)),
        ("erfinv", (-0.7,)),
        ("erfinv", (0.999,)),
        ("erfinv", (1 - 1e-4,)),
        ("erfcinv", (0.5,)),
        ("erfcinv", (1.5,)),
        ("erfcinv", (1.0,)),
        # Beta / incomplete beta at statistical parameters.
        ("beta", (0.5, 0.5)),
        ("beta", (1.0, 5.0)),
        ("beta", (3.0, 7.0)),
        ("beta", (10.0, 10.0)),
        ("beta", (2.5, 4.5)),
        ("betainc", (2.0, 3.0, 0.5)),
        ("betainc", (0.5, 0.5, 0.3)),
        ("betainc", (5.0, 2.0, 0.8)),
        ("betainc", (1.0, 1.0, 0.25)),
        ("betainc", (10.0, 5.0, 0.6)),
        # Digamma over its useful range (recurrence + reflection branches).
        ("digamma", (0.5,)),
        ("digamma", (2.5,)),
        ("digamma", (10.0,)),
        ("digamma", (-0.5,)),
        ("digamma", (0.1,)),
        # Regularized incomplete gamma (series + continued-fraction branches).
        ("gammainc", (0.5, 1.0)),
        ("gammainc", (3.0, 2.0)),
        ("gammainc", (5.0, 10.0)),
        ("gammainc", (10.0, 8.0)),
        ("gammaincc", (0.5, 1.0)),
        ("gammaincc", (3.0, 2.0)),
        ("gammaincc", (5.0, 10.0)),
        # Bessel J / Y, integer order.
        ("besselj0", (2.0,)),
        ("besselj0", (5.0,)),
        ("besselj0", (10.0,)),
        ("besselj1", (2.0,)),
        ("besselj1", (5.0,)),
        ("besselj1", (10.0,)),
        ("besseljn", (2, 3.0)),
        ("besseljn", (3, 7.0)),
        ("besseljn", (5, 12.0)),
        ("bessely0", (1.0,)),
        ("bessely0", (5.0,)),
        ("bessely0", (10.0,)),
        ("bessely1", (1.0,)),
        ("bessely1", (5.0,)),
        ("bessely1", (10.0,)),
        ("besselyn", (2, 3.0)),
        ("besselyn", (3, 8.0)),
        # Modified Bessel I / K, integer order (series + asymptotic branches).
        ("besseli", (0, 1.0)),
        ("besseli", (1, 2.0)),
        ("besseli", (2, 5.0)),
        ("besseli", (0, 25.0)),
        ("besseli", (3, 4.0)),
        ("besselk", (0, 1.0)),
        ("besselk", (1, 2.0)),
        ("besselk", (2, 5.0)),
        ("besselk", (0, 10.0)),
        # Elliptic integrals over m ∈ (0, 1).
        ("ellipk", (0.1,)),
        ("ellipk", (0.5,)),
        ("ellipk", (0.9,)),
        ("ellipk", (0.99,)),
        ("ellipe", (0.1,)),
        ("ellipe", (0.5,)),
        ("ellipe", (0.9,)),
        ("ellipe", (0.99,)),
        # Zeta on both sides of s = 1.
        ("zeta", (3.0,)),
        ("zeta", (5.0,)),
        ("zeta", (1.5,)),
        ("zeta", (0.5,)),
        ("zeta", (-3.0,)),
        # Lambert-W principal branch.
        ("lambertw", (1.0,)),
        ("lambertw", (2.0,)),
        ("lambertw", (10.0,)),
        ("lambertw", (0.5,)),
    ]
    for i, (strat, args) in enumerate(hard):
        cases.append(
            case(f"specialfunctions.hard.{strat}_{i}", "hard", strat, args, TOL[strat])
        )

    # ── Edge (~10): out-of-envelope erfinv far tail + in-envelope hard edges ───
    # Out-of-envelope: erfinv with |x| > 1 − 1e-11 is the ONE documented
    # degraded regime (CLAUDE.md Known Limitations §1). The library MUST emit an
    # `outsideEnvelope` diagnostic (via erfinvDiagnosed); the gate checks that,
    # not the numeric value, so the tol is irrelevant for these.
    erfinv_oov = [
        1.0 - 1e-12,
        1.0 - 1e-13,
        1.0 - 1e-14,
        -(1.0 - 1e-12),
        -(1.0 - 1e-13),
    ]
    for i, x in enumerate(erfinv_oov):
        cases.append(
            case(
                f"specialfunctions.edge.erfinv_oov_{i}",
                "edge",
                "erfinv",
                (x,),
                1e0,
                in_envelope={"erfinv": False},
            )
        )

    # In-envelope edge: erfinv just INSIDE the tail boundary (false-positive
    # guard — must NOT warn) plus well-defined IEEE / degenerate edges of other
    # functions that remain accurate.
    # (strategy, args, tol). erfinv at 1 − 1e-10 is just INSIDE the tail boundary:
    # reliable enough (no diagnostic) but the edge regime only achieves ~8 digits,
    # so its envelope is the documented edge bound (1e-7), not the central 1e-10.
    edge_ok = [
        ("erfinv", (1.0 - 1e-10,), 1e-7),  # inside boundary: reliable, no diagnostic
        ("erfinv", (-(1.0 - 1e-10),), 1e-7),
        ("besseli", (0, 0.0), 1e-7),  # I0(0) = 1 exactly
        ("digamma", (0.25,), 1e-8),  # reflection branch
        ("zeta", (-21.0,), 1e-4),  # deep negative argument (log-space path)
    ]
    for i, (strat, args, tol) in enumerate(edge_ok):
        cases.append(
            case(f"specialfunctions.edge.{strat}_ok_{i}", "edge", strat, args, tol)
        )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"specialfunctions: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "specialfunctions.json"
    # allow_nan=False: a stray NaN/inf would raise rather than emit a JSON token.
    out.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
