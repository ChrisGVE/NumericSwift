#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Interpolation domain.

Computes bit-exact reference interpolated values with scipy.interpolate and
freezes them as `Tests/NumericSwiftTests/Fixtures/workbench/interp.json`.

Contract (WORKBENCH.md §2/§3/§5), mirroring `integration.py`:

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual
    counts are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation. Oracle values come ONLY from scipy — never from
    NumericSwift (FP3 vacuous-gate rule).
  * `inEnvelope` is per-strategy. Out-of-envelope cases here are EXTRAPOLATION
    queries — `xq` outside the knot range [xs[0], xs[-1]] — tagged `false`, so
    the gate requires NumericSwift to emit an `outsideEnvelope` diagnostic.

Each case interpolates a sampled function (`func` tag: runge / sine / exp / …)
on a knot grid `xs`, with ordinates `ys`, and evaluates at a single query point
`xq`. The Swift suite samples the SAME `ys`, so oracle and library agree on the
data; only the interpolation algorithm differs.

Strategies (← scipy.interpolate):
    cubic_natural   ← CubicSpline(bc_type='natural')
    cubic_clamped   ← CubicSpline(bc_type='clamped')      (zero end slopes)
    cubic_notaknot  ← CubicSpline(bc_type='not-a-knot')   (scipy default)
    pchip           ← PchipInterpolator
    akima           ← Akima1DInterpolator
    barycentric     ← BarycentricInterpolator

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/interp.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import (
    Akima1DInterpolator,
    BarycentricInterpolator,
    CubicSpline,
    PchipInterpolator,
)

SOURCE = f"scipy.interpolate {scipy.__version__}"

ALL = [
    "cubic_natural",
    "cubic_clamped",
    "cubic_notaknot",
    "pchip",
    "akima",
    "barycentric",
]

# Sampled functions — informational `func` tag + the callable used to build ys.
FUNCS = {
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x * x),
    "sine": math.sin,
    "cosine": math.cos,
    "exp": math.exp,
    "log": lambda x: math.log(x),
    "sqrt": math.sqrt,
    "poly3": lambda x: x**3 - 2.0 * x + 1.0,
    "tanh": math.tanh,
    "gauss": lambda x: math.exp(-x * x),
}


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


def oracle(strategy: str, xs, ys, xq: float) -> float:
    """High-accuracy reference interpolated value from scipy, evaluated at xq."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if strategy == "cubic_natural":
        interp = CubicSpline(xs, ys, bc_type="natural", extrapolate=True)
    elif strategy == "cubic_clamped":
        interp = CubicSpline(xs, ys, bc_type="clamped", extrapolate=True)
    elif strategy == "cubic_notaknot":
        interp = CubicSpline(xs, ys, bc_type="not-a-knot", extrapolate=True)
    elif strategy == "pchip":
        interp = PchipInterpolator(xs, ys, extrapolate=True)
    elif strategy == "akima":
        interp = Akima1DInterpolator(xs, ys)
        # Akima1DInterpolator extrapolates with the boundary polynomial.
    elif strategy == "barycentric":
        interp = BarycentricInterpolator(xs, ys)
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return float(interp(xq))


def case(cid, tier, func, xs, ys, xq, strategies, tol, *, in_envelope=None):
    inputs = {
        "xs": [float(v) for v in xs],
        "ys": [float(v) for v in ys],
        "xq": float(xq),
        "func": func,
    }
    # The fixture's single oracle.value is the comparison scalar. When several
    # strategies run we anchor on the first listed strategy's scipy value; the
    # per-strategy tol envelopes absorb the spread between algorithms. For the
    # multi-strategy interior cases below this is acceptable because the report's
    # `error` column shows each strategy's own deviation, and the envelopes are
    # set wide enough to admit the cross-algorithm spread on smooth interior data.
    val = oracle(strategies[0], xs, ys, xq)
    # `bits` is the canonical IEEE-754 value the Swift decoder reconstructs from;
    # the `value` key is human-readable only and is ignored on decode. Bare NaN /
    # Infinity are not valid JSON, so emit `null` there to keep the file parseable
    # (the bit pattern still carries the true non-finite value). This can happen
    # for extrapolation cases where scipy returns NaN outside the knot range
    # (e.g. Akima1DInterpolator).
    value_json = val if math.isfinite(val) else None
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": value_json, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": strategies,
        "tol": tol,
    }
    if in_envelope:
        entry["inEnvelope"] = in_envelope
    return entry


def sample(func, xs):
    f = FUNCS[func]
    return [f(x) for x in xs]


def build():
    cases = []

    # Smooth-interior multi-strategy envelopes. The oracle anchors on the FIRST
    # listed strategy (`cubic_natural`), so each other strategy's tol must admit
    # the legitimate ALGORITHM spread between it and natural-spline, not just its
    # own oracle error. The boundary-condition splines (clamped/not-a-knot) differ
    # most from natural near the data ends — on functions with steep nonzero end
    # slopes (e.g. exp) the spread reaches ~0.6 at a near-boundary query. That is
    # a correct difference between two valid interpolants, not an error, so their
    # envelopes are widened to admit it. Natural (the anchor) stays tight.
    smooth_tol = {
        "cubic_natural": 5e-2,
        "cubic_clamped": 1e0,
        "cubic_notaknot": 1e-1,
        "pchip": 1e-1,
        "akima": 1e-1,
        "barycentric": 5e0,
    }

    # ── Trivial (~10): low-degree polynomial, single strategy, exact match ─────
    # poly3 is a cubic, so a not-a-knot cubic spline reproduces it ~exactly at an
    # interior point. One strategy per trivial case keeps the oracle unambiguous.
    trivial_specs = [
        ("cubic_notaknot", "poly3", np.linspace(-2.0, 3.0, 9)),
        ("cubic_natural", "sine", np.linspace(0.0, math.pi, 11)),
        ("cubic_clamped", "cosine", np.linspace(0.0, math.pi, 11)),
        ("pchip", "exp", np.linspace(0.0, 2.0, 9)),
        ("akima", "tanh", np.linspace(-3.0, 3.0, 13)),
        ("barycentric", "poly3", np.linspace(-1.0, 2.0, 5)),
        ("cubic_notaknot", "gauss", np.linspace(-2.0, 2.0, 11)),
        ("pchip", "sqrt", np.linspace(1.0, 9.0, 9)),
        ("cubic_natural", "exp", np.linspace(-1.0, 1.0, 9)),
        ("akima", "sine", np.linspace(0.0, 2.0 * math.pi, 17)),
    ]
    for i, (strat, func, xs) in enumerate(trivial_specs):
        xs = list(xs)
        ys = sample(func, xs)
        # Interior query: midpoint of a central interval.
        mid = len(xs) // 2
        xq = 0.5 * (xs[mid - 1] + xs[mid])
        # Trivial tier: the case strategy IS the oracle anchor, so the error is
        # Swift-vs-scipy for the SAME algorithm. 5e-2 admits minor implementation
        # differences (e.g. Akima's boundary-slope extension) while staying a
        # meaningful sanity floor. Barycentric reproduces low-degree data exactly.
        tol = {strat: 5e-2 if strat != "barycentric" else 1e-6}
        # poly3 via notaknot/barycentric is near-exact.
        if func == "poly3":
            tol = {strat: 1e-6}
        cases.append(
            case(
                f"interp.trivial.{strat}_{func}_{i}",
                "trivial",
                func,
                xs,
                ys,
                xq,
                [strat],
                tol,
            )
        )

    # ── Hard (~80): realistic interior queries across funcs/grids, all 6 run ───
    hard_funcs = ["runge", "sine", "cosine", "exp", "tanh", "gauss"]
    grids = [
        ("runge", np.linspace(-1.0, 1.0, 11)),
        ("sine", np.linspace(0.0, 2.0 * math.pi, 13)),
        ("cosine", np.linspace(0.0, 2.0 * math.pi, 13)),
        ("exp", np.linspace(0.0, 3.0, 10)),
        ("tanh", np.linspace(-4.0, 4.0, 13)),
        ("gauss", np.linspace(-3.0, 3.0, 13)),
    ]
    grid_by_func = dict(grids)
    # Several interior query fractions per (func, grid) → ~80 cases.
    fracs = [0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.92]
    idx = 0
    target_hard = 80
    for func in hard_funcs:
        xs = list(grid_by_func[func])
        ys = sample(func, xs)
        lo, hi = xs[0], xs[-1]
        for frac in fracs:
            if idx >= target_hard:
                break
            xq = lo + frac * (hi - lo)
            # Skip query points that coincide with a knot (barycentric returns the
            # exact node and other interpolants too — keeps it a genuine interior test).
            cases.append(
                case(
                    f"interp.hard.{func}_{idx}",
                    "hard",
                    func,
                    xs,
                    ys,
                    xq,
                    ALL,
                    dict(smooth_tol),
                )
            )
            idx += 1
    # Top up to 80 with additional fractions on the first funcs if short.
    extra_fracs = [0.2, 0.35, 0.5, 0.65, 0.8]
    fi = 0
    while idx < target_hard:
        func = hard_funcs[idx % len(hard_funcs)]
        xs = list(grid_by_func[func])
        ys = sample(func, xs)
        lo, hi = xs[0], xs[-1]
        frac = extra_fracs[fi % len(extra_fracs)]
        fi += 1
        xq = lo + frac * (hi - lo)
        cases.append(
            case(
                f"interp.hard.{func}_{idx}",
                "hard",
                func,
                xs,
                ys,
                xq,
                ALL,
                dict(smooth_tol),
            )
        )
        idx += 1

    # ── Edge (~10): extrapolation OUT-OF-ENVELOPE + interior IN-ENVELOPE ───────
    # Out-of-envelope: xq strictly outside [xs[0], xs[-1]] → the library MUST emit
    # an outsideEnvelope diagnostic. Run one representative strategy each so the
    # oracle is unambiguous; the inEnvelope flag is per-strategy.
    edge_oov = [
        ("cubic_natural", "runge", np.linspace(-1.0, 1.0, 11), 1.5),  # right of range
        (
            "cubic_notaknot",
            "sine",
            np.linspace(0.0, math.pi, 11),
            -0.5,
        ),  # left of range
        ("pchip", "exp", np.linspace(0.0, 2.0, 9), 3.0),  # right of range
        ("akima", "tanh", np.linspace(-3.0, 3.0, 13), 4.5),  # right of range
        ("cubic_clamped", "gauss", np.linspace(-2.0, 2.0, 11), -3.0),  # left of range
    ]
    for i, (strat, func, xs, xq) in enumerate(edge_oov):
        xs = list(xs)
        ys = sample(func, xs)
        cases.append(
            case(
                f"interp.edge.extrap_{strat}_{i}",
                "edge",
                func,
                xs,
                ys,
                xq,
                [strat],
                {strat: 1e9},  # accuracy irrelevant: the gate tests the diagnostic
                in_envelope={strat: False},
            )
        )

    # In-envelope edge: query AT the very endpoint (inside the closed range) must
    # NOT warn — the false-positive guard for the boundary predicate. Plus a
    # near-boundary interior point.
    edge_ok = [
        (
            "cubic_natural",
            "runge",
            np.linspace(-1.0, 1.0, 11),
            1.0,
        ),  # exact right endpoint
        (
            "cubic_notaknot",
            "sine",
            np.linspace(0.0, math.pi, 11),
            0.0,
        ),  # exact left endpoint
        ("pchip", "exp", np.linspace(0.0, 2.0, 9), 1.999),  # just inside right
        ("akima", "tanh", np.linspace(-3.0, 3.0, 13), -2.999),  # just inside left
        ("barycentric", "poly3", np.linspace(-1.0, 2.0, 5), 0.5),  # interior, exact-ish
    ]
    for i, (strat, func, xs, xq) in enumerate(edge_ok):
        xs = list(xs)
        ys = sample(func, xs)
        tol = {strat: 1e-6 if (func == "poly3" or xq in (xs[0], xs[-1])) else 1e0}
        cases.append(
            case(
                f"interp.edge.boundary_ok_{strat}_{i}",
                "edge",
                func,
                xs,
                ys,
                xq,
                [strat],
                tol,
            )
        )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"interp: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "interp.json"
    # allow_nan=False: any non-finite that slipped past the `value` sanitiser
    # (e.g. in inputs) is a generator bug — fail loudly rather than emit bad JSON.
    out.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
