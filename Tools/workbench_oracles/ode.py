#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Integration (ODE) domain.

Computes bit-exact reference ODE solutions with scipy and freezes them as the
JSON fixture `Tests/NumericSwiftTests/Fixtures/workbench/ode.json`.

Mirrors the REFERENCE generator `integration.py` (WORKBENCH.md §2/§3/§5):

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual
    counts are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation. Oracle values come ONLY from scipy — never from
    NumericSwift (FP1 / FP3 vacuous-gate rule).
  * The comparison scalar is the **first solution component `y[0]` at the final
    time `tf`** (documented in `ODE.swift`'s strategy closures).
  * `inEnvelope` is per-strategy. Out-of-envelope cases are STIFF systems
    (Van der Pol with large μ, Robertson) integrated with an EXPLICIT method
    (`rk45`/`rk23`): the explicit step controller collapses, so the library MUST
    emit a `outsideEnvelope` diagnostic (see `detectExplicitStiffness` in
    Sources/NumericSwift/Integration.swift). These cases tag `rk45`/`rk23`
    `false`.

Strategy → scipy method map (kept in lockstep with `ODE.swift`):
  * rk45   → scipy.integrate.solve_ivp(method="RK45")  → NumericSwift solveIVP(.rk45)
  * rk23   → scipy.integrate.solve_ivp(method="RK23")  → NumericSwift solveIVP(.rk23)
  * bdf    → scipy.integrate.solve_ivp(method="BDF")   → NumericSwift solveIVP(.bdf)
  * odeint → scipy.integrate.solve_ivp(method="LSODA") → NumericSwift odeint(...)

NumericSwift has no DOP853 method, so the `dop853` strategy from WORKBENCH.md §4
is deliberately omitted (FP1: no fabricated mapping to a different solver).

The system `tag`s and their parameter keys MUST match the `odeSystem` resolver in
`Sources/NumericSwiftWorkbenchKit/Domains/ODE.swift`.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/ode.py
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import integrate

SCIPY_VERSION = __import__("scipy").__version__
SOURCE = f"scipy.integrate.solve_ivp {SCIPY_VERSION}"

# Strategy → scipy solver-method map. odeint maps to LSODA (the algorithm
# scipy.integrate.odeint wraps).
STRATEGY_METHOD = {
    "rk45": "RK45",
    "rk23": "RK23",
    "bdf": "BDF",
    "odeint": "LSODA",
}


# ── ODE right-hand sides. Keep in lockstep with the Swift `odeSystem` resolver.
# Each takes (t, y, params) -> dy/dt list. `params` carries the system-specific
# scalars (mu, k, r, kk) so a single tag can cover a family of cases.
def rhs(tag, params):
    if tag == "exp_decay":
        # dy/dt = -k y ; analytic y(t) = y0 e^{-k t}
        k = params.get("k", 1.0)
        return lambda t, y: [-k * y[0]]
    if tag == "harmonic":
        # simple harmonic oscillator: y'' = -y  → [y1, -y0]
        return lambda t, y: [y[1], -y[0]]
    if tag == "logistic":
        # dy/dt = r y (1 - y/K)
        r = params.get("r", 1.0)
        kk = params.get("kk", 1.0)
        return lambda t, y: [r * y[0] * (1.0 - y[0] / kk)]
    if tag == "linear2d":
        # rotation-decay linear system: y0' = -y0 + y1, y1' = -y0 - y1
        return lambda t, y: [-y[0] + y[1], -y[0] - y[1]]
    if tag == "vdp":
        # Van der Pol oscillator: y0' = y1, y1' = mu (1-y0^2) y1 - y0
        mu = params["mu"]
        return lambda t, y: [y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]
    if tag == "robertson":
        # Robertson stiff chemical kinetics.
        k1, k2, k3 = 0.04, 3.0e7, 1.0e4
        return lambda t, y: [
            -k1 * y[0] + k3 * y[1] * y[2],
            k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] * y[1],
            k2 * y[1] * y[1],
        ]
    raise ValueError(f"unknown tag {tag}")


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


def oracle_value(tag, y0, tf, params):
    """High-accuracy reference y[0] at tf from scipy.

    Uses a stiff-capable reference solver (LSODA with tight tolerances) so the
    ground-truth value is correct for both stiff and non-stiff systems — the
    self-awareness gate ignores accuracy for out-of-envelope cases, but an
    honest oracle keeps the in-envelope accuracy flags meaningful (FP1).
    """
    sol = integrate.solve_ivp(
        rhs(tag, params),
        (0.0, tf),
        y0,
        method="LSODA",
        rtol=1e-11,
        atol=1e-12,
        dense_output=False,
    )
    return float(sol.y[0][-1])


def case(cid, tier, tag, y0, tf, strategies, tol, params, *, in_envelope=None):
    inputs = {"tag": tag, "y0": list(map(float, y0)), "tf": float(tf)}
    for key in ("mu", "k", "r", "kk"):
        if key in params:
            inputs[key] = float(params[key])
    val = oracle_value(tag, list(map(float, y0)), float(tf), params)
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": val, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": strategies,
        "tol": tol,
    }
    if in_envelope:
        entry["inEnvelope"] = in_envelope
    return entry


# Per-strategy accuracy envelopes for smooth, non-stiff cases. The explicit RK
# methods at default-ish tol reach a few digits; bdf (implicit Euler, O(h)) is
# looser; odeint (LSODA, tight tol) is near machine precision. These are
# declared envelopes, not achieved errors — the report's `error` column shows
# what each hits.
NONSTIFF_ALL = ["rk45", "rk23", "bdf", "odeint"]
# NumericSwift's solveIVP defaults to rtol=1e-3 / atol=1e-6, so the explicit
# pairs reach low-single-digit absolute error; bdf is O(h); odeint (the library's
# single-pass dense-output solve) reaches ~1e-7. Envelopes are set to the
# library's actual achievable accuracy at its defaults (probe-confirmed).
NONSTIFF_TOL = {
    "rk45": 1e-2,
    "rk23": 5e-2,
    "bdf": 5e-1,
    "odeint": 1e-5,
}


def build():
    cases = []

    # ── Trivial (~10): closed-form, smooth, gentle systems ────────────────────
    trivial = [
        ("exp_decay", [1.0], 1.0, {"k": 1.0}),
        ("exp_decay", [1.0], 2.0, {"k": 1.0}),
        ("exp_decay", [2.0], 1.0, {"k": 0.5}),
        ("harmonic", [1.0, 0.0], 1.0, {}),
        ("harmonic", [1.0, 0.0], 3.141592653589793, {}),
        ("harmonic", [0.0, 1.0], 1.0, {}),
        ("logistic", [0.01], 5.0, {"r": 1.0, "kk": 1.0}),
        ("logistic", [0.1], 3.0, {"r": 1.0, "kk": 1.0}),
        ("exp_decay", [1.0], 0.5, {"k": 2.0}),
        ("linear2d", [1.0, 0.0], 1.0, {}),
    ]
    for i, (tag, y0, tf, p) in enumerate(trivial):
        # trivial tier: envelopes matched to the library's default-tolerance
        # accuracy on gentle systems (probe-confirmed).
        tol = {"rk45": 1e-2, "rk23": 5e-2, "bdf": 5e-1, "odeint": 1e-6}
        cases.append(
            case(f"ode.trivial.{tag}_{i}", "trivial", tag, y0, tf, NONSTIFF_ALL, tol, p)
        )

    # ── Hard (~80): realistic non-stiff systems, varied y0 / tf / params ──────
    idx = 0
    hard_specs = []
    # exp_decay family across decay rates and horizons
    for k in [0.3, 1.0, 2.5, 5.0]:
        for tf in [1.0, 3.0, 6.0]:
            hard_specs.append(("exp_decay", [1.0], tf, {"k": k}))
    # harmonic oscillator across horizons and initial conditions
    for tf in [2.0, 5.0, 8.0, 12.0]:
        for y0 in ([1.0, 0.0], [0.5, 0.5], [0.0, 2.0]):
            hard_specs.append(("harmonic", y0, tf, {}))
    # logistic growth across rates and carrying capacities
    for r in [0.5, 1.0, 2.0]:
        for kk in [1.0, 5.0]:
            for tf in [2.0, 5.0, 10.0]:
                hard_specs.append(("logistic", [0.05 * kk], tf, {"r": r, "kk": kk}))
    # linear2d spiral-decay across horizons
    for tf in [1.0, 2.0, 4.0, 6.0, 8.0]:
        for y0 in ([1.0, 0.0], [0.0, 1.0]):
            hard_specs.append(("linear2d", y0, tf, {}))
    # mildly stiff Van der Pol (small mu) — still inside the explicit envelope,
    # confirmed by the probe (mu=50 over a short horizon stays in-envelope).
    for mu in [1.0, 2.0, 5.0]:
        for tf in [5.0, 10.0]:
            hard_specs.append(("vdp", [2.0, 0.0], tf, {"mu": mu}))
    # extra exp_decay / harmonic to fill toward 80
    for k in [0.7, 1.5, 3.5]:
        for tf in [2.0, 4.0, 7.0]:
            hard_specs.append(("exp_decay", [1.5], tf, {"k": k}))
    for tf in [3.0, 6.0, 9.0]:
        hard_specs.append(("harmonic", [2.0, -1.0], tf, {}))
    # final fill toward the ~80 target: more exp_decay / logistic / linear2d
    for k in [0.4, 1.2]:
        for tf in [3.0, 8.0]:
            hard_specs.append(("exp_decay", [0.8], tf, {"k": k}))
    for r in [1.5, 2.5]:
        for tf in [4.0, 8.0]:
            hard_specs.append(("logistic", [0.02], tf, {"r": r, "kk": 1.0}))
    for tf in [3.0, 10.0]:
        hard_specs.append(("linear2d", [2.0, -1.0], tf, {}))

    for tag, y0, tf, p in hard_specs:
        if idx >= 80:
            break
        # vdp with small mu over short horizons gets a slightly wider explicit
        # envelope (mild stiffness costs the explicit pair some accuracy).
        tol = dict(NONSTIFF_TOL)
        if tag == "vdp":
            tol["rk45"] = 1e-1
            tol["rk23"] = 5e-1
            tol["bdf"] = 1e0
        cases.append(
            case(f"ode.hard.{tag}_{idx}", "hard", tag, y0, tf, NONSTIFF_ALL, tol, p)
        )
        idx += 1

    # ── Edge (~10): out-of-envelope stiff (explicit) + in-envelope guards ─────
    # Out-of-envelope: STIFF systems integrated with EXPLICIT methods. The
    # explicit step controller collapses (probe-confirmed: vdp μ≥100 over a long
    # horizon and Robertson over tf=40 exhaust the 10000-step budget for both
    # rk45 and rk23), so the library MUST emit `outsideEnvelope`. Accuracy is
    # NOT gated for these — tol is wide so no spurious accuracy flag fires.
    edge_oov = [
        ("vdp", [2.0, 0.0], 3000.0, {"mu": 1000.0}),  # classic stiff Van der Pol
        ("vdp", [2.0, 0.0], 300.0, {"mu": 100.0}),
        ("robertson", [1.0, 0.0, 0.0], 40.0, {}),  # Robertson stiff kinetics
    ]
    for i, (tag, y0, tf, p) in enumerate(edge_oov):
        cases.append(
            case(
                f"ode.edge.stiff_oov_{i}",
                "edge",
                tag,
                y0,
                tf,
                ["rk45", "rk23"],
                {"rk45": 1e9, "rk23": 1e9},
                p,
                in_envelope={"rk45": False, "rk23": False},
            )
        )
    # The same stiff systems solved with the implicit `.bdf` method are
    # IN-envelope: BDF is the library's documented stiff solver (issue #15) and
    # must NOT emit a diagnostic (false-positive guard). `odeint` is intentionally
    # NOT used here — NumericSwift's odeint is RK45-based, so it is no more
    # stiff-capable than the explicit pair (FP1: do not claim stiff-safety the
    # library does not have). BDF-1 is O(h), so its accuracy flag is wide.
    edge_stiff_ok = [
        ("vdp", [2.0, 0.0], 300.0, {"mu": 100.0}),
        ("robertson", [1.0, 0.0, 0.0], 40.0, {}),
    ]
    for i, (tag, y0, tf, p) in enumerate(edge_stiff_ok):
        cases.append(
            case(
                f"ode.edge.stiff_ok_{i}",
                "edge",
                tag,
                y0,
                tf,
                ["bdf"],
                {"bdf": 1e9},
                p,
            )
        )
    # In-envelope explicit edge: well-behaved non-stiff systems over LONG
    # horizons — explicit rk45/rk23 stay reliable and must NOT warn.
    edge_nonstiff_ok = [
        ("harmonic", [1.0, 0.0], 50.0, {}),
        ("exp_decay", [1.0], 20.0, {"k": 1.0}),
        ("logistic", [0.001], 20.0, {"r": 1.0, "kk": 1.0}),
        ("linear2d", [1.0, 1.0], 15.0, {}),
        ("harmonic", [0.0, 1.0], 30.0, {}),
    ]
    for i, (tag, y0, tf, p) in enumerate(edge_nonstiff_ok):
        cases.append(
            case(
                f"ode.edge.nonstiff_ok_{i}",
                "edge",
                tag,
                y0,
                tf,
                ["rk45", "rk23"],
                {"rk45": 1e-2, "rk23": 1e-1},
                p,
            )
        )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"ode: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ode.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
