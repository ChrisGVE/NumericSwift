#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Integration (quadrature) domain.

Computes bit-exact reference integrals with scipy and freezes them as the JSON
fixture `Tests/NumericSwiftTests/Fixtures/workbench/integration.json`.

This is the REFERENCE generator every other domain copies. Contract (WORKBENCH.md
§2/§3/§5):

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual counts
    are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation. Oracle values come ONLY from scipy — never from NumericSwift
    (FP3 vacuous-gate rule).
  * `inEnvelope` is per-strategy. Out-of-envelope cases (here: `quad` forced to a
    tiny subdivision limit on a hard integrand) are tagged `false`, so the gate
    requires NumericSwift to emit an `outsideEnvelope` diagnostic for them.

Integrand `tag`s MUST match `integrationIntegrand` in
`Sources/NumericSwiftWorkbenchKit/Domains/Integration.swift`.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/integration.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import integrate

SOURCE = f"scipy.integrate.quad {__import__('scipy').__version__}"

# Integrands — keep in lockstep with the Swift `integrationIntegrand` resolver.
INTEGRANDS = {
    "gaussian_bell": lambda x: math.exp(-x * x),
    "sine": math.sin,
    "cosine": math.cos,
    "polynomial_deg2": lambda x: x * x + 2 * x + 1,
    "exp": math.exp,
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x * x),
    "oscillatory": lambda x: math.sin(50.0 * x),
    "inverse_sqrt": lambda x: 0.0 if x <= 0 else 1.0 / math.sqrt(x),
}


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


def oracle(tag: str, a: float, b: float) -> float:
    """High-accuracy reference integral from scipy."""
    f = INTEGRANDS[tag]
    val, _ = integrate.quad(f, a, b, limit=200, epsabs=1e-13, epsrel=1e-13)
    return val


def case(
    cid,
    tier,
    tag,
    a,
    b,
    strategies,
    tol,
    *,
    n=None,
    order=None,
    limit=None,
    epsabs=None,
    epsrel=None,
    in_envelope=None,
):
    inputs = {"a": a, "b": b, "tag": tag}
    if n is not None:
        inputs["n"] = n
    if order is not None:
        inputs["order"] = order
    if limit is not None:
        inputs["limit"] = limit
    if epsabs is not None:
        inputs["epsabs"] = epsabs
    if epsrel is not None:
        inputs["epsrel"] = epsrel
    val = oracle(tag, a, b)
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


def build():
    cases = []
    ALL = ["quad", "romberg", "simps", "trapz", "fixed_quad"]
    # Realistic per-strategy envelopes for smooth hard cases. quad/romberg are
    # near machine precision; simps is 4th-order; trapz is 1st-order; fixed_quad
    # is a single fixed-order Gauss-Legendre rule (no adaptivity) so it gets a
    # wide envelope on broad ranges. These are the declared envelopes, not the
    # achieved errors — the report's `error` column still shows what each hits.
    smooth_tol = {
        "quad": 1e-7,
        "romberg": 1e-4,
        "simps": 1e-2,
        "trapz": 5e0,
        "fixed_quad": 1e0,
    }

    # ── Trivial (~10): closed-form, smooth, fine grids ────────────────────────
    trivial = [
        ("polynomial_deg2", 0.0, 1.0),
        ("polynomial_deg2", -1.0, 2.0),
        ("sine", 0.0, math.pi),
        ("cosine", 0.0, math.pi / 2),
        ("exp", 0.0, 1.0),
        ("exp", -1.0, 1.0),
        ("sine", 0.0, 2 * math.pi),
        ("polynomial_deg2", 0.0, 5.0),
        ("cosine", -math.pi / 2, math.pi / 2),
        ("exp", 0.0, 2.0),
    ]
    for i, (tag, a, b) in enumerate(trivial):
        # trivial tier: tight tolerances, fine grid (n=1000)
        cases.append(
            case(
                f"integration.trivial.{tag}_{i}",
                "trivial",
                tag,
                a,
                b,
                ALL,
                {
                    "quad": 1e-12,
                    "romberg": 1e-10,
                    "simps": 1e-9,
                    "trapz": 1e-4,
                    "fixed_quad": 1e-6,
                },
                n=1000,
                order=8,
            )
        )

    # ── Hard (~80): realistic, varied tags / ranges / grids ───────────────────
    hard_tags = ["gaussian_bell", "runge", "sine", "cosine", "exp", "oscillatory"]
    ranges = [(-3.0, 3.0), (-1.0, 1.0), (0.0, 2.0), (-2.0, 5.0), (0.5, 4.0)]
    n_choices = [200, 400, 800]
    idx = 0
    for tag in hard_tags:
        for a, b in ranges:
            for n in n_choices:
                if idx >= 80:
                    break
                # Oscillatory / Runge integrands defeat the low-order and
                # fixed-order rules: widen their envelopes accordingly.
                tol = dict(smooth_tol)
                if tag in ("oscillatory", "runge"):
                    tol["romberg"] = 1e0
                    tol["simps"] = 1e0
                    tol["trapz"] = 1e1
                    tol["fixed_quad"] = 1e1
                cases.append(
                    case(
                        f"integration.hard.{tag}_{idx}",
                        "hard",
                        tag,
                        a,
                        b,
                        ALL,
                        tol,
                        n=n,
                        order=8,
                    )
                )
                idx += 1

    # ── Edge (~10): out-of-envelope quad (tiny limit) + hard in-envelope ──────
    # Out-of-envelope: quad with a 3-subdivision budget on hard integrands MUST
    # trip the IntegrationWarning-equivalent diagnostic. Run quad only.
    # Asymmetric / singular only: a symmetric oscillatory range lets gk15's
    # symmetric nodes cancel, yielding a tiny error estimate that is accepted
    # without subdivision (no diagnostic). Endpoint singularities and asymmetric
    # high-frequency integrands reliably exhaust the 3-subdivision budget.
    edge_oov = [
        ("oscillatory", 0.0, 10.0),
        ("inverse_sqrt", 0.0, 1.0),
        ("inverse_sqrt", 0.0, 4.0),
        ("inverse_sqrt", 0.0, 9.0),
        ("oscillatory", 0.3, 12.0),
    ]
    for i, (tag, a, b) in enumerate(edge_oov):
        cases.append(
            case(
                f"integration.edge.quad_oov_{i}",
                "edge",
                tag,
                a,
                b,
                ["quad"],
                {"quad": 1e-5},
                limit=3,
                in_envelope={"quad": False},
            )
        )

    # In-envelope edge: quad with a generous budget stays reliable, no diagnostic.
    edge_ok = [
        ("gaussian_bell", -8.0, 8.0),
        ("runge", -10.0, 10.0),
        ("exp", -3.0, 3.0),
        ("sine", 0.0, 8 * math.pi),
        ("cosine", 0.0, 8 * math.pi),
    ]
    for i, (tag, a, b) in enumerate(edge_ok):
        # quad only: this block is the false-positive guard — quad with a generous
        # budget stays inside its envelope and must NOT emit a diagnostic.
        cases.append(
            case(
                f"integration.edge.quad_ok_{i}",
                "edge",
                tag,
                a,
                b,
                ["quad"],
                {"quad": 1e-6},
            )
        )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"integration: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "integration.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
