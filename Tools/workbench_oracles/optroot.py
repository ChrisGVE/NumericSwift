#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Optimization (root finding).

Computes bit-exact reference roots with scipy and freezes them as the JSON fixture
`Tests/NumericSwiftTests/Fixtures/workbench/optroot.json`.

Strategies compared (WORKBENCH.md §4): `bisect`, `newton`, `secant`, `brentq`.
The bracketing methods (`bisect`, `brentq`) take a sign-changing bracket [a, b];
the open methods (`newton`, `secant`) take an initial guess x0.

Contract (WORKBENCH.md §2/§3/§5):

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual counts
    are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation. Oracle roots come ONLY from scipy.optimize.brentq on a
    sign-changing bracket — never from NumericSwift (FP1 / FP3 vacuous-gate rule).
  * `inEnvelope` is per-strategy. Out-of-envelope cases are tagged `false`, so the
    gate requires NumericSwift to emit an `outsideEnvelope` diagnostic for them:
      - bisect / brentq on a bracket with NO sign change (f(a)·f(b) > 0);
      - newton / secant that hit a near-zero derivative or diverge (maxiter budget).

Function `tag`s MUST match `optrootFunction` in
`Sources/NumericSwiftWorkbenchKit/Domains/OptRoot.swift`.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/optroot.py
"""

import json
import math
import struct
import sys
from pathlib import Path

from scipy import optimize

SOURCE = f"scipy.optimize.brentq {__import__('scipy').__version__}"

# Test functions — keep in lockstep with the Swift `optrootFunction` resolver.
# Each entry: tag -> callable f(x). All have at least one simple real root that
# scipy.optimize.brentq can locate from a sign-changing bracket.
FUNCS = {
    "x2_minus_2": lambda x: x * x - 2.0,  # root √2 ≈ 1.4142135624
    "cos_minus_x": lambda x: math.cos(x) - x,  # root  ≈ 0.7390851332
    "cubic_x3_x_2": lambda x: x * x * x - x - 2.0,  # root  ≈ 1.5213797068
    "exp_minus_3": lambda x: math.exp(x) - 3.0,  # root ln 3 ≈ 1.0986122887
    "sin": lambda x: math.sin(x),  # roots at kπ
    "cubic_x3_2x_5": lambda x: x * x * x - 2.0 * x - 5.0,  # root  ≈ 2.0945514815
    "log_minus_1": lambda x: math.log(x) - 1.0,  # root e ≈ 2.7182818285
    "quad_x2_2x_3": lambda x: x * x - 2.0 * x - 3.0,  # roots 3 and -1
}


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


def root_oracle(tag: str, a: float, b: float) -> float:
    """High-accuracy reference root from scipy on the sign-changing bracket [a, b]."""
    f = FUNCS[tag]
    return optimize.brentq(
        f, a, b, xtol=1e-15, rtol=4 * sys.float_info.epsilon, maxiter=200
    )


def case(cid, tier, tag, root, strategies, tol, inputs, in_envelope=None):
    inputs = dict(inputs)
    inputs["tag"] = tag
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": root, "bits": bits_hex(root)},
        "source": SOURCE,
        "strategies": strategies,
        "tol": tol,
    }
    if in_envelope:
        entry["inEnvelope"] = in_envelope
    return entry


# Per-tier tolerances. The library's default convergence test is the absolute
# step tolerance xtol = 1e-8, so the located root sits within ~1e-8 of the true
# root for every method (bisection stops at |b-a| < xtol; the open/Brent methods
# at |dx| < xtol). The declared envelopes sit a safe order of magnitude above the
# achieved ~1e-8 — tight enough to catch a real regression, loose enough not to
# flag the expected step-tolerance residual. Newton's quadratic convergence
# routinely lands at machine precision, so it earns a slightly tighter bound.
TRIVIAL_TOL = {"bisect": 1e-6, "newton": 1e-8, "secant": 1e-6, "brentq": 1e-6}
HARD_TOL = {"bisect": 1e-6, "newton": 1e-8, "secant": 1e-6, "brentq": 1e-6}
EDGE_TOL = {"bisect": 1e-5, "newton": 1e-7, "secant": 1e-5, "brentq": 1e-5}

ALL = ["bisect", "newton", "secant", "brentq"]
BRACKETING = ["bisect", "brentq"]
OPEN = ["newton", "secant"]


def build():
    cases = []

    # ── Trivial (~10): textbook, well-separated roots, tight brackets ─────────
    # (tag, a, b, x0): a/b is the sign-changing bracket; x0 the open-method guess.
    trivial = [
        ("x2_minus_2", 0.0, 2.0, 1.0),
        ("cos_minus_x", 0.0, 1.0, 0.5),
        ("exp_minus_3", 0.0, 2.0, 1.0),
        ("cubic_x3_x_2", 1.0, 2.0, 1.5),
        ("sin", 3.0, 4.0, 3.0),
        ("cubic_x3_2x_5", 2.0, 3.0, 2.0),
        ("log_minus_1", 1.0, 4.0, 2.0),
        ("quad_x2_2x_3", 1.0, 5.0, 4.0),
        ("x2_minus_2", -2.0, 0.0, -1.0),  # negative root −√2
        ("sin", 2.5, 4.0, 3.2),
    ]
    for i, (tag, a, b, x0) in enumerate(trivial):
        root = root_oracle(tag, a, b)
        cases.append(
            case(
                f"optroot.trivial.{tag}_{i}",
                "trivial",
                tag,
                root,
                ALL,
                TRIVIAL_TOL,
                {"a": a, "b": b, "x0": x0},
            )
        )

    # ── Hard (~80): varied functions, brackets shifted/widened, multiple guesses
    # All brackets straddle a root (sign change verified by brentq below); all x0
    # guesses lie in a basin that converges. These are the bulk in-envelope cases.
    hard_specs = [
        # (tag, [(a, b, x0), ...])
        (
            "x2_minus_2",
            [
                (0.0, 2.0, 1.0),
                (1.0, 3.0, 1.2),
                (0.5, 5.0, 2.0),
                (1.0, 1.5, 1.4),
                (-2.0, -0.5, -1.3),
                (-3.0, 0.0, -1.5),
            ],
        ),
        (
            "cos_minus_x",
            [
                (0.0, 1.0, 0.5),
                (0.5, 1.5, 0.7),
                (0.0, 1.5, 0.6),
                (0.3, 0.9, 0.74),
                (-1.0, 1.0, 0.5),
                (0.0, 2.0, 1.0),
            ],
        ),
        (
            "exp_minus_3",
            [
                (0.0, 2.0, 1.0),
                (0.5, 1.5, 1.1),
                (1.0, 1.2, 1.1),
                (0.0, 3.0, 0.8),
                (-1.0, 2.0, 1.0),
                (0.9, 1.3, 1.1),
            ],
        ),
        (
            "cubic_x3_x_2",
            [
                (1.0, 2.0, 1.5),
                (1.4, 1.6, 1.5),
                (0.0, 3.0, 1.8),
                (1.0, 5.0, 2.0),
                (1.3, 1.7, 1.52),
                (1.5, 2.5, 1.6),
            ],
        ),
        (
            "cubic_x3_2x_5",
            [
                (2.0, 3.0, 2.0),
                (2.0, 2.2, 2.1),
                (1.0, 4.0, 2.5),
                (2.05, 2.15, 2.1),
                (0.0, 5.0, 3.0),
                (2.0, 10.0, 2.2),
            ],
        ),
        (
            "log_minus_1",
            [
                (1.0, 4.0, 2.0),
                (2.0, 3.0, 2.5),
                (2.5, 3.0, 2.7),
                (1.5, 5.0, 3.0),
                (2.0, 4.0, 2.7),
                (0.5, 4.0, 2.0),
            ],
        ),
        (
            "quad_x2_2x_3",
            [
                (1.0, 5.0, 4.0),
                (2.5, 4.0, 3.2),
                (2.9, 3.1, 3.0),
                (0.0, 6.0, 4.0),
                (-3.0, 0.0, -1.5),
                (-2.0, 0.0, -0.8),
            ],
        ),
        (
            "sin",
            [
                (3.0, 4.0, 3.0),
                (2.8, 3.5, 3.1),
                (3.0, 3.2, 3.1),
                (-0.5, 0.5, 0.2),
                (5.5, 7.0, 6.3),
                (2.5, 4.0, 3.4),
            ],
        ),
    ]
    idx = 0
    # Round-robin over functions so all are represented before any repeats,
    # filling up to 80 hard cases.
    flattened = []
    for tag, specs in hard_specs:
        for a, b, x0 in specs:
            flattened.append((tag, a, b, x0))
    # We have 8 funcs × 6 = 48 base specs. Extend with shifted brackets to reach 80.
    extra = [
        ("x2_minus_2", 1.0, 2.0, 1.41),
        ("cos_minus_x", 0.6, 0.8, 0.73),
        ("exp_minus_3", 1.05, 1.15, 1.1),
        ("cubic_x3_x_2", 1.51, 1.53, 1.52),
        ("cubic_x3_2x_5", 2.09, 2.10, 2.094),
        ("log_minus_1", 2.7, 2.75, 2.71),
        ("quad_x2_2x_3", 2.99, 3.01, 3.0),
        ("sin", 3.13, 3.15, 3.14),
        ("x2_minus_2", 0.0, 10.0, 3.0),
        ("cos_minus_x", 0.0, 0.9, 0.6),
        ("exp_minus_3", 0.0, 5.0, 2.0),
        ("cubic_x3_x_2", 0.5, 2.5, 1.7),
        ("cubic_x3_2x_5", 1.5, 3.5, 2.3),
        ("log_minus_1", 1.0, 10.0, 4.0),
        ("quad_x2_2x_3", 2.0, 10.0, 5.0),
        ("sin", 2.9, 3.3, 3.0),
        ("x2_minus_2", -3.0, -1.0, -1.41),
        ("cos_minus_x", 0.1, 1.2, 0.5),
        ("exp_minus_3", 0.8, 1.4, 1.0),
        ("cubic_x3_x_2", 1.45, 1.6, 1.5),
        ("cubic_x3_2x_5", 2.0, 4.0, 2.5),
        ("log_minus_1", 2.0, 5.0, 3.0),
        ("quad_x2_2x_3", 2.8, 3.2, 3.1),
        ("sin", 3.0, 3.5, 3.2),
        ("x2_minus_2", 1.2, 1.6, 1.4),
        ("cos_minus_x", 0.5, 1.0, 0.75),
        ("exp_minus_3", 0.9, 1.2, 1.05),
        ("cubic_x3_x_2", 1.0, 1.7, 1.5),
        ("cubic_x3_2x_5", 2.0, 2.5, 2.1),
        ("log_minus_1", 2.5, 3.5, 2.8),
        ("quad_x2_2x_3", 1.5, 4.5, 3.5),
        ("sin", 6.0, 7.0, 6.3),
    ]
    flattened.extend(extra)
    flattened = flattened[:80]

    for tag, a, b, x0 in flattened:
        root = root_oracle(tag, a, b)
        cases.append(
            case(
                f"optroot.hard.{tag}_{idx}",
                "hard",
                tag,
                root,
                ALL,
                HARD_TOL,
                {"a": a, "b": b, "x0": x0},
            )
        )
        idx += 1

    # ── Edge (~10): out-of-envelope (must warn) + in-envelope guards ──────────
    #
    # OUT-OF-ENVELOPE — the library MUST emit an `outsideEnvelope` diagnostic.
    #
    #  (1) bracketing on a bracket with NO sign change (f(a)·f(b) > 0). We pick a
    #      bracket where the function keeps one sign, so brentq/bisect are invalid.
    #      Oracle value is irrelevant for the gate (numeric value is not checked
    #      out-of-envelope); we store the nearest true root for the report column.
    #  (2) newton with a near-zero derivative at the start (f'(x0) ≈ 0).
    #  (3) newton / secant that diverge (guess in a runaway basin → maxiter).

    # (1) Invalid bracket for bisect + brentq. cos(x)-x is positive on [-2,-1]
    # (cos≈0.54..., x negative ⇒ f>0) and on [-1,0] as well: no sign change.
    cases.append(
        case(
            "optroot.edge.bisect_no_sign_change",
            "edge",
            "x2_minus_2",
            root_oracle("x2_minus_2", 0.0, 2.0),
            BRACKETING,
            {"bisect": EDGE_TOL["bisect"], "brentq": EDGE_TOL["brentq"]},
            {"a": 2.0, "b": 5.0},  # x²−2 > 0 on all of [2,5] → no sign change
            in_envelope={"bisect": False, "brentq": False},
        )
    )
    cases.append(
        case(
            "optroot.edge.brentq_no_sign_change",
            "edge",
            "exp_minus_3",
            root_oracle("exp_minus_3", 0.0, 2.0),
            BRACKETING,
            {"bisect": EDGE_TOL["bisect"], "brentq": EDGE_TOL["brentq"]},
            {"a": 2.0, "b": 4.0},  # exp(x)−3 > 0 on [2,4] → no sign change
            in_envelope={"bisect": False, "brentq": False},
        )
    )
    cases.append(
        case(
            "optroot.edge.bracketing_same_sign_quad",
            "edge",
            "quad_x2_2x_3",
            root_oracle("quad_x2_2x_3", 1.0, 5.0),
            BRACKETING,
            {"bisect": EDGE_TOL["bisect"], "brentq": EDGE_TOL["brentq"]},
            {"a": 4.0, "b": 6.0},  # x²−2x−3 > 0 on [4,6] (both past root 3) → same sign
            in_envelope={"bisect": False, "brentq": False},
        )
    )

    # (2) newton at a stationary point: f(x)=x²−2, f'(x)=2x, so x0=0 ⇒ f'(0)=0.
    cases.append(
        case(
            "optroot.edge.newton_zero_derivative",
            "edge",
            "x2_minus_2",
            root_oracle("x2_minus_2", 0.0, 2.0),
            ["newton"],
            {"newton": EDGE_TOL["newton"]},
            {"x0": 0.0},  # derivative 2·0 = 0 → ill-defined Newton step
            in_envelope={"newton": False},
        )
    )
    # newton on sin near a flat extremum: f'(x)=cos(x), x0=π/2 ⇒ cos=0.
    cases.append(
        case(
            "optroot.edge.newton_flat_cos",
            "edge",
            "sin",
            root_oracle("sin", 3.0, 4.0),
            ["newton"],
            {"newton": EDGE_TOL["newton"]},
            {"x0": math.pi / 2},  # cos(π/2)=0 → near-zero derivative
            in_envelope={"newton": False},
        )
    )

    # (3) Divergence: exp(x)−3 has f'(x)=exp(x); a large negative x0 makes the
    # Newton step exp(x0)−3 over exp(x0) ≈ 1 − 3·exp(−x0) drive x further negative
    # (exp(x)→0, root never reached) → maxiter without convergence.
    cases.append(
        case(
            "optroot.edge.newton_diverge_exp",
            "edge",
            "exp_minus_3",
            root_oracle("exp_minus_3", 0.0, 2.0),
            ["newton"],
            {"newton": EDGE_TOL["newton"]},
            {"x0": -40.0},  # runaway: exp tail is flat, iterate drifts off
            in_envelope={"newton": False},
        )
    )
    # secant divergence on the same flat exp tail with two nearby negative guesses.
    cases.append(
        case(
            "optroot.edge.secant_diverge_exp",
            "edge",
            "exp_minus_3",
            root_oracle("exp_minus_3", 0.0, 2.0),
            ["secant"],
            {"secant": EDGE_TOL["secant"]},
            {"x0": -50.0, "x1": -49.0},  # flat region ⇒ tiny slope, drifts to maxiter
            in_envelope={"secant": False},
        )
    )

    # IN-ENVELOPE edge guards (must NOT warn): tight valid brackets / good guesses
    # on geometrically harder roots. These are the false-positive guards.
    cases.append(
        case(
            "optroot.edge.tight_bracket_ok",
            "edge",
            "cubic_x3_x_2",
            root_oracle("cubic_x3_x_2", 1.5, 1.55),
            ALL,
            EDGE_TOL,
            {"a": 1.5, "b": 1.55, "x0": 1.52},
        )
    )
    cases.append(
        case(
            "optroot.edge.newton_good_guess_ok",
            "edge",
            "cos_minus_x",
            root_oracle("cos_minus_x", 0.0, 1.0),
            OPEN,
            {"newton": EDGE_TOL["newton"], "secant": EDGE_TOL["secant"]},
            {"x0": 0.739},  # essentially at the root → fast clean convergence
        )
    )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"optroot: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "optroot.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
