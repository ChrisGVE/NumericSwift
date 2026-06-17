#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Distributions domain.

Computes bit-exact reference values with scipy.stats and freezes them as the
JSON fixture `Tests/NumericSwiftTests/Fixtures/workbench/distributions.json`.

Copied from the REFERENCE generator `integration.py`. Contract (WORKBENCH.md
§2/§3/§5):

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual
    counts are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation. Oracle values come ONLY from scipy.stats — never from
    NumericSwift (FP1 / FP3 vacuous-gate rule).
  * The domain is single-strategy-per-function: each distribution function is a
    strategy id (`normal_cdf`, `t_ppf`, `chi2_cdf`, …). `inEnvelope` is per
    strategy. The one documented limitation (CLAUDE.md *Known Limitations §1*) is
    the Student-t `ppf` extreme tails (`p < 1e-4` or `p > 0.9999`): those cases
    are tagged `inEnvelope: {"t_ppf": false}`, so the gate requires NumericSwift
    to emit an `outsideEnvelope` diagnostic for them. Every other function/case
    is in-envelope (a correctness-vs-scipy gate).

Strategy ids and their parameters MUST match `registerDistributionsStrategies`
in `Sources/NumericSwiftWorkbenchKit/Domains/Distributions.swift`.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/distributions.py
"""

import json
import struct
import sys
from pathlib import Path

import scipy
from scipy import stats

SOURCE = f"scipy.stats {scipy.__version__}"


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


# ── Oracle dispatch ─────────────────────────────────────────────────────────
# One scipy.stats frozen distribution per family. Each strategy id is
# "<family>_<func>"; the function is cdf / ppf / pdf.


def _frozen(family: str, params: dict):
    """Construct the scipy.stats frozen distribution for a family."""
    loc = params.get("loc", 0.0)
    scale = params.get("scale", 1.0)
    if family == "normal":
        return stats.norm(loc=loc, scale=scale)
    if family == "uniform":
        return stats.uniform(loc=loc, scale=scale)
    if family == "expon":
        return stats.expon(loc=loc, scale=scale)
    if family == "t":
        return stats.t(params["df"], loc=loc, scale=scale)
    if family == "chi2":
        return stats.chi2(params["df"], loc=loc, scale=scale)
    if family == "f":
        return stats.f(params["dfn"], params["dfd"], loc=loc, scale=scale)
    if family == "gamma":
        return stats.gamma(params["shape"], loc=loc, scale=scale)
    if family == "beta":
        return stats.beta(params["a"], params["b"], loc=loc, scale=scale)
    raise ValueError(f"unknown family {family}")


def oracle(family: str, func: str, arg: float, params: dict) -> float:
    """High-accuracy reference value from scipy.stats."""
    d = _frozen(family, params)
    return float(getattr(d, func)(arg))


def case(cid, tier, family, func, arg, *, params, tol, in_envelope=None):
    """Build one fixture entry.

    `arg` is the evaluation point: `x` for cdf/pdf, `p` for ppf. It is carried
    in the inputs bag under the function-appropriate key so the Swift strategy
    closure reads the same value.
    """
    strategy = f"{family_alias(family)}_{func}"
    inputs = {"func": func, "family": family_alias(family)}
    # cdf/pdf take x; ppf takes p.
    if func == "ppf":
        inputs["p"] = arg
    else:
        inputs["x"] = arg
    # Distribution parameters.
    for k, v in params.items():
        inputs[k] = v
    val = oracle(family, func, arg, params)
    # NaN/inf edge values: store bit-exact, write `value` as JSON null
    # (guard allow_nan=False at dump time like the other generators).
    json_value = val if (val == val and abs(val) != float("inf")) else None
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": json_value, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": [strategy],
        "tol": {strategy: tol},
    }
    if in_envelope is not None:
        entry["inEnvelope"] = {strategy: in_envelope}
    return entry


# Swift-side strategy prefix per scipy family.
_ALIAS = {
    "normal": "normal",
    "uniform": "uniform",
    "expon": "expon",
    "t": "t",
    "chi2": "chi2",
    "f": "f",
    "gamma": "gamma",
    "beta": "beta",
}


def family_alias(family: str) -> str:
    return _ALIAS[family]


# Per-function default tolerances (absolute error vs scipy). cdf/pdf are
# closed-form-ish and very tight; ppf is Newton-Raphson and a touch looser.
TOL = {"cdf": 1e-10, "pdf": 1e-10, "ppf": 1e-7}


def build():
    cases = []

    # ── Trivial (~10): textbook smoke values, standard params ─────────────────
    trivial = [
        ("normal", "cdf", 0.0, {}),  # 0.5
        ("normal", "pdf", 0.0, {}),  # 1/sqrt(2pi)
        ("normal", "ppf", 0.5, {}),  # 0.0
        ("normal", "cdf", 1.96, {}),  # ~0.975
        ("uniform", "cdf", 0.5, {}),  # 0.5
        ("expon", "cdf", 1.0, {}),  # 1 - 1/e
        ("t", "cdf", 0.0, {"df": 10.0}),  # 0.5
        ("chi2", "cdf", 1.0, {"df": 1.0}),  # ~0.6827
        ("f", "cdf", 1.0, {"dfn": 5.0, "dfd": 10.0}),
        ("gamma", "cdf", 1.0, {"shape": 2.0}),
    ]
    for i, (fam, func, arg, params) in enumerate(trivial):
        cases.append(
            case(
                f"distributions.trivial.{family_alias(fam)}_{func}_{i}",
                "trivial",
                fam,
                func,
                arg,
                params=params,
                tol=TOL[func],
            )
        )

    # ── Hard (~80): realistic params / args across all families & funcs ───────
    # A grid of (family, func, [args], [param-sets]) yielding non-degenerate,
    # mid-distribution evaluations — the bulk of correctness coverage vs scipy.
    idx = 0
    # The grid is sized so the full sweep lands at ~80 cases with EVERY family
    # and EVERY function covered (no family is dropped by the cap). cdf carries
    # the most args (the bulk-correctness workhorse); pdf/ppf a few each.
    hard_specs = [
        # Normal — shifted/scaled.
        ("normal", "cdf", [-2.5, -1.0, 0.7, 2.3], [{"loc": 1.0, "scale": 2.0}]),
        ("normal", "pdf", [-1.0, 0.4, 1.8], [{"loc": 0.5, "scale": 1.5}]),
        ("normal", "ppf", [0.05, 0.25, 0.6, 0.9], [{"loc": 2.0, "scale": 3.0}]),
        # Student-t — central / near-tail (all in-envelope: |p| <= 0.9999).
        ("t", "cdf", [-3.0, -0.5, 1.2, 4.0], [{"df": 3.0}]),
        ("t", "pdf", [-2.0, 0.0, 2.5], [{"df": 5.0}, {"df": 100.0}]),
        ("t", "ppf", [0.01, 0.1, 0.4, 0.75, 0.95, 0.99], [{"df": 4.0}]),
        # Chi-squared.
        ("chi2", "cdf", [0.5, 2.0, 6.0, 15.0], [{"df": 3.0}]),
        ("chi2", "pdf", [1.0, 4.0, 9.0], [{"df": 2.0}]),
        ("chi2", "ppf", [0.05, 0.5, 0.9, 0.99], [{"df": 5.0}]),
        # F.
        ("f", "cdf", [0.5, 1.5, 3.0, 8.0], [{"dfn": 3.0, "dfd": 12.0}]),
        ("f", "pdf", [0.8, 2.0, 5.0], [{"dfn": 4.0, "dfd": 8.0}]),
        ("f", "ppf", [0.1, 0.5, 0.9, 0.99], [{"dfn": 5.0, "dfd": 15.0}]),
        # Gamma.
        ("gamma", "cdf", [0.5, 2.0, 5.0, 10.0], [{"shape": 2.0}]),
        ("gamma", "pdf", [1.0, 3.0, 7.0], [{"shape": 3.0}]),
        ("gamma", "ppf", [0.1, 0.5, 0.9], [{"shape": 2.5}]),
        # Beta.
        ("beta", "cdf", [0.2, 0.5, 0.8], [{"a": 2.0, "b": 5.0}, {"a": 0.5, "b": 0.5}]),
        ("beta", "pdf", [0.3, 0.6], [{"a": 3.0, "b": 2.0}]),
        ("beta", "ppf", [0.1, 0.5, 0.9], [{"a": 2.0, "b": 3.0}]),
        # Exponential / uniform tail of the grid.
        ("expon", "cdf", [0.5, 2.0, 5.0], [{"scale": 2.0}]),
        ("expon", "ppf", [0.1, 0.5, 0.9], [{"scale": 1.5}]),
        ("uniform", "cdf", [0.2, 0.5, 0.9], [{"loc": -1.0, "scale": 4.0}]),
        ("uniform", "ppf", [0.25, 0.75], [{"loc": 2.0, "scale": 6.0}]),
    ]
    for fam, func, args, param_sets in hard_specs:
        for params in param_sets:
            for arg in args:
                if idx >= 80:
                    break
                cases.append(
                    case(
                        f"distributions.hard.{family_alias(fam)}_{func}_{idx}",
                        "hard",
                        fam,
                        func,
                        arg,
                        params=params,
                        tol=TOL[func],
                    )
                )
                idx += 1

    # ── Edge (~10): the documented t_ppf extreme-tail out-of-envelope cases ───
    # CLAUDE.md Known Limitations §1: Student-t ppf is ~5 digits for |p| > 0.9999.
    # These MUST trip TDistribution.ppfDiagnosed's outsideEnvelope diagnostic.
    # Loose tol (1e-4) reflects the documented ~5-digit accuracy; the gate is
    # the *diagnostic*, not the numeric value.
    t_oov = [
        (1e-5, {"df": 5.0}),
        (5e-6, {"df": 10.0}),
        (1.0 - 1e-5, {"df": 5.0}),
        (1.0 - 5e-6, {"df": 20.0}),
        (1e-6, {"df": 3.0}),
    ]
    for i, (p, params) in enumerate(t_oov):
        cases.append(
            case(
                f"distributions.edge.t_ppf_oov_{i}",
                "edge",
                "t",
                "ppf",
                p,
                params=params,
                tol=1e-4,
                in_envelope=False,
            )
        )

    # In-envelope edge: t_ppf just INSIDE the envelope (|p| == 0.9999 boundary
    # region) and other-family extremes — must NOT emit a diagnostic, and must
    # stay within full-precision tol. This is the false-positive guard.
    edge_ok = [
        ("t", "ppf", 0.999, {"df": 5.0}, TOL["ppf"]),  # just inside (p < 0.9999)
        ("t", "ppf", 0.001, {"df": 5.0}, TOL["ppf"]),  # just inside (p > 1e-4)
        ("normal", "ppf", 0.99999, {}, 1e-7),  # normal has no documented tail limit
        ("chi2", "cdf", 50.0, {"df": 4.0}, 1e-10),  # far tail cdf, full precision
        ("gamma", "cdf", 30.0, {"shape": 2.0}, 1e-10),
    ]
    for i, (fam, func, arg, params, tol) in enumerate(edge_ok):
        cases.append(
            case(
                f"distributions.edge.{family_alias(fam)}_{func}_ok_{i}",
                "edge",
                fam,
                func,
                arg,
                params=params,
                tol=tol,
            )
        )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"distributions: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "distributions.json"
    # allow_nan=False: NaN/inf oracle values are stored bit-exact in `bits` and
    # written as JSON null in `value` (see `case`), so the dump must be finite.
    out.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
