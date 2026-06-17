#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Series domain.

Computes bit-exact reference values with numpy / sympy and freezes them as the
JSON fixture `Tests/NumericSwiftTests/Fixtures/workbench/series.json`.

This is a **single-strategy-per-function correctness** domain (WORKBENCH.md §4,
"Single-strategy domains") with ONE out-of-envelope strategy:

  * polyval     → numpy.polynomial.polynomial.polyval(x, coeffs)     [ascending]
  * polyder     → npp.polyval(x, npp.polyder(coeffs))                [eval'd at x]
  * polyint     → npp.polyval(x, npp.polyint(coeffs, k=[0]))         [eval'd at x]
  * polyadd     → npp.polyval(x, npp.polyadd(p, q))                  [eval'd at x]
  * polymul     → npp.polyval(x, npp.polymul(p, q))                  [eval'd at x]
  * seriesSum   → exact closed-form / high-precision partial sum (numpy/sympy)
  * divdiff     → numpy Newton-form leading divided-difference coefficient
  * taylor      → sympy TRUE Taylor series of the named function, eval'd at x

Coefficient ordering — IMPORTANT
─────────────────────────────────
NumericSwift `Series.polyval([c0, c1, c2, ...], at: x)` evaluates
`c0 + c1·x + c2·x² + …`, i.e. coefficients are in **ASCENDING** power order. This
matches `numpy.polynomial.polynomial` (NOT the legacy `numpy.polyval`, which is
descending). All polynomial oracles below therefore use
`numpy.polynomial.polynomial.*` so the convention is aligned. Coefficient arrays
are carried in the fixture under `inputs` (`coeffs`, or `p`/`q`) in ascending
order, exactly as the Swift suite passes them to `Series.*`.

The taylor strategy is the ONLY out-of-envelope source (WORKBENCH.md §5). The
`Series.taylor` generator for `tan` hard-codes coefficients only to index 11
(12 terms); requesting more terms silently returns 0 for every higher index —
including the genuinely non-zero x¹³ coefficient — so the truncated series is
materially wrong near x = ±π/2 (documented in CLAUDE.md "Code Review Findings →
Series.swift"). Those cases are tagged `inEnvelope:false`, requiring NumericSwift
to emit an `outsideEnvelope` diagnostic; the oracle there is the TRUE math.tan /
sympy value, so the fixture is non-vacuous (FP1: never NumericSwift; FP3).

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/series.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import numpy.polynomial.polynomial as npp
import sympy as sp

SOURCE = f"numpy {np.__version__} / numpy.polynomial / sympy {sp.__version__}"


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


# ── sympy Taylor reference ────────────────────────────────────────────────────
# The named functions the Swift `Series.knownTaylorSeries` generator supports.
# Maps the Swift generator name → the sympy expression in the variable `x`.
_X = sp.symbols("x")
_SYMPY_FUNC = {
    "sin": sp.sin(_X),
    "cos": sp.cos(_X),
    "exp": sp.exp(_X),
    "sinh": sp.sinh(_X),
    "cosh": sp.cosh(_X),
    "tan": sp.tan(_X),
    "atan": sp.atan(_X),
    "log1p": sp.log(1 + _X),
}


def taylor_oracle(func: str, x: float) -> float:
    """TRUE value of the function at x (the converged target the series chases).

    The Taylor series of these functions converges to the function value inside
    its radius of convergence, so the authoritative oracle is the function value
    itself — computed with sympy at high precision, NEVER from NumericSwift.
    """
    expr = _SYMPY_FUNC[func]
    return float(expr.subs(_X, x).evalf(30))


# ── numpy polynomial references (ascending coefficient order) ─────────────────


def poly_eval(coeffs, x: float) -> float:
    return float(npp.polyval(x, np.asarray(coeffs, dtype=float)))


def polyder_eval(coeffs, x: float) -> float:
    return float(npp.polyval(x, npp.polyder(np.asarray(coeffs, dtype=float))))


def polyint_eval(coeffs, x: float) -> float:
    # k=[0.0] → integration constant 0, matching Series.polyint.
    return float(npp.polyval(x, npp.polyint(np.asarray(coeffs, dtype=float), k=[0.0])))


def polyadd_eval(p, q, x: float) -> float:
    s = npp.polyadd(np.asarray(p, dtype=float), np.asarray(q, dtype=float))
    return float(npp.polyval(x, s))


def polymul_eval(p, q, x: float) -> float:
    s = npp.polymul(np.asarray(p, dtype=float), np.asarray(q, dtype=float))
    return float(npp.polyval(x, s))


# ── seriesSum references (closed-form / high precision) ───────────────────────
# Each entry: a Swift `Series.seriesSum` "tag" → a function (frm, to) → exact sum.
# The Swift suite reconstructs the SAME term generator from the tag; the oracle
# computes the sum with sympy/numpy at high precision.


def series_sum_oracle(tag: str, frm: int, to: int) -> float:
    n = sp.symbols("n", integer=True)
    if tag == "inv_square":  # sum 1/n^2
        expr = 1 / n**2
    elif tag == "geometric_half":  # sum (1/2)^n
        expr = sp.Rational(1, 2) ** n
    elif tag == "inv_factorial":  # sum 1/n!  (→ e from n=0)
        expr = 1 / sp.factorial(n)
    elif tag == "alt_harmonic":  # sum (-1)^(n+1)/n  (→ ln 2 from n=1)
        expr = sp.Integer(-1) ** (n + 1) / n
    elif tag == "inv_fourth":  # sum 1/n^4
        expr = 1 / n**4
    else:
        raise ValueError(f"unknown seriesSum tag {tag}")
    return float(sp.summation(expr, (n, frm, to)).evalf(30))


# ── divided-difference reference ──────────────────────────────────────────────
# Series.dividedDifferences(xs, ys) returns Newton-form coefficients; the
# comparison scalar is the LEADING (top-order) divided difference, i.e. the last
# Newton coefficient. numpy is the oracle via a straightforward DD table.


def divdiff_leading(xs, ys) -> float:
    xs = list(map(float, xs))
    ys = list(map(float, ys))
    n = len(xs)
    table = [ys[:]]
    for j in range(1, n):
        prev = table[j - 1]
        col = [(prev[i + 1] - prev[i]) / (xs[i + j] - xs[i]) for i in range(n - j)]
        table.append(col)
    return float(table[-1][0])  # top-order coefficient


# ── case builders ─────────────────────────────────────────────────────────────


def poly_case(cid, tier, strategy, x, *, coeffs=None, p=None, q=None, tol):
    if strategy == "polyval":
        inputs = {"coeffs": coeffs, "x": x}
        val = poly_eval(coeffs, x)
    elif strategy == "polyder":
        inputs = {"coeffs": coeffs, "x": x}
        val = polyder_eval(coeffs, x)
    elif strategy == "polyint":
        inputs = {"coeffs": coeffs, "x": x}
        val = polyint_eval(coeffs, x)
    elif strategy == "polyadd":
        inputs = {"p": p, "q": q, "x": x}
        val = polyadd_eval(p, q, x)
    elif strategy == "polymul":
        inputs = {"p": p, "q": q, "x": x}
        val = polymul_eval(p, q, x)
    else:
        raise ValueError(strategy)
    return _emit(cid, tier, strategy, inputs, val, tol)


def seriessum_case(cid, tier, tag, frm, to, *, tol):
    val = series_sum_oracle(tag, frm, to)
    inputs = {"tag": tag, "from": frm, "to": to}
    return _emit(cid, tier, "seriesSum", inputs, val, tol)


def divdiff_case(cid, tier, xs, ys, *, tol):
    val = divdiff_leading(xs, ys)
    inputs = {"xs": list(map(float, xs)), "ys": list(map(float, ys))}
    return _emit(cid, tier, "divdiff", inputs, val, tol)


def taylor_case(cid, tier, func, x, terms, *, tol, in_env=True):
    val = taylor_oracle(func, x)
    inputs = {"func": func, "x": x, "terms": terms}
    c = _emit(cid, tier, "taylor", inputs, val, tol)
    if not in_env:
        c["inEnvelope"] = {"taylor": False}
    return c


def _emit(cid, tier, strategy, inputs, val, tol):
    # Normalise coefficient / array inputs to plain float lists.
    norm = {}
    for k, v in inputs.items():
        if isinstance(v, (list, tuple)):
            norm[k] = list(map(float, v))
        else:
            norm[k] = v
    return {
        "id": cid,
        "tier": tier,
        "inputs": norm,
        "oracle": {"value": val, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": [strategy],
        "tol": {strategy: tol},
    }


def build():
    cases = []

    # ── Trivial (~10): textbook smoke cases, closed-form answers ──────────────
    cases += [
        poly_case(
            "series.trivial.polyval_linear",
            "trivial",
            "polyval",
            2.0,
            coeffs=[1.0, 2.0],
            tol=1e-12,
        ),  # 1 + 2x at 2 = 5
        poly_case(
            "series.trivial.polyval_quad",
            "trivial",
            "polyval",
            2.0,
            coeffs=[1.0, 2.0, 3.0],
            tol=1e-12,
        ),  # 1+2x+3x²
        poly_case(
            "series.trivial.polyder_quad",
            "trivial",
            "polyder",
            2.0,
            coeffs=[1.0, 2.0, 3.0],
            tol=1e-12,
        ),  # d/dx → 2+6x
        poly_case(
            "series.trivial.polyint_quad",
            "trivial",
            "polyint",
            2.0,
            coeffs=[2.0, 6.0],
            tol=1e-12,
        ),  # ∫ → 2x+3x²
        poly_case(
            "series.trivial.polyadd",
            "trivial",
            "polyadd",
            1.5,
            p=[1.0, 2.0, 3.0],
            q=[0.0, 1.0],
            tol=1e-12,
        ),
        poly_case(
            "series.trivial.polymul",
            "trivial",
            "polymul",
            1.5,
            p=[1.0, 1.0],
            q=[1.0, 1.0],
            tol=1e-12,
        ),  # (1+x)²
        seriessum_case(
            "series.trivial.geom_finite", "trivial", "geometric_half", 0, 5, tol=1e-12
        ),
        seriessum_case(
            "series.trivial.inv_factorial_finite",
            "trivial",
            "inv_factorial",
            0,
            8,
            tol=1e-12,
        ),
        divdiff_case(
            "series.trivial.divdiff_line",
            "trivial",
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 5.0],
            tol=1e-12,
        ),  # linear → 0 lead
        taylor_case("series.trivial.taylor_sin", "trivial", "sin", 0.5, 12, tol=1e-9),
    ]

    # ── Hard (~80): realistic, varied polynomials/series ──────────────────────
    rng = np.random.default_rng(20260617)
    hard = []

    # Random polynomials of various degrees, evaluated at several x.
    for deg in (2, 3, 4, 5, 6, 8):
        coeffs = list(np.round(rng.uniform(-5.0, 5.0, deg + 1), 6))
        for x in (-1.3, -0.4, 0.7, 1.9, 3.1):
            hard.append(("polyval", x, {"coeffs": coeffs}, f"d{deg}_x{x}"))
        # derivative & integral evaluated at one interior point each
        hard.append(("polyder", 1.25, {"coeffs": coeffs}, f"d{deg}"))
        hard.append(("polyint", 0.85, {"coeffs": coeffs}, f"d{deg}"))

    # polyadd / polymul over random coefficient pairs.
    for k in range(6):
        p = list(np.round(rng.uniform(-4.0, 4.0, rng.integers(2, 6)), 6))
        q = list(np.round(rng.uniform(-4.0, 4.0, rng.integers(2, 6)), 6))
        x = float(np.round(rng.uniform(-2.0, 2.0), 4))
        hard.append(("polyadd", x, {"p": p, "q": q}, f"k{k}"))
        hard.append(("polymul", x, {"p": p, "q": q}, f"k{k}"))

    # seriesSum: convergent series summed to a large finite N (compared to the
    # high-precision partial sum oracle — exact for the same N).
    ssum = [
        ("inv_square", 1, 200, "N200"),
        ("inv_square", 1, 1000, "N1000"),
        ("inv_fourth", 1, 200, "N200"),
        ("inv_fourth", 1, 500, "N500"),
        ("geometric_half", 0, 40, "N40"),
        ("inv_factorial", 0, 15, "N15"),
        ("inv_factorial", 0, 20, "N20"),
        ("alt_harmonic", 1, 100, "N100"),
        ("alt_harmonic", 1, 500, "N500"),
        ("inv_square", 1, 50, "N50"),
    ]
    for tag, frm, to, lab in ssum:
        hard.append(("seriesSum", None, {"tag": tag, "from": frm, "to": to}, lab))

    # dividedDifferences: leading coefficient over varied node/value sets.
    for k in range(8):
        m = int(rng.integers(3, 7))
        xs = sorted(np.round(rng.uniform(-3.0, 3.0, m), 4))
        # ensure distinct nodes
        xs = list(np.unique(xs))
        while len(xs) < m:
            xs = list(np.unique(sorted(xs + [round(float(rng.uniform(-3, 3)), 4)])))
        ys = list(np.round(rng.uniform(-5.0, 5.0, len(xs)), 6))
        hard.append(("divdiff", None, {"xs": xs, "ys": ys}, f"k{k}"))

    # taylor (in-envelope): well-supported generators, terms within support, x
    # well inside radius of convergence.
    taylor_hard = [
        ("sin", 0.7, 14),
        ("sin", 1.1, 18),
        ("cos", 0.6, 14),
        ("cos", 1.2, 16),
        ("exp", 0.5, 16),
        ("exp", 1.3, 20),
        ("sinh", 0.4, 14),
        ("cosh", 0.5, 14),
        ("atan", 0.3, 18),
        ("log1p", 0.4, 24),
        ("tan", 0.5, 12),
        ("tan", 0.3, 12),
    ]
    for func, x, terms in taylor_hard:
        hard.append(
            ("taylor", x, {"func": func, "terms": terms}, f"{func}_x{x}_t{terms}")
        )

    # Trim to ~80 hard cases (proportion enforcement, ±a few).
    hard = hard[:80]
    for i, (strat, x, kw, label) in enumerate(hard):
        cid = f"series.hard.{strat}_{label}_{i}"
        if strat in ("polyval", "polyder", "polyint", "polyadd", "polymul"):
            tol = 1e-9
            cases.append(poly_case(cid, "hard", strat, x, tol=tol, **kw))
        elif strat == "seriesSum":
            cases.append(
                seriessum_case(cid, "hard", kw["tag"], kw["from"], kw["to"], tol=1e-9)
            )
        elif strat == "divdiff":
            cases.append(divdiff_case(cid, "hard", kw["xs"], kw["ys"], tol=1e-8))
        elif strat == "taylor":
            cases.append(taylor_case(cid, "hard", kw["func"], x, kw["terms"], tol=1e-6))

    # ── Edge (~10): degenerate / out-of-envelope cases ────────────────────────
    edge = []
    # Empty / single coefficient polynomials (closed-form, in-envelope).
    edge.append(
        poly_case(
            "series.edge.polyval_constant",
            "edge",
            "polyval",
            99.0,
            coeffs=[7.0],
            tol=1e-12,
        )
    )
    edge.append(
        poly_case(
            "series.edge.polyval_empty_like",
            "edge",
            "polyval",
            5.0,
            coeffs=[0.0],
            tol=1e-12,
        )
    )
    edge.append(
        poly_case(
            "series.edge.polyder_constant",
            "edge",
            "polyder",
            3.0,
            coeffs=[5.0],
            tol=1e-12,
        )
    )  # d/dx const = 0
    # Large-magnitude evaluation (cancellation but still closed-form).
    edge.append(
        poly_case(
            "series.edge.polyval_largex",
            "edge",
            "polyval",
            1e3,
            coeffs=[1.0, -2.0, 1.0],
            tol=1e-3,
        )
    )  # (x-1)² at 1e3
    # seriesSum near the tail.
    edge.append(
        seriessum_case(
            "series.edge.inv_square_N5000", "edge", "inv_square", 1, 5000, tol=1e-9
        )
    )

    # ── Out-of-envelope: taylor `tan` requested BEYOND its 12-term support ────
    # The generator silently caps coefficients at index 11, so terms>12 drops the
    # genuinely non-zero x¹³+ coefficients. Near x=±π/2 the result is materially
    # wrong. NumericSwift MUST emit `outsideEnvelope`. Oracle = TRUE tan value.
    edge.append(
        taylor_case(
            "series.edge.tan_oob_t16_x13", "edge", "tan", 1.3, 16, tol=1e9, in_env=False
        )
    )  # huge tol: value irrelevant
    edge.append(
        taylor_case(
            "series.edge.tan_oob_t20_x14", "edge", "tan", 1.4, 20, tol=1e9, in_env=False
        )
    )
    edge.append(
        taylor_case(
            "series.edge.tan_oob_t24_x15", "edge", "tan", 1.5, 24, tol=1e9, in_env=False
        )
    )
    edge.append(
        taylor_case(
            "series.edge.tan_oob_t30_x10", "edge", "tan", 1.0, 30, tol=1e9, in_env=False
        )
    )
    # In-envelope tan boundary: exactly 12 terms must NOT warn. At x=1.0 the
    # 12-term (index 0..11) truncated tan series differs from true tan by ~6e-3 —
    # legitimate truncation WITHIN the supported regime, so the case is
    # in-envelope (no diagnostic) and its tol admits that residual.
    edge.append(
        taylor_case(
            "series.edge.tan_in_t12_x10", "edge", "tan", 1.0, 12, tol=1e-2, in_env=True
        )
    )

    cases += edge
    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    oob = 0
    for c in cases:
        tiers[c["tier"]] += 1
        ie = c.get("inEnvelope", {})
        if any(v is False for v in ie.values()):
            oob += 1
    print(
        f"series: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']} "
        f"({oob} out-of-envelope)",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "series.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
