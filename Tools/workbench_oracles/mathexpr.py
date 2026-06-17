#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — MathExpr (expression eval) domain.

Evaluates expression STRINGS with variable bindings and freezes the reference
scalar (computed by Python's `eval` over the `math` module — never NumericSwift,
FP1 / FP3) as the JSON fixture
`Tests/NumericSwiftTests/Fixtures/workbench/mathexpr.json`.

This is a **single-strategy correctness** domain (WORKBENCH.md §4, "Single-strategy
domains"): the one strategy id is `eval`, the comparison scalar is the evaluated
`Double`, and the oracle is the value of the EQUIVALENT Python expression with the
SAME variable bindings. Every case is in-envelope — the expression evaluator either
parses + evaluates an expression exactly (to within floating-point tolerance) or it
does not, so there are ZERO out-of-envelope cases and the gate is a pure
correctness-vs-Python check.

## Library ↔ Python translation contract

The NumericSwift `MathExpr.eval(expr, variables:)` evaluator and Python's `eval`
agree on these once the function/operator names are mapped 1:1:

  operators : + - * /  (standard); `^` ↔ `pow`/`**`; `%` ↔ math.fmod (the library's
              `%` is `truncatingRemainder`, i.e. C fmod, NOT Python's `%`);
              unary `-`/`+`; postfix `!` ↔ math.gamma(x+1) (factorial via tgamma).
  constants : pi ↔ math.pi, e ↔ math.e.
  functions (NumericSwift name → Python):
    sin/cos/tan/asin/acos/atan        → math.*
    arcsin/arccos/arctan              → math.asin/acos/atan
    atan2(y,x)                        → math.atan2
    sinh/cosh/tanh/asinh/acosh/atanh  → math.*
    exp                               → math.exp
    log, ln                           → math.log            (natural log; SciPy convention)
    log10                             → math.log10
    log2, lg                          → math.log2
    sqrt                              → math.sqrt
    cbrt                              → math.cbrt  (Py 3.11+) / x**(1/3)
    pow(a,b)                          → math.pow
    hypot(a,b)                        → math.hypot
    abs                               → abs
    sign, sgn                         → 1/-1/0 (zero → 0.0, matching the library)
    floor/ceil/trunc                  → math.* (return float)
    round                             → away-from-zero (Foundation.round), NOT banker's
    min/max (n-ary)                   → min/max
    clamp(x,lo,hi)                    → min(max(x,lo),hi)
    lerp(a,b,t)                       → a + (b-a)*t
    rad(x) → x*pi/180 ; deg(x) → x*180/pi

`round` is the one trap: the library uses round-half-AWAY-from-zero while Python's
builtin `round` is banker's rounding. The oracle below implements the library's
semantics (`_round_away`) so the two agree, and corpus `round` cases avoid exact
.5 fractions anyway for clarity.

IEEE edge cases (NaN / ±inf, negative-real `sqrt`/`log`) store the value bit-exact
via `oracle.bits`; the JSON `value` is set to null (JSON cannot encode NaN/inf) and
the Swift decoder reconstructs the Double from `bits`. The real `MathExpr.evaluate`
contract returns NaN for a negative radicand under `sqrt`/`log`, so those cases
carry the frozen NaN bit pattern and are still in-envelope (the documented real
contract — NOT an out-of-envelope condition).

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/mathexpr.py
"""

import json
import math
import struct
import sys
from pathlib import Path

SOURCE = "python math (CPython %d.%d eval)" % sys.version_info[:2]


# ── Helpers ───────────────────────────────────────────────────────────────────


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _round_away(x: float) -> float:
    """Round half away from zero, matching Foundation.round (the library's `round`)."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _cbrt(x: float) -> float:
    return (
        math.cbrt(x) if hasattr(math, "cbrt") else math.copysign(abs(x) ** (1 / 3), x)
    )


# The `eval` namespace: names that may appear in a corpus expression after the
# library→Python rewrite. Mirrors the library's evaluator exactly (see contract
# above). `^` and `!` are pre-rewritten by the caller into pow()/_fact() calls.
EVAL_NS = {
    "__builtins__": {},
    # constants
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    # trig
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    # exp / log
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    # power / roots
    "sqrt": math.sqrt,
    "cbrt": _cbrt,
    "pow": math.pow,
    "hypot": math.hypot,
    # rounding / sign
    "abs": abs,
    "sign": _sign,
    "floor": lambda x: float(math.floor(x)),
    "ceil": lambda x: float(math.ceil(x)),
    "trunc": lambda x: float(math.trunc(x)),
    "round": _round_away,
    # min / max / interpolation
    "fmin": min,
    "fmax": max,
    "clamp": lambda x, lo, hi: min(max(x, lo), hi),
    "lerp": lambda a, b, t: a + (b - a) * t,
    # angle conversion
    "rad": lambda x: x * math.pi / 180.0,
    "deg": lambda x: x * 180.0 / math.pi,
    # operator helpers
    "_fmod": math.fmod,
    "_fact": lambda x: math.gamma(x + 1.0),
}


def py_eval(py_expr: str, bindings: dict) -> float:
    """Reference value of the rewritten Python expression with `bindings`.

    math.sqrt(-1) / math.log(-2) / math.acos(2) raise ValueError in Python, whereas
    the library's real `MathExpr.evaluate` returns NaN (its documented real-domain
    contract). Catch the domain error and return NaN so the oracle matches.
    """
    ns = dict(EVAL_NS)
    ns.update(bindings)
    try:
        return float(eval(py_expr, ns))  # noqa: S307 — controlled namespace, no builtins
    except ValueError:
        return float("nan")


def case(cid, tier, expr, py_expr, bindings, *, tol=1e-12, ieee=False):
    """Build one fixture case.

    - `expr`     : the expression string fed to NumericSwift `MathExpr.eval`.
    - `py_expr`  : the equivalent Python expression (library names rewritten:
                   `^`→pow(), `!`→_fact(), `%`→_fmod(), min/max→fmin/fmax).
    - `bindings` : variable name → value, passed identically to both evaluators.
    - `ieee`     : when True the value is NaN/±inf; JSON `value` is null and the
                   bits carry the canonical pattern.

    The bindings are emitted as parallel `inputs.varNames` / `inputs.varValues`
    arrays (see body) — the Swift suite zips them back into `[String: Double]`.
    """
    val = py_eval(py_expr, bindings)
    # Bindings are stored as two parallel arrays (`varNames` / `varValues`) rather
    # than a nested object, because the workbench `InputValue` decoder models JSON
    # scalars and arrays but not nested objects. The Swift suite zips them back
    # into the `[String: Double]` the evaluator expects.
    names = sorted(bindings)
    inputs = {
        "expr": expr,
        "varNames": names,
        "varValues": [float(bindings[k]) for k in names],
    }
    oracle_value = None if (ieee or not math.isfinite(val)) else val
    return {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": oracle_value, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": ["eval"],
        "tol": {"eval": tol},
    }


# ── Corpus ──────────────────────────────────────────────────────────────────


def build():
    cases = []

    # ── Trivial (~10): simple arithmetic, precedence, parentheses, unary minus ──
    trivial = [
        ("1 + 2 * 3", "1 + 2 * 3", {}),
        ("(1 + 2) * 3", "(1 + 2) * 3", {}),
        ("10 - 4 - 3", "10 - 4 - 3", {}),
        ("2 * 3 + 4 * 5", "2 * 3 + 4 * 5", {}),
        ("-5 + 3", "-5 + 3", {}),
        ("-(2 + 3)", "-(2 + 3)", {}),
        ("100 / 4 / 5", "100 / 4 / 5", {}),
        ("2 ^ 10", "pow(2, 10)", {}),
        ("3 + 4 * 2 - 1", "3 + 4 * 2 - 1", {}),
        ("(8 - 3) * (2 + 1)", "(8 - 3) * (2 + 1)", {}),
    ]
    for i, (expr, py, b) in enumerate(trivial):
        cases.append(case(f"mathexpr.trivial.arith_{i}", "trivial", expr, py, b))

    # ── Hard (~80): functions, nesting, chains, multi-variable, constants ──────
    hard = []

    # Single-arg transcendental functions over assorted real arguments.
    one_arg = [
        ("sin", "sin", 0.7),
        ("cos", "cos", 1.3),
        ("tan", "tan", 0.5),
        ("exp", "exp", 1.5),
        ("log", "log", 3.0),
        ("sqrt", "sqrt", 2.0),
        ("sinh", "sinh", 0.9),
        ("cosh", "cosh", 0.4),
        ("tanh", "tanh", 1.1),
        ("asin", "asin", 0.3),
        ("acos", "acos", 0.6),
        ("atan", "atan", 2.0),
        ("log10", "log10", 50.0),
        ("log2", "log2", 12.0),
        ("cbrt", "cbrt", 27.0),
        ("abs", "abs", -4.25),
        ("floor", "floor", 3.7),
        ("ceil", "ceil", 3.2),
        ("trunc", "trunc", -3.7),
        ("sign", "sign", -8.0),
        ("rad", "rad", 90.0),
        ("deg", "deg", math.pi / 3),
        ("asinh", "asinh", 1.5),
        ("atanh", "atanh", 0.4),
    ]
    for fn, py_fn, arg in one_arg:
        hard.append((f"{fn}({arg})", f"{py_fn}({arg})", {}, f"{fn}"))

    # Two-arg functions.
    two_arg = [
        ("pow", "pow", (2.0, 8.0)),
        ("hypot", "hypot", (3.0, 4.0)),
        ("atan2", "atan2", (1.0, 1.0)),
        ("min", "fmin", (5.0, 2.0)),
        ("max", "fmax", (5.0, 2.0)),
        ("pow", "pow", (9.0, 0.5)),
        ("hypot", "hypot", (5.0, 12.0)),
        ("atan2", "atan2", (-1.0, 2.0)),
    ]
    for fn, py_fn, (a, b) in two_arg:
        hard.append((f"{fn}({a}, {b})", f"{py_fn}({a}, {b})", {}, f"{fn}2"))

    # Three-arg functions.
    three_arg = [
        ("clamp", "clamp", (5.0, 0.0, 3.0)),
        ("clamp", "clamp", (-2.0, 0.0, 3.0)),
        ("clamp", "clamp", (1.5, 0.0, 3.0)),
        ("lerp", "lerp", (0.0, 10.0, 0.25)),
        ("lerp", "lerp", (2.0, 8.0, 0.5)),
        ("lerp", "lerp", (-4.0, 4.0, 0.75)),
    ]
    for fn, py_fn, (a, b, c) in three_arg:
        hard.append((f"{fn}({a}, {b}, {c})", f"{py_fn}({a}, {b}, {c})", {}, f"{fn}3"))

    # Constants in expressions.
    const_cases = [
        ("2 * pi", "2 * pi", {}),
        ("pi / 4", "pi / 4", {}),
        ("e ^ 2", "pow(e, 2)", {}),
        ("sin(pi / 6)", "sin(pi / 6)", {}),
        ("cos(pi)", "cos(pi)", {}),
        ("log(e)", "log(e)", {}),
        ("exp(1) - e", "exp(1) - e", {}),
        ("pi * 2 + e", "pi * 2 + e", {}),
    ]
    for expr, py, b in const_cases:
        hard.append((expr, py, b, "const"))

    # Nested calls.
    nested = [
        ("sin(cos(0.5))", "sin(cos(0.5))", {}),
        ("sqrt(exp(2))", "sqrt(exp(2))", {}),
        ("log(exp(3.5))", "log(exp(3.5))", {}),
        ("abs(sin(2) - cos(2))", "abs(sin(2) - cos(2))", {}),
        ("exp(log(5) + log(2))", "exp(log(5) + log(2))", {}),
        ("sqrt(pow(3, 2) + pow(4, 2))", "sqrt(pow(3, 2) + pow(4, 2))", {}),
        ("max(sin(1), cos(1))", "fmax(sin(1), cos(1))", {}),
        ("clamp(sqrt(50), 0, 7)", "clamp(sqrt(50), 0, 7)", {}),
        ("hypot(sin(1), cos(1))", "hypot(sin(1), cos(1))", {}),
        ("floor(exp(2)) + ceil(log(10))", "floor(exp(2)) + ceil(log(10))", {}),
    ]
    for expr, py, b in nested:
        hard.append((expr, py, b, "nested"))

    # Long operator chains and mixed precedence.
    chains = [
        ("1 + 2 * 3 - 4 / 2 + 5 * 6 - 7", "1 + 2 * 3 - 4 / 2 + 5 * 6 - 7", {}),
        ("2 ^ 3 ^ 1 + 1", "pow(2, pow(3, 1)) + 1", {}),
        ("(1 + 2) * (3 + 4) / (5 - 2)", "(1 + 2) * (3 + 4) / (5 - 2)", {}),
        ("10 % 3 + 2", "_fmod(10, 3) + 2", {}),
        ("17 % 5", "_fmod(17, 5)", {}),
        ("2 * 3 + 4 * 5 + 6 * 7 + 8 * 9", "2 * 3 + 4 * 5 + 6 * 7 + 8 * 9", {}),
        ("100 / 5 / 2 / 2", "100 / 5 / 2 / 2", {}),
        ("-2 ^ 2 + 3", "-pow(2, 2) + 3", {}),
        ("4 !", "_fact(4)", {}),
        ("3 ! + 2 !", "_fact(3) + _fact(2)", {}),
    ]
    for expr, py, b in chains:
        hard.append((expr, py, b, "chain"))

    # Multi-variable expressions (variable substitution).
    multivar = [
        ("x + y", "x + y", {"x": 3.0, "y": 4.0}),
        ("x * y - z", "x * y - z", {"x": 2.0, "y": 5.0, "z": 3.0}),
        ("a ^ 2 + b ^ 2", "pow(a, 2) + pow(b, 2)", {"a": 3.0, "b": 4.0}),
        ("sin(theta) * r", "sin(theta) * r", {"theta": 0.6, "r": 10.0}),
        ("(x + y) / (x - y)", "(x + y) / (x - y)", {"x": 7.0, "y": 2.0}),
        ("sqrt(x * x + y * y)", "sqrt(x * x + y * y)", {"x": 5.0, "y": 12.0}),
        ("exp(-k * t)", "exp(-k * t)", {"k": 0.5, "t": 2.0}),
        ("m * x + b", "m * x + b", {"m": 2.5, "x": 4.0, "b": 1.0}),
        ("log(p / q)", "log(p / q)", {"p": 8.0, "q": 2.0}),
        ("clamp(v, lo, hi)", "clamp(v, lo, hi)", {"v": 12.0, "lo": 0.0, "hi": 10.0}),
        ("hypot(dx, dy)", "hypot(dx, dy)", {"dx": 8.0, "dy": 15.0}),
        (
            "a * sin(w * t + phi)",
            "a * sin(w * t + phi)",
            {"a": 3.0, "w": 2.0, "t": 1.5, "phi": 0.3},
        ),
        ("n * (n + 1) / 2", "n * (n + 1) / 2", {"n": 100.0}),
        ("x ^ 3 - 2 * x + 1", "pow(x, 3) - 2 * x + 1", {"x": 1.7}),
        ("pow(base, exponent)", "pow(base, exponent)", {"base": 1.5, "exponent": 6.0}),
        ("atan2(y, x)", "atan2(y, x)", {"y": 3.0, "x": -4.0}),
        (
            "lerp(start, stop, frac)",
            "lerp(start, stop, frac)",
            {"start": 10.0, "stop": 50.0, "frac": 0.3},
        ),
        ("(a + b + c) / 3", "(a + b + c) / 3", {"a": 4.0, "b": 7.0, "c": 13.0}),
    ]
    for i, (expr, py, b) in enumerate(multivar):
        hard.append((expr, py, b, f"var{i}"))

    # Trim to ~80 hard (proportion enforcement, ±a few). Tolerance: transcendental
    # / chained expressions get a looser 1e-9; the rest 1e-12.
    hard = hard[:80]
    transc = {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
        "log10",
        "log2",
        "cbrt",
        "rad",
        "deg",
        "asinh",
        "atanh",
        "atan2",
        "pow",
        "hypot",
        "const",
        "nested",
        "chain",
        "lerp3",
    }
    for i, (expr, py, b, label) in enumerate(hard):
        tol = 1e-9 if label in transc or label.startswith("var") else 1e-12
        cases.append(case(f"mathexpr.hard.{label}_{i}", "hard", expr, py, b, tol=tol))

    # ── Edge (~10): deep nesting, extreme magnitudes, IEEE edges ──────────────
    # Deep nesting + very large / very small finite magnitudes (within domain).
    cases.append(
        case(
            "mathexpr.edge.deep_nest_0",
            "edge",
            "sin(cos(sin(cos(sin(0.5)))))",
            "sin(cos(sin(cos(sin(0.5)))))",
            {},
            tol=1e-9,
        )
    )
    cases.append(
        case(
            "mathexpr.edge.deep_paren_1",
            "edge",
            "((((1 + 2) * 3) - 4) / 5)",
            "((((1 + 2) * 3) - 4) / 5)",
            {},
            tol=1e-12,
        )
    )
    cases.append(
        case(
            "mathexpr.edge.large_mag_2",
            "edge",
            "exp(700)",
            "exp(700)",
            {},
            tol=1e-9 * 1e304,
        )
    )  # ~1e304; relative-scale tol
    cases.append(
        case(
            "mathexpr.edge.small_mag_3",
            "edge",
            "exp(-700)",
            "exp(-700)",
            {},
            tol=1e-310,
        )
    )
    cases.append(
        case(
            "mathexpr.edge.big_pow_4",
            "edge",
            "10 ^ 308",
            "pow(10, 308)",
            {},
            tol=1e-9 * 1e308,
        )
    )
    cases.append(
        case(
            "mathexpr.edge.tiny_diff_5", "edge", "1e-15 + 1", "1e-15 + 1", {}, tol=1e-16
        )
    )

    # IEEE edge cases: NaN / +inf. value null, bits authoritative (allow_nan=False
    # in the JSON dump, so a non-finite value would raise — hence value=None).
    cases.append(
        case(
            "mathexpr.edge.sqrt_neg_nan_6",
            "edge",
            "sqrt(-1)",
            "sqrt(-1)",
            {},
            ieee=True,
        )
    )  # real contract → NaN
    cases.append(
        case("mathexpr.edge.log_neg_nan_7", "edge", "log(-2)", "log(-2)", {}, ieee=True)
    )  # real contract → NaN
    cases.append(
        case(
            "mathexpr.edge.pos_inf_8",
            "edge",
            "inf",
            "inf",
            {},
            ieee=True,
        )
    )  # +inf constant
    cases.append(
        case(
            "mathexpr.edge.acos_domain_nan_9",
            "edge",
            "acos(2)",
            "acos(2)",
            {},
            ieee=True,
        )
    )  # acos out of [-1,1] → NaN

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"mathexpr: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']} "
        f"(0 out-of-envelope)",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mathexpr.json"
    out.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
