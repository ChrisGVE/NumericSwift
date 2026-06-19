#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Complex domain.

Computes bit-exact reference values for Complex arithmetic + transcendental
functions with Python's `cmath` / native complex arithmetic and freezes them as
the JSON fixture `Tests/NumericSwiftTests/Fixtures/workbench/complex.json`.

This domain is **single-strategy-per-component correctness** (WORKBENCH.md §4,
"Single-strategy domains"). A Complex result has TWO components, so each
operation contributes TWO strategy ids — the real part suffixed `_re` and the
imaginary part suffixed `_im` (e.g. `cexp_re`, `cexp_im`). The comparison scalar
for a `*_re` strategy is `result.real`; for `*_im` it is `result.imag`. The
oracle is Python `cmath` (split into `.real` / `.imag`), NEVER NumericSwift
(FP1 / FP3 — non-vacuous because the oracle is cmath).

Branch conventions: NumericSwift follows the numpy/SciPy **principal** branch
(C99 Annex G), which is exactly what `cmath` uses — `sqrt(-1) = +1j`,
`log(-1) = +pi*1j`. So cmath is a faithful oracle for the library's documented
conventions (CLAUDE.md design-philosophy #1, Known Limitations §2 note that
`sqrt(-1) = +i`).

Complex arithmetic + transcendentals are exact-up-to-rounding correctness, so
there is NO documented limitation envelope — EVERY case is in-envelope and there
are ZERO out-of-envelope cases. The gate is a pure correctness-vs-cmath check;
every strategy is expected to emit NO diagnostic, and the Swift suite returns
empty diagnostics accordingly.

Strategy ids ↔ Sources/NumericSwift/Complex.swift:

  add_re / add_im     → lhs + rhs            (Complex + Complex)
  sub_re / sub_im     → lhs - rhs
  mul_re / mul_im     → lhs * rhs            (Smith-protected operator)
  div_re / div_im     → lhs / rhs            (Smith's algorithm)
  abs                 → z.abs                (real scalar — single id)
  arg                 → z.arg                (real scalar — single id)
  cexp_re / cexp_im   → z.exp / cexp(z)
  clog_re / clog_im   → z.log / clog(z)
  csqrt_re / csqrt_im → z.sqrt / csqrt(z)    (incl. negative-real radicand)
  csin_re / csin_im   → z.sin
  ccos_re / ccos_im   → z.cos
  ctan_re / ctan_im   → z.tan
  csinh_re / csinh_im → z.sinh
  ccosh_re / ccosh_im → z.cosh
  ctanh_re / ctanh_im → z.tanh
  cpow_re / cpow_im   → cpow(z, w)           (complex base ^ complex exponent)

Inputs carry the complex operand(s) as plain real/imag doubles:
`z_re`, `z_im` for unary ops; additionally `w_re`, `w_im` for binary ops
(`add`/`sub`/`mul`/`div`/`cpow`). IEEE inf/nan results are frozen bit-exact via
`oracle.bits`; `oracle.value` is JSON-null for non-finite values
(`allow_nan=False`), the Swift decoder reconstructs from `bits` regardless.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/complex.py
"""

import cmath
import json
import math
import struct
import sys
from pathlib import Path

SOURCE = f"python cmath {sys.version_info.major}.{sys.version_info.minor}"


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def json_value(v: float):
    """JSON-safe value: finite floats pass through; non-finite become null.

    The canonical value always lives in `bits`; `value` is human-readability
    only, and JSON cannot represent NaN/Inf with `allow_nan=False`.
    """
    return v if math.isfinite(v) else None


# ── Component oracle ─────────────────────────────────────────────────────────
# Each (op, operands) → the complex (or real) reference value from cmath.


def op_complex(op: str, z: complex, w: complex) -> complex:
    """Reference complex result for a complex-valued op."""
    if op == "add":
        return z + w
    if op == "sub":
        return z - w
    if op == "mul":
        return z * w
    if op == "div":
        return z / w
    if op == "cexp":
        return cmath.exp(z)
    if op == "clog":
        # cmath.log(0) raises; the C99 / NumericSwift principal value is
        # (-inf, arg(z)). Mirror the library's log = (log|z|, arg) directly so
        # the zero radicand is a well-defined non-finite oracle.
        return complex(
            math.log(abs(z)) if abs(z) != 0.0 else float("-inf"), cmath.phase(z)
        )
    if op == "csqrt":
        return cmath.sqrt(z)
    if op == "csin":
        return cmath.sin(z)
    if op == "ccos":
        return cmath.cos(z)
    if op == "ctan":
        return cmath.tan(z)
    if op == "csinh":
        return cmath.sinh(z)
    if op == "ccosh":
        return cmath.cosh(z)
    if op == "ctanh":
        return cmath.tanh(z)
    if op == "cpow":
        return z**w
    raise ValueError(f"unknown complex op {op}")


def op_real(op: str, z: complex) -> float:
    """Reference real-scalar result for a real-valued op (abs / arg)."""
    if op == "abs":
        return abs(z)
    if op == "arg":
        return cmath.phase(z)
    raise ValueError(f"unknown real op {op}")


def oracle_entry(value: float, strategy: str, tol: float, *, inputs, cid, tier):
    return {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": json_value(value), "bits": bits_hex(value)},
        "source": SOURCE,
        "strategies": [strategy],
        "tol": {strategy: tol},
    }


# Binary ops take z and w; unary ops take z only.
BINARY_OPS = {"add", "sub", "mul", "div", "cpow"}
COMPLEX_OPS = [
    "add",
    "sub",
    "mul",
    "div",
    "cexp",
    "clog",
    "csqrt",
    "csin",
    "ccos",
    "ctan",
    "csinh",
    "ccosh",
    "ctanh",
    "cpow",
]
REAL_OPS = ["abs", "arg"]


def make_cases(op, z, w, *, tier, name, tol_re, tol_im):
    """Emit the `_re` + `_im` (or single real) cases for one op on (z, w)."""
    inputs = {"z_re": z.real, "z_im": z.imag}
    if op in BINARY_OPS:
        inputs["w_re"] = w.real
        inputs["w_im"] = w.imag

    out = []
    if op in REAL_OPS:
        val = op_real(op, z)
        out.append(
            oracle_entry(
                val,
                op,
                tol_re,
                inputs=inputs,
                cid=f"complex.{tier}.{op}_{name}",
                tier=tier,
            )
        )
        return out

    res = op_complex(op, z, w)
    out.append(
        oracle_entry(
            res.real,
            f"{op}_re",
            tol_re,
            inputs=inputs,
            cid=f"complex.{tier}.{op}_re_{name}",
            tier=tier,
        )
    )
    out.append(
        oracle_entry(
            res.imag,
            f"{op}_im",
            tol_im,
            inputs=inputs,
            cid=f"complex.{tier}.{op}_im_{name}",
            tier=tier,
        )
    )
    return out


def build():
    cases = []

    # Per-component tolerances. Arithmetic (add/sub/mul/div) is exact up to a few
    # ULP → very tight. Transcendentals carry a couple more ULP from the
    # underlying libm calls. Edge cases near branch cuts / large magnitude get a
    # looser bound. These mirror integration.py's per-tier tuning so in-envelope
    # cases PASS.
    ARITH = 1e-12
    TRANS = 1e-12
    EDGE = 1e-9

    # ── Trivial (~10): textbook smoke values, hand-checkable ──────────────────
    trivial = [
        ("add", complex(1, 2), complex(3, 4), "1p2i_3p4i"),
        ("sub", complex(5, 6), complex(2, 1), "5p6i_2p1i"),
        ("mul", complex(1, 0), complex(0, 1), "one_i"),  # i
        ("div", complex(1, 0), complex(0, 1), "one_over_i"),  # -i
        ("abs", complex(3, 4), None, "3p4i"),  # 5
        ("arg", complex(0, 1), None, "i"),  # pi/2
        ("cexp", complex(0, 0), None, "zero"),  # 1
        ("clog", complex(1, 0), None, "one"),  # 0
        ("csqrt", complex(4, 0), None, "four"),  # 2
        ("csin", complex(0, 0), None, "zero"),  # 0
    ]
    for op, z, w, name in trivial:
        cases.extend(
            make_cases(
                op,
                z,
                w,
                tier="trivial",
                name=name,
                tol_re=ARITH if op in ("add", "sub", "mul", "div") else 1e-13,
                tol_im=ARITH if op in ("add", "sub", "mul", "div") else 1e-13,
            )
        )

    # ── Hard (~80): realistic operands across all ops ─────────────────────────
    # A deterministic spread of finite operands sweeping all four quadrants,
    # several magnitudes, and irrational components — the bulk of coverage.
    operands = [
        complex(1.5, -2.5),
        complex(-3.25, 0.75),
        complex(0.5, 0.5),
        complex(-1.0, -1.0),
        complex(2.0, 3.0),
        complex(-0.3, 1.7),
        complex(4.2, -0.1),
        complex(0.001, 0.002),
        complex(-7.5, 8.25),
        complex(math.pi, math.e),
        complex(1.0, 0.0),
        complex(0.0, 2.0),
        complex(-2.0, 0.0),  # negative real — exercises branch cuts in log/sqrt
        complex(10.0, -10.0),
    ]
    second = complex(0.7, -1.3)  # fixed second operand for binary ops

    # Emit hard cases until the EMITTED count reaches ~80 (each COMPLEX op emits
    # 2 cases — `_re` + `_im`; each REAL op emits 1). We count emitted cases, not
    # loop iterations, so the 10/80/10 proportion holds exactly.
    emitted = 0
    target = 80
    idx = 0
    for z in operands:
        for op in COMPLEX_OPS + REAL_OPS:
            if emitted >= target:
                break
            # cpow with a large/negative magnitude base and the irrational
            # exponent can grow fast but stays finite for these operands.
            tol_re = TRANS
            tol_im = TRANS
            if op in ("add", "sub", "mul", "div"):
                tol_re = tol_im = ARITH
            if op == "cpow":
                tol_re = tol_im = 1e-9  # exp(w*log z) compounds rounding
            new = make_cases(
                op, z, second, tier="hard", name=f"q{idx}", tol_re=tol_re, tol_im=tol_im
            )
            cases.extend(new)
            emitted += len(new)
            idx += 1
        if emitted >= target:
            break

    # ── Edge (~10): branch cuts, ±0 imaginary, inf/nan, large magnitude ───────
    # All remain well-defined principal-branch values vs cmath (in-envelope, no
    # diagnostic). Branch-cut sign behaviour on the negative real axis and signed
    # zero is exactly where cmath and the C99 library agree.
    # Inputs are kept finite and JSON-representable (the `InputValue` decoder has
    # no non-finite scalar form); the NON-finite IEEE behaviour is exercised on
    # the OUTPUT side — `log(0) = (-inf, 0)` is the principal-value limit, frozen
    # bit-exact (`value` becomes JSON-null, the Swift decoder rebuilds from
    # `bits`). The upper/lower branch-cut sign (positive vs tiny-negative
    # imaginary) and the `sqrt(-1)=+i` convention (Known Limitations §2) are the
    # substantive edges.
    edge = [
        ("clog", complex(-1.0, 0.0), None, "log_neg_real"),  # +pi*i (upper branch)
        (
            "csqrt",
            complex(-1.0, 0.0),
            None,
            "sqrt_neg_one",
        ),  # +1j (Known Limitations §2)
        # Lower-branch cut: a tiny NEGATIVE imaginary part puts us just below the
        # negative-real axis → -iπ / -2j. (A literal -0.0 input cannot be used:
        # the workbench `InputValue` JSON decoder coerces -0.0 to Int 0, dropping
        # the sign — a shared-harness limitation, not a library defect. The
        # representable -1e-300 lands on the same lower branch and round-trips.)
        (
            "clog",
            complex(-1.0, -1e-300),
            None,
            "log_neg_real_lower",
        ),  # lower side → -pi*i
        ("csqrt", complex(-4.0, -1e-300), None, "sqrt_neg_lower"),  # lower side → -2j
        ("clog", complex(0.0, 0.0), None, "log_zero"),  # (-inf, 0) — non-finite output
        (
            "ctanh",
            complex(0.0, math.pi / 4),
            None,
            "tanh_imag",
        ),  # purely imaginary argument
        (
            "csin",
            complex(0.0, 20.0),
            None,
            "sin_large_imag",
        ),  # large |im| → big magnitude
        ("abs", complex(-1e150, 1e150), None, "abs_large_mag"),  # ~1.4e150, finite
        (
            "div",
            complex(1.0, 0.0),
            complex(1e-300, 0.0),
            "div_tiny",
        ),  # large but finite
        (
            "cpow",
            complex(0.0, 1.0),
            complex(0.0, 1.0),
            "i_pow_i",
        ),  # i^i = e^{-pi/2} real
    ]
    for op, z, w, name in edge:
        cases.extend(
            make_cases(op, z, w, tier="edge", name=name, tol_re=EDGE, tol_im=EDGE)
        )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"complex: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']} "
        f"(0 out-of-envelope)",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "complex.json"
    # allow_nan=False: NaN/Inf cannot appear as JSON `value` — they live in `bits`.
    out.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
