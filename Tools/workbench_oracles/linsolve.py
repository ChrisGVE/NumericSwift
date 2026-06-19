#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — LinAlg linear-system solvers.

Computes bit-exact reference solutions with numpy/scipy and freezes them as the
JSON fixture `Tests/NumericSwiftTests/Fixtures/workbench/linsolve.json`.

Mirrors the reference generator `integration.py`. Contract (WORKBENCH.md §2/§3/§5):

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); actual counts are
    printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value plus a
    `source` citation. Oracle values come ONLY from numpy/scipy — never from
    NumericSwift (FP1, FP3 vacuous-gate rule).
  * `inEnvelope` is per-strategy. Out-of-envelope cases (ill-conditioned /
    near-singular A; non-SPD input for choSolve) are tagged `false`, so the gate
    requires NumericSwift to emit an `outsideEnvelope` diagnostic for them.

## Comparison scalar

The strategies return a solution vector x; the workbench compares a single scalar.
We use **x[0]** — the first solution component — a deterministic functional of x.
The Swift suite (`Domains/LinSolve.swift`) reconstructs the matrix from the flat
row-major `inputs` arrays, runs the diagnostic-bearing solver, and reports x[0].

## Input encoding

  * solve / solveTriangular : square A as flat row-major `A`, dim `n`, RHS `b`.
    solveTriangular cases also carry `lower` (bool).
  * lstsq                   : A as flat row-major `A`, dims `rows`/`cols`, RHS `b`.
  * choSolve                : the Cholesky factor `L` (flat row-major), dim `n`,
    RHS `b`. The implied system matrix is A = L·Lᵀ; the oracle solves that system.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/linsolve.py
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as sla

SOURCE = (
    f"numpy.linalg.solve/lstsq {np.__version__}; "
    f"scipy.linalg.cho_solve/solve_triangular {scipy.__version__}"
)

# Condition-number envelope shared with Swift `LinAlg.solveConditionEnvelope`.
COND_ENVELOPE = 1e12


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def flat(M):
    """Row-major flat list of a 2-D numpy array."""
    return [float(v) for v in np.asarray(M, dtype=float).reshape(-1)]


def vec(v):
    return [float(x) for x in np.asarray(v, dtype=float).reshape(-1)]


# ── Oracle solvers (numpy / scipy ONLY) ──────────────────────────────────────


def oracle_solve(A, b):
    return float(np.linalg.solve(np.asarray(A, float), np.asarray(b, float))[0])


def oracle_lstsq(A, b):
    x, *_ = np.linalg.lstsq(np.asarray(A, float), np.asarray(b, float), rcond=None)
    return float(x[0])


def oracle_cho(L, b):
    # The Swift `choSolve(L, b)` solves the system A = L·Lᵀ given factor L.
    A = np.asarray(L, float) @ np.asarray(L, float).T
    b = np.asarray(b, float)
    try:
        c, low = sla.cho_factor(A, lower=True)
        return float(sla.cho_solve((c, low), b)[0])
    except sla.LinAlgError:
        # Non-SPD / singular A (an out-of-envelope case): cho_factor is undefined.
        # The gate for these cases is the DIAGNOSTIC, not the value (accuracy is
        # not scored for inEnvelope:false). Provide a least-squares reference so
        # the fixture still carries a finite, citable oracle value.
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(x[0])


def oracle_triangular(A, b, lower):
    return float(
        sla.solve_triangular(np.asarray(A, float), np.asarray(b, float), lower=lower)[0]
    )


# ── Case builders ────────────────────────────────────────────────────────────


def case_solve(cid, tier, A, b, tol, in_env=None):
    n = len(A)
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": {"A": flat(A), "n": n, "b": vec(b)},
        "oracle": _ov(oracle_solve(A, b)),
        "source": SOURCE,
        "strategies": ["solve"],
        "tol": {"solve": tol},
    }
    if in_env is not None:
        entry["inEnvelope"] = {"solve": in_env}
    return entry


def case_lstsq(cid, tier, A, b, tol, in_env=None):
    A = np.asarray(A, float)
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": {"A": flat(A), "rows": A.shape[0], "cols": A.shape[1], "b": vec(b)},
        "oracle": _ov(oracle_lstsq(A, b)),
        "source": SOURCE,
        "strategies": ["lstsq"],
        "tol": {"lstsq": tol},
    }
    if in_env is not None:
        entry["inEnvelope"] = {"lstsq": in_env}
    return entry


def case_cho(cid, tier, L, b, tol, in_env=None):
    n = len(L)
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": {"L": flat(L), "n": n, "b": vec(b)},
        "oracle": _ov(oracle_cho(L, b)),
        "source": SOURCE,
        "strategies": ["choSolve"],
        "tol": {"choSolve": tol},
    }
    if in_env is not None:
        entry["inEnvelope"] = {"choSolve": in_env}
    return entry


def case_tri(cid, tier, A, b, lower, tol, in_env=None):
    n = len(A)
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": {"A": flat(A), "n": n, "b": vec(b), "lower": bool(lower)},
        "oracle": _ov(oracle_triangular(A, b, lower)),
        "source": SOURCE,
        "strategies": ["solveTriangular"],
        "tol": {"solveTriangular": tol},
    }
    if in_env is not None:
        entry["inEnvelope"] = {"solveTriangular": in_env}
    return entry


def _ov(v):
    return {"value": float(v), "bits": bits_hex(v)}


# ── Random helpers (seeded — deterministic fixtures) ──────────────────────────

RNG = np.random.default_rng(20260617)


def well_conditioned(n):
    """A random square matrix with a controlled, modest condition number."""
    while True:
        A = RNG.standard_normal((n, n))
        if np.linalg.cond(A) < 1e3:
            return A


def random_lower(n, diag_lo=0.5, diag_hi=2.0):
    """A random lower-triangular Cholesky factor with positive diagonal."""
    L = np.tril(RNG.standard_normal((n, n)))
    for i in range(n):
        L[i, i] = RNG.uniform(diag_lo, diag_hi)
    return L


def hilbert(n):
    return sla.hilbert(n)


def build():
    cases = []

    # ── Trivial (~10): tiny textbook systems, well-conditioned ────────────────
    cases.append(
        case_solve(
            "linsolve.trivial.identity2",
            "trivial",
            [[1.0, 0.0], [0.0, 1.0]],
            [3.0, 5.0],
            1e-12,
        )
    )
    cases.append(
        case_solve(
            "linsolve.trivial.diag2",
            "trivial",
            [[2.0, 0.0], [0.0, 4.0]],
            [6.0, 8.0],
            1e-12,
        )
    )
    cases.append(
        case_solve(
            "linsolve.trivial.simple2",
            "trivial",
            [[3.0, 2.0], [1.0, 2.0]],
            [7.0, 5.0],
            1e-10,
        )
    )
    cases.append(
        case_solve(
            "linsolve.trivial.simple3",
            "trivial",
            [[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]],
            [4.0, 5.0, 6.0],
            1e-10,
        )
    )
    cases.append(
        case_lstsq(
            "linsolve.trivial.overdet",
            "trivial",
            [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
            [6.0, 5.0, 7.0],
            1e-10,
        )
    )
    cases.append(
        case_lstsq(
            "linsolve.trivial.square_ls",
            "trivial",
            [[2.0, 0.0], [0.0, 2.0]],
            [4.0, 6.0],
            1e-10,
        )
    )
    cases.append(
        case_cho(
            "linsolve.trivial.chol_eye",
            "trivial",
            [[1.0, 0.0], [0.0, 1.0]],
            [2.0, 3.0],
            1e-10,
        )
    )
    cases.append(
        case_cho(
            "linsolve.trivial.chol_diag",
            "trivial",
            [[2.0, 0.0], [0.0, 3.0]],
            [4.0, 9.0],
            1e-9,
        )
    )
    cases.append(
        case_tri(
            "linsolve.trivial.lower2",
            "trivial",
            [[2.0, 0.0], [1.0, 3.0]],
            [4.0, 5.0],
            True,
            1e-10,
        )
    )
    cases.append(
        case_tri(
            "linsolve.trivial.upper2",
            "trivial",
            [[2.0, 1.0], [0.0, 3.0]],
            [5.0, 9.0],
            False,
            1e-10,
        )
    )

    # ── Hard (~80): realistic well-conditioned systems across all 4 solvers ───
    # solve: 24 random well-conditioned square systems of growing order.
    for i in range(24):
        n = [3, 4, 5, 6, 8, 10][i % 6]
        A = well_conditioned(n)
        b = RNG.standard_normal(n)
        cases.append(case_solve(f"linsolve.hard.solve_{i}", "hard", A, b, 1e-7))

    # lstsq: 20 overdetermined / square full-rank systems.
    for i in range(20):
        m = [5, 6, 8, 10][i % 4]
        n = [2, 3, 4][i % 3]
        A = RNG.standard_normal((m, n))
        b = RNG.standard_normal(m)
        cases.append(case_lstsq(f"linsolve.hard.lstsq_{i}", "hard", A, b, 1e-7))

    # choSolve: 18 well-conditioned SPD systems via random Cholesky factors.
    for i in range(18):
        n = [3, 4, 5, 6][i % 4]
        L = random_lower(n)
        b = RNG.standard_normal(n)
        cases.append(case_cho(f"linsolve.hard.cho_{i}", "hard", L, b, 1e-7))

    # solveTriangular: 18 well-conditioned triangular systems, mixed lower/upper.
    for i in range(18):
        n = [3, 4, 5, 6][i % 4]
        lower = i % 2 == 0
        L = random_lower(n)
        A = L if lower else L.T
        b = RNG.standard_normal(n)
        cases.append(case_tri(f"linsolve.hard.tri_{i}", "hard", A, b, lower, 1e-7))

    # ── Edge (~10): out-of-envelope ill-conditioning + in-envelope guards ─────
    # OUT-OF-ENVELOPE: cond(A) >> COND_ENVELOPE → library MUST emit a diagnostic.
    # Hilbert matrices have cond growing super-exponentially with order; H6 ≈ 1.5e7,
    # H8 ≈ 1.5e10, H10 ≈ 1.6e13, H12 ≈ 1.7e16 — orders 10/12 clear the 1e12 envelope.
    for n in (10, 12):
        H = hilbert(n)
        b = np.ones(n)
        cases.append(
            case_solve(
                f"linsolve.edge.hilbert{n}_solve", "edge", H, b, 1e-2, in_env=False
            )
        )
    # lstsq on an ill-conditioned tall matrix: a 20×10 vertical stack of two
    # Hilbert(10) blocks — cond ≈ 1.6e13, clearing the 1e12 envelope.
    A_ill = np.vstack([hilbert(10), hilbert(10)])
    cases.append(
        case_lstsq(
            "linsolve.edge.hilbert_lstsq",
            "edge",
            A_ill,
            np.ones(20),
            1e-2,
            in_env=False,
        )
    )
    # choSolve OUT-OF-ENVELOPE: non-SPD input. A singular Cholesky factor (zeroed
    # last diagonal) makes A = L·Lᵀ rank-deficient → not positive-DEFINITE.
    L_sing = random_lower(5)
    L_sing[4, 4] = 0.0  # singular → A = L·Lᵀ is only PSD, not PD
    cases.append(
        case_cho(
            "linsolve.edge.cho_nonspd",
            "edge",
            L_sing,
            RNG.standard_normal(5),
            1e-2,
            in_env=False,
        )
    )
    # choSolve OUT-OF-ENVELOPE: ill-conditioned SPD (tiny diagonal entry).
    L_ill = random_lower(5)
    L_ill[4, 4] = 1e-7  # A = L·Lᵀ has cond ~ 1e14
    cases.append(
        case_cho(
            "linsolve.edge.cho_illcond",
            "edge",
            L_ill,
            RNG.standard_normal(5),
            1e-2,
            in_env=False,
        )
    )
    # solveTriangular OUT-OF-ENVELOPE: near-singular diagonal → huge cond.
    T_ill = random_lower(5)
    T_ill[4, 4] = 1e-13
    cases.append(
        case_tri(
            "linsolve.edge.tri_illcond",
            "edge",
            T_ill,
            RNG.standard_normal(5),
            True,
            1e0,
            in_env=False,
        )
    )

    # IN-ENVELOPE guards: well-conditioned edge-ish cases must NOT warn.
    A_ok = well_conditioned(6)
    cases.append(
        case_solve("linsolve.edge.solve_ok", "edge", A_ok, RNG.standard_normal(6), 1e-6)
    )
    cases.append(
        case_lstsq(
            "linsolve.edge.lstsq_ok",
            "edge",
            RNG.standard_normal((10, 4)),
            RNG.standard_normal(10),
            1e-6,
        )
    )
    L_ok = random_lower(6)
    cases.append(
        case_cho("linsolve.edge.cho_ok", "edge", L_ok, RNG.standard_normal(6), 1e-6)
    )
    cases.append(
        case_tri(
            "linsolve.edge.tri_ok",
            "edge",
            random_lower(6),
            RNG.standard_normal(6),
            True,
            1e-6,
        )
    )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"linsolve: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "linsolve.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
