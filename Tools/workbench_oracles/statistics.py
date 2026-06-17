#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Statistics domain.

Computes bit-exact reference descriptive statistics with numpy / scipy and
freezes them as the JSON fixture
`Tests/NumericSwiftTests/Fixtures/workbench/statistics.json`.

This domain is **single-strategy-per-function correctness** (WORKBENCH.md §4,
"Single-strategy domains"): each statistic IS a strategy id, the comparison
scalar is that statistic's output over an input array, and the oracle is the
matching numpy/scipy reference. The statistics covered here are all exact /
closed-form, so EVERY case is in-envelope — there are ZERO out-of-envelope
cases and the gate is a pure correctness-vs-numpy check (non-vacuous: the
oracle is numpy, never NumericSwift; FP1 / FP3).

Strategy ids ↔ Stats.* (Sources/NumericSwift/Statistics.swift):

  mean        → np.mean
  median      → np.median
  variance    → np.var(ddof=)           (input `ddof`)
  stddev      → np.std(ddof=)           (input `ddof`)
  percentile  → np.percentile(q, linear) (input `q`, 0..100)
  gmean       → scipy.stats.gmean
  hmean       → scipy.stats.hmean
  mode        → scipy.stats.mode (smallest on tie)

Each case carries its data array under `inputs.data` plus any params
(`ddof`, `q`). The Swift suite reads these and calls the matching `Stats.*`.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/statistics.py
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SOURCE = f"numpy {np.__version__} / scipy.stats {__import__('scipy').__version__}"


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


# ── Oracle dispatch ──────────────────────────────────────────────────────────
# Each statistic's reference value, computed ONLY from numpy/scipy.


def oracle(strategy: str, data, ddof: int, q: float) -> float:
    arr = np.asarray(data, dtype=float)
    if strategy == "mean":
        return float(np.mean(arr))
    if strategy == "median":
        return float(np.median(arr))
    if strategy == "variance":
        return float(np.var(arr, ddof=ddof))
    if strategy == "stddev":
        return float(np.std(arr, ddof=ddof))
    if strategy == "percentile":
        return float(np.percentile(arr, q))
    if strategy == "gmean":
        return float(stats.gmean(arr))
    if strategy == "hmean":
        return float(stats.hmean(arr))
    if strategy == "mode":
        return float(stats.mode(arr, keepdims=False).mode)
    raise ValueError(f"unknown strategy {strategy}")


def case(cid, tier, strategy, data, *, ddof=0, q=50.0, tol=1e-12):
    """Build one fixture case for a single statistic over `data`.

    Every case is in-envelope (exact/closed-form vs numpy), so `inEnvelope` is
    omitted (defaults to true in the Swift decoder).
    """
    inputs = {"data": list(map(float, data))}
    # Only attach the params the strategy actually consumes, keeping fixtures lean.
    if strategy in ("variance", "stddev"):
        inputs["ddof"] = ddof
    if strategy == "percentile":
        inputs["q"] = q
    val = oracle(strategy, data, ddof, q)
    return {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": val, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": [strategy],
        "tol": {strategy: tol},
    }


def build():
    cases = []

    # ── Trivial (~10): textbook smoke datasets, closed-form answers ───────────
    # Small integer arrays where the statistic is hand-checkable.
    trivial = [
        ("mean", [1.0, 2.0, 3.0, 4.0, 5.0], {}),
        ("median", [3.0, 1.0, 2.0, 5.0, 4.0], {}),
        ("variance", [1.0, 2.0, 3.0, 4.0, 5.0], {"ddof": 0}),
        ("stddev", [2.0, 4.0, 6.0, 8.0], {"ddof": 0}),
        ("percentile", [1.0, 2.0, 3.0, 4.0], {"q": 50.0}),
        ("gmean", [1.0, 2.0, 4.0, 8.0], {}),
        ("hmean", [1.0, 2.0, 4.0], {}),
        ("mode", [2.0, 2.0, 3.0, 4.0], {}),
        ("mean", [10.0, 20.0, 30.0], {}),
        ("median", [1.0, 2.0, 3.0, 4.0], {}),
    ]
    for i, (strat, data, kw) in enumerate(trivial):
        cases.append(
            case(f"statistics.trivial.{strat}_{i}", "trivial", strat, data, **kw)
        )

    # ── Hard (~80): realistic, varied datasets across all statistics ──────────
    # Deterministic pseudo-random datasets (fixed seeds) so the fixture is
    # reproducible; numpy is the oracle for each. We sweep dataset shapes,
    # sizes, scales, ddof values, and percentile quantiles.
    rng = np.random.default_rng(20260617)

    hard = []
    # Random normal / uniform / lognormal datasets, several sizes & scales.
    datasets = []
    for n in (7, 13, 25, 50, 100):
        datasets.append(("normal", list(rng.normal(0.0, 1.0, n))))
        datasets.append(("normal_scaled", list(rng.normal(50.0, 12.5, n))))
        datasets.append(("uniform_pos", list(rng.uniform(0.5, 20.0, n))))
        datasets.append(("lognormal", list(rng.lognormal(0.0, 0.75, n))))

    # mean / median / variance(ddof0,1) / stddev(ddof0,1) over each dataset.
    di = 0
    for label, data in datasets:
        hard.append(("mean", data, {}, f"{label}_n{len(data)}"))
        hard.append(("median", data, {}, f"{label}_n{len(data)}"))
        hard.append(("variance", data, {"ddof": 0}, f"{label}_n{len(data)}_d0"))
        hard.append(("variance", data, {"ddof": 1}, f"{label}_n{len(data)}_d1"))
        hard.append(("stddev", data, {"ddof": 1}, f"{label}_n{len(data)}_d1"))
        di += 1

    # percentile over positive datasets at several quantiles.
    pos_sets = [d for lab, d in datasets if lab in ("uniform_pos", "lognormal")]
    for k, data in enumerate(pos_sets):
        for q in (10.0, 25.0, 75.0, 90.0):
            hard.append(("percentile", data, {"q": q}, f"pos{k}_q{int(q)}"))

    # gmean / hmean over the strictly-positive datasets.
    for k, data in enumerate(pos_sets):
        hard.append(("gmean", data, {}, f"pos{k}"))
        hard.append(("hmean", data, {}, f"pos{k}"))

    # mode over integer-valued datasets with repeats (incl. ties → smallest).
    mode_sets = [
        [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0],
        [5.0, 5.0, 1.0, 1.0, 9.0],  # tie 1 vs 5 → 1 (smallest)
        list(rng.integers(0, 5, 40).astype(float)),
        list(rng.integers(-3, 4, 60).astype(float)),
        [7.0, 7.0, 7.0, 2.0, 2.0, 9.0],
    ]
    for k, data in enumerate(mode_sets):
        hard.append(("mode", data, {}, f"set{k}"))

    # Trim/pad to ~80 hard cases (proportion enforcement, ±a few).
    hard = hard[:80]
    for i, (strat, data, kw, label) in enumerate(hard):
        cases.append(
            case(f"statistics.hard.{strat}_{label}_{i}", "hard", strat, data, **kw)
        )

    # ── Edge (~10): degenerate but well-defined inputs (still exact) ──────────
    # Single element, two elements, large magnitudes, tiny magnitudes,
    # negative values, percentile endpoints. All remain closed-form vs numpy,
    # so they are in-envelope (no diagnostic expected).
    edge = [
        ("mean", [42.0], {}, "single"),
        ("median", [7.0], {}, "single"),
        ("variance", [3.0, 3.0, 3.0], {"ddof": 0}, "constant"),
        ("stddev", [1e8, 1e8 + 1.0, 1e8 + 2.0], {"ddof": 1}, "large_mag"),
        ("percentile", [1.0, 2.0, 3.0, 4.0, 5.0], {"q": 0.0}, "q0"),
        ("percentile", [1.0, 2.0, 3.0, 4.0, 5.0], {"q": 100.0}, "q100"),
        ("gmean", [1e-6, 1e-3, 1.0, 1e3], {}, "wide_scale"),
        ("hmean", [1e-3, 1.0, 1e3], {}, "wide_scale"),
        ("mode", [5.0], {}, "single"),
        ("mean", [-1e6, 1e6], {}, "cancellation"),
    ]
    for i, (strat, data, kw, label) in enumerate(edge):
        cases.append(
            case(f"statistics.edge.{strat}_{label}_{i}", "edge", strat, data, **kw)
        )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"statistics: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']} "
        f"(0 out-of-envelope)",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "statistics.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
