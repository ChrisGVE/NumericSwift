#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Spatial k-NN domain.

Computes bit-exact reference k-th-nearest-neighbour distances with scipy and
freezes them as the JSON fixture
`Tests/NumericSwiftTests/Fixtures/workbench/spatialknn.json`.

Contract (WORKBENCH.md §2/§3/§5), mirroring the reference `integration.py`:

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual
    counts are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation. Oracle values come ONLY from scipy — never from
    NumericSwift (FP3 vacuous-gate rule).
  * Comparison scalar = the distance to the k-th nearest neighbour of the query
    point: `scipy.spatial.cKDTree(points).query(q, k)` taking the k-th returned
    distance (1-indexed: k=1 is the nearest neighbour). Both Swift strategies
    (kdTree, bruteForce) are EXACT and must reproduce this to machine precision.
  * `inEnvelope` is per-strategy. Out-of-envelope cases — DEGENERATE queries that
    cannot return k valid neighbours (k > n, empty point set, or k <= 0) — are
    tagged `false` for both strategies, so the gate requires NumericSwift to emit
    an `outsideEnvelope` diagnostic for them. Their oracle scalar is NaN (the
    k-th neighbour does not exist).

Inputs per case (kept in lockstep with `spatialKNNSample` in
`Sources/NumericSwiftWorkbenchKit/Domains/SpatialKNN.swift`):
  * `points` — flat row-major coordinate array (n * dims doubles).
  * `dims`   — coordinate dimensionality d.
  * `query`  — flat query coordinates (d doubles).
  * `k`      — requested neighbour count.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/spatialknn.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SOURCE = f"scipy.spatial.cKDTree {__import__('scipy').__version__}"


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


def kth_nn_distance(points, query, k) -> float:
    """Distance to the k-th nearest neighbour (1-indexed) via scipy cKDTree.

    Returns NaN for a degenerate query (empty set, k <= 0, or k > n) — the k-th
    neighbour does not exist, which is exactly the out-of-envelope regime.
    """
    n = len(points)
    if n == 0 or k <= 0 or k > n:
        return float("nan")
    tree = cKDTree(np.asarray(points, dtype=float))
    dists, _ = tree.query(np.asarray(query, dtype=float), k=k)
    # scipy returns a scalar for k=1, a length-k array otherwise.
    dists = np.atleast_1d(dists)
    return float(dists[k - 1])


def case(cid, tier, points, dims, query, k, *, in_envelope=None):
    val = kth_nn_distance(points, query, k)
    flat = [float(c) for pt in points for c in pt]
    # The canonical oracle value is `bits` (IEEE-754, exact). The `value` field
    # is human-readable only and is ignored by the Swift decoder — but a non-finite
    # value (NaN on degenerate queries) is not valid JSON, so emit `null` there.
    value_json = val if math.isfinite(val) else None
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": {
            "points": flat,
            "dims": dims,
            "query": [float(c) for c in query],
            "k": k,
        },
        "oracle": {"value": value_json, "bits": bits_hex(val)},
        "source": SOURCE,
        "strategies": ["kdTree", "bruteForce"],
        # Both strategies are exact; the per-case tol is uniformly tight.
        "tol": {"kdTree": 1e-12, "bruteForce": 1e-12},
    }
    if in_envelope:
        entry["inEnvelope"] = in_envelope
    return entry


def grid_points(nx, ny, spacing=1.0):
    """A regular nx x ny 2-D grid — deterministic, no RNG."""
    return [[i * spacing, j * spacing] for i in range(nx) for j in range(ny)]


def build():
    cases = []
    rng = np.random.default_rng(20260617)

    # ── Trivial (~10): tiny, well-separated sets; nearest-neighbour (k small) ──
    # Closed-form-checkable: a unit grid where the nearest neighbour is the
    # adjacent grid point at distance 1, the 2nd is the diagonal at sqrt(2), etc.
    trivial_specs = [
        # (points, query, k)
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.1, 0.1], 1),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.5, 0.5], 1),
        ([[0.0, 0.0], [3.0, 4.0]], [0.0, 0.0], 1),
        ([[0.0, 0.0], [3.0, 4.0]], [0.0, 0.0], 2),
        ([[-1.0], [0.0], [2.0], [5.0]], [0.2], 1),
        ([[-1.0], [0.0], [2.0], [5.0]], [0.2], 2),
        (grid_points(3, 3), [1.0, 1.0], 1),
        (grid_points(3, 3), [1.0, 1.0], 4),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [0.0, 0.0, 0.0], 2),
        ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], [1.4, 1.6], 3),
    ]
    for i, (pts, q, k) in enumerate(trivial_specs):
        dims = len(pts[0])
        cases.append(case(f"spatialknn.trivial.{i}", "trivial", pts, dims, q, k))

    # ── Hard (~80): random point clouds, varied n / dims / k ──────────────────
    # Random clouds in 1-5 D so the k-th-neighbour distance has no special
    # structure — a realistic NN-search workload. All in-envelope (k <= n).
    idx = 0
    n_choices = [10, 25, 50, 100, 200]
    dim_choices = [1, 2, 3, 5]
    k_choices = [1, 3, 5, 10]
    for n in n_choices:
        for d in dim_choices:
            for k in k_choices:
                if idx >= 80:
                    break
                if k > n:
                    continue
                pts = rng.standard_normal((n, d)).tolist()
                q = rng.standard_normal(d).tolist()
                cases.append(case(f"spatialknn.hard.{idx}", "hard", pts, d, q, k))
                idx += 1
            if idx >= 80:
                break
        if idx >= 80:
            break
    # Top up to 80 with extra random clouds if the nested loop fell short.
    while idx < 80:
        n = int(rng.integers(8, 150))
        d = int(rng.integers(1, 6))
        k = int(rng.integers(1, min(n, 12) + 1))
        pts = rng.standard_normal((n, d)).tolist()
        q = rng.standard_normal(d).tolist()
        cases.append(case(f"spatialknn.hard.{idx}", "hard", pts, d, q, k))
        idx += 1

    # ── Edge (~10): degenerate (out-of-envelope) + boundary (in-envelope) ─────
    # Out-of-envelope: queries that cannot return k valid neighbours. The library
    # MUST emit an outsideEnvelope diagnostic; the oracle scalar is NaN.
    oov = [{"kdTree": False, "bruteForce": False}]
    # k > n
    cases.append(
        case(
            "spatialknn.edge.k_gt_n_0",
            "edge",
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            2,
            [0.5, 0.5],
            5,
            in_envelope=oov[0],
        )
    )
    cases.append(
        case(
            "spatialknn.edge.k_gt_n_1",
            "edge",
            [[0.0]],
            1,
            [10.0],
            3,
            in_envelope=oov[0],
        )
    )
    cases.append(
        case(
            "spatialknn.edge.k_eq_n_plus_1",
            "edge",
            grid_points(2, 2),
            2,
            [0.5, 0.5],
            5,
            in_envelope=oov[0],
        )
    )
    # empty point set
    cases.append(
        case(
            "spatialknn.edge.empty_0", "edge", [], 2, [0.0, 0.0], 1, in_envelope=oov[0]
        )
    )
    cases.append(
        case(
            "spatialknn.edge.empty_1",
            "edge",
            [],
            3,
            [1.0, 2.0, 3.0],
            2,
            in_envelope=oov[0],
        )
    )
    # k <= 0
    cases.append(
        case(
            "spatialknn.edge.k_zero",
            "edge",
            [[0.0, 0.0], [1.0, 1.0]],
            2,
            [0.5, 0.5],
            0,
            in_envelope=oov[0],
        )
    )
    cases.append(
        case(
            "spatialknn.edge.k_negative",
            "edge",
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            2,
            [0.5, 0.5],
            -2,
            in_envelope=oov[0],
        )
    )

    # In-envelope edge: well-posed boundary cases that must NOT warn (the
    # false-positive guard) — k exactly equal to n, single point with k=1,
    # coincident points (zero distance), and large k = n on a small set.
    cases.append(
        case("spatialknn.edge.k_eq_n", "edge", grid_points(2, 2), 2, [0.5, 0.5], 4)
    )
    cases.append(
        case("spatialknn.edge.single_point", "edge", [[3.0, 4.0]], 2, [0.0, 0.0], 1)
    )
    cases.append(
        case(
            "spatialknn.edge.coincident",
            "edge",
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            2,
            [1.0, 1.0],
            2,
        )
    )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"spatialknn: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "spatialknn.json"
    # allow_nan=False: any stray non-finite value is a generator bug — fail loudly
    # rather than emit invalid JSON. (Non-finite oracle scalars are written `null`.)
    out.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
