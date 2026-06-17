#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Cluster (clustering) domain.

Computes bit-exact reference scalars with scikit-learn and freezes them as the
JSON fixture `Tests/NumericSwiftTests/Fixtures/workbench/cluster.json`.

Mirrors the reference `integration.py` generator. Contract (WORKBENCH.md
§2/§3/§5):

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual counts
    are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation. Oracle values come ONLY from scikit-learn — never from
    NumericSwift (FP3 vacuous-gate rule).
  * `inEnvelope` is per-strategy. Out-of-envelope cases (degenerate requests:
    k > n / k <= 0 for kmeans, all-noise for dbscan, nClusters > n for
    hierarchical) are tagged `false`, so the gate requires NumericSwift to emit
    an `outsideEnvelope` diagnostic for them.

## Determinism

Clustering is otherwise non-deterministic, so each comparison scalar is chosen
to be reproducible and to genuinely discriminate a correct clustering from a
broken one (FP3 — non-vacuous gate):

  * `kmeans`           → final INERTIA (sum of squared distances to the assigned
                         centroid). The SAME fixed initial centroids are passed
                         to both NumericSwift (`Cluster.kmeans(init…)`) and
                         sklearn `KMeans(init=<array>, n_init=1, max_iter=…)`, so
                         the Lloyd trajectories — and the inertia — coincide.
  * `dbscan`           → number of clusters found at fixed `eps` / `minSamples`.
  * `hierarchical_*`   → the SIZE OF THE LARGEST CLUSTER at a fixed `nClusters`
                         cut. The cluster *count* alone is vacuous (a cut into
                         `nClusters` always yields `nClusters` groups), so we use
                         the largest-group size, which depends on the actual
                         merge structure and therefore discriminates a correct
                         linkage from a broken one.

Input encoding (carried through the JSON `inputs` bag):
    points  — flat [Double] of n*dims coordinates (row-major).
    dims    — dimensionality d (so n = len(points)/d).
    k       — number of clusters (kmeans).
    init    — flat [Double] of k*dims initial centroids (kmeans).
    eps     — neighbourhood radius (dbscan).
    minPts  — minimum points for a dense region (dbscan).
    nClusters — cut size (hierarchical).
    maxIter — kmeans Lloyd iteration cap.
    tol     — kmeans convergence tolerance.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/cluster.py
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

import sklearn

SOURCE = f"sklearn.cluster {sklearn.__version__}"

KMEANS_MAX_ITER = 100
KMEANS_TOL = 1e-4

LINKAGE_FOR = {
    "hierarchical_single": "single",
    "hierarchical_complete": "complete",
    "hierarchical_average": "average",
    "hierarchical_ward": "ward",
}


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


# ── Deterministic synthetic data generators ──────────────────────────────────


def blobs(rng, n_per, centers, spread=0.6):
    """Stack `n_per` Gaussian points around each center."""
    pts = []
    for c in centers:
        c = np.asarray(c, dtype=float)
        pts.append(rng.normal(loc=c, scale=spread, size=(n_per, len(c))))
    return np.vstack(pts)


def grid_centers(d, spacing=5.0, per_axis=None):
    """Well-separated centers on a coarse grid (d-dimensional)."""
    if d == 2:
        base = [(0, 0), (spacing, 0), (0, spacing), (spacing, spacing)]
    elif d == 3:
        base = [(0, 0, 0), (spacing, 0, 0), (0, spacing, 0), (0, 0, spacing)]
    else:
        base = [tuple([spacing * (i == j) for j in range(d)]) for i in range(d + 1)]
    return base


# ── Scalar computations (sklearn oracles) ────────────────────────────────────


def kmeans_inertia(X, init):
    """Final inertia of KMeans seeded with the EXACT `init` centroids."""
    km = KMeans(
        n_clusters=len(init),
        init=np.asarray(init, dtype=float),
        n_init=1,
        max_iter=KMEANS_MAX_ITER,
        tol=KMEANS_TOL,
        algorithm="lloyd",
    )
    km.fit(X)
    return float(km.inertia_)


def dbscan_nclusters(X, eps, min_samples):
    """Number of clusters DBSCAN finds (noise label -1 excluded)."""
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    return float(len(set(labels) - {-1}))


def hierarchical_largest(X, linkage, n_clusters):
    """Size of the largest cluster of an Agglomerative cut into `n_clusters`."""
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X)
    counts = np.bincount(ac.labels_)
    return float(counts.max())


# ── Case builder ──────────────────────────────────────────────────────────────


def make_case(cid, tier, strategy, inputs, value, in_envelope=None, tol=None):
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": float(value), "bits": bits_hex(value)},
        "source": SOURCE,
        "strategies": [strategy],
        "tol": {strategy: (tol if tol is not None else 1e-9)},
    }
    if in_envelope is not None:
        entry["inEnvelope"] = {strategy: in_envelope}
    return entry


def flat(X):
    X = np.asarray(X, dtype=float)
    return [float(v) for v in X.reshape(-1)]


def kmeans_inputs(X, init, dims):
    return {
        "points": flat(X),
        "dims": dims,
        "k": len(init),
        "init": flat(init),
        "maxIter": KMEANS_MAX_ITER,
        "tol": KMEANS_TOL,
    }


def dbscan_inputs(X, dims, eps, min_samples):
    return {
        "points": flat(X),
        "dims": dims,
        "eps": eps,
        "minPts": min_samples,
    }


def hier_inputs(X, dims, n_clusters):
    return {
        "points": flat(X),
        "dims": dims,
        "nClusters": n_clusters,
    }


# Pick the first point of each blob group as the seed centroid — a fixed,
# reproducible init that matches what we hand to both libraries.
def seed_init(X, n_per, k):
    return np.asarray([X[i * n_per] for i in range(k)], dtype=float)


def build():
    cases = []
    rng = np.random.default_rng(20260617)

    HIER = list(LINKAGE_FOR.keys())

    # ── Trivial (~10): tiny, textbook, very well separated ────────────────────
    # 4 kmeans + 2 dbscan + 4 hierarchical (one per linkage).
    triv_centers2 = grid_centers(2, spacing=8.0)
    Xt = blobs(rng, 8, triv_centers2, spread=0.3)
    dims_t = 2

    for i, k in enumerate([2, 3, 4, 4]):
        init = seed_init(Xt, 8, k)
        cases.append(
            make_case(
                f"cluster.trivial.kmeans_{i}",
                "trivial",
                "kmeans",
                kmeans_inputs(Xt, init, dims_t),
                kmeans_inertia(Xt, init),
                tol=1e-6,
            )
        )

    for i, (eps, mp) in enumerate([(1.5, 4), (2.0, 3)]):
        cases.append(
            make_case(
                f"cluster.trivial.dbscan_{i}",
                "trivial",
                "dbscan",
                dbscan_inputs(Xt, dims_t, eps, mp),
                dbscan_nclusters(Xt, eps, mp),
                tol=0.0,
            )
        )

    for strat in HIER:
        cases.append(
            make_case(
                f"cluster.trivial.{strat}",
                "trivial",
                strat,
                hier_inputs(Xt, dims_t, 4),
                hierarchical_largest(Xt, LINKAGE_FOR[strat], 4),
                tol=0.0,
            )
        )

    # ── Hard (~80): realistic blobs, varied dims / spread / k / params ────────
    hard = []

    # kmeans: many seeds × k over 2-D and 3-D blobs.
    for trial in range(20):
        d = 2 if trial % 2 == 0 else 3
        centers = grid_centers(d, spacing=rng.uniform(4.0, 7.0))
        k = rng.integers(2, len(centers) + 1)
        n_per = int(rng.integers(10, 25))
        X = blobs(rng, n_per, centers[: max(k, 2)], spread=rng.uniform(0.4, 0.9))
        init = seed_init(X, n_per, k)
        hard.append(
            make_case(
                f"cluster.hard.kmeans_{trial}",
                "hard",
                "kmeans",
                kmeans_inputs(X, init, d),
                kmeans_inertia(X, init),
                tol=1e-6,
            )
        )

    # dbscan: blobs with eps/minPts chosen to actually form clusters.
    for trial in range(16):
        d = 2 if trial % 2 == 0 else 3
        centers = grid_centers(d, spacing=6.0)
        n_per = int(rng.integers(12, 22))
        X = blobs(rng, n_per, centers, spread=rng.uniform(0.4, 0.8))
        eps = float(rng.uniform(1.2, 2.2))
        mp = int(rng.integers(3, 6))
        hard.append(
            make_case(
                f"cluster.hard.dbscan_{trial}",
                "hard",
                "dbscan",
                dbscan_inputs(X, d, eps, mp),
                dbscan_nclusters(X, eps, mp),
                tol=0.0,
            )
        )

    # hierarchical: every linkage × several blob configurations.
    hcfg = []
    for trial in range(44):
        strat = HIER[trial % 4]
        d = 2 if trial % 2 == 0 else 3
        centers = grid_centers(d, spacing=rng.uniform(5.0, 8.0))
        n_per = int(rng.integers(8, 16))
        nc = int(rng.integers(2, len(centers) + 1))
        X = blobs(rng, n_per, centers, spread=rng.uniform(0.3, 0.7))
        hcfg.append((trial, strat, d, X, nc))
    for trial, strat, d, X, nc in hcfg:
        hard.append(
            make_case(
                f"cluster.hard.{strat}_{trial}",
                "hard",
                strat,
                hier_inputs(X, d, nc),
                hierarchical_largest(X, LINKAGE_FOR[strat], nc),
                tol=0.0,
            )
        )

    cases.extend(hard)

    # ── Edge (~10): degenerate / out-of-envelope (library MUST warn) ──────────
    # These are tagged inEnvelope=false → the library must emit outsideEnvelope.
    edge_centers = grid_centers(2, spacing=8.0)
    Xe = blobs(rng, 6, edge_centers, spread=0.3)  # n = 24
    dims_e = 2
    n_e = Xe.shape[0]

    oov = []

    # kmeans k > n: ask for more clusters than points. sklearn would raise, so
    # the oracle scalar is just 0.0 (the library returns an empty best-effort
    # result); the gate is the diagnostic, not the value.
    bigk = n_e + 5
    init_bigk = np.vstack([Xe, Xe[:5]])  # bigk seed rows (arbitrary, unused path)
    oov.append(
        make_case(
            "cluster.edge.kmeans_k_gt_n",
            "edge",
            "kmeans",
            kmeans_inputs(Xe, init_bigk, dims_e),
            0.0,
            in_envelope=False,
            tol=0.0,
        )
    )

    # kmeans k <= 0.
    oov.append(
        make_case(
            "cluster.edge.kmeans_k_zero",
            "edge",
            "kmeans",
            {
                "points": flat(Xe),
                "dims": dims_e,
                "k": 0,
                "init": [],
                "maxIter": KMEANS_MAX_ITER,
                "tol": KMEANS_TOL,
            },
            0.0,
            in_envelope=False,
            tol=0.0,
        )
    )

    # kmeans empty input.
    oov.append(
        make_case(
            "cluster.edge.kmeans_empty",
            "edge",
            "kmeans",
            {
                "points": [],
                "dims": dims_e,
                "k": 2,
                "init": flat(Xe[:2]),
                "maxIter": KMEANS_MAX_ITER,
                "tol": KMEANS_TOL,
            },
            0.0,
            in_envelope=False,
            tol=0.0,
        )
    )

    # dbscan all-noise: eps tiny → no dense region → zero clusters.
    oov.append(
        make_case(
            "cluster.edge.dbscan_all_noise",
            "edge",
            "dbscan",
            dbscan_inputs(Xe, dims_e, 0.01, 5),
            0.0,
            in_envelope=False,
            tol=0.0,
        )
    )

    # dbscan empty input.
    oov.append(
        make_case(
            "cluster.edge.dbscan_empty",
            "edge",
            "dbscan",
            {"points": [], "dims": dims_e, "eps": 1.0, "minPts": 3},
            0.0,
            in_envelope=False,
            tol=0.0,
        )
    )

    # hierarchical nClusters > n.
    oov.append(
        make_case(
            "cluster.edge.hier_nc_gt_n",
            "edge",
            "hierarchical_ward",
            hier_inputs(Xe, dims_e, n_e + 3),
            0.0,
            in_envelope=False,
            tol=0.0,
        )
    )

    cases.extend(oov)

    # In-envelope edge guards (must NOT warn): well-posed but stressed requests
    # at the boundary (k == n is valid; a single tight blob with one cluster).
    ok = []

    # k == n exactly: every point is its own cluster (valid, inertia ~0).
    Xkn = blobs(rng, 1, grid_centers(2, spacing=8.0), spread=0.0)  # n = 4
    init_kn = Xkn.copy()
    ok.append(
        make_case(
            "cluster.edge.kmeans_k_eq_n",
            "edge",
            "kmeans",
            kmeans_inputs(Xkn, init_kn, 2),
            kmeans_inertia(Xkn, init_kn),
            tol=1e-6,
        )
    )

    # hierarchical nClusters == n (every leaf its own cluster, largest size 1).
    ok.append(
        make_case(
            "cluster.edge.hier_nc_eq_n",
            "edge",
            "hierarchical_complete",
            hier_inputs(Xe, dims_e, n_e),
            hierarchical_largest(Xe, "complete", n_e),
            tol=0.0,
        )
    )

    # dbscan that DOES form a cluster on the tight edge data (false-positive guard).
    ok.append(
        make_case(
            "cluster.edge.dbscan_ok",
            "edge",
            "dbscan",
            dbscan_inputs(Xe, dims_e, 1.5, 4),
            dbscan_nclusters(Xe, 1.5, 4),
            tol=0.0,
        )
    )

    cases.extend(ok)

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"cluster: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cluster.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
