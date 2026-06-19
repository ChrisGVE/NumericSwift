#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Geometry circle-fit domain.

Generates synthetic circle-fitting problems and freezes them as the JSON fixture
`Tests/NumericSwiftTests/Fixtures/workbench/circlefit.json`. Mirrors the
reference generator `integration.py`. Contract (WORKBENCH.md §2/§3/§5):

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); the actual
    counts are printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value and a
    `source` citation.

  ## Comparison scalar & oracle (FP1)

  The comparison scalar is the **fitted circle RADIUS**. Each well-posed case is
  a synthetic circle of KNOWN centre + radius, sampled at evenly-spaced angles
  with small zero-mean Gaussian radial noise (fixed seed → reproducible). The
  oracle radius is the **TRUE radius** that generated the points — analytic
  ground truth, never NumericSwift's own output (FP3 vacuous-gate rule). This is
  an independent reference: the library is judged on recovering the radius that
  actually produced the samples. (Both kasa and taubin are unbiased radius
  estimators for low isotropic noise, so the true radius is the right oracle for
  both strategies; the per-tier `tol` widens with the noise level.)

  ## Inputs

  Points are carried in `inputs` as a flat `[x0, y0, x1, y1, …]` array under
  `points`, with `count` = number of points, plus the synthetic truth
  (`true_cx`, `true_cy`, `true_r`) for documentation/debugging. The Swift suite
  reconstructs `[Vec2]` from `points`.

  ## Out-of-envelope (self-awareness gate, §5)

  Out-of-envelope cases are tagged `inEnvelope: {kasa:false, taubin:false}`:
  collinear / near-collinear point sets, and fewer-than-3-point sets. The library
  MUST emit an `outsideEnvelope` diagnostic for these (collinearity / insufficient
  data make the circle undetermined). Well-posed circle fits must NOT warn.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/circlefit.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

SOURCE = f"synthetic circle (analytic true radius); numpy {np.__version__}"

# Fixed seed so the noisy fixtures are bit-reproducible across regenerations.
SEED = 20260617
RNG = np.random.default_rng(SEED)

STRATEGIES = ["kasa", "taubin"]


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


def sample_circle(cx, cy, r, n, noise, *, full=True):
    """Sample `n` points on a circle (cx, cy, r) with Gaussian radial noise.

    `full=True` spreads the angles over a complete revolution; `full=False`
    restricts them to a short arc (a harder, less-conditioned configuration).
    Returns a flat [x0, y0, x1, y1, …] list.
    """
    if full:
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    else:
        # Short 60° arc — points span only part of the circle.
        angles = np.linspace(0.0, math.pi / 3.0, n)
    radii = r + noise * RNG.standard_normal(n) if noise > 0 else np.full(n, r)
    pts = []
    for a, rr in zip(angles, radii):
        pts.append(cx + rr * math.cos(a))
        pts.append(cy + rr * math.sin(a))
    return pts


def case(cid, tier, points, count, true_cx, true_cy, true_r, tol, *, in_envelope=None):
    inputs = {
        "points": points,
        "count": count,
        "true_cx": true_cx,
        "true_cy": true_cy,
        "true_r": true_r,
    }
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": {"value": true_r, "bits": bits_hex(true_r)},
        "source": SOURCE,
        "strategies": STRATEGIES,
        "tol": tol,
    }
    if in_envelope:
        entry["inEnvelope"] = in_envelope
    return entry


def build():
    cases = []

    # ── Trivial (~10): clean (noise-free) full circles, exact recovery ────────
    # noise = 0 → both estimators recover the true radius to ~machine precision.
    trivial_specs = [
        (0.0, 0.0, 1.0, 16),
        (0.0, 0.0, 5.0, 24),
        (3.0, -2.0, 2.5, 20),
        (-4.0, 4.0, 10.0, 32),
        (1.0, 1.0, 0.5, 12),
        (10.0, 10.0, 7.0, 40),
        (0.0, 0.0, 100.0, 48),
        (-5.0, 0.0, 3.0, 16),
        (2.5, 2.5, 1.25, 24),
        (0.0, -7.0, 8.0, 36),
    ]
    for i, (cx, cy, r, n) in enumerate(trivial_specs):
        pts = sample_circle(cx, cy, r, n, 0.0, full=True)
        cases.append(
            case(
                f"circlefit.trivial.clean_{i}",
                "trivial",
                pts,
                n,
                cx,
                cy,
                r,
                {"kasa": 1e-6, "taubin": 1e-6},
            )
        )

    # ── Hard (~80): noisy full circles across radii / centres / counts ────────
    # Realistic measurement scenario: zero-mean Gaussian radial noise. tol is the
    # declared envelope (a few × the noise level / sqrt(n)), not the achieved
    # error. Both estimators stay well inside it for isotropic full-circle noise.
    centres = [(0.0, 0.0), (3.0, -2.0), (-5.0, 5.0), (12.0, 1.0)]
    radii = [1.0, 4.0, 10.0, 25.0]
    counts = [16, 32, 64]
    # noise as a fraction of the radius
    noise_fracs = [0.01, 0.03]
    idx = 0
    for cx, cy in centres:
        for r in radii:
            for n in counts:
                for nf in noise_fracs:
                    if idx >= 80:
                        break
                    noise = nf * r
                    pts = sample_circle(cx, cy, r, n, noise, full=True)
                    # Envelope: a few sigma on the mean radius. With n samples the
                    # radius estimate's std ≈ noise / sqrt(n); allow 6× plus a
                    # floor for the small-n cases.
                    bound = max(6.0 * noise / math.sqrt(n), 0.05 * r)
                    cases.append(
                        case(
                            f"circlefit.hard.noisy_{idx}",
                            "hard",
                            pts,
                            n,
                            cx,
                            cy,
                            r,
                            {"kasa": bound, "taubin": bound},
                        )
                    )
                    idx += 1

    # ── Edge (~10): out-of-envelope (collinear / <3) + in-envelope hard ───────
    # Out-of-envelope: the library MUST emit outsideEnvelope. Oracle radius is
    # set to the synthetic generating radius where one exists, or 0 for the
    # degenerate point sets (the numeric value is irrelevant for these cases —
    # the gate checks the diagnostic, not the deviation, §5).
    oov = []

    # Exactly collinear: points on the line y = x.
    line1 = []
    for t in np.linspace(-5.0, 5.0, 8):
        line1.append(float(t))
        line1.append(float(t))
    oov.append(("collinear_diag", line1, 8, 0.0, 0.0, 0.0))

    # Exactly collinear: points on a horizontal line.
    line2 = []
    for t in np.linspace(0.0, 10.0, 6):
        line2.append(float(t))
        line2.append(3.0)
    oov.append(("collinear_horiz", line2, 6, 0.0, 0.0, 0.0))

    # Near-collinear: points strung along y = 2x with a minuscule perpendicular
    # jitter (1e-7 of the along-line extent). The transverse scatter relative to
    # the along-line scatter collapses far below the collinearity threshold, so
    # the circle is undetermined and the library must warn.
    nc = []
    nc_jitter = RNG.standard_normal(9) * 1e-7
    for t, j in zip(np.linspace(-4.0, 4.0, 9), nc_jitter):
        nc.append(float(t))
        nc.append(float(2.0 * t + j))
    oov.append(("near_collinear", nc, 9, 0.0, 0.0, 0.0))

    # Fewer than 3 points: two distinct points — a circle is undetermined.
    two = [0.0, 0.0, 1.0, 1.0]
    oov.append(("two_points", two, 2, 0.0, 0.0, 0.0))

    for name, pts, n, cx, cy, r in oov:
        cases.append(
            case(
                f"circlefit.edge.{name}",
                "edge",
                pts,
                n,
                cx,
                cy,
                r,
                # tol is irrelevant for an out-of-envelope case (the gate checks the
                # diagnostic) but a value is required by the schema; keep it loose.
                {"kasa": 1e9, "taubin": 1e9},
                in_envelope={"kasa": False, "taubin": False},
            )
        )

    # In-envelope edge: well-posed but stressful — minimal 3-point exact circle,
    # very small and very large radii, high noise on a full circle. These guard
    # against FALSE POSITIVES: the library must NOT warn here.
    # Minimal exact circle: 3 non-collinear points on the unit circle.
    three = sample_circle(0.0, 0.0, 1.0, 3, 0.0, full=True)
    cases.append(
        case(
            "circlefit.edge.three_points",
            "edge",
            three,
            3,
            0.0,
            0.0,
            1.0,
            {"kasa": 1e-6, "taubin": 1e-6},
        )
    )
    # Small radius (well-posed; both estimators recover it). Kept above the
    # Taubin det floor so this stays a false-positive guard, not an error probe.
    tiny = sample_circle(0.0, 0.0, 0.1, 24, 0.0, full=True)
    cases.append(
        case(
            "circlefit.edge.small_radius",
            "edge",
            tiny,
            24,
            0.0,
            0.0,
            0.1,
            {"kasa": 1e-7, "taubin": 1e-7},
        )
    )
    # Large radius.
    large = sample_circle(0.0, 0.0, 1e4, 48, 0.0, full=True)
    cases.append(
        case(
            "circlefit.edge.large_radius",
            "edge",
            large,
            48,
            0.0,
            0.0,
            1e4,
            {"kasa": 1e-2, "taubin": 1e-2},
        )
    )
    # High-noise full circle: still well-posed (the gate must not warn), wide tol.
    hn_r = 5.0
    hn = sample_circle(0.0, 0.0, hn_r, 64, 0.10 * hn_r, full=True)
    cases.append(
        case(
            "circlefit.edge.high_noise",
            "edge",
            hn,
            64,
            0.0,
            0.0,
            hn_r,
            {
                "kasa": 6.0 * 0.10 * hn_r / math.sqrt(64) + 0.10 * hn_r,
                "taubin": 6.0 * 0.10 * hn_r / math.sqrt(64) + 0.10 * hn_r,
            },
        )
    )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    oov = sum(
        1
        for c in cases
        if c.get("inEnvelope") and any(v is False for v in c["inEnvelope"].values())
    )
    print(
        f"circlefit: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']} "
        f"(out-of-envelope={oov})",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "circlefit.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
