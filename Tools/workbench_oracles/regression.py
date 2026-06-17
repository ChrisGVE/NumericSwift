#!/usr/bin/env python3
"""Oracle generator for the NumericSwift workbench — Regression domain.

Computes bit-exact reference regression coefficients with **statsmodels** and
freezes them as the JSON fixture
`Tests/NumericSwiftTests/Fixtures/workbench/regression.json`.

Mirrors the reference generator `integration.py`. Contract (WORKBENCH.md §2/§3/§5):

  * ~100 cases partitioned 10 / 80 / 10 (trivial / hard / edge); actual counts are
    printed so a thin tier is never silently shipped.
  * Each case carries `oracle.bits` (IEEE-754 hex) as the canonical value plus a
    `source` citation. Oracle values come ONLY from statsmodels — never from
    NumericSwift (FP1, FP3 vacuous-gate rule).
  * `inEnvelope` is per-strategy. Out-of-envelope cases (ill-conditioned /
    multicollinear design for ols/wls/glm; series too short for the requested
    (p,d,q) for arima) are tagged `false`, so the gate requires NumericSwift to
    emit an `outsideEnvelope` diagnostic for them.

## Comparison scalar

  * ols / wls / glm : the **slope** params[1] — the first non-intercept
    coefficient. Every design matrix has a leading intercept column, so
    params[0] is the intercept and params[1] is the first regressor's effect.
  * arima           : the **AR(1) coefficient** arParams[0] (every ARIMA fixture
    requests p ≥ 1). The oracle uses statsmodels' CSS estimator to match the
    library's CSS / Hannan-Rissanen estimator within the declared envelope.

## Input encoding

  * ols   : X as flat row-major `X`, dims `n`/`k`, response `y`.
  * wls   : as ols, plus `weights` (length n).
  * glm   : as ols, plus `family` (gaussian|binomial|poisson|gamma); the
    family's canonical link is used (matching the Swift suite).
  * arima : the series `y`, orders `p`/`d`/`q`.

Run:
    /tmp/.nsoracle/bin/python Tools/workbench_oracles/regression.py
"""

import json
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import statsmodels
import statsmodels.api as sm

warnings.simplefilter("ignore")  # suppress statsmodels convergence/cond warnings

SOURCE = f"statsmodels {statsmodels.__version__} (sm.OLS/WLS/GLM/tsa.ARIMA)"

# Design-matrix condition-number envelope shared with Swift
# `regressionConditionEnvelope`. cond(X) above this → out-of-envelope.
COND_ENVELOPE = 1e10

RNG = np.random.default_rng(20260617)


def bits_hex(value: float) -> str:
    """IEEE-754 bit pattern of a double as a 0x-prefixed 16-hex-digit string."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _ov(v):
    return {"value": float(v), "bits": bits_hex(v)}


def flat(M):
    return [float(v) for v in np.asarray(M, dtype=float).reshape(-1)]


def vec(v):
    return [float(x) for x in np.asarray(v, dtype=float).reshape(-1)]


# ── Oracle estimators (statsmodels ONLY) ─────────────────────────────────────


def oracle_ols(X, y):
    return float(sm.OLS(np.asarray(y, float), np.asarray(X, float)).fit().params[1])


def oracle_wls(X, y, w):
    return float(
        sm.WLS(np.asarray(y, float), np.asarray(X, float), weights=np.asarray(w, float))
        .fit()
        .params[1]
    )


_FAMILIES = {
    "gaussian": sm.families.Gaussian(),
    "binomial": sm.families.Binomial(),
    "poisson": sm.families.Poisson(),
    "gamma": sm.families.Gamma(),
}


def oracle_glm(X, y, family):
    fam = _FAMILIES[family]
    return float(
        sm.GLM(np.asarray(y, float), np.asarray(X, float), family=fam).fit().params[1]
    )


def oracle_arima(y, p, d, q):
    # CSS / Hannan-Rissanen to match the Swift CSS / Hannan-Rissanen estimator.
    # trend='n' (no constant) mirrors the library, which models the differenced
    # series with no intercept. Return the AR(1) coefficient.
    #
    # For out-of-envelope (series-too-short) cases statsmodels' estimators fail
    # outright (NaN/Inf in the parameter vector). Those cases are NOT
    # accuracy-scored — the gate is the DIAGNOSTIC, not the value (WORKBENCH.md
    # §5) — but the fixture still needs a finite, citable oracle value. Fall back
    # to the lag-1 sample autocorrelation (the AR(1) Yule-Walker estimate).
    y = np.asarray(y, float)
    yd = np.diff(y, n=d) if d > 0 else y
    try:
        model = sm.tsa.ARIMA(y, order=(p, d, q), trend="n")
        res = (
            model.fit(method="hannan_rissanen")
            if q > 0
            else model.fit(method="yule_walker")
        )
        val = float(res.arparams[0])
        if not np.isfinite(val):
            raise ValueError("non-finite arparam")
        return val
    except Exception:
        acf = sm.tsa.acf(yd, nlags=1, fft=False)
        return float(acf[1]) if np.isfinite(acf[1]) else 0.0


# ── Case builders ────────────────────────────────────────────────────────────


def case_ls(cid, tier, strategy, X, y, tol, *, weights=None, in_env=None):
    X = np.asarray(X, float)
    n, k = X.shape
    inputs = {"X": flat(X), "n": int(n), "k": int(k), "y": vec(y)}
    if strategy == "wls":
        inputs["weights"] = vec(weights)
        value = oracle_wls(X, y, weights)
    else:
        value = oracle_ols(X, y)
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": inputs,
        "oracle": _ov(value),
        "source": SOURCE,
        "strategies": [strategy],
        "tol": {strategy: tol},
    }
    if in_env is not None:
        entry["inEnvelope"] = {strategy: in_env}
    return entry


def case_glm(cid, tier, X, y, family, tol, in_env=None):
    X = np.asarray(X, float)
    n, k = X.shape
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": {
            "X": flat(X),
            "n": int(n),
            "k": int(k),
            "y": vec(y),
            "family": family,
        },
        "oracle": _ov(oracle_glm(X, y, family)),
        "source": SOURCE,
        "strategies": ["glm"],
        "tol": {"glm": tol},
    }
    if in_env is not None:
        entry["inEnvelope"] = {"glm": in_env}
    return entry


def case_arima(cid, tier, y, p, d, q, tol, in_env=None):
    entry = {
        "id": cid,
        "tier": tier,
        "inputs": {"y": vec(y), "p": int(p), "d": int(d), "q": int(q)},
        "oracle": _ov(oracle_arima(y, p, d, q)),
        "source": SOURCE,
        "strategies": ["arima"],
        "tol": {"arima": tol},
    }
    if in_env is not None:
        entry["inEnvelope"] = {"arima": in_env}
    return entry


# ── Data synthesis helpers ────────────────────────────────────────────────────


def linear_design(n, k_regressors, noise=0.3, betas=None):
    """An n×(1+k) design with a leading intercept column and a clean linear y."""
    Xr = RNG.standard_normal((n, k_regressors))
    X = np.column_stack([np.ones(n), Xr])
    if betas is None:
        betas = RNG.uniform(-2.0, 2.0, size=1 + k_regressors)
    y = X @ betas + noise * RNG.standard_normal(n)
    return X, y


def ar_series(n, phi, burn=200):
    """A stationary AR(1) series with coefficient phi."""
    e = RNG.standard_normal(n + burn)
    y = np.zeros(n + burn)
    for t in range(1, n + burn):
        y[t] = phi * y[t - 1] + e[t]
    return y[burn:]


def arma_series(n, phi, theta, burn=300):
    """A stationary ARMA(1,1) series with AR coef phi and MA coef theta."""
    e = RNG.standard_normal(n + burn)
    y = np.zeros(n + burn)
    for t in range(1, n + burn):
        y[t] = phi * y[t - 1] + e[t] + theta * e[t - 1]
    return y[burn:]


def collinear_design(n, base_cols=2):
    """A near-perfectly-collinear design: cond(X) >> 1e10 (out-of-envelope)."""
    X = np.column_stack(
        [np.ones(n)] + [RNG.standard_normal(n) for _ in range(base_cols)]
    )
    # Append a column that is the first regressor plus a tiny perturbation —
    # near-perfect collinearity drives cond(X) past the 1e10 envelope.
    X = np.column_stack([X, X[:, 1] + 1e-11 * RNG.standard_normal(n)])
    y = X[:, 1] * 2.0 + 0.1 * RNG.standard_normal(n)
    return X, y


def build():
    cases = []

    # ── Trivial (~10): tiny, well-conditioned, clean designs ──────────────────
    # ols: y = 1 + 2x exactly.
    x = np.arange(1.0, 8.0)
    cases.append(
        case_ls(
            "regression.trivial.ols_line",
            "trivial",
            "ols",
            np.column_stack([np.ones_like(x), x]),
            1.0 + 2.0 * x,
            1e-9,
        )
    )
    # ols two-regressor clean fit.
    Xt, yt = linear_design(40, 2, noise=0.0, betas=np.array([1.0, 2.0, -1.0]))
    cases.append(
        case_ls("regression.trivial.ols_clean2", "trivial", "ols", Xt, yt, 1e-8)
    )
    cases.append(
        case_ls(
            "regression.trivial.ols_clean3",
            "trivial",
            "ols",
            *linear_design(50, 3, noise=0.0),
            1e-8,
        )
    )
    # wls: uniform weights → equals ols.
    Xw, yw = linear_design(40, 2, noise=0.0, betas=np.array([0.5, 1.5, 0.0]))
    cases.append(
        case_ls(
            "regression.trivial.wls_uniform",
            "trivial",
            "wls",
            Xw,
            yw,
            1e-8,
            weights=np.ones(40),
        )
    )
    Xw2, yw2 = linear_design(45, 2, noise=0.1)
    cases.append(
        case_ls(
            "regression.trivial.wls_varied",
            "trivial",
            "wls",
            Xw2,
            yw2,
            1e-8,
            weights=RNG.uniform(0.5, 2.0, 45),
        )
    )
    # glm gaussian/identity → equals ols.
    Xg, yg = linear_design(50, 2, noise=0.2)
    cases.append(
        case_glm("regression.trivial.glm_gaussian", "trivial", Xg, yg, "gaussian", 1e-6)
    )
    # glm poisson with a small positive-count response.
    Xp = np.column_stack([np.ones(60), RNG.standard_normal(60) * 0.3])
    mu_p = np.exp(0.5 + 0.4 * Xp[:, 1])
    yp = RNG.poisson(mu_p).astype(float)
    cases.append(
        case_glm("regression.trivial.glm_poisson", "trivial", Xp, yp, "poisson", 1e-4)
    )
    # arima AR(1).
    cases.append(
        case_arima(
            "regression.trivial.arima_ar1",
            "trivial",
            ar_series(400, 0.6),
            1,
            0,
            0,
            5e-2,
        )
    )
    cases.append(
        case_arima(
            "regression.trivial.arima_ar1b",
            "trivial",
            ar_series(500, -0.4),
            1,
            0,
            0,
            5e-2,
        )
    )
    cases.append(
        case_arima(
            "regression.trivial.arima_ar1c",
            "trivial",
            ar_series(450, 0.3),
            1,
            0,
            0,
            5e-2,
        )
    )

    # ── Hard (~80): realistic noisy designs across all 4 strategies ───────────
    # ols: 24 noisy linear designs of growing size / regressor count.
    for i in range(24):
        n = [30, 40, 50, 60, 80, 100][i % 6]
        kr = [1, 2, 3, 4][i % 4]
        X, y = linear_design(n, kr, noise=0.5)
        cases.append(case_ls(f"regression.hard.ols_{i}", "hard", "ols", X, y, 1e-6))

    # wls: 18 noisy designs with random positive weights.
    for i in range(18):
        n = [40, 50, 60, 80][i % 4]
        kr = [1, 2, 3][i % 3]
        X, y = linear_design(n, kr, noise=0.5)
        w = RNG.uniform(0.3, 3.0, n)
        cases.append(
            case_ls(f"regression.hard.wls_{i}", "hard", "wls", X, y, 1e-6, weights=w)
        )

    # glm: 20 cases across gaussian / poisson / binomial / gamma families.
    for i in range(20):
        fam = ["gaussian", "poisson", "binomial", "gamma"][i % 4]
        n = [50, 60, 80][i % 3]
        xr = RNG.standard_normal(n) * 0.4
        X = np.column_stack([np.ones(n), xr])
        if fam == "gaussian":
            y = 0.5 + 0.8 * xr + 0.3 * RNG.standard_normal(n)
            tol = 1e-4
        elif fam == "poisson":
            y = RNG.poisson(np.exp(0.4 + 0.5 * xr)).astype(float)
            tol = 1e-3
        elif fam == "binomial":
            p_ = 1.0 / (1.0 + np.exp(-(0.2 + 1.0 * xr)))
            y = RNG.binomial(1, p_).astype(float)
            tol = 1e-3
        else:  # gamma — positive continuous response, inverse link
            mu_ = 1.0 / (0.5 + 0.3 * (xr - xr.min() + 0.5))
            y = RNG.gamma(shape=4.0, scale=mu_ / 4.0)
            tol = 1e-2
        cases.append(case_glm(f"regression.hard.glm_{fam}_{i}", "hard", X, y, fam, tol))

    # arima: 18 AR(1) / ARMA(1,1) series.
    for i in range(18):
        if i % 3 == 2:
            phi = RNG.uniform(-0.6, 0.6)
            theta = RNG.uniform(-0.5, 0.5)
            y = arma_series(400, phi, theta)
            cases.append(
                case_arima(f"regression.hard.arma_{i}", "hard", y, 1, 0, 1, 2e-1)
            )
        else:
            phi = RNG.uniform(-0.7, 0.7)
            y = ar_series(400, phi)
            cases.append(
                case_arima(f"regression.hard.ar_{i}", "hard", y, 1, 0, 0, 1e-1)
            )

    # ── Edge (~10): out-of-envelope + in-envelope guards ──────────────────────
    # OUT-OF-ENVELOPE: near-perfectly-collinear design → cond(X) >> 1e10 → the
    # library MUST emit an outsideEnvelope diagnostic. ols / wls / glm.
    Xc, yc = collinear_design(60)
    cases.append(
        case_ls(
            "regression.edge.ols_collinear", "edge", "ols", Xc, yc, 1e-1, in_env=False
        )
    )
    Xc2, yc2 = collinear_design(70)
    cases.append(
        case_ls(
            "regression.edge.wls_collinear",
            "edge",
            "wls",
            Xc2,
            yc2,
            1e-1,
            weights=RNG.uniform(0.5, 2.0, 70),
            in_env=False,
        )
    )
    Xc3, yc3 = collinear_design(80)
    # gaussian GLM on the collinear design — same envelope breach as ols.
    cases.append(
        case_glm(
            "regression.edge.glm_collinear",
            "edge",
            Xc3,
            yc3,
            "gaussian",
            1e-1,
            in_env=False,
        )
    )
    # OUT-OF-ENVELOPE: ARIMA series too short for the requested order. An ARMA(2,2)
    # needs ≥ 3·(2+2)+1 = 13 effective observations; 10 points is far too few.
    short = ar_series(10, 0.5)
    cases.append(
        case_arima(
            "regression.edge.arima_short", "edge", short, 2, 0, 2, 1e0, in_env=False
        )
    )
    short2 = ar_series(9, -0.3)
    cases.append(
        case_arima(
            "regression.edge.arima_short2", "edge", short2, 2, 0, 1, 1e0, in_env=False
        )
    )

    # IN-ENVELOPE guards: well-conditioned edge-ish cases must NOT warn.
    Xok, yok = linear_design(100, 3, noise=0.4)
    cases.append(case_ls("regression.edge.ols_ok", "edge", "ols", Xok, yok, 1e-5))
    Xok2, yok2 = linear_design(90, 2, noise=0.4)
    cases.append(
        case_ls(
            "regression.edge.wls_ok",
            "edge",
            "wls",
            Xok2,
            yok2,
            1e-5,
            weights=RNG.uniform(0.5, 2.0, 90),
        )
    )
    Xok3 = np.column_stack([np.ones(80), RNG.standard_normal(80) * 0.4])
    yok3 = RNG.poisson(np.exp(0.3 + 0.5 * Xok3[:, 1])).astype(float)
    cases.append(
        case_glm("regression.edge.glm_ok", "edge", Xok3, yok3, "poisson", 1e-3)
    )
    # A long AR(1) — comfortably in-envelope, must NOT warn.
    cases.append(
        case_arima(
            "regression.edge.arima_ok", "edge", ar_series(600, 0.5), 1, 0, 0, 5e-2
        )
    )
    cases.append(
        case_arima(
            "regression.edge.arima_ok2", "edge", ar_series(550, -0.5), 1, 0, 0, 5e-2
        )
    )

    return cases


def main():
    cases = build()
    tiers = {"trivial": 0, "hard": 0, "edge": 0}
    for c in cases:
        tiers[c["tier"]] += 1
    print(
        f"regression: {len(cases)} cases — "
        f"trivial={tiers['trivial']} hard={tiers['hard']} edge={tiers['edge']}",
        file=sys.stderr,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "Tests/NumericSwiftTests/Fixtures/workbench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "regression.json"
    out.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
