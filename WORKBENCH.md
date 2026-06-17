# NumericSwift E2E Functional Workbench — Design Spec

> Status: DRAFT for Chris's approval. Authored 2026-06-17 during the
> `hardening-sweep` campaign. This spec gates Phase 2 (workbench build).
> Untracked/local until approved, then committed alongside the implementation.

## 1. Purpose

A **functional** (end-to-end) test layer that complements the existing unit
tests. Unit tests assert one function on a few inputs; the workbench exercises
each domain the way a user would — realistic problems, graded by difficulty, with
results checked against an authoritative reference (FP1) and, where a domain
offers several algorithms, **all strategies run on the same problems and compared
against each strategy's documented limitation envelope**.

Two consumers:
1. `NumericSwiftWorkbench` executable — rich human-readable report (per-domain
   pass/fail, cross-strategy accuracy table, envelope violations).
2. `WorkbenchGateTests` (XCTest) — asserts zero envelope violations so CI fails
   on regression.

Neither is part of the library product → remote SPM consumers (LuaSwift) are
unaffected, exactly like `NumericSwiftBench`.

## 2. Difficulty taxonomy — 10 / 80 / 10 per domain

Every domain's case set is partitioned:

| Tier | Share | Definition |
|------|-------|-----------|
| `trivial` | ~10% | Textbook smoke cases; closed-form answers; sanity floor. |
| `hard` | ~80% | Realistic, non-degenerate problems — the bulk. Cases are lifted from the **SciPy / statsmodels own test suites** wherever they exist (FP1), plus representative applied problems. |
| `edge` | ~10% | Very difficult / degenerate: extreme tails, ill-conditioning, near-singular, stiff, collinear, empty/1-element, IEEE inf/nan/signed-zero, branch cuts. |

The generator enforces the proportions (±a few cases) and `log()`s the actual
counts so a thin tier is never silently shipped.

## 3. Oracle & fixture discipline (FP1 + FP3)

- Oracle values are computed **offline** by committed Python generators under
  `Tools/workbench_oracles/<domain>.py` (scipy / numpy / statsmodels), and frozen
  as JSON fixtures under `Tests/NumericSwiftTests/Fixtures/workbench/<domain>.json`.
- **No runtime Python dependency** — the Swift workbench only reads JSON. Mirrors
  the existing `LegacySnapshot.json` / `ParityCorpus` precedent.
- IEEE-754 edge values stored bit-exact via `UInt64` `bitPattern` (same convention
  as `LegacySnapshot.json`).
- Regeneration is explicit and gated (`NUMERICSWIFT_REGENERATE_WORKBENCH=1`), and
  **never** derived from this library's own output (FP3 vacuous-gate rule). Each
  fixture entry carries a `source` citation (e.g. `scipy.integrate.quad 1.17.1`).

### Fixture schema (per case)

```json
{
  "id": "integration.hard.gaussian_bell",
  "tier": "hard",
  "inputs": { "...": "domain-specific" },
  "oracle": { "value": 1.7724538509055159, "bits": "0x3FFC5BF891B4EF6A" },
  "source": "scipy.integrate.quad 1.17.1",
  "strategies": ["quad", "romberg", "simps"],
  "tol": { "quad": 1e-10, "romberg": 1e-8, "simps": 1e-3 }
}
```

`tol` is the **per-strategy, per-case expected accuracy** — the limitation
envelope (§5).

## 4. Domains & strategy registry

One suite per module. Multi-strategy domains (run ALL, compare):

| Domain | Strategies compared |
|--------|--------------------|
| Integration (quadrature) | `quad`, `romberg`, `simps`, `trapz`, `fixed_quad` |
| Integration (ODE) | `rk45`, `rk23`, `dop853`, `odeint` |
| Optimization (root) | `bisect`, `newton`, `secant`, `brent` |
| Optimization (minimize) | `nelderMead`, `bfgs`, `goldenSection` |
| Interpolation | `cubic(natural\|clamped\|notAKnot)`, `pchip`, `akima`, `barycentric` |
| Cluster | `kmeans`, `dbscan`, `hierarchical(single\|complete\|average\|ward)` |
| LinAlg (solve) | `solve(LU)`, `lstsq`, `choSolve`, `solveTriangular` |
| Spatial (kNN) | `kdTree`, `bruteForce` |
| Geometry (circle fit) | `kasa`, `taubin` |
| Regression | `ols`, `wls`, `glm`, `arima` |

Single-strategy domains (correctness vs oracle only): Complex, Constants,
Distributions, NumberTheory, Series, SpecialFunctions, Statistics, MathExpr.

## 5. Limitation envelopes & cross-strategy comparison

Each strategy declares a **documented limitation** (from CLAUDE.md Known
Limitations + SciPy docs + the audit), encoded as the per-tier `tol`. The
workbench, per case:

1. Runs every applicable strategy.
2. Asserts each result is within its declared `tol` of the oracle → **envelope
   check** (a strategy must meet its claimed accuracy).
3. Ranks strategies by error and emits a comparison row.
4. **Flags** two failure modes: (a) a strategy **exceeds** its envelope (worse
   than claimed) — a real regression/bug; (b) a strategy is **unexpectedly
   better** across the board, signalling a stale/too-loose declared limitation to
   tighten.

Example declared envelopes: `simps` exact to deg≤2 on non-uniform grids; T-dist
`ppf` ~5 digits for |p|>0.9999 (known limitation); matrix `logm/sqrtm` valid only
for diagonalizable inputs (until #7 lands); `expm` full double precision
(post-CR-D1).

## 6. Swift structure

```
Sources/NumericSwiftWorkbench/
  main.swift                 // arg parse, run all/one domain, print report, exit code
  WorkbenchCase.swift        // Codable fixture model + bit-exact decode
  Strategy.swift             // strategy id → closure registry per domain
  Envelope.swift             // per-(strategy,tier) tolerance + violation model
  Report.swift               // comparison table + summary rendering
  Domains/<Domain>Suite.swift// one per domain: maps fixture inputs → strategy calls
Tools/workbench_oracles/
  <domain>.py                // committed generators
Tests/NumericSwiftTests/
  WorkbenchGateTests.swift    // XCTest: load fixtures, run, assert 0 violations
  Fixtures/workbench/<domain>.json
```

## 7. Reporting

Executable output per domain: a table `case | tier | strategy | error | tol |
status`, a per-domain rollup, and a final `N domains, M cases, K violations`.
Nonzero exit on any violation. The XCTest gate asserts `K == 0`.

## 8. Build / run

```bash
swift build --product NumericSwiftWorkbench
.build/debug/NumericSwiftWorkbench            # all domains
.build/debug/NumericSwiftWorkbench integration optimization   # selected
swift test --filter WorkbenchGateTests        # CI gate
NUMERICSWIFT_REGENERATE_WORKBENCH=1 ...        # regen fixtures (offline, scipy)
```

## 9. Open decisions for Chris

1. **Gate strictness:** should an envelope violation fail CI hard (XCTest), or
   only warn in the executable for the first iteration while envelopes settle?
   (Proposed: hard gate, with envelopes tuned during Phase 2.)
2. **Case volume per domain:** target count (e.g. 30–60 cases/domain →
   ~500–900 total)? Affects fixture size + generation time.
3. **Tracking:** commit WORKBENCH.md + generators + fixtures to the repo
   (proposed yes — shipped, reviewable infra).
