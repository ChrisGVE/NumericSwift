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

## 2. Difficulty taxonomy — 100 cases/domain, 10 / 80 / 10

**Target volume: 100 cases per domain** (owner decision), partitioned 10/80/10
→ ~10 trivial, ~80 hard, ~10 edge. Across ~16 domains ≈ **1,600 cases total**.

| Tier | Count/domain | Definition |
|------|-------------|-----------|
| `trivial` | ~10 | Textbook smoke cases; closed-form answers; sanity floor. |
| `hard` | ~80 | Realistic, non-degenerate problems — the bulk. Cases are lifted from the **SciPy / statsmodels own test suites** wherever they exist (FP1), plus representative applied problems. |
| `edge` | ~10 | Very difficult / degenerate: extreme tails, ill-conditioning, near-singular, stiff, collinear, empty/1-element, IEEE inf/nan/signed-zero, branch cuts. |

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

## 5. Limitation envelopes & the self-awareness gate (owner reframing)

Each strategy declares a **documented limitation** (from CLAUDE.md Known
Limitations + SciPy docs + the audit), encoded as the per-tier `tol`. The
workbench, per case:

1. Runs every applicable strategy.
2. Records each result's error vs the oracle.
3. Ranks strategies by error and emits a comparison row.

**The gate is NOT "is the result within tol".** Per the owner: when a strategy is
applied *outside its valid envelope*, the dangerous outcome is the library
**silently returning a plausible-but-wrong answer with no signal**. So the real
test is the library's **self-awareness**:

> For every case that lies outside a strategy's declared limitation envelope, the
> library MUST surface an appropriate **warning — or, better, a recoverable
> error/diagnostic** — telling the caller "this result may be unreliable here."
> **The test FAILS when the library gives a wrong (or even a right) answer in an
> out-of-envelope regime WITHOUT emitting that signal.** A numeric deviation
> alone is not the failure; the *missing diagnostic* is.

So each fixture case is tagged `inEnvelope: true|false` per strategy. The
workbench asserts:
- in-envelope → result within `tol` AND no spurious limitation-diagnostic;
- out-of-envelope → the library emitted the expected limitation-diagnostic
  (a recoverable error or a warning), regardless of the numeric value.

This requires a **limitation-diagnostics mechanism in the library** (§5b) — the
single new piece of public surface this workbench introduces. Cross-strategy, the
report still flags a strategy that is **unexpectedly better** than its declared
envelope (a stale/too-loose limitation to tighten).

Example declared envelopes: `simps` exact to deg≤2 on non-uniform grids; T-dist
`ppf` ~5 digits for |p|>0.9999; `logm/sqrtm` defective/complex-eigenvalue support
(post-#7); `expm` full double precision (post-CR-D1); BDF-1 stiff only / O(√rtol)
(post-#15).

## 5b. Limitation-diagnostics mechanism (NEW public surface — needs owner sign-off)

Swift has no runtime "warning". Candidate mechanisms (one decision, library-wide):

1. **Recoverable-error / Result channel (owner's stated preference "or better, a
   recoverable error").** Fallible/limited entry points return a value PLUS an
   optional `[NumericDiagnostic]`, e.g. a `Diagnosed<T>` wrapper
   `{ value: T, diagnostics: [NumericDiagnostic] }`, or throw a recoverable
   `NumericDiagnostic.outsideEnvelope(...)` the caller can catch and still read a
   value from. Explicit, testable, no global state. Cost: touches many signatures
   (mitigated by additive overloads + deprecation of the bare ones).
2. **Diagnostics context/handler.** A `NumericDiagnostics.withHandler { ... }`
   scope (or a settable sink) the library calls when entering a degraded regime;
   the workbench installs a collecting handler. Minimal signature churn; relies on
   scoped/thread state.
3. **Result-type field.** Only the result-struct-returning APIs (QuadResult,
   ODEResult, OLSResult, …) gain a `diagnostics: [NumericDiagnostic]` field. Clean
   for those; doesn't cover bare-value functions (simps, ppf, …).

**Proposed:** a hybrid — `NumericDiagnostic` enum + a `Diagnosed<T>` return on the
specific entry points that have known envelopes (additive overloads; bare ones
kept + deprecated), and the existing result structs gain a `diagnostics` field.
This is an architecture-level addition → **awaiting owner pick of mechanism**
before the workbench build proceeds past the harness skeleton.

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
inEnvelope | diagnostic? | status`, a per-domain rollup, and a final
`N domains, M cases, K self-awareness failures`. A **self-awareness failure** =
an out-of-envelope case where the library emitted NO limitation-diagnostic (the
gated condition, §5). A numeric `tol` miss while in-envelope is a separate
regression flag. The XCTest gate (`WorkbenchGateTests`) asserts zero
self-awareness failures; envelope/accuracy stats are reported, and (per the owner)
a missing diagnostic is the hard failure, not the deviation itself.

## 8. Build / run

```bash
swift build --product NumericSwiftWorkbench
.build/debug/NumericSwiftWorkbench            # all domains
.build/debug/NumericSwiftWorkbench integration optimization   # selected
swift test --filter WorkbenchGateTests        # CI gate
NUMERICSWIFT_REGENERATE_WORKBENCH=1 ...        # regen fixtures (offline, scipy)
```

## 9. Decisions

- ✅ **Case volume:** 100/domain (10/80/10) ≈ 1,600 total. (Owner.)
- ✅ **Gate semantics:** the gate tests the library's *self-awareness* of its
  limitations — a missing limitation-diagnostic on an out-of-envelope case is the
  hard failure; numeric deviation alone is only a reported flag. (Owner, §5.)
- ✅ **Tracking:** commit WORKBENCH.md + generators + fixtures + the workbench
  target to the repo (shipped, reviewable infra).
- ⏳ **OPEN — diagnostics mechanism (§5b):** which mechanism the library uses to
  surface a limitation-diagnostic (recoverable `Diagnosed<T>` / context-handler /
  result-struct field / hybrid). This is new public API surface and gates the
  workbench build past the skeleton. **Owner pick needed.**
