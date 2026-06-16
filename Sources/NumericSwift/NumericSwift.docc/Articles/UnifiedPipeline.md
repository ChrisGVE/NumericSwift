# Unified Numeric Pipeline

The `NumericValue` type tower, `NumericDispatch` routing surface, and unified evaluator front door for mathematical expression evaluation over scalars, complex numbers, real matrices, and complex matrices.

## Overview

The unified numeric pipeline (NumericSwift 0.3.0) is a single-pass evaluation engine that handles all numeric kinds in one recursive traversal. The key components are:

- **`NumericValue`** — a discriminated union (enum) over the four numeric kinds.
- **`NumericDispatch`** — the dispatch surface routing `(operator, kinds)` → result.
- **`MathExpr.evaluateUnified`** — the public entry point for AST evaluation.

Together they replace the previous split between a scalar evaluator (`evaluate`) and a separate linear-algebra path, while keeping the legacy public API unchanged.

## The NumericValue Type Tower

`NumericValue` unifies the four numeric kinds the pipeline produces and consumes:

| Kind | Swift type | Coercion lattice position |
|------|-----------|--------------------------|
| `.scalar(Double)` | `Double` | Narrowest |
| `.complex(Complex)` | `Complex` | Above scalar |
| `.matrix(LinAlg.Matrix)` | row-major `[Double]` | Above scalar |
| `.complexMatrix(LinAlg.ComplexMatrix)` | two `[Double]` arrays | Widest |

Widening (`scalar → complex`, `matrix → complexMatrix`, `scalar → complexMatrix`) is implicit and automatic. Narrowing never happens implicitly — with exactly one documented exception (§4.3a, see below).

### Equality

`NumericValue` does **not** conform to `Equatable`. Use the explicit methods instead:

```swift
let a = NumericValue.scalar(1.0)
let b = NumericValue.scalar(1.0)
let c = NumericValue.scalar(Double.nan)

a.isExactlyEqual(to: b)      // true
c.isExactlyEqual(to: c)      // false  — NaN is non-reflexive (IEEE 754 §5.11)
a.isApproximatelyEqual(to: b, tolerance: 1e-10)  // true
```

**`isExactlyEqual` IEEE-754 semantics:**
- NaN in any component returns `false` — even `nan.isExactlyEqual(to: nan)`.
- `+0.0` and `−0.0` compare as **equal** (IEEE 754 §5.10 value equality). Use `bitPattern` if sign-of-zero identity is required.
- Matrix elements are compared with exact `Double ==`, **bypassing** `LinAlg.Matrix`'s tolerance-based `==` operator (which uses a hardcoded 1e-10 tolerance).
- Matrices of differing shape return `false`.

**`isApproximatelyEqual` semantics:**
- Each component must satisfy `|a − b| ≤ tolerance` (default 1e-10).
- NaN propagates: `|NaN − x|` is NaN, never ≤ tolerance, so NaN-containing values return `false`.
- The relation is **non-transitive**: `a ≈ b` and `b ≈ c` does not imply `a ≈ c`.

### Accessors and Shape Introspection

```swift
let v = NumericValue.matrix(LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3]))

v.kind              // .matrix
v.isMatrix          // true
v.asMatrix          // Optional<LinAlg.Matrix>
v.rows              // Optional(3)
v.cols              // Optional(1)
v.shape             // Optional((rows: 3, cols: 1))
v.elementCount      // 3
v.is1x1             // false  (3×1 is not 1×1)
v.typeAndShapeDescription  // "matrix(3x1)"
```

`rows` / `cols` return `nil` for `.scalar` and `.complex` — scalars are dimensionless in the pipeline, **not** treated as 1×1. Use `is1x1` to test for a 1×1 matrix (the §4.3a coercion gate).

Throwing extractors are available for pipeline stages that cannot handle a kind mismatch:

```swift
let x = try v.asMatrixThrowing()   // or throws AccessorError.kindMismatch
```

## The Dispatch Contract

`NumericDispatch` is a pure `enum` namespace (no stored state, fully re-entrant). Three public entry points cover all dispatch needs:

```swift
// Binary operators: +, −, *, /, **, %
let result = try NumericDispatch.applyBinary(.mul, lhs: a, rhs: b)

// Unary operators: neg, pos, !, transpose
let neg = try NumericDispatch.applyUnary(.neg, operand: v)

// Named functions: sin, det, inv, dot, hadamard, …
let d = try NumericDispatch.applyFunction("det", args: [m])
```

### Operator Semantics

- `*` (`BinaryOp.mul`) between two matrices is **matrix multiplication** (matmul), not element-wise. Element-wise multiplication uses the `hadamard` named function.
- `dot(u, v)` for two column vectors returns a `.scalar` after the 1×1 → scalar coercion (§4.3a).
- `dot(u, v)` for complex column vectors is **bilinear** (Σ uᵢ·vᵢ, no conjugation). Hermitian/conjugated inner product is deferred to a future release.

### The §15 Routing Truth Table

Every `(op, lhsKind, rhsKind)` combination maps to an entry in the §15 truth table. The routing discriminant is `NumericValue.Kind` from `NumericValue+Accessors.swift`. No competing kind enum exists — all dispatch switches on `value.kind`.

## The Two-Mechanism Error Model

The dispatch layer uses two distinct error mechanisms, depending on the nature of the operation:

### Group-A — Pre-validate, then call (catchable)

Group-A operators use `precondition` internally inside `LinAlg`. The dispatcher **pre-validates** shapes and divisors, throwing `MathExprError` **before** any `LinAlg` call, so a shape mismatch is always a recoverable `Error`, never a process trap:

- `MathExprError.shapeMismatch(String)` — incompatible matrix shapes for add/sub/hadamard/matmul/dot.
- `MathExprError.divisionByZero` — scalar or matrix ÷ 0.
- `MathExprError.invalidArguments(String)` — undefined `(lhsKind, rhsKind)` combination.

### Group-B — Propagate LinAlg errors (catchable)

Group-B named functions (`inv`, `det`, `trace`, `expm`, `logm`, `sqrtm`, `cdet`, `cinv`) throw `LinAlgError` internally. The dispatcher calls them with `try` and propagates the error unmodified:

- `LinAlgError.notSquare` — non-square matrix passed to a square-only function.

```swift
do {
    let inv = try NumericDispatch.applyFunction("inv", args: [nonSquareMatrix])
} catch let err as LinAlg.LinAlgError {
    // Group-B: LinAlgError.notSquare propagated here
} catch let err as MathExprError {
    // Group-A: e.g. kindMismatch, divisionByZero
}
```

## The Two-Tier Size Cap

The pipeline enforces two independent size limits to prevent allocator exhaustion from expression-driven inputs:

### HARD cap — `LinAlg.hardMaxMatrixElementCount`

- Value: `Int(Int32.max)` = 2 147 483 647 (the LAPACK `int32` element-count boundary).
- Enforced by a `precondition` inside every `LinAlg.Matrix` and `LinAlg.ComplexMatrix` constructor.
- **Not catchable.** A matrix exceeding this limit is a programmer error, not a runtime condition.

### SOFT cap — `LinAlg.maxEvaluatorMatrixElements`

- Default value: 16 777 216 (4096²).
- Checked by `LinAlg.checkSoftCap(rows:cols:)` **before** allocating any matrix from an expression.
- Throws `LinAlgError.invalidParameter` (never `MathExprError`) when the limit is exceeded.
- Tunable from Swift host code only via `LinAlg.setMaxEvaluatorMatrixElements(_:)`.

> Important: The soft cap setter is **host-configuration only** (SEC-05). It must not be registered in, or called from, the mathlex/Lua evaluator bridge — doing so would allow untrusted script code to bypass the resource guard.

```swift
// Raise the soft cap for batch-processing large matrices
try LinAlg.setMaxEvaluatorMatrixElements(64 * 1024 * 1024)  // 64M elements

// The hard cap is always a ceiling
print(LinAlg.hardMaxMatrixElementCount)  // 2147483647
```

The soft cap bounds each **individual result matrix**, not the cumulative working set of a chained expression. Cumulative working-set bounding is deferred to a future release.

## The §4.3a 1×1 → Scalar Coercion

The pipeline collapses a 1×1 matrix result to `.scalar` (or `.complexMatrix` result to `.complex`) at **exactly two sites**:

1. `applyBinary(.mul, lhs: .matrix, rhs: .matrix)` — when `M * M` produces a 1×1 matrix (i.e. vec·vec via matmul).
2. `applyFunction("dotProduct", args: [u, v])` — when `dot(u, v)` on column vectors produces a 1×1 matrix.

The complex analogues collapse 1×1 `.complexMatrix` → `.complex` at the CM * CM and `dotProduct(CM, CM)` sites.

**The collapse does NOT fire:**
- For user-constructed 1×1 matrices used as operands — they stay `.matrix`.
- For 1×1 results of add, sub, hadamard, elementDiv, transpose, neg, inv, expm, logm, or sqrtm.
- Globally or at any other site.

Use `is1x1` to test whether a `NumericValue` holds a 1×1 matrix:

```swift
let m = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 1, data: [42.0]))
m.is1x1   // true  — this is a 1×1 matrix (coercible by §4.3a)
m.isScalar // false — it is NOT already a scalar
```

## The Unified Evaluator Entry Point

`MathExpr.evaluateUnified` is the single front door for the unified pipeline:

```swift
// Matrix multiplication via variable binding
let A = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
let B = LinAlg.Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])

let ast = try MathExpr.parse("A * B")
let result = try MathExpr.evaluateUnified(ast, values: [
    "A": .matrix(A),
    "B": .matrix(B)
])
// result == .matrix([[19, 22], [43, 50]])

// det and inv
let det  = try MathExpr.evaluateUnified(try MathExpr.parse("det(A)"), values: ["A": .matrix(A)])
// det == .scalar(-2.0)
```

### Fallback-Parser Bracket Limitation

The default-build pure-Swift parser has **no bracket tokenizer**. Expressions such as `[1, 2, 3]` or `[[1, 2], [3, 4]]` cannot be parsed and will throw `MathExprError.parseError` from `MathExpr.parse(_:)`.

- On the default build, matrix values enter the pipeline through the `values:` binding dictionary (as shown above).
- Bracket-literal parsing (`[1, 2; 3, 4]`) is only available with the opt-in MathLex Rust backend (`NUMERICSWIFT_INCLUDE_MATHLEX=1`).
- **Imaginary literals are not affected** — expressions such as `2*i`, `3.5*i`, and `1 + 2*i` parse and evaluate correctly on the default build.

### Legacy Wrappers

The public `MathExpr.evaluate` and `MathExpr.evaluateComplex` wrappers delegate to `evaluateUnified` and extract the result. These wrappers preserve the pre-0.3.0 API contract. See ``MathExpr`` for details.

## Complex-Context Promotion (issue #1, resolved)

`evaluateUnified` takes a `complexMode: Bool = false` flag, threaded through `NumericDispatch.applyBinary`/`applyFunction`/`applyPow`. `evaluateComplex` sets it `true`, so a negative-real `sqrt`/`log`/`ln` scalar argument — and the `^` operator with a negative base and a non-integer exponent — is promoted to the complex principal value (`sqrt(-1) ≈ 0 + 1i`) instead of NaN. The real `evaluate` leaves the flag `false`, preserving the IEEE-754 NaN contract.

The promotion set is narrow by design (the complex-native names only); `pow(x,y)` as a function, `log10`/`log2`, and inverse-trig still return NaN for negative reals, matching the legacy complex evaluator. Results follow the numpy/SciPy principal (upper) branch. Resolved GitHub issue [#1](https://github.com/ChrisGVE/NumericSwift/issues/1); see ``MathExpr`` for the full convention note.

## Topics

### Type Tower

- ``NumericValue``
- ``NumericValue/Kind-swift.enum``
- ``NumericValue/AccessorError``

### Dispatch Surface

- ``NumericDispatch``
- ``NumericDispatch/applyBinary(_:lhs:rhs:complexMode:)``
- ``NumericDispatch/applyUnary(_:operand:)``
- ``NumericDispatch/applyFunction(_:args:complexMode:)``

### Evaluator Front Door

- ``MathExpr/evaluateUnified(_:values:complexMode:)``

### Size Caps

- ``LinAlg/hardMaxMatrixElementCount``
- ``LinAlg/maxEvaluatorMatrixElements``
- ``LinAlg/setMaxEvaluatorMatrixElements(_:)``
- ``LinAlg/checkSoftCap(rows:cols:)``
- ``LinAlg/checkSoftCap(shape:)``
