# Mathematical Expressions

Mathematical expression parsing and evaluation, including the unified numeric pipeline.

## Overview

The `MathExpr` module provides parsing and evaluation for mathematical expressions
given as strings. Two evaluation paths are available:

- **Legacy scalar/complex path** — `evaluate(_:variables:)` and `evaluateComplex(_:variables:complexVariables:)`
  for expressions that produce a single `Double` or `Complex` value.
- **Unified pipeline path** — `evaluateUnified(_:values:complexMode:)`, the single front
  door that traverses the full AST and produces a `NumericValue`, supporting scalars,
  complex numbers, real matrices, and complex matrices in one call.

## Unified Evaluator (Phase 3)

The unified evaluator accepts a `[String: NumericValue]` variable binding dictionary,
so matrix and complex values can be supplied directly alongside scalars.

```swift
// Matrix multiplication via variable binding
let A = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
let B = LinAlg.Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])
let ast = try MathExpr.parse("A * B")
// * = matmul (SciPy/scipy.linalg convention)
let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(A), "B": .matrix(B)])
// result == .matrix([[19,22],[43,50]])

// Mixed scalar and matrix
let ast2 = try MathExpr.parse("2 * A")
let r2 = try MathExpr.evaluateUnified(ast2, values: ["A": .matrix(A)])
// r2 == .matrix([[2,4],[6,8]])

// Complex variables
let z = Complex(re: 1, im: 2)
let ast3 = try MathExpr.parse("z * z")
let r3 = try MathExpr.evaluateUnified(ast3, values: ["z": .complex(z)])
// r3 == .complex(-3 + 4i)
```

## Supported AST Node Kinds

The unified evaluator handles:

| Node kind | Produces |
|-----------|----------|
| `.integer`, `.float` | `.scalar` |
| `.constant` (pi, e, inf, nan) | `.scalar` |
| `.constant(.i)` | `.complex(0+1i)` |
| `.variable(name)` | The bound `NumericValue` |
| `.rational(n, d)` | Result of n / d |
| `.complex(re, im)` | `.complex` |
| `.binary(op, l, r)` | Via `NumericDispatch.applyBinary` |
| `.unary(op, operand)` | Via `NumericDispatch.applyUnary` |
| `.function(name, args)` | Via `NumericDispatch.applyFunction` |
| `.vector([...])` | `.matrix` column vector (mathlex build) |
| `.matrix([[...]])` | `.matrix` (mathlex build) |
| `.dotProduct(l, r)` | `.scalar` after 1×1 coercion (§4.3a) |
| `.determinant(m)` | `.scalar` |
| `.matrixInverse(m)` | `.matrix` |
| `.trace(m)` | `.scalar` |
| `.conjugateTranspose(m)` | `.matrix` or `.complexMatrix` |
| `.rank(m)` | `.scalar` (numerical rank via SVD) |

Calculus, logic, set-theory, tensor, and quaternion nodes throw `.unsupportedNode`.

## Operator Semantics

`*` between two matrices is **matrix multiplication** (matmul), not element-wise.
This matches `LinAlg.Matrix * Matrix` (`LinAlg.swift:2022`) and scipy.linalg conventions.
Element-wise multiplication uses the `hadamard` named function.

`dot(u, v)` on two column vectors returns a `.scalar` after the automatic 1×1 → scalar
coercion (§4.3a).

## Parser Backends

```swift
// Default build (pure-Swift shunting-yard parser):
let ast = try MathExpr.parse("sin(x) + cos(x)")

// With MathLex Rust backend (NUMERICSWIFT_INCLUDE_MATHLEX=1):
// Full grammar including LaTeX and matrix literal syntax [1, 2; 3, 4]
let ast2 = try MathExpr.parseLatex(#"\frac{x^2}{2}"#)
```

On the default build the fallback parser has no bracket tokenizer, so `.vector` and
`.matrix` literal AST nodes are never emitted. Matrix values are supplied via the
`values:` dictionary instead and the evaluator handles them identically.

## Error Surface

```swift
do {
    let result = try MathExpr.evaluateUnified(ast, values: ["x": .scalar(3.0)])
} catch MathExprError.undefinedVariable(let name) {
    print("Variable '\(name)' not bound")
} catch MathExprError.unknownFunction(let name) {
    print("No function named '\(name)'")
} catch MathExprError.divisionByZero {
    print("Division by zero")
} catch MathExprError.invalidArguments(let msg) {
    print("Argument error: \(msg)")
} catch MathExprError.unsupportedNode(let kind) {
    print("AST node '\(kind)' not evaluable")
} catch MathExprError.nonFiniteFloat {
    print("NaN sentinel in AST (float(nil))")
} catch MathExprError.shapeMismatch(let msg) {
    print("Matrix shape error: \(msg)")
} catch let laErr as LinAlg.LinAlgError {
    // Group-B functions propagate their own notSquare error
    print("Linear algebra error: \(laErr)")
}
```

## Bilinear Dot Product Semantics

`dot(u, v)` for two complex column vectors computes the **bilinear** inner product
Σ uᵢ·vᵢ — there is **no conjugation** of the first argument. This differs from the
Hermitian (conjugate-linear) inner product ⟨u, v⟩ = Σ ū_i · vᵢ used in physics
and much of numerical linear algebra. Hermitian/conjugated dot product is deferred
to a future release.

## Complex-Context Promotion (resolved — GitHub issue #1)

`MathExpr.evaluateComplex` evaluates in **complex mode**: a negative-real
`sqrt`, `log`, or `ln` scalar argument — and the `^` operator with a negative
base and a non-integer exponent — is promoted to its complex principal value
rather than returning NaN.

```swift
let z = try MathExpr.evaluateComplex(MathExpr.parse("sqrt(-1)"))   // ≈ 0 + 1i
```

Mechanism: `evaluateUnified` takes a `complexMode: Bool = false` flag threaded
through `NumericDispatch.applyBinary`/`applyFunction`/`applyPow`. `evaluateComplex`
sets it `true`; the real `evaluate` leaves it `false`, so the real path keeps its
IEEE-754 NaN contract (`eval("sqrt(-4)")` is still NaN).

The promotion set is deliberately narrow — exactly the names whose legacy complex
path was complex-native. The **`pow(x, y)` function**, `log10`/`log2`, and the
inverse-trig functions still return NaN for negative-real arguments, matching the
legacy complex evaluator's real-fallback behaviour. Only the `^` **operator**
promotes a negative base.

**Branch convention:** results are the numpy/SciPy *principal* (upper) branch —
`sqrt(-1) = +i`, `log(-1) = +iπ` — per design-philosophy #1. This matches legacy
in magnitude; the imaginary sign differs from legacy's `-i`/`-iπ`, which were an
incidental signed-zero artifact, not an intended convention.

## Legacy Scalar Evaluation

The legacy scalar and complex paths are unchanged and remain the primary API for
expressions that return `Double` or `Complex`:

```swift
// Scalar evaluation
let result: Double = try MathExpr.eval("sin(x) + 2", variables: ["x": 0.5])

// Complex evaluation
let z: Complex = try MathExpr.evaluateComplex(
    try MathExpr.parse("(1 + i) * (1 - i)"))
```

## Topics

### Parsing

- ``MathExpr/parse(_:)``
- ``MathExpr/parseLatex(_:)``
- ``MathLexExpression``

### Unified Evaluation (Phase 3)

- ``MathExpr/evaluateUnified(_:values:complexMode:)``

### Legacy Scalar/Complex Evaluation

- ``MathExpr/evaluate(_:variables:)``
- ``MathExpr/eval(_:variables:)``
- ``MathExpr/evaluateComplex(_:variables:complexVariables:)``

### Utilities

- ``MathExpr/findVariables(in:)-4c5q0``
- ``MathExpr/findVariables(in:)-6fkpu``
- ``MathExpr/substitute(_:with:)``
- ``MathExpr/toString(_:)``

### Errors

- ``MathExprError``
