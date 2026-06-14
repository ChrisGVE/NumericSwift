//
//  NumericDispatch+CoercionLattice.swift
//  NumericSwift
//
//  Coercion lattice helpers for the unified numeric pipeline.
//
//  ## §15 coercion lattice — partial order and promotion rules
//
//  The pipeline enforces a **widening-only** coercion lattice over the four
//  NumericValue kinds:
//
//      scalar  →  complex
//         ↓          ↓
//      matrix  →  complexMatrix
//
//  Widening always produces the more-expressive kind and never loses information.
//  Narrowing never happens implicitly (no `.complex → .scalar` without an
//  explicit extraction), with exactly one documented **dimensional collapse**
//  (§4.3a, see below).
//
//  ### Promotion rules
//
//  | Situation                        | Rule               | Helper                      |
//  |----------------------------------|--------------------|-----------------------------|
//  | scalar operand alongside complex | scalar → complex   | `promoteScalarToComplex`    |
//  | matrix operand alongside complex | matrix → complexMatrix | `promoteToComplexMatrix`|
//  | matrix alongside complexMatrix   | matrix → complexMatrix | `promoteToComplexMatrix`|
//
//  ### §4.3a 1×1-matrix → scalar collapse (single documented coercion site)
//
//  `LinAlg.dot` of two column vectors returns a 1×1 `Matrix` (not a scalar).
//  The pipeline coerces that 1×1 result to `.scalar` (or `.complex`) per §4.3a.
//
//  **This collapse fires ONLY at the dot/vec·vec result site.** Specifically:
//
//  - `applyMul` — `M * M` path (matmul via `LinAlg.dot`): calls `coerce1x1`
//  - `applyDotProduct` — `dotProduct(M, M)` function path: calls `coerce1x1`
//  - Complex analogue: `evalComplexMatrixDotProduct` and the `CM * CM` matmul
//    path MUST call `coerce1x1Complex` when the result is 1×1 (§4.3a, PRD §15
//    truth table: "1×1 → coerce to C").
//
//  **The collapse does NOT fire:**
//
//  - For a user-constructed 1×1 matrix used as an operand — it stays `.matrix`.
//  - For a 1×1 result of add/sub/hadamard/elementDiv — it stays `.matrix`.
//  - For a 1×1 result of transpose/neg/inv/expm/logm/sqrtm — it stays `.matrix`.
//  - Globally or anywhere outside the named dot result sites above.
//
//  ### M → CM promotion site
//
//  `promoteToComplexMatrix` converts `.matrix` → `.complexMatrix` by creating
//  a `LinAlg.ComplexMatrix` with the real array copied and a zero imaginary array.
//  Call sites (all in EVAL cells per §15 truth table):
//    • `complex ± matrix` / `matrix ± complex`
//    • `complex * matrix` / `matrix * complex`
//    • `matrix ± complexMatrix` / `complexMatrix ± matrix`
//    • `matrix * complexMatrix` / `complexMatrix * matrix`
//    • complex vec·vec dot (promote real vec → complex vec)
//
//  ### Scalar → complex promotion site
//
//  `promoteScalarToComplex` wraps a `Double` in a `Complex` with zero imaginary
//  part. This is used wherever a scalar operand appears alongside a complex value.
//  For the binary dispatch cells this promotion is performed inline (directly in
//  each case arm) because the extract-and-promote is a single expression. The
//  helper is provided here for use by multi-step EVAL cells that promote before
//  looping over elements (e.g. scalar ± complexMatrix broadcast).
//
//  Licensed under the Apache License, Version 2.0.
//

// MARK: - §4.3a 1×1 collapse helpers

extension NumericDispatch {

    // MARK: Real 1×1 collapse

    /// Collapse a 1×1 `.matrix` result to `.scalar`.
    ///
    /// ## §4.3a coercion rule
    ///
    /// This is the **only** site where a matrix is implicitly narrowed to a
    /// scalar. The PRD (§4.3a / §15) prescribes exactly two call sites:
    ///
    /// 1. `applyMul` — the `M * M` matmul path producing a 1×1 result.
    /// 2. `applyDotProduct` — the `dotProduct(M, M)` function path.
    ///
    /// **Do not call `coerce1x1` anywhere else.** A user-constructed 1×1
    /// matrix, or a 1×1 result from any other operation (add, hadamard,
    /// transpose, inv, …), stays `.matrix`.
    ///
    /// A 1×1 `.complexMatrix` is handled by ``coerce1x1Complex(_:)`` on
    /// the complex dot / complex matmul paths.
    ///
    /// - Parameter value: The `NumericValue` produced by a dot/matmul call.
    /// - Returns: `.scalar(data[0,0])` when `value` is a 1×1 `.matrix`;
    ///   otherwise returns `value` unchanged.
    @inline(__always)
    static func coerce1x1(_ value: NumericValue) -> NumericValue {
        if case .matrix(let m) = value, m.rows == 1, m.cols == 1 {
            return .scalar(m[0, 0])
        }
        return value
    }

    // MARK: Complex 1×1 collapse

    /// Collapse a 1×1 `.complexMatrix` result to `.complex`.
    ///
    /// ## §4.3a coercion rule (complex analogue)
    ///
    /// Prescribed call sites (PRD §15 truth table):
    ///
    /// 1. `evalComplexMatrixMulComplexMatrix` — `CM * CM` matmul path.
    /// 2. `evalComplexMatrixDotProduct` — `dotProduct(CM, CM)` path.
    ///
    /// **Do not call `coerce1x1Complex` anywhere else.** A 1×1 complex
    /// matrix produced by any other operation stays `.complexMatrix`.
    ///
    /// - Parameter value: The `NumericValue` produced by a complex dot/matmul call.
    /// - Returns: `.complex(re:im:)` when `value` is a 1×1 `.complexMatrix`;
    ///   otherwise returns `value` unchanged.
    @inline(__always)
    static func coerce1x1Complex(_ value: NumericValue) -> NumericValue {
        if case .complexMatrix(let cm) = value, cm.rows == 1, cm.cols == 1 {
            return .complex(Complex(re: cm[0, 0].re, im: cm[0, 0].im))
        }
        return value
    }
}

// MARK: - Promotion helpers

extension NumericDispatch {

    // MARK: M → CM promotion

    /// Promote a real `.matrix` to `.complexMatrix` with zero imaginary part.
    ///
    /// The promotion allocates a new `LinAlg.ComplexMatrix` from the existing
    /// `LinAlg.Matrix` using `ComplexMatrix(_ matrix:)` (LinAlg.swift :380), which
    /// copies the real data array and fills the imaginary array with zeros.
    /// This is the canonical M → CM widening step per the §15 coercion lattice.
    ///
    /// Call this helper in every EVAL cell that must handle a mixed
    /// (matrix, complexMatrix) pair — the real matrix is promoted to complex
    /// before any element-wise complex operation runs.
    ///
    /// - Parameter matrix: The real matrix to promote.
    /// - Returns: A `.complexMatrix` value containing the same data with zero
    ///   imaginary parts.
    static func promoteToComplexMatrix(_ matrix: LinAlg.Matrix) -> LinAlg.ComplexMatrix {
        LinAlg.ComplexMatrix(matrix)
    }

    // MARK: Scalar → complex promotion

    /// Promote a real scalar (`Double`) to a `Complex` with zero imaginary part.
    ///
    /// This is the S → C widening step per the §15 coercion lattice.
    /// Inline promotion (`Complex(scalar)`) is preferred for single-scalar
    /// expression arms; use this helper in multi-step EVAL cells that extract
    /// the scalar once and then iterate over elements.
    ///
    /// - Parameter scalar: The real scalar to promote.
    /// - Returns: `Complex(re: scalar, im: 0)`.
    @inline(__always)
    static func promoteScalarToComplex(_ scalar: Double) -> Complex {
        Complex(scalar)
    }
}
