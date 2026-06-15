//
//  LinAlg.swift
//  NumericSwift
//
//  Linear algebra operations using Accelerate framework for hardware-accelerated computation.
//  Follows NumPy/scipy.linalg patterns with namespaced API.
//
//  This file declares the `LinAlg` namespace, error types, and the two-tier
//  matrix size-cap system.  All concrete operations live in the LinAlg+*.swift
//  extension files:
//    - LinAlg+Matrix.swift         — Matrix struct + operators
//    - LinAlg+ComplexMatrix.swift  — ComplexMatrix struct
//    - LinAlg+Arithmetic.swift     — Factory + element-wise ops + matmul
//    - LinAlg+Properties.swift     — trace, det, inv, rank, cond, pinv, norm
//    - LinAlg+Decompositions.swift — lu, qr, svd, eig, eigvals, cholesky
//    - LinAlg+Solvers.swift        — solve, lstsq, solveTriangular, choSolve, luSolve
//    - LinAlg+MatrixFunctions.swift— expm, logm, sqrtm, funm
//    - LinAlg+Complex.swift        — csolve, csvd, ceig, ceigvals, cdet, cinv
//    - LinAlg+Internal.swift       — internal cross-file helpers
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation
import Accelerate

// MARK: - LinAlg Namespace

/// Linear algebra operations namespace.
///
/// Provides matrix operations, decompositions, and solvers using the Accelerate framework.
/// All operations use hardware-accelerated LAPACK/BLAS routines.
///
/// Example usage:
/// ```swift
/// let A = LinAlg.Matrix([[1, 2], [3, 4]])
/// let b = LinAlg.Matrix([5, 6])
/// let x = LinAlg.solve(A, b)
/// ```
public enum LinAlg {

    // MARK: - Errors

    /// Recoverable error conditions raised by fallible linear-algebra operations.
    ///
    /// These model *runtime* conditions a caller can sensibly catch and recover
    /// from — a shape supplied at runtime that an operation cannot accept. They
    /// are distinct from programmer errors (out-of-range subscripts, mismatched
    /// operator shapes, malformed literal `Matrix` construction), which remain
    /// `precondition` failures that trap, matching the Swift standard library's
    /// treatment of `Array` and `SIMD`.
    ///
    /// Numerical failure of an otherwise well-shaped problem (a singular system,
    /// a non-positive-definite matrix) is *not* reported here: those operations
    /// continue to return `nil`, because `Optional` already models "well-formed
    /// input, no answer."
    public enum LinAlgError: Error, Equatable, Sendable {
        /// An operation requiring a square matrix received a non-square one.
        case notSquare(rows: Int, cols: Int)
        /// Two operands have shapes that cannot be combined; the message names
        /// the specific dimension constraint that failed.
        case dimensionMismatch(String)
        /// A scalar parameter was outside its valid domain (e.g. a zero `step`
        /// or a `count` below the minimum); the message names the parameter.
        case invalidParameter(String)
    }

    // MARK: - Matrix Size Caps

    // -------------------------------------------------------------------------
    // Two-tier matrix size cap
    //
    // HARD cap  — `hardMaxMatrixElementCount`
    //   • Equals Int(Int32.max) = 2 147 483 647 — the LAPACK int32 element-count
    //     boundary.  Enforced via `precondition` inside every Matrix and
    //     ComplexMatrix constructor.  Overflow or exceeding this limit is a
    //     *programmer error* and is not catchable.
    //
    // SOFT cap  — `maxEvaluatorMatrixElements` (mutable, default 4096² = 16 777 216)
    //   • Enforced by the evaluator pre-pass (Task 20) before allocating any
    //     matrix from a user-facing expression.  Violation throws
    //     `LinAlgError.invalidParameter` (CONS-07) — *never* MathExprError.
    //   • Tunable from Swift host code via `setMaxEvaluatorMatrixElements(_:)`.
    //     That setter is deliberately NOT registered in the mathlex/Lua callable-
    //     symbol table (SEC-05); it is host-configuration only.
    // -------------------------------------------------------------------------

    /// The HARD element-count ceiling enforced at matrix construction time.
    ///
    /// Equal to `Int(Int32.max)` — the LAPACK `int32` boundary.  Constructing a
    /// matrix whose `rows * cols` (computed overflow-safely) equals or exceeds
    /// this value triggers a `precondition` trap.  The trap is **not catchable**;
    /// it signals a programmer error, not a runtime condition.
    public static let hardMaxMatrixElementCount: Int = Int(Int32.max)

    /// Default value for ``maxEvaluatorMatrixElements`` (4096 × 4096 = 16 777 216).
    private static let defaultEvaluatorMatrixElements: Int = 16_777_216

    /// Backing store for the tunable soft-cap; isolated to this file.
    private static var _maxEvaluatorMatrixElements: Int = defaultEvaluatorMatrixElements

    /// The SOFT element-count ceiling checked by the evaluator pre-pass.
    ///
    /// Matrices whose `rows * cols` exceeds this value throw
    /// ``LinAlgError/invalidParameter(_:)`` when validated by the evaluator
    /// (Task 20, `checkSoftCap(rows:cols:)`).  Default: 16 777 216 (4096²).
    ///
    /// Modify only from Swift host code via ``setMaxEvaluatorMatrixElements(_:)``.
    /// This getter is readable from both the host and the evaluator pre-pass.
    public static var maxEvaluatorMatrixElements: Int { _maxEvaluatorMatrixElements }

    /// Set the evaluator-level soft-cap threshold.
    ///
    /// - Parameter n: New cap (must be > 0 and ≤ ``hardMaxMatrixElementCount``).
    /// - Throws: ``LinAlgError/invalidParameter(_:)`` when `n` is out of range.
    ///
    /// > Important: This function is **host-configuration only** (SEC-05).  It
    /// > must not be registered in, or called from, the mathlex/Lua evaluator
    /// > bridge; doing so would let untrusted script code bypass the resource
    /// > guard entirely.
    public static func setMaxEvaluatorMatrixElements(_ n: Int) throws {
        guard n > 0 else {
            throw LinAlgError.invalidParameter(
                "maxEvaluatorMatrixElements must be positive, got \(n)")
        }
        guard n <= hardMaxMatrixElementCount else {
            throw LinAlgError.invalidParameter(
                "maxEvaluatorMatrixElements (\(n)) must not exceed hardMaxMatrixElementCount (\(hardMaxMatrixElementCount))")
        }
        _maxEvaluatorMatrixElements = n
    }

    // MARK: - Size-cap helpers (internal)

    /// Compute `rows * cols` using overflow-safe multiplication.
    ///
    /// - Returns: `(count: Int, overflow: Bool)`.  If either dimension is
    ///   negative the overflow flag is set, treating negative dims as invalid.
    static func elementCount(rows: Int, cols: Int) -> (count: Int, overflow: Bool) {
        guard rows >= 0, cols >= 0 else { return (0, true) }
        let result = rows.multipliedReportingOverflow(by: cols)
        return (count: result.partialValue, overflow: result.overflow)
    }

    /// Trap when the element count of a prospective matrix overflows `Int` or
    /// exceeds ``hardMaxMatrixElementCount``.
    ///
    /// Call this as the *first* guard in every Matrix and ComplexMatrix
    /// constructor, before any raw `rows * cols` arithmetic, to avoid UB.
    static func assertWithinHardCap(rows: Int, cols: Int) {
        let (count, overflow) = elementCount(rows: rows, cols: cols)
        precondition(
            !overflow && count <= hardMaxMatrixElementCount,
            "Matrix element count must not exceed hardMaxMatrixElementCount (Int32.max = \(hardMaxMatrixElementCount)); got rows=\(rows) cols=\(cols)")
    }

    /// Validate a prospective matrix shape against the soft-cap, throwing when
    /// it would be rejected by the evaluator pre-pass (AC3.6 / CONS-07).
    ///
    /// ## Two-tier interaction (AC7.2 / §4.10)
    ///
    /// The size-guard system has two tiers:
    ///
    /// - **SOFT cap** (this function): checks `rows * cols` against
    ///   ``maxEvaluatorMatrixElements``.  A finite product that exceeds the cap
    ///   throws ``LinAlgError/invalidParameter(_:)`` — a *catchable* error.
    ///   An Int-overflowing product is also thrown here as a secondary defence
    ///   (the product is meaningless, so the cap comparison cannot be meaningful).
    ///
    /// - **HARD cap** (constructor ``assertWithinHardCap(rows:cols:)``): uses
    ///   `precondition` to trap when `rows * cols` overflows `Int` or exceeds
    ///   ``hardMaxMatrixElementCount``.  This trap is **not catchable** — it is
    ///   a programmer error, not a runtime condition.
    ///
    /// The evaluator pre-pass calls this function BEFORE delegating to any LinAlg
    /// or LAPACK routine; an over-cap allocation is never attempted (AC3.6).
    /// The check is placed at the **value-construction boundary** — i.e. on every
    /// result-shape that the dispatcher predicts, not inside the LinAlg primitives.
    ///
    /// ## Scope honesty (MF-5 / §5)
    ///
    /// This guard bounds the element count of **each individual result matrix**.
    /// It does **not** bound the cumulative working set of a chained expression:
    /// *k* at-cap intermediates held simultaneously consume *k × maxEvaluatorMatrixElements*
    /// doubles, which is neither detected nor rejected by this function.
    /// Cumulative working-set bounding is deferred to a future release (§14 / v-next).
    ///
    /// - Parameters:
    ///   - rows: Prospective row count.
    ///   - cols: Prospective column count.
    /// - Throws: ``LinAlgError/invalidParameter(_:)`` when `rows * cols` exceeds
    ///   ``maxEvaluatorMatrixElements`` or when the product overflows `Int`.
    ///
    /// Per CONS-07 the error type is always ``LinAlgError`` — never MathExprError.
    public static func checkSoftCap(rows: Int, cols: Int) throws {
        let (count, overflow) = elementCount(rows: rows, cols: cols)
        guard !overflow && count <= maxEvaluatorMatrixElements else {
            let desc = overflow ? "overflows Int" : "\(count)"
            throw LinAlgError.invalidParameter(
                "matrix element count \(desc) exceeds soft cap \(maxEvaluatorMatrixElements)")
        }
    }

    /// Convenience overload accepting a shape tuple; delegates to
    /// ``checkSoftCap(rows:cols:)``.
    public static func checkSoftCap(shape: (rows: Int, cols: Int)) throws {
        try checkSoftCap(rows: shape.rows, cols: shape.cols)
    }
}
