//
//  NumericDispatch+MatrixPower.swift
//  NumericSwift
//
//  Matrix exponentiation EVAL implementations for the unified numeric pipeline.
//
//  Covers:
//    - Real matrix integer power (exponentiation-by-squaring, O(log |n|))
//    - Complex matrix integer power (exponentiation-by-squaring; cinv for n < 0)
//
//  Caller contracts (enforced by NumericDispatch+BinaryOps.swift `applyPow`):
//    - `matrix` is square (rows == cols)
//    - `exponent` has no fractional part (exponent == exponent.rounded())
//
//  Licensed under the Apache License, Version 2.0.
//

// MARK: - Pow EVAL implementations

extension NumericDispatch {

    /// Raise a square real matrix to an integer power via exponentiation-by-squaring.
    ///
    /// Contracts enforced by the caller (`applyPow`) before this function is invoked:
    ///   - `matrix` is square (`rows == cols`)
    ///   - `exponent` has no fractional part (`exponent == exponent.rounded()`)
    ///
    /// Semantics:
    ///   - `n > 0`: repeated matrix multiplication using exponentiation-by-squaring,
    ///     O(log n) multiplications.
    ///   - `n == 0`: identity matrix of the same size (A⁰ = I by convention).
    ///   - `n < 0`: `inv(A^|n|)`; throws `MathExprError.invalidArguments("inverse of singular
    ///     matrix")` when A is singular.
    ///
    /// - Parameters:
    ///   - matrix:   A square `LinAlg.Matrix`.
    ///   - exponent: Integer-valued `Double` exponent (may be negative).
    /// - Returns: `NumericValue.matrix(_)` containing the result.
    /// - Throws: `MathExprError.invalidArguments` when the matrix is singular and `n < 0`;
    ///           `LinAlgError.notSquare` propagated from `LinAlg.inv` if shapes are wrong
    ///           (defensive — caller already checked squareness).
    static func evalMatrixPow(
        matrix: LinAlg.Matrix, exponent: Double
    ) throws -> NumericValue {
        // Self-protect the `Int(exponent)` cast: a finite integer-valued Double
        // beyond ±2^53 (e.g. 1e20) passes the caller's `== rounded()` guard but
        // overflows `Int` and would TRAP (process kill). 2^53 is also the largest
        // Double that represents consecutive integers exactly, so anything beyond
        // is a meaningless matrix exponent. (Audit CR — issue #1 follow-up.)
        guard exponent.magnitude <= 9_007_199_254_740_992 else {
            throw MathExprError.invalidArguments(
                "matrix power exponent \(exponent) exceeds ±2^53; "
                + "an exponent this large is not a meaningful matrix power")
        }
        let n = Int(exponent)       // caller guarantees no fractional part

        // n == 0 → A⁰ = identity regardless of A (even singular)
        if n == 0 {
            return .matrix(LinAlg.eye(matrix.rows))
        }

        // For negative exponents compute A^|n| then invert
        let absN = n < 0 ? -n : n

        // Exponentiation by squaring: O(log |n|) multiplications
        var result = LinAlg.eye(matrix.rows)    // accumulator starts as identity
        var base   = matrix                     // running square
        var remaining = absN
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = LinAlg.dot(result, base)
            }
            base      = LinAlg.dot(base, base)
            remaining >>= 1
        }

        if n < 0 {
            // Negative power: invert the positive-power result
            guard let invResult = try LinAlg.inv(result) else {
                throw MathExprError.invalidArguments(
                    "matrix power A^\(n): the matrix (or A^\(absN)) is singular; "
                    + "negative powers require an invertible matrix")
            }
            return .matrix(invResult)
        }
        return .matrix(result)
    }

    /// Raise a square complex matrix to an integer power via exponentiation-by-squaring.
    ///
    /// Contracts enforced by the caller (`applyPow`) before this function is invoked:
    ///   - `cm` is square (`rows == cols`)
    ///   - `exponent` has no fractional part (`exponent == exponent.rounded()`)
    ///
    /// Semantics mirror ``evalMatrixPow(matrix:exponent:)``:
    ///   - `n > 0`: exponentiation-by-squaring (O(log n) complex multiplies).
    ///   - `n == 0`: complex identity matrix of the same size (A⁰ = I).
    ///   - `n < 0`: `cinv(A^|n|)`; throws `MathExprError.invalidArguments` when the
    ///     matrix (or its positive power) is singular.
    ///
    /// - Throws: `MathExprError.invalidArguments` when `|exponent| > 2^53` (the
    ///   `Int(exponent)` cast would otherwise trap) or when a negative power is
    ///   requested for a singular matrix.
    static func evalComplexMatrixPow(
        cm: LinAlg.ComplexMatrix, exponent: Double
    ) throws -> NumericValue {
        // Self-protect the `Int(exponent)` cast (same rationale as evalMatrixPow):
        // a finite integer-valued Double beyond ±2^53 passes the caller's
        // `== rounded()` guard but overflows `Int` and would TRAP.
        guard exponent.magnitude <= 9_007_199_254_740_992 else {
            throw MathExprError.invalidArguments(
                "matrix power exponent \(exponent) exceeds ±2^53; "
                + "an exponent this large is not a meaningful matrix power")
        }
        let n = Int(exponent)       // caller guarantees no fractional part
        let dim = cm.rows           // caller guarantees square

        // n == 0 → A⁰ = identity regardless of A (even singular)
        if n == 0 {
            return .complexMatrix(LinAlg.ComplexMatrix(LinAlg.eye(dim)))
        }

        let absN = n < 0 ? -n : n

        // Peak-aware admission control. Each `complexMatrixMultiply` allocates the
        // same six result-sized buffers as `complexMatmul` (4 real products + 2
        // output arrays) but skips its soft-cap check; the caller (`applyPow`) only
        // validated the result shape (dim×dim ≤ cap), not the 6× peak. The shape is
        // constant dim×dim through the squaring loop, so check the peak once here.
        let elements = dim * dim
        let cap = LinAlg.maxEvaluatorMatrixElements
        guard elements <= cap / complexMatmulWorkingSetMultiplier else {
            throw LinAlg.LinAlgError.invalidParameter(
                "complexMatrix^scalar peak working set (\(elements) × "
                + "\(complexMatmulWorkingSetMultiplier) elements) exceeds soft cap \(cap) "
                + "(per-matrix limit \(cap / complexMatmulWorkingSetMultiplier) for complex matmul)")
        }

        // Exponentiation by squaring: O(log |n|) complex multiplications.
        var result = LinAlg.ComplexMatrix(LinAlg.eye(dim))   // accumulator = identity
        var base   = cm                                      // running square
        var remaining = absN
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = complexMatrixMultiply(result, base)
            }
            base      = complexMatrixMultiply(base, base)
            remaining >>= 1
        }

        if n < 0 {
            // Negative power: invert the positive-power result.
            guard let invResult = try LinAlg.cinv(result) else {
                throw MathExprError.invalidArguments(
                    "matrix power A^\(n): the matrix (or A^\(absN)) is singular; "
                    + "negative powers require an invertible matrix")
            }
            return .complexMatrix(invResult)
        }
        return .complexMatrix(result)
    }
}
