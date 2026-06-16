//
//  NumericDispatch+MatrixPower.swift
//  NumericSwift
//
//  Matrix exponentiation EVAL implementations for the unified numeric pipeline.
//
//  Covers:
//    - Real matrix integer power (exponentiation-by-squaring, O(log |n|))
//    - Complex matrix integer power (deferred stub — throws .unsupportedNode)
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

    /// complexMatrix^n integer power — deferred to a future task.
    ///
    /// Complex-matrix integer power is deferred; the hard preconditions (square,
    /// integer exponent) are already checked by the caller (`applyPow`).
    static func evalComplexMatrixPow(
        cm: LinAlg.ComplexMatrix, exponent: Double
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix^scalar (Task 13)")
    }
}
