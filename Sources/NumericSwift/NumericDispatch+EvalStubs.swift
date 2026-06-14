//
//  NumericDispatch+EvalStubs.swift
//  NumericSwift
//
//  EVAL stub placeholders for the §15 truth-table cells that require
//  evaluator-implemented arithmetic (§4.8). Each stub throws a typed
//  `MathExprError.unsupportedNode` with a message naming the task that will
//  replace it. Tasks 10–16 provide `extension NumericDispatch` in their own
//  files, replacing these bodies with real implementations.
//
//  All functions are `internal` (not `private`) so that extension files in
//  other source files can shadow them via Task N's extension. If a Task N
//  extension matches the same signature via a separate `extension NumericDispatch`
//  in a new file, Swift resolves the call at the call site using the more
//  specifically-typed overload. For routing functions that need complete
//  replacement, Tasks 10-16 use extension files.
//
//  Seam contract:
//    • Signature must be preserved exactly — callers in NumericDispatch.swift
//      depend on each name/parameter label/type.
//    • Replace the throw with real arithmetic; remove the SEAM comment.
//    • Do NOT change access level (internal is the minimum required).
//    • File this stubs file alongside the implementation once all stubs are
//      replaced; delete the file when all cells are implemented (Task 16).
//
//  Licensed under the Apache License, Version 2.0.
//

// MARK: - EVAL stubs (Tasks 10-16 implement)

extension NumericDispatch {

    // MARK: - Add/sub EVAL stubs (Task 10)

    /// SEAM: Task 10 — scalar±matrix via `vDSP_vsaddD`.
    static func evalScalarPlusMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: scalar±matrix (Task 10)")
    }

    /// SEAM: Task 10 — scalar±complexMatrix broadcast.
    static func evalScalarPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: scalar±complexMatrix (Task 10)")
    }

    /// SEAM: Task 10 — complex±matrix (promote M→CM, then element-wise).
    static func evalComplexPlusMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complex±matrix (Task 10)")
    }

    /// SEAM: Task 10 — complex±complexMatrix broadcast.
    static func evalComplexPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complex±complexMatrix (Task 10)")
    }

    /// SEAM: Task 10 — matrix±complexMatrix (promote M→CM, then element-wise).
    static func evalMatrixPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: matrix±complexMatrix (Task 10)")
    }

    /// SEAM: Task 10 — complexMatrix±complexMatrix element-wise.
    static func evalComplexMatrixPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix±complexMatrix (Task 10)")
    }

    // MARK: - Mul EVAL stubs (Task 11)

    /// SEAM: Task 11 — scalar * complexMatrix broadcast.
    static func evalScalarMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: scalar*complexMatrix (Task 11)")
    }

    /// SEAM: Task 11 — complex * matrix (promote M→CM, then element-wise mul).
    static func evalComplexMulMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complex*matrix (Task 11)")
    }

    /// SEAM: Task 11 — complex * complexMatrix broadcast.
    static func evalComplexMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complex*complexMatrix (Task 11)")
    }

    /// SEAM: Task 11 — matrix * complexMatrix (promote M→CM, complex matmul).
    static func evalMatrixMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: matrix*complexMatrix (Task 11)")
    }

    /// SEAM: Task 15 — complexMatrix * complexMatrix complex matmul (§4.8).
    ///
    /// **§4.3a coercion contract:** after computing the complex matmul result,
    /// call `coerce1x1Complex(result)` before returning so that a 1×1 result
    /// (vec·vec) is collapsed to `.complex` per §15 truth table.
    static func evalComplexMatrixMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix*complexMatrix (Task 15)")
    }

    // MARK: - Div EVAL stubs (Task 11)

    /// SEAM: Task 11 — matrix / complex scalar (element-wise division by complex).
    static func evalMatrixDivComplex(
        matrix: LinAlg.Matrix, divisor: Complex
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: matrix/complex (Task 11)")
    }

    /// SEAM: Task 11 — complexMatrix / scalar (element-wise division by real).
    static func evalComplexMatrixDivScalar(
        cm: LinAlg.ComplexMatrix, scalar: Double
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix/scalar (Task 11)")
    }

    /// SEAM: Task 11 — complexMatrix / complex scalar element-wise.
    static func evalComplexMatrixDivComplex(
        cm: LinAlg.ComplexMatrix, divisor: Complex
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix/complex (Task 11)")
    }

    // MARK: - Pow EVAL stubs (Task 12)

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

    /// SEAM: Task 13 — complexMatrix^n integer power via exponentiation-by-squaring.
    ///
    /// Complex-matrix integer power is deferred to Task 13 (complex-matrix arithmetic).
    static func evalComplexMatrixPow(
        cm: LinAlg.ComplexMatrix, exponent: Double
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix^scalar (Task 13)")
    }

    // MARK: - Unary EVAL stubs (Task 11)

    /// SEAM: Task 11 — neg(complexMatrix): element-wise negate both re and im arrays.
    static func evalNegComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: neg(complexMatrix) (Task 11)")
    }

    /// SEAM: Task 11 — plain (non-Hermitian) transpose of complexMatrix.
    ///
    /// There is no `ComplexMatrix.T` in `LinAlg`. The transpose swaps rows and
    /// cols without conjugation (conjugate-transpose is deferred to v-next §14).
    static func evalTransposeComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: transpose(complexMatrix) (Task 11)")
    }

    // MARK: - Function EVAL stubs (Task 11)

    /// SEAM: Task 11 — abs(complexMatrix): complex Frobenius norm.
    ///
    /// Formula: sqrt(Σ|z_ij|²) = sqrt(Σ(re_ij² + im_ij²)) per Golub & Van Loan §2.3.2.
    static func evalAbsComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: abs(complexMatrix) complex Frobenius norm (Task 11)")
    }

    /// SEAM: Task 11 — trace(complexMatrix): sum of complex diagonal elements.
    static func evalTraceComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: trace(complexMatrix) (Task 11)")
    }

    /// SEAM: Task 15 — dotProduct(CM, CM): bilinear complex dot (DOM-06).
    ///
    /// Uses the bilinear (no-conjugation) form: Σ aᵢ·bᵢ in the complex sense.
    /// The conjugate form (vdot) is deferred to v-next (§14).
    ///
    /// **§4.3a coercion contract:** after computing the complex dot result,
    /// call `coerce1x1Complex(result)` before returning so that a vec·vec
    /// result is collapsed to `.complex` per §15 truth table (PRD §4.3a).
    static func evalComplexMatrixDotProduct(
        lhs: LinAlg.ComplexMatrix, rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: dotProduct(complexMatrix, complexMatrix) (Task 15)")
    }

    /// SEAM: Task 11 — hadamard(CM, CM): element-wise complex product.
    static func evalComplexHadamard(
        lhs: LinAlg.ComplexMatrix, rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: hadamard(complexMatrix, complexMatrix) (Task 11)")
    }
}
