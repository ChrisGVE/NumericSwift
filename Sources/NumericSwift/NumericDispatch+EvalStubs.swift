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

    /// SEAM: Task 11 — complexMatrix * complexMatrix complex matmul (§4.8).
    static func evalComplexMatrixMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix*complexMatrix (Task 11)")
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

    /// SEAM: Task 12 — matrix^n integer power via exponentiation-by-squaring.
    ///
    /// Pre-conditions the caller checks (before this stub is invoked):
    ///   - exponent must be an exact integer (no fractional part)
    ///   - matrix must be square (`rows == cols`)
    static func evalMatrixPow(
        matrix: LinAlg.Matrix, exponent: Double
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: matrix^scalar integer power (Task 12)")
    }

    /// SEAM: Task 12 — complexMatrix^n integer power via exponentiation-by-squaring.
    static func evalComplexMatrixPow(
        cm: LinAlg.ComplexMatrix, exponent: Double
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix^scalar (Task 12)")
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

    /// SEAM: Task 11 — dotProduct(CM, CM): bilinear complex dot (DOM-06).
    ///
    /// Uses the bilinear (no-conjugation) form: Σ aᵢ·bᵢ in the complex sense.
    /// The conjugate form (vdot) is deferred to v-next (§14).
    static func evalComplexMatrixDotProduct(
        lhs: LinAlg.ComplexMatrix, rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: dotProduct(complexMatrix, complexMatrix) (Task 11)")
    }

    /// SEAM: Task 11 — hadamard(CM, CM): element-wise complex product.
    static func evalComplexHadamard(
        lhs: LinAlg.ComplexMatrix, rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        throw MathExprError.unsupportedNode(
            "not yet implemented: hadamard(complexMatrix, complexMatrix) (Task 11)")
    }
}
