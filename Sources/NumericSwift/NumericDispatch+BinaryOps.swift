//
//  NumericDispatch+BinaryOps.swift
//  NumericSwift
//
//  Binary-operator sub-dispatchers for the unified numeric pipeline.
//
//  Each function handles one operator family across all (lhsKind × rhsKind) pairs
//  defined in the §15 truth table. EVAL cells delegate to the complex-matrix
//  implementations in NumericDispatch+ComplexMatrix{Arithmetic,Functions,Helpers}.swift
//  and NumericDispatch+MatrixPower.swift; DELEG cells call LinAlg directly after
//  pre-validating shapes (Group-A) or propagate LinAlg's own errors (Group-B).
//
//  Group-A pre-validation contract (AC2.2/§4.5):
//    The dispatcher pre-validates shapes and THROWS `MathExprError.shapeMismatch`
//    (or `.divisionByZero`) BEFORE any LinAlg precondition can fire. The shared
//    `validateShapes(_:lhs:rhs:rule:)` helper centralises this gate.
//
//  All methods are `internal` so that extension files produced by Tasks 10–12 can
//  reference sibling helpers within the same module.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Shape pre-validation

extension NumericDispatch {

    /// Shape-validation rule for Group-A real-matrix operators.
    enum ShapeRule {
        /// add / sub / hadamard / elementDiv — both operands must have identical shape.
        case equalDims
        /// dot (matmul / mat·vec / vec·vec) — mirrors LinAlg.dot's three sub-cases:
        ///   - vec·vec (both cols==1): lhs.rows == rhs.rows
        ///   - mat·vec (rhs.cols==1) or mat·mat: lhs.cols == rhs.rows
        case dotProduct
    }

    /// Pre-validate two matrix shapes against a Group-A operator rule.
    ///
    /// Throws `MathExprError.shapeMismatch` when the shapes are incompatible,
    /// so no `LinAlg` precondition can fire for script-driven inputs.
    ///
    /// - Parameters:
    ///   - op:  Human-readable operator name for the error message.
    ///   - lhs: Left operand matrix.
    ///   - rhs: Right operand matrix.
    ///   - rule: The shape compatibility rule.
    /// - Throws: `MathExprError.shapeMismatch` on incompatible shapes.
    static func validateShapes(
        _ op: String,
        lhs: LinAlg.Matrix,
        rhs: LinAlg.Matrix,
        rule: ShapeRule
    ) throws {
        switch rule {
        case .equalDims:
            guard lhs.rows == rhs.rows && lhs.cols == rhs.cols else {
                throw MathExprError.shapeMismatch(
                    "\(op): shapes (\(lhs.rows)×\(lhs.cols)) "
                    + "and (\(rhs.rows)×\(rhs.cols)) must match")
            }
        case .dotProduct:
            // Mirrors the three branches in LinAlg.dot (LinAlg.swift ~577):
            //   1. vec·vec (both cols==1): equal row count
            //   2. mat·vec (rhs.cols==1) or mat·mat: inner dims must match
            if lhs.cols == 1 && rhs.cols == 1 {
                // Column-vector dot product
                guard lhs.rows == rhs.rows else {
                    throw MathExprError.shapeMismatch(
                        "\(op): vectors must have the same length "
                        + "(\(lhs.rows) vs \(rhs.rows))")
                }
            } else {
                guard lhs.cols == rhs.rows else {
                    throw MathExprError.shapeMismatch(
                        "\(op): lhs.cols (\(lhs.cols)) must equal rhs.rows (\(rhs.rows))")
                }
            }
        }
    }
}

// MARK: - Binary sub-dispatchers

extension NumericDispatch {

    // MARK: + / -

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    static func applyAddSub(
        _ op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        let isAdd = op == .add
        let opName = isAdd ? "add" : "sub"

        switch (lhs.kind, rhs.kind) {
        case (.scalar, .scalar):
            let l = lhs.asScalar!, r = rhs.asScalar!
            return .scalar(isAdd ? l + r : l - r)

        case (.scalar, .complex):
            let l = Complex(lhs.asScalar!), r = rhs.asComplex!
            return .complex(isAdd ? l + r : l - r)
        case (.complex, .scalar):
            let l = lhs.asComplex!, r = Complex(rhs.asScalar!)
            return .complex(isAdd ? l + r : l - r)
        case (.complex, .complex):
            let l = lhs.asComplex!, r = rhs.asComplex!
            return .complex(isAdd ? l + r : l - r)

        // scalar ± matrix / matrix ± scalar — SEAM Task 10
        case (.scalar, .matrix), (.matrix, .scalar):
            return try evalScalarPlusMatrix(lhs: lhs, rhs: rhs, op: op)

        // scalar ± complexMatrix / complexMatrix ± scalar — SEAM Task 10
        case (.scalar, .complexMatrix), (.complexMatrix, .scalar):
            return try evalScalarPlusComplexMatrix(lhs: lhs, rhs: rhs, op: op)

        // complex ± matrix / matrix ± complex — SEAM Task 10
        case (.complex, .matrix), (.matrix, .complex):
            return try evalComplexPlusMatrix(lhs: lhs, rhs: rhs, op: op)

        // complex ± complexMatrix / complexMatrix ± complex — SEAM Task 10
        case (.complex, .complexMatrix), (.complexMatrix, .complex):
            return try evalComplexPlusComplexMatrix(lhs: lhs, rhs: rhs, op: op)

        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            // Group-A: pre-validate before LinAlg.add/sub precondition fires
            try validateShapes(opName, lhs: l, rhs: r, rule: .equalDims)
            // Soft-cap: result has same shape as operands
            try LinAlg.checkSoftCap(rows: l.rows, cols: l.cols)
            return .matrix(isAdd ? LinAlg.add(l, r) : LinAlg.sub(l, r))

        // matrix ± complexMatrix / complexMatrix ± matrix — SEAM Task 10
        case (.matrix, .complexMatrix), (.complexMatrix, .matrix):
            return try evalMatrixPlusComplexMatrix(lhs: lhs, rhs: rhs, op: op)

        // complexMatrix ± complexMatrix — SEAM Task 10
        case (.complexMatrix, .complexMatrix):
            return try evalComplexMatrixPlusComplexMatrix(lhs: lhs, rhs: rhs, op: op)
        }
    }

    // MARK: *

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    static func applyMul(lhs: NumericValue, rhs: NumericValue) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.scalar, .scalar):
            return .scalar(lhs.asScalar! * rhs.asScalar!)

        case (.scalar, .complex):
            return .complex(Complex(lhs.asScalar!) * rhs.asComplex!)
        case (.complex, .scalar):
            return .complex(lhs.asComplex! * Complex(rhs.asScalar!))
        case (.complex, .complex):
            return .complex(lhs.asComplex! * rhs.asComplex!)

        // scalar * matrix  / matrix * scalar — delegated to LinAlg scalar overloads
        case (.scalar, .matrix):
            // Soft-cap: result has same shape as the matrix operand
            let m = rhs.asMatrix!
            try LinAlg.checkSoftCap(rows: m.rows, cols: m.cols)
            return .matrix(LinAlg.mul(lhs.asScalar!, m))
        case (.matrix, .scalar):
            // Soft-cap: result has same shape as the matrix operand
            let m = lhs.asMatrix!
            try LinAlg.checkSoftCap(rows: m.rows, cols: m.cols)
            return .matrix(LinAlg.mul(m, rhs.asScalar!))

        // scalar * complexMatrix / complexMatrix * scalar — SEAM Task 11
        case (.scalar, .complexMatrix), (.complexMatrix, .scalar):
            return try evalScalarMulComplexMatrix(lhs: lhs, rhs: rhs)

        // complex * matrix / matrix * complex — SEAM Task 11
        case (.complex, .matrix), (.matrix, .complex):
            return try evalComplexMulMatrix(lhs: lhs, rhs: rhs)

        // complex * complexMatrix / complexMatrix * complex — SEAM Task 11
        case (.complex, .complexMatrix), (.complexMatrix, .complex):
            return try evalComplexMulComplexMatrix(lhs: lhs, rhs: rhs)

        case (.matrix, .matrix):
            // `*` between matrices = matmul (dot); element-wise = hadamard().
            // Covers: mat·mat, mat·vec (rhs.cols==1), vec·mat (lhs.cols==1),
            //         vec·vec (both cols==1, yields 1×1 → scalar via coerce1x1).
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            // Group-A: pre-validate inner dimensions before LinAlg.dot precondition
            try validateShapes("*", lhs: l, rhs: r, rule: .dotProduct)
            // Soft-cap: guard result size before allocating
            try LinAlg.checkSoftCap(rows: l.rows, cols: r.cols)
            return coerce1x1(.matrix(LinAlg.dot(l, r)))

        // matrix * complexMatrix / complexMatrix * matrix — SEAM Task 11
        case (.matrix, .complexMatrix), (.complexMatrix, .matrix):
            return try evalMatrixMulComplexMatrix(lhs: lhs, rhs: rhs)

        // complexMatrix * complexMatrix — SEAM Task 11
        case (.complexMatrix, .complexMatrix):
            return try evalComplexMatrixMulComplexMatrix(lhs: lhs, rhs: rhs)
        }
    }

    // MARK: /

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    static func applyDiv(lhs: NumericValue, rhs: NumericValue) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.scalar, .scalar):
            let r = rhs.asScalar!
            if r == 0 { throw MathExprError.divisionByZero }
            return .scalar(lhs.asScalar! / r)

        case (.scalar, .complex):
            let r = rhs.asComplex!
            if r.re == 0 && r.im == 0 { throw MathExprError.divisionByZero }
            return .complex(Complex(lhs.asScalar!) / r)

        case (.scalar, .matrix):
            throw MathExprError.invalidArguments(
                "scalar/matrix is undefined; matrices have no reciprocal in this sense")
        case (.scalar, .complexMatrix):
            throw MathExprError.invalidArguments(
                "scalar/complexMatrix is undefined")

        case (.complex, .scalar):
            let r = rhs.asScalar!
            if r == 0 { throw MathExprError.divisionByZero }
            return .complex(lhs.asComplex! / Complex(r))
        case (.complex, .complex):
            let r = rhs.asComplex!
            if r.re == 0 && r.im == 0 { throw MathExprError.divisionByZero }
            return .complex(lhs.asComplex! / r)
        case (.complex, .matrix):
            throw MathExprError.invalidArguments("complex/matrix is undefined")
        case (.complex, .complexMatrix):
            throw MathExprError.invalidArguments("complex/complexMatrix is undefined")

        case (.matrix, .scalar):
            let r = rhs.asScalar!
            // Group-A: pre-validate divisor before LinAlg.div precondition(scalar != 0)
            if r == 0 { throw MathExprError.divisionByZero }
            // Soft-cap: result has same shape as the matrix operand
            let m = lhs.asMatrix!
            try LinAlg.checkSoftCap(rows: m.rows, cols: m.cols)
            return .matrix(LinAlg.div(m, r))

        // matrix / complex — SEAM Task 11
        case (.matrix, .complex):
            return try evalMatrixDivComplex(matrix: lhs.asMatrix!, divisor: rhs.asComplex!)

        case (.matrix, .matrix):
            throw MathExprError.invalidArguments(
                "matrix/matrix is undefined; use inv(A) * B or solve(A, B)")
        case (.matrix, .complexMatrix):
            throw MathExprError.invalidArguments("matrix/complexMatrix is undefined")

        // complexMatrix / scalar — SEAM Task 11
        case (.complexMatrix, .scalar):
            let r = rhs.asScalar!
            if r == 0 { throw MathExprError.divisionByZero }
            return try evalComplexMatrixDivScalar(cm: lhs.asComplexMatrix!, scalar: r)

        // complexMatrix / complex — SEAM Task 11
        case (.complexMatrix, .complex):
            let r = rhs.asComplex!
            if r.re == 0 && r.im == 0 { throw MathExprError.divisionByZero }
            return try evalComplexMatrixDivComplex(cm: lhs.asComplexMatrix!, divisor: r)

        case (.complexMatrix, .matrix):
            throw MathExprError.invalidArguments("complexMatrix/matrix is undefined")
        case (.complexMatrix, .complexMatrix):
            throw MathExprError.invalidArguments(
                "complexMatrix/complexMatrix is undefined; use cinv(A) * B")
        }
    }

    // MARK: ^

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    static func applyPow(
        lhs: NumericValue,
        rhs: NumericValue,
        complexMode: Bool = false
    ) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.scalar, .scalar):
            let base = lhs.asScalar!, exponent = rhs.asScalar!
            // Complex-context promotion (issue #1): a negative-real base with a
            // non-integer exponent is exactly the case where the real `pow`
            // returns NaN. In complex mode, promote the base and take the complex
            // `(.complex, .scalar)` path below — matching the legacy `^` operator
            // `(exponent * base.log).exp`. Integer exponents stay on the real path
            // (e.g. (-2)^3 = -8 exactly), which the frozen snapshot oracle leaves
            // unconstrained and where the exact real value is preferable.
            if complexMode && base < 0 && exponent != exponent.rounded() {
                return try applyPow(lhs: .complex(Complex(base)), rhs: rhs, complexMode: complexMode)
            }
            return .scalar(pow(base, exponent))
        case (.scalar, .complex):
            // s^c = exp(c * log(s))
            let s = Complex(lhs.asScalar!), c = rhs.asComplex!
            return .complex((c * s.log).exp)
        case (.scalar, .matrix), (.scalar, .complexMatrix):
            throw MathExprError.invalidArguments(
                "scalar^matrix is undefined")

        case (.complex, .scalar):
            let b = lhs.asComplex!, e = rhs.asScalar!
            // Zero-base special case (mirrors legacy evalComplexBinary line 378).
            if b.re == 0 && b.im == 0 { return .complex(Complex(0)) }
            return .complex((b.log * Complex(e)).exp)
        case (.complex, .complex):
            let b = lhs.asComplex!, e = rhs.asComplex!
            // Zero-base special case per legacy evalComplexBinary line 378:
            // 0^z = Complex(0) to avoid -inf/NaN from log(0).
            if b.re == 0 && b.im == 0 { return .complex(Complex(0)) }
            return .complex((b.log * e).exp)
        case (.complex, .matrix), (.complex, .complexMatrix):
            throw MathExprError.invalidArguments("complex^matrix is undefined")

        case (.matrix, .scalar):
            let m = lhs.asMatrix!, e = rhs.asScalar!
            guard m.rows == m.cols else {
                throw MathExprError.invalidArguments(
                    "matrix power requires a square matrix (\(m.rows)×\(m.cols) is not square)")
            }
            guard e.isFinite && e == e.rounded() else {
                throw MathExprError.invalidArguments(
                    "matrix power requires a finite integer exponent; got \(e)")
            }
            // Soft-cap for result (same shape as input)
            try LinAlg.checkSoftCap(rows: m.rows, cols: m.cols)
            return try evalMatrixPow(matrix: m, exponent: e)

        case (.matrix, .complex), (.matrix, .matrix), (.matrix, .complexMatrix):
            throw MathExprError.invalidArguments(
                "matrix^non-scalar-integer is undefined")

        case (.complexMatrix, .scalar):
            let cm = lhs.asComplexMatrix!, e = rhs.asScalar!
            guard cm.rows == cm.cols else {
                throw MathExprError.invalidArguments(
                    "complexMatrix power requires a square matrix")
            }
            guard e.isFinite && e == e.rounded() else {
                throw MathExprError.invalidArguments(
                    "complexMatrix power requires a finite integer exponent; got \(e)")
            }
            try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
            return try evalComplexMatrixPow(cm: cm, exponent: e)

        case (.complexMatrix, .complex),
             (.complexMatrix, .matrix),
             (.complexMatrix, .complexMatrix):
            throw MathExprError.invalidArguments(
                "complexMatrix^non-scalar-integer is undefined")
        }
    }

    // MARK: %

    static func applyMod(lhs: NumericValue, rhs: NumericValue) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.scalar, .scalar):
            // Bit-identical to MathExpr.swift evalBinary(.mod) for scalar operands
            return .scalar(lhs.asScalar!.truncatingRemainder(dividingBy: rhs.asScalar!))
        case (.scalar, .complex), (.complex, .scalar), (.complex, .complex):
            // Preserves MathExpr.swift:381 behaviour for complex operands
            throw MathExprError.unsupportedNode("modulo over complex numbers")
        default:
            // Preserves the original error type for non-scalar modulo
            throw MathExprError.unsupportedNode(
                "modulo requires scalar operands; got \(lhs.kind) % \(rhs.kind)")
        }
    }

    // MARK: - dotProduct / hadamard / elementDiv helpers

    static func applyDotProduct(
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            // Group-A: pre-validate before LinAlg.dot precondition
            try validateShapes("dotProduct", lhs: l, rhs: r, rule: .dotProduct)
            try LinAlg.checkSoftCap(rows: l.rows, cols: r.cols)
            return coerce1x1(.matrix(LinAlg.dot(l, r)))
        case (.complexMatrix, .complexMatrix):
            return try evalComplexMatrixDotProduct(
                lhs: lhs.asComplexMatrix!, rhs: rhs.asComplexMatrix!)
        default:
            throw MathExprError.invalidArguments(
                "dotProduct requires two matrix arguments of compatible shape; "
                + "got \(lhs.kind) and \(rhs.kind)")
        }
    }

    static func applyHadamard(
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            // Group-A: pre-validate before LinAlg.hadamard precondition
            try validateShapes("hadamard", lhs: l, rhs: r, rule: .equalDims)
            // Soft-cap: result has same shape as operands
            try LinAlg.checkSoftCap(rows: l.rows, cols: l.cols)
            return .matrix(LinAlg.hadamard(l, r))
        case (.complexMatrix, .complexMatrix):
            return try evalComplexHadamard(lhs: lhs.asComplexMatrix!, rhs: rhs.asComplexMatrix!)
        default:
            throw MathExprError.invalidArguments(
                "hadamard requires two matrix arguments of the same shape; "
                + "got \(lhs.kind) and \(rhs.kind)")
        }
    }

    /// Element-wise matrix division (`./ ` in MATLAB notation).
    ///
    /// Both operands must be matrices of identical shape. Each element in the
    /// result is `lhs[i,j] / rhs[i,j]`. Division by zero propagates as `inf`
    /// or `nan` per IEEE 754 (consistent with element-wise scalar division).
    ///
    /// - Throws: `MathExprError.shapeMismatch` when shapes differ;
    ///           `MathExprError.invalidArguments` when kinds are unsupported.
    static func applyElementDiv(
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            // Group-A: pre-validate before LinAlg.elementDiv precondition
            try validateShapes("elementDiv", lhs: l, rhs: r, rule: .equalDims)
            // Soft-cap: result has same shape as operands
            try LinAlg.checkSoftCap(rows: l.rows, cols: l.cols)
            return .matrix(LinAlg.elementDiv(l, r))
        default:
            throw MathExprError.invalidArguments(
                "elementDiv requires two real matrices of the same shape; "
                + "got \(lhs.kind) and \(rhs.kind)")
        }
    }
}
