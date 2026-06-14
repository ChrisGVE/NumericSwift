//
//  NumericDispatch+BinaryOps.swift
//  NumericSwift
//
//  Binary-operator sub-dispatchers for the unified numeric pipeline.
//
//  Each function handles one operator family across all (lhsKind × rhsKind) pairs
//  defined in the §15 truth table. EVAL cells call stubs in
//  NumericDispatch+EvalStubs.swift; DELEG cells call LinAlg directly after
//  pre-validating shapes (Group-A) or propagate LinAlg's own errors (Group-B).
//
//  All methods are `internal` so that extension files produced by Tasks 10–12 can
//  reference sibling helpers within the same module.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

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
            guard l.rows == r.rows && l.cols == r.cols else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "matrix shapes (\(l.rows)×\(l.cols)) and (\(r.rows)×\(r.cols)) "
                    + "must match for \(isAdd ? "add" : "sub")")
            }
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
            return .matrix(LinAlg.mul(lhs.asScalar!, rhs.asMatrix!))
        case (.matrix, .scalar):
            return .matrix(LinAlg.mul(lhs.asMatrix!, rhs.asScalar!))

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
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            // Group-A: pre-validate inner dimensions before LinAlg.dot precondition
            guard l.cols == r.rows else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "matrix multiply: lhs.cols (\(l.cols)) must equal rhs.rows (\(r.rows))")
            }
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
            return .matrix(LinAlg.div(lhs.asMatrix!, r))

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
    static func applyPow(lhs: NumericValue, rhs: NumericValue) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.scalar, .scalar):
            return .scalar(pow(lhs.asScalar!, rhs.asScalar!))
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
            guard e == e.rounded() else {
                throw MathExprError.invalidArguments(
                    "matrix power requires an integer exponent; got \(e)")
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
            guard e == e.rounded() else {
                throw MathExprError.invalidArguments(
                    "complexMatrix power requires an integer exponent; got \(e)")
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

    // MARK: - dotProduct / hadamard helpers

    static func applyDotProduct(
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            // Group-A: pre-validate before LinAlg.dot precondition
            guard l.cols == r.rows else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "dotProduct: lhs.cols (\(l.cols)) must equal rhs.rows (\(r.rows))")
            }
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
            guard l.rows == r.rows && l.cols == r.cols else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "hadamard: shapes (\(l.rows)×\(l.cols)) and (\(r.rows)×\(r.cols)) must match")
            }
            return .matrix(LinAlg.hadamard(l, r))
        case (.complexMatrix, .complexMatrix):
            return try evalComplexHadamard(lhs: lhs.asComplexMatrix!, rhs: rhs.asComplexMatrix!)
        default:
            throw MathExprError.invalidArguments(
                "hadamard requires two matrix arguments of the same shape; "
                + "got \(lhs.kind) and \(rhs.kind)")
        }
    }
}
