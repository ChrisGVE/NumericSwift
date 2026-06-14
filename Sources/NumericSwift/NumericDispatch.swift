//
//  NumericDispatch.swift
//  NumericSwift
//
//  Central dispatch surface for the unified numeric pipeline.
//
//  Routes (operator/function, NumericValue kinds) → typed NumericValue result.
//  Implements the §15 truth table for all operator × kind combinations.
//
//  Design: each public entry point delegates to a private per-operator or
//  per-function-family sub-function. Cells marked "SEAM: Task N" are stubs that
//  throw `.unsupportedNode("not yet implemented: … (Task N)")`. Tasks 10–16 will
//  replace those stub bodies in extension files.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// NOTE (9.2): NumericValue.Kind (.scalar/.complex/.matrix/.complexMatrix) serves as the
// numeric-kind discriminant for routing. No new enum is introduced; see NumericValue+Accessors.swift.

// MARK: - NumericDispatch

/// Central dispatch surface for the unified numeric pipeline.
///
/// Routes `(operator/function, NumericValue kinds)` → typed result following the
/// §15 truth table. Three public entry points cover all dispatch needs:
///
/// - ``applyBinary(_:lhs:rhs:)`` — binary arithmetic/algebraic operators
/// - ``applyUnary(_:operand:)``  — unary prefix/postfix operators
/// - ``applyFunction(_:args:)``  — named function calls (1-arg and 2-arg)
///
/// Cells in the truth table not yet implemented are stubs that throw
/// `MathExprError.unsupportedNode("not yet implemented: … (Task N)")`.
/// Tasks 10–16 replace those stub bodies in extension files.
public enum NumericDispatch {

    // MARK: - Binary dispatch

    /// Route a binary operator over two `NumericValue` operands.
    ///
    /// - Parameters:
    ///   - op:  The binary operator.
    ///   - lhs: Left-hand operand.
    ///   - rhs: Right-hand operand.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError` on invalid combination or unsupported cell.
    public static func applyBinary(
        _ op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        switch op {
        case .add, .sub:
            return try applyAddSub(op, lhs: lhs, rhs: rhs)
        case .mul:
            return try applyMul(lhs: lhs, rhs: rhs)
        case .div:
            return try applyDiv(lhs: lhs, rhs: rhs)
        case .pow:
            return try applyPow(lhs: lhs, rhs: rhs)
        case .mod:
            return try applyMod(lhs: lhs, rhs: rhs)
        case .plusMinus, .minusPlus:
            throw MathExprError.unsupportedNode(
                "plusMinus/minusPlus are display-only operators")
        }
    }

    // MARK: - Unary dispatch

    /// Route a unary operator over a `NumericValue` operand.
    ///
    /// - Parameters:
    ///   - op:      The unary operator.
    ///   - operand: The operand.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError` on invalid combination or unsupported cell.
    public static func applyUnary(
        _ op: UnaryOp,
        operand: NumericValue
    ) throws -> NumericValue {
        switch op {
        case .neg:
            return try applyNeg(operand: operand)
        case .pos:
            return operand
        case .factorial:
            return try applyFactorial(operand: operand)
        case .transpose:
            return try applyTransposeUnary(operand: operand)
        }
    }

    // MARK: - Function dispatch

    /// Route a named function call over one or more `NumericValue` arguments.
    ///
    /// - Parameters:
    ///   - name: The function name (case-sensitive, matching MathExpr conventions).
    ///   - args: The evaluated argument list.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError` on unknown function, invalid arguments, or unsupported cell.
    public static func applyFunction(
        _ name: String,
        args: [NumericValue]
    ) throws -> NumericValue {
        let trigNames: Set<String> = [
            "sin", "cos", "tan",
            "asin", "acos", "atan",
            "sinh", "cosh", "tanh",
            "asinh", "acosh", "atanh",
        ]
        if trigNames.contains(name) {
            return try applyTrigFunction(name, args: args)
        }
        switch name {
        case "exp", "log", "sqrt":
            return try applyExpLogSqrt(name, args: args)
        case "abs", "inv", "det", "trace":
            return try applyAbsInvDetTrace(name, args: args)
        case "transpose":
            return try applyTransposeFunction(args: args)
        case "dotProduct", "hadamard":
            return try applyMultiArgFunction(name, args: args)
        case "crossProduct":
            throw MathExprError.unsupportedNode("crossProduct not yet implemented")
        case "min":
            return try applyMinMax(name: "min", args: args)
        case "max":
            return try applyMinMax(name: "max", args: args)
        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    // MARK: - Private helpers

    /// Coerce a 1×1 `.matrix` to `.scalar`. All other values pass through unchanged.
    ///
    /// This is the §4.3a coercion gate: matrix dot-product results that happen to
    /// be 1×1 are collapsed to a plain scalar so downstream code doesn't have to
    /// special-case them.
    private static func coerce1x1(_ value: NumericValue) -> NumericValue {
        if case .matrix(let m) = value, m.rows == 1, m.cols == 1 {
            return .scalar(m[0, 0])
        }
        return value
    }

    // MARK: - applyAddSub

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    private static func applyAddSub(
        _ op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        let isAdd = op == .add

        switch (lhs.kind, rhs.kind) {
        // scalar ± scalar
        case (.scalar, .scalar):
            let l = lhs.asScalar!, r = rhs.asScalar!
            return .scalar(isAdd ? l + r : l - r)

        // scalar ± complex  /  complex ± scalar
        case (.scalar, .complex):
            let l = Complex(lhs.asScalar!), r = rhs.asComplex!
            return .complex(isAdd ? l + r : l - r)
        case (.complex, .scalar):
            let l = lhs.asComplex!, r = Complex(rhs.asScalar!)
            return .complex(isAdd ? l + r : l - r)

        // complex ± complex
        case (.complex, .complex):
            let l = lhs.asComplex!, r = rhs.asComplex!
            return .complex(isAdd ? l + r : l - r)

        // scalar ± matrix  /  matrix ± scalar
        case (.scalar, .matrix), (.matrix, .scalar):
            return try evalScalarPlusMatrix(lhs: lhs, rhs: rhs, op: op)

        // scalar ± complexMatrix  /  complexMatrix ± scalar
        case (.scalar, .complexMatrix), (.complexMatrix, .scalar):
            return try evalScalarPlusComplexMatrix(lhs: lhs, rhs: rhs, op: op)

        // complex ± matrix  /  matrix ± complex
        case (.complex, .matrix), (.matrix, .complex):
            return try evalComplexPlusMatrix(lhs: lhs, rhs: rhs, op: op)

        // complex ± complexMatrix  /  complexMatrix ± complex
        case (.complex, .complexMatrix), (.complexMatrix, .complex):
            return try evalComplexPlusComplexMatrix(lhs: lhs, rhs: rhs, op: op)

        // matrix ± matrix
        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            guard l.rows == r.rows && l.cols == r.cols else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "matrix shapes (\(l.rows)x\(l.cols)) and (\(r.rows)x\(r.cols)) must match for add/sub")
            }
            let result = isAdd ? LinAlg.add(l, r) : LinAlg.sub(l, r)
            return .matrix(result)

        // matrix ± complexMatrix  /  complexMatrix ± matrix
        case (.matrix, .complexMatrix), (.complexMatrix, .matrix):
            return try evalMatrixPlusComplexMatrix(lhs: lhs, rhs: rhs, op: op)

        // complexMatrix ± complexMatrix
        case (.complexMatrix, .complexMatrix):
            return try evalComplexMatrixPlusComplexMatrix(lhs: lhs, rhs: rhs, op: op)
        }
    }

    // MARK: - applyMul

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    private static func applyMul(lhs: NumericValue, rhs: NumericValue) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        // scalar * scalar
        case (.scalar, .scalar):
            return .scalar(lhs.asScalar! * rhs.asScalar!)

        // scalar * complex  /  complex * scalar
        case (.scalar, .complex):
            let l = Complex(lhs.asScalar!), r = rhs.asComplex!
            return .complex(l * r)
        case (.complex, .scalar):
            let l = lhs.asComplex!, r = Complex(rhs.asScalar!)
            return .complex(l * r)

        // complex * complex
        case (.complex, .complex):
            return .complex(lhs.asComplex! * rhs.asComplex!)

        // scalar * matrix  /  matrix * scalar
        case (.scalar, .matrix):
            return .matrix(LinAlg.mul(lhs.asScalar!, rhs.asMatrix!))
        case (.matrix, .scalar):
            return .matrix(LinAlg.mul(lhs.asMatrix!, rhs.asScalar!))

        // scalar * complexMatrix  /  complexMatrix * scalar
        case (.scalar, .complexMatrix), (.complexMatrix, .scalar):
            return try evalScalarMulComplexMatrix(lhs: lhs, rhs: rhs)

        // complex * matrix  /  matrix * complex
        case (.complex, .matrix), (.matrix, .complex):
            return try evalComplexMulMatrix(lhs: lhs, rhs: rhs)

        // complex * complexMatrix  /  complexMatrix * complex
        case (.complex, .complexMatrix), (.complexMatrix, .complex):
            return try evalComplexMulComplexMatrix(lhs: lhs, rhs: rhs)

        // matrix * matrix
        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            guard l.cols == r.rows else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "matrix multiply: lhs.cols (\(l.cols)) must equal rhs.rows (\(r.rows))")
            }
            try LinAlg.checkSoftCap(rows: l.rows, cols: r.cols)
            let result = LinAlg.dot(l, r)
            return coerce1x1(.matrix(result))

        // matrix * complexMatrix  /  complexMatrix * matrix
        case (.matrix, .complexMatrix), (.complexMatrix, .matrix):
            return try evalMatrixMulComplexMatrix(lhs: lhs, rhs: rhs)

        // complexMatrix * complexMatrix
        case (.complexMatrix, .complexMatrix):
            return try evalComplexMatrixMulComplexMatrix(lhs: lhs, rhs: rhs)
        }
    }

    // MARK: - applyDiv

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    private static func applyDiv(lhs: NumericValue, rhs: NumericValue) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        // scalar / scalar
        case (.scalar, .scalar):
            let r = rhs.asScalar!
            if r == 0 { throw MathExprError.divisionByZero }
            return .scalar(lhs.asScalar! / r)

        // scalar / complex
        case (.scalar, .complex):
            let r = rhs.asComplex!
            if r.re == 0 && r.im == 0 { throw MathExprError.divisionByZero }
            return .complex(Complex(lhs.asScalar!) / r)

        // scalar / matrix  /  scalar / complexMatrix
        case (.scalar, .matrix):
            throw MathExprError.invalidArguments(
                "matrix divisor undefined; scalar/matrix has no meaning")
        case (.scalar, .complexMatrix):
            throw MathExprError.invalidArguments(
                "complexMatrix divisor undefined; scalar/complexMatrix has no meaning")

        // complex / scalar
        case (.complex, .scalar):
            let r = rhs.asScalar!
            if r == 0 { throw MathExprError.divisionByZero }
            return .complex(lhs.asComplex! / r)

        // complex / complex
        case (.complex, .complex):
            let r = rhs.asComplex!
            if r.re == 0 && r.im == 0 { throw MathExprError.divisionByZero }
            return .complex(lhs.asComplex! / r)

        // complex / matrix  /  complex / complexMatrix
        case (.complex, .matrix):
            throw MathExprError.invalidArguments(
                "matrix divisor undefined; complex/matrix has no meaning")
        case (.complex, .complexMatrix):
            throw MathExprError.invalidArguments(
                "complexMatrix divisor undefined; complex/complexMatrix has no meaning")

        // matrix / scalar
        case (.matrix, .scalar):
            let r = rhs.asScalar!
            if r == 0 { throw MathExprError.divisionByZero }
            return .matrix(LinAlg.div(lhs.asMatrix!, r))

        // matrix / complex
        case (.matrix, .complex):
            return try evalMatrixDivComplex(matrix: lhs.asMatrix!, divisor: rhs.asComplex!)

        // matrix / matrix  /  matrix / complexMatrix
        case (.matrix, .matrix):
            throw MathExprError.invalidArguments(
                "matrix/matrix division undefined; use inv() for matrix inversion")
        case (.matrix, .complexMatrix):
            throw MathExprError.invalidArguments(
                "matrix/complexMatrix division undefined; use inv() for matrix inversion")

        // complexMatrix / scalar
        case (.complexMatrix, .scalar):
            return try evalComplexMatrixDivScalar(cm: lhs.asComplexMatrix!, scalar: rhs.asScalar!)

        // complexMatrix / complex
        case (.complexMatrix, .complex):
            return try evalComplexMatrixDivComplex(
                cm: lhs.asComplexMatrix!, divisor: rhs.asComplex!)

        // complexMatrix / matrix  /  complexMatrix / complexMatrix
        case (.complexMatrix, .matrix):
            throw MathExprError.invalidArguments(
                "complexMatrix/matrix division undefined; use inv() for matrix inversion")
        case (.complexMatrix, .complexMatrix):
            throw MathExprError.invalidArguments(
                "complexMatrix/complexMatrix division undefined; use inv() for matrix inversion")
        }
    }

    // MARK: - applyPow

    // swiftlint:disable:next cyclomatic_complexity
    private static func applyPow(lhs: NumericValue, rhs: NumericValue) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        // scalar ^ scalar
        case (.scalar, .scalar):
            return .scalar(pow(lhs.asScalar!, rhs.asScalar!))

        // scalar ^ complex
        case (.scalar, .complex):
            let base = Complex(lhs.asScalar!)
            return .complex(base.pow(rhs.asComplex!))

        // complex ^ scalar
        case (.complex, .scalar):
            return .complex(lhs.asComplex!.pow(rhs.asScalar!))

        // complex ^ complex
        case (.complex, .complex):
            return .complex(lhs.asComplex!.pow(rhs.asComplex!))

        // matrix ^ scalar
        case (.matrix, .scalar):
            return try evalMatrixPow(matrix: lhs.asMatrix!, exponent: rhs.asScalar!)

        // complexMatrix ^ scalar
        case (.complexMatrix, .scalar):
            return try evalComplexMatrixPow(cm: lhs.asComplexMatrix!, exponent: rhs.asScalar!)

        // invalid combinations
        case (.scalar, .matrix), (.scalar, .complexMatrix):
            throw MathExprError.invalidArguments(
                "scalar^matrix/complexMatrix is undefined")
        case (.complex, .matrix), (.complex, .complexMatrix):
            throw MathExprError.invalidArguments(
                "complex^matrix/complexMatrix is undefined")
        case (.matrix, .complex):
            throw MathExprError.invalidArguments(
                "matrix^complex exponent is undefined")
        case (.matrix, .matrix):
            throw MathExprError.invalidArguments(
                "matrix^matrix is undefined")
        case (.matrix, .complexMatrix):
            throw MathExprError.invalidArguments(
                "matrix^complexMatrix is undefined")
        case (.complexMatrix, .complex):
            throw MathExprError.invalidArguments(
                "complexMatrix^complex exponent is undefined")
        case (.complexMatrix, .matrix):
            throw MathExprError.invalidArguments(
                "complexMatrix^matrix is undefined")
        case (.complexMatrix, .complexMatrix):
            throw MathExprError.invalidArguments(
                "complexMatrix^complexMatrix is undefined")
        }
    }

    // MARK: - applyMod

    private static func applyMod(lhs: NumericValue, rhs: NumericValue) throws -> NumericValue {
        guard case .scalar(let l) = lhs, case .scalar(let r) = rhs else {
            throw MathExprError.unsupportedNode("modulo requires scalar operands")
        }
        return .scalar(l.truncatingRemainder(dividingBy: r))
    }

    // MARK: - Unary sub-dispatchers

    private static func applyNeg(operand: NumericValue) throws -> NumericValue {
        switch operand.kind {
        case .scalar:
            return .scalar(-operand.asScalar!)
        case .complex:
            let z = operand.asComplex!
            return .complex(Complex(re: -z.re, im: -z.im))
        case .matrix:
            return .matrix(LinAlg.neg(operand.asMatrix!))
        case .complexMatrix:
            return try evalNegComplexMatrix(cm: operand.asComplexMatrix!)
        }
    }

    private static func applyFactorial(operand: NumericValue) throws -> NumericValue {
        switch operand.kind {
        case .scalar:
            let v = operand.asScalar!
            guard v >= 0 else {
                throw MathExprError.invalidArguments(
                    "factorial requires non-negative argument, got \(v)")
            }
            return .scalar(tgamma(v + 1))
        case .complex:
            throw MathExprError.invalidArguments("factorial is not defined for complex numbers")
        case .matrix:
            throw MathExprError.invalidArguments("factorial is not defined for matrices")
        case .complexMatrix:
            throw MathExprError.invalidArguments(
                "factorial is not defined for complex matrices")
        }
    }

    private static func applyTransposeUnary(operand: NumericValue) throws -> NumericValue {
        switch operand.kind {
        case .scalar:
            return operand
        case .complex:
            return operand
        case .matrix:
            return .matrix(operand.asMatrix!.T)
        case .complexMatrix:
            return try evalTransposeComplexMatrix(cm: operand.asComplexMatrix!)
        }
    }

    // MARK: - Function sub-dispatchers

    private static func applyTrigFunction(
        _ name: String,
        args: [NumericValue]
    ) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 1 argument, got \(args.count)")
        }
        let arg = args[0]
        switch arg.kind {
        case .scalar:
            return .scalar(try evalScalarTrig(name, arg.asScalar!))
        case .complex:
            return .complex(try evalComplexTrig(name, arg.asComplex!))
        case .matrix, .complexMatrix:
            throw MathExprError.invalidArguments(
                "trig functions are not defined for matrices; use elementwise ops")
        }
    }

    private static func evalScalarTrig(_ name: String, _ x: Double) throws -> Double {
        switch name {
        case "sin":   return Foundation.sin(x)
        case "cos":   return Foundation.cos(x)
        case "tan":   return Foundation.tan(x)
        case "asin":  return Foundation.asin(x)
        case "acos":  return Foundation.acos(x)
        case "atan":  return Foundation.atan(x)
        case "sinh":  return Foundation.sinh(x)
        case "cosh":  return Foundation.cosh(x)
        case "tanh":  return Foundation.tanh(x)
        case "asinh": return Foundation.asinh(x)
        case "acosh": return Foundation.acosh(x)
        case "atanh": return Foundation.atanh(x)
        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    private static func evalComplexTrig(_ name: String, _ z: Complex) throws -> Complex {
        switch name {
        case "sin":   return z.sin
        case "cos":   return z.cos
        case "tan":   return z.tan
        case "asin":  return z.asin
        case "acos":  return z.acos
        case "atan":  return z.atan
        case "sinh":  return z.sinh
        case "cosh":  return z.cosh
        case "tanh":  return z.tanh
        case "asinh": return z.asinh
        case "acosh": return z.acosh
        case "atanh": return z.atanh
        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    // swiftlint:disable:next cyclomatic_complexity
    private static func applyExpLogSqrt(
        _ name: String,
        args: [NumericValue]
    ) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 1 argument, got \(args.count)")
        }
        let arg = args[0]
        switch (name, arg.kind) {
        // exp
        case ("exp", .scalar):
            return .scalar(Foundation.exp(arg.asScalar!))
        case ("exp", .complex):
            return .complex(arg.asComplex!.exp)
        case ("exp", .matrix):
            let result = try LinAlg.expm(arg.asMatrix!)
            return .matrix(result)
        case ("exp", .complexMatrix):
            throw MathExprError.invalidArguments(
                "exp is not supported for complex matrices")

        // log
        case ("log", .scalar):
            return .scalar(Foundation.log(arg.asScalar!))
        case ("log", .complex):
            return .complex(arg.asComplex!.log)
        case ("log", .matrix):
            guard let result = try LinAlg.logm(arg.asMatrix!) else {
                throw MathExprError.invalidArguments(
                    "matrix logarithm failed to converge")
            }
            return .matrix(result)
        case ("log", .complexMatrix):
            throw MathExprError.invalidArguments(
                "log is not supported for complex matrices")

        // sqrt
        case ("sqrt", .scalar):
            return .scalar(Foundation.sqrt(arg.asScalar!))
        case ("sqrt", .complex):
            return .complex(arg.asComplex!.sqrt)
        case ("sqrt", .matrix):
            guard let result = try LinAlg.sqrtm(arg.asMatrix!) else {
                throw MathExprError.invalidArguments(
                    "matrix square root failed: eigenvalues are negative")
            }
            return .matrix(result)
        case ("sqrt", .complexMatrix):
            throw MathExprError.invalidArguments(
                "sqrt is not supported for complex matrices")

        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    private static func applyAbsInvDetTrace(
        _ name: String,
        args: [NumericValue]
    ) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 1 argument, got \(args.count)")
        }
        let arg = args[0]
        switch (name, arg.kind) {
        // abs
        case ("abs", .scalar):
            let sv = arg.asScalar!
            return .scalar(sv < 0 ? -sv : sv)
        case ("abs", .complex):
            return .scalar(arg.asComplex!.abs)
        case ("abs", .matrix):
            return .scalar(LinAlg.frobeniusNorm(arg.asMatrix!))
        case ("abs", .complexMatrix):
            return try evalAbsComplexMatrix(cm: arg.asComplexMatrix!)

        // inv
        case ("inv", .scalar):
            throw MathExprError.invalidArguments(
                "inv() requires a matrix argument; for scalar use 1/x")
        case ("inv", .complex):
            throw MathExprError.invalidArguments(
                "inv() requires a matrix argument; for complex use 1/z")
        case ("inv", .matrix):
            guard let result = try LinAlg.inv(arg.asMatrix!) else {
                throw MathExprError.invalidArguments("inverse of singular matrix")
            }
            return .matrix(result)
        case ("inv", .complexMatrix):
            guard let result = try LinAlg.cinv(arg.asComplexMatrix!) else {
                throw MathExprError.invalidArguments("inverse of singular complex matrix")
            }
            return .complexMatrix(result)

        // det
        case ("det", .scalar):
            throw MathExprError.invalidArguments(
                "det() requires a matrix argument; determinant is undefined for scalars")
        case ("det", .complex):
            throw MathExprError.invalidArguments(
                "det() requires a matrix argument; determinant is undefined for complex scalars")
        case ("det", .matrix):
            let value = try LinAlg.det(arg.asMatrix!)
            return .scalar(value)
        case ("det", .complexMatrix):
            guard let cdResult = try LinAlg.cdet(arg.asComplexMatrix!) else {
                throw MathExprError.invalidArguments(
                    "determinant failed: LAPACK error on complex matrix")
            }
            return .complex(Complex(re: cdResult.re, im: cdResult.im))

        // trace
        case ("trace", .scalar):
            throw MathExprError.invalidArguments(
                "trace() requires a matrix argument; trace is undefined for scalars")
        case ("trace", .complex):
            throw MathExprError.invalidArguments(
                "trace() requires a matrix argument; trace is undefined for complex scalars")
        case ("trace", .matrix):
            let value = try LinAlg.trace(arg.asMatrix!)
            return .scalar(value)
        case ("trace", .complexMatrix):
            return try evalTraceComplexMatrix(cm: arg.asComplexMatrix!)

        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    private static func applyTransposeFunction(args: [NumericValue]) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "transpose() requires exactly 1 argument, got \(args.count)")
        }
        return try applyTransposeUnary(operand: args[0])
    }

    private static func applyMultiArgFunction(
        _ name: String,
        args: [NumericValue]
    ) throws -> NumericValue {
        guard args.count == 2 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 2 arguments, got \(args.count)")
        }
        switch name {
        case "dotProduct":
            return try applyDotProduct(lhs: args[0], rhs: args[1])
        case "hadamard":
            return try applyHadamard(lhs: args[0], rhs: args[1])
        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    private static func applyDotProduct(
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            guard l.cols == r.rows else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "dotProduct: lhs.cols (\(l.cols)) must equal rhs.rows (\(r.rows))")
            }
            try LinAlg.checkSoftCap(rows: l.rows, cols: r.cols)
            let result = LinAlg.dot(l, r)
            return coerce1x1(.matrix(result))
        case (.complexMatrix, .complexMatrix):
            return try evalComplexMatrixDotProduct(
                lhs: lhs.asComplexMatrix!, rhs: rhs.asComplexMatrix!)
        default:
            throw MathExprError.invalidArguments(
                "dotProduct requires matrix arguments of compatible shape")
        }
    }

    private static func applyHadamard(
        lhs: NumericValue,
        rhs: NumericValue
    ) throws -> NumericValue {
        switch (lhs.kind, rhs.kind) {
        case (.matrix, .matrix):
            let l = lhs.asMatrix!, r = rhs.asMatrix!
            guard l.rows == r.rows && l.cols == r.cols else {
                throw LinAlg.LinAlgError.dimensionMismatch(
                    "hadamard: shapes (\(l.rows)x\(l.cols)) and (\(r.rows)x\(r.cols)) must match")
            }
            return .matrix(LinAlg.hadamard(l, r))
        case (.complexMatrix, .complexMatrix):
            return try evalComplexHadamard(lhs: lhs.asComplexMatrix!, rhs: rhs.asComplexMatrix!)
        default:
            throw MathExprError.invalidArguments(
                "hadamard requires matrix arguments of the same shape")
        }
    }

    private static func applyMinMax(name: String, args: [NumericValue]) throws -> NumericValue {
        switch args.count {
        case 1:
            guard case .scalar(let x) = args[0] else {
                throw MathExprError.invalidArguments(
                    "\(name)(x) with 1 argument requires a scalar")
            }
            return .scalar(x)
        case 2:
            guard case .scalar(let a) = args[0], case .scalar(let b) = args[1] else {
                throw MathExprError.invalidArguments(
                    "\(name)(a, b) requires scalar arguments")
            }
            return .scalar(name == "min" ? Swift.min(a, b) : Swift.max(a, b))
        default:
            throw MathExprError.invalidArguments(
                "\(name) requires 1 or 2 scalar arguments, got \(args.count)")
        }
    }

    // MARK: - EVAL stubs (Tasks 10-16 implement)

    private static func evalScalarPlusMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        // SEAM: Task 10 implements scalar±matrix via vDSP
        throw MathExprError.unsupportedNode(
            "not yet implemented: scalar±matrix (Task 10)")
    }

    private static func evalScalarPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        // SEAM: Task 10 implements scalar±complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: scalar±complexMatrix (Task 10)")
    }

    private static func evalComplexPlusMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        // SEAM: Task 10 implements complex±matrix (promote + EVAL)
        throw MathExprError.unsupportedNode(
            "not yet implemented: complex±matrix (Task 10)")
    }

    private static func evalComplexPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        // SEAM: Task 10 implements complex±complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: complex±complexMatrix (Task 10)")
    }

    private static func evalMatrixPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        // SEAM: Task 10 implements matrix±complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: matrix±complexMatrix (Task 10)")
    }

    private static func evalComplexMatrixPlusComplexMatrix(
        lhs: NumericValue, rhs: NumericValue, op: BinaryOp
    ) throws -> NumericValue {
        // SEAM: Task 10 implements complexMatrix±complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix±complexMatrix (Task 10)")
    }

    private static func evalScalarMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        // SEAM: Task 11 implements scalar*complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: scalar*complexMatrix (Task 11)")
    }

    private static func evalComplexMulMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complex*matrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: complex*matrix (Task 11)")
    }

    private static func evalComplexMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complex*complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: complex*complexMatrix (Task 11)")
    }

    private static func evalMatrixMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        // SEAM: Task 11 implements matrix*complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: matrix*complexMatrix (Task 11)")
    }

    private static func evalComplexMatrixMulComplexMatrix(
        lhs: NumericValue, rhs: NumericValue
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complexMatrix*complexMatrix (complex matmul)
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix*complexMatrix (Task 11)")
    }

    private static func evalMatrixDivComplex(
        matrix: LinAlg.Matrix, divisor: Complex
    ) throws -> NumericValue {
        // SEAM: Task 11 implements matrix/complex
        throw MathExprError.unsupportedNode(
            "not yet implemented: matrix/complex (Task 11)")
    }

    private static func evalComplexMatrixDivScalar(
        cm: LinAlg.ComplexMatrix, scalar: Double
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complexMatrix/scalar
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix/scalar (Task 11)")
    }

    private static func evalComplexMatrixDivComplex(
        cm: LinAlg.ComplexMatrix, divisor: Complex
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complexMatrix/complex
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix/complex (Task 11)")
    }

    private static func evalMatrixPow(
        matrix: LinAlg.Matrix, exponent: Double
    ) throws -> NumericValue {
        // SEAM: Task 12 implements integer matrix power
        throw MathExprError.unsupportedNode(
            "not yet implemented: matrix^scalar integer power (Task 12)")
    }

    private static func evalComplexMatrixPow(
        cm: LinAlg.ComplexMatrix, exponent: Double
    ) throws -> NumericValue {
        // SEAM: Task 12 implements complexMatrix^scalar
        throw MathExprError.unsupportedNode(
            "not yet implemented: complexMatrix^scalar (Task 12)")
    }

    private static func evalNegComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // SEAM: Task 11 implements negation of complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: neg(complexMatrix) (Task 11)")
    }

    private static func evalTransposeComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // SEAM: Task 11 implements plain (non-Hermitian) transpose of complexMatrix
        throw MathExprError.unsupportedNode(
            "not yet implemented: transpose(complexMatrix) (Task 11)")
    }

    private static func evalAbsComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complex Frobenius norm
        throw MathExprError.unsupportedNode(
            "not yet implemented: abs(complexMatrix) complex Frobenius norm (Task 11)")
    }

    private static func evalTraceComplexMatrix(
        cm: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complex trace
        throw MathExprError.unsupportedNode(
            "not yet implemented: trace(complexMatrix) (Task 11)")
    }

    private static func evalComplexMatrixDotProduct(
        lhs: LinAlg.ComplexMatrix, rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complex matrix dot product
        throw MathExprError.unsupportedNode(
            "not yet implemented: dotProduct(complexMatrix, complexMatrix) (Task 11)")
    }

    private static func evalComplexHadamard(
        lhs: LinAlg.ComplexMatrix, rhs: LinAlg.ComplexMatrix
    ) throws -> NumericValue {
        // SEAM: Task 11 implements complex Hadamard product
        throw MathExprError.unsupportedNode(
            "not yet implemented: hadamard(complexMatrix, complexMatrix) (Task 11)")
    }
}
