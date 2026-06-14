//
//  NumericDispatch+UnaryFunctions.swift
//  NumericSwift
//
//  Unary-operator and named-function sub-dispatchers for the unified pipeline.
//
//  Covers:
//    • applyNeg / applyFactorial / applyTransposeUnary (UnaryOp routing)
//    • applyTrigFunction / evalScalarTrig / evalComplexTrig
//    • applyExpLogSqrt / applyAbsInvDetTrace
//    • applyTransposeFunction / applyMultiArgFunction / applyMinMax
//
//  All methods are `internal` so that Tasks 10-16 extension files can reference
//  them from the same module without re-exporting them.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Unary sub-dispatchers

extension NumericDispatch {

    static func applyNeg(operand: NumericValue) throws -> NumericValue {
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

    static func applyFactorial(operand: NumericValue) throws -> NumericValue {
        switch operand.kind {
        case .scalar:
            let v = operand.asScalar!
            guard v >= 0 else {
                throw MathExprError.invalidArguments(
                    "factorial requires a non-negative argument, got \(v)")
            }
            return .scalar(tgamma(v + 1))
        case .complex:
            throw MathExprError.invalidArguments(
                "factorial is not defined for complex numbers")
        case .matrix:
            throw MathExprError.invalidArguments(
                "factorial is not defined for matrices")
        case .complexMatrix:
            throw MathExprError.invalidArguments(
                "factorial is not defined for complex matrices")
        }
    }

    static func applyTransposeUnary(operand: NumericValue) throws -> NumericValue {
        switch operand.kind {
        case .scalar:
            return operand           // transpose of a scalar is the scalar itself
        case .complex:
            return operand           // transpose of a complex scalar is the scalar
        case .matrix:
            return .matrix(operand.asMatrix!.T)
        case .complexMatrix:
            return try evalTransposeComplexMatrix(cm: operand.asComplexMatrix!)
        }
    }
}

// MARK: - Function sub-dispatchers

extension NumericDispatch {

    static func applyTrigFunction(
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
                "\(name) is not defined for matrices; "
                + "use element-wise operations if needed")
        }
    }

    static func evalScalarTrig(_ name: String, _ x: Double) throws -> Double {
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

    static func evalComplexTrig(_ name: String, _ z: Complex) throws -> Complex {
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
    static func applyExpLogSqrt(
        _ name: String,
        args: [NumericValue]
    ) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 1 argument, got \(args.count)")
        }
        let arg = args[0]
        switch (name, arg.kind) {
        case ("exp", .scalar):
            return .scalar(Foundation.exp(arg.asScalar!))
        case ("exp", .complex):
            return .complex(arg.asComplex!.exp)
        case ("exp", .matrix):
            // Group-B: expm already throws .notSquare; propagate
            return .matrix(try LinAlg.expm(arg.asMatrix!))
        case ("exp", .complexMatrix):
            throw MathExprError.invalidArguments(
                "exp(complexMatrix) is not supported; expm is defined for real matrices only")

        case ("log", .scalar):
            return .scalar(Foundation.log(arg.asScalar!))
        case ("log", .complex):
            return .complex(arg.asComplex!.log)
        case ("log", .matrix):
            // Group-B: logm throws .notSquare; nil → non-diagonalizable
            guard let result = try LinAlg.logm(arg.asMatrix!) else {
                throw MathExprError.invalidArguments(
                    "matrix logarithm failed: matrix is not diagonalizable "
                    + "with real positive eigenvalues")
            }
            return .matrix(result)
        case ("log", .complexMatrix):
            throw MathExprError.invalidArguments(
                "log(complexMatrix) is not supported")

        case ("sqrt", .scalar):
            return .scalar(Foundation.sqrt(arg.asScalar!))
        case ("sqrt", .complex):
            return .complex(arg.asComplex!.sqrt)
        case ("sqrt", .matrix):
            // Group-B: sqrtm throws .notSquare; nil → negative eigenvalues
            guard let result = try LinAlg.sqrtm(arg.asMatrix!) else {
                throw MathExprError.invalidArguments(
                    "matrix square root failed: eigenvalues are negative or complex")
            }
            return .matrix(result)
        case ("sqrt", .complexMatrix):
            throw MathExprError.invalidArguments(
                "sqrt(complexMatrix) is not supported")

        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    static func applyAbsInvDetTrace(
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
            // Frobenius norm per MF-2; NOT spectral norm
            return .scalar(LinAlg.frobeniusNorm(arg.asMatrix!))
        case ("abs", .complexMatrix):
            return try evalAbsComplexMatrix(cm: arg.asComplexMatrix!)

        // inv
        case ("inv", .scalar):
            throw MathExprError.invalidArguments(
                "inv() requires a matrix; for scalar use 1/x")
        case ("inv", .complex):
            throw MathExprError.invalidArguments(
                "inv() requires a matrix; for complex use 1/z")
        case ("inv", .matrix):
            // Group-B: inv throws .notSquare; nil → singular
            guard let result = try LinAlg.inv(arg.asMatrix!) else {
                throw MathExprError.invalidArguments("inverse of singular matrix")
            }
            return .matrix(result)
        case ("inv", .complexMatrix):
            // Group-B: cinv throws .notSquare; nil → singular
            guard let result = try LinAlg.cinv(arg.asComplexMatrix!) else {
                throw MathExprError.invalidArguments("inverse of singular complex matrix")
            }
            return .complexMatrix(result)

        // det
        case ("det", .scalar):
            throw MathExprError.invalidArguments(
                "det() requires a matrix; determinant is undefined for scalars")
        case ("det", .complex):
            throw MathExprError.invalidArguments(
                "det() requires a matrix; determinant is undefined for complex scalars")
        case ("det", .matrix):
            // Group-B: det throws .notSquare; total over square (no Optional)
            return .scalar(try LinAlg.det(arg.asMatrix!))
        case ("det", .complexMatrix):
            // Group-B: cdet throws .notSquare; nil → LAPACK failure (MF-9)
            guard let tuple = try LinAlg.cdet(arg.asComplexMatrix!) else {
                throw LinAlg.LinAlgError.invalidParameter(
                    "cdet: LAPACK failed (info < 0)")
            }
            // (0,0) is a valid result for exactly-singular matrix (DOM-01)
            return .complex(Complex(re: tuple.re, im: tuple.im))

        // trace
        case ("trace", .scalar):
            throw MathExprError.invalidArguments(
                "trace() requires a matrix; trace is undefined for scalars")
        case ("trace", .complex):
            throw MathExprError.invalidArguments(
                "trace() requires a matrix; trace is undefined for complex scalars")
        case ("trace", .matrix):
            // Group-B: trace throws .notSquare
            return .scalar(try LinAlg.trace(arg.asMatrix!))
        case ("trace", .complexMatrix):
            return try evalTraceComplexMatrix(cm: arg.asComplexMatrix!)

        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    static func applyTransposeFunction(args: [NumericValue]) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "transpose() requires exactly 1 argument, got \(args.count)")
        }
        return try applyTransposeUnary(operand: args[0])
    }

    static func applyMultiArgFunction(
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

    static func applyMinMax(name: String, args: [NumericValue]) throws -> NumericValue {
        switch args.count {
        case 1:
            guard case .scalar(let x) = args[0] else {
                throw MathExprError.invalidArguments(
                    "\(name)(x) with 1 argument requires a scalar, got \(args[0].kind)")
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
}
