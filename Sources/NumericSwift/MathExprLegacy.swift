//
//  MathExprLegacy.swift
//  NumericSwift
//
//  Legacy scalar and complex recursive evaluators — snapshot oracle only.
//
//  These functions are the ORIGINAL evaluation implementations, unchanged
//  from before the Phase 4 delegation refactor. They are kept here so that
//  `LegacySnapshotGenerator` can regenerate `LegacySnapshot.json` from the
//  true legacy path — never from the unified evaluator — preserving the
//  parity contract (no vacuous-gate bug).
//
//  DO NOT call these from production code. The public entry points
//  (`MathExpr.evaluate`, `MathExpr.evaluateComplex`) are the correct path
//  for all callers.
//
//  Visibility: `internal` (not `private`) so the test target can access
//  them via `@testable import NumericSwift`.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Legacy evaluator entry points

extension MathExpr {

    /// Legacy real-scalar recursive evaluator — snapshot oracle only.
    ///
    /// This is the pre-Phase-4 implementation of `evaluate(_:variables:)`,
    /// preserved verbatim. Used exclusively by `LegacySnapshotGenerator` to
    /// regenerate the frozen parity baseline from the authentic legacy path.
    // swiftlint:disable:next function_body_length
    static func legacyScalarEvaluate(
        _ ast: MathLexExpression, variables: [String: Double] = [:]
    ) throws -> Double {
        switch ast {
        case .integer(let n):
            return Double(n)
        case .float(let v):
            guard let v else { throw MathExprError.nonFiniteFloat }
            return v
        case .variable(let name):
            guard let value = variables[name] else {
                throw MathExprError.undefinedVariable(name)
            }
            return value
        case .constant(let c):
            return try legacyResolveConstant(c)
        case .rational(let num, let den):
            return try legacyScalarEvaluate(num, variables: variables)
                / legacyScalarEvaluate(den, variables: variables)
        case .binary(let op, let left, let right):
            return try legacyEvalBinary(op, left, right, variables: variables)
        case .unary(let op, let operand):
            return try legacyEvalUnary(op, operand, variables: variables)
        case .function(let name, let args):
            let vals = try args.map { try legacyScalarEvaluate($0, variables: variables) }
            return try legacyEvalFunction(name, args: vals)
        default:
            throw MathExprError.unsupportedNode(nodeLabel(ast))
        }
    }

    /// Legacy complex-scalar recursive evaluator — snapshot oracle only.
    ///
    /// This is the pre-Phase-4 implementation of
    /// `evaluateComplex(_:variables:complexVariables:)`, preserved verbatim.
    // swiftlint:disable:next function_body_length
    static func legacyComplexEvaluate(
        _ ast: MathLexExpression,
        variables: [String: Double] = [:],
        complexVariables: [String: Complex] = [:]
    ) throws -> Complex {
        switch ast {
        case .integer(let n):
            return Complex(Double(n))
        case .float(let v):
            guard let v else { throw MathExprError.nonFiniteFloat }
            return Complex(v)
        case .variable(let name):
            if let cval = complexVariables[name] { return cval }
            if let rval = variables[name] { return Complex(rval) }
            throw MathExprError.undefinedVariable(name)
        case .constant(let c):
            return try legacyResolveComplexConstant(c)
        case .rational(let num, let den):
            let n = try legacyComplexEvaluate(
                num, variables: variables, complexVariables: complexVariables)
            let d = try legacyComplexEvaluate(
                den, variables: variables, complexVariables: complexVariables)
            return n / d
        case .complex(let re, let im):
            let r = try legacyComplexEvaluate(
                re, variables: variables, complexVariables: complexVariables)
            let i = try legacyComplexEvaluate(
                im, variables: variables, complexVariables: complexVariables)
            return Complex(re: r.re, im: i.re)
        case .binary(let op, let left, let right):
            let l = try legacyComplexEvaluate(
                left, variables: variables, complexVariables: complexVariables)
            let r = try legacyComplexEvaluate(
                right, variables: variables, complexVariables: complexVariables)
            return try legacyEvalComplexBinary(op, l, r)
        case .unary(let op, let operand):
            let v = try legacyComplexEvaluate(
                operand, variables: variables, complexVariables: complexVariables)
            return try legacyEvalComplexUnary(op, v)
        case .function(let name, let args):
            let vals = try args.map {
                try legacyComplexEvaluate($0, variables: variables, complexVariables: complexVariables)
            }
            return try legacyEvalComplexFunction(name, args: vals)
        default:
            throw MathExprError.unsupportedNode(nodeLabel(ast))
        }
    }
}

// MARK: - Legacy helpers (internal)

extension MathExpr {

    static func legacyResolveConstant(_ c: MathLexConstant) throws -> Double {
        switch c {
        case .pi: return .pi
        case .e: return M_E
        case .infinity: return .infinity
        case .negInfinity: return -.infinity
        case .nan: return .nan
        case .i, .j, .k:
            throw MathExprError.invalidArguments(
                "imaginary/quaternion constants require complex evaluation")
        }
    }

    static func legacyEvalBinary(
        _ op: BinaryOp,
        _ left: MathLexExpression,
        _ right: MathLexExpression,
        variables: [String: Double]
    ) throws -> Double {
        let l = try legacyScalarEvaluate(left, variables: variables)
        let r = try legacyScalarEvaluate(right, variables: variables)
        switch op {
        case .add: return l + r
        case .sub: return l - r
        case .mul: return l * r
        case .div:
            if r == 0 { throw MathExprError.divisionByZero }
            return l / r
        case .pow: return pow(l, r)
        case .mod: return l.truncatingRemainder(dividingBy: r)
        case .plusMinus, .minusPlus:
            throw MathExprError.unsupportedNode("BinaryOp(\(op))")
        }
    }

    static func legacyEvalUnary(
        _ op: UnaryOp,
        _ operand: MathLexExpression,
        variables: [String: Double]
    ) throws -> Double {
        let v = try legacyScalarEvaluate(operand, variables: variables)
        switch op {
        case .neg: return -v
        case .pos: return v
        case .factorial:
            guard v >= 0 else {
                throw MathExprError.invalidArguments("factorial requires non-negative argument")
            }
            return tgamma(v + 1)
        case .transpose:
            throw MathExprError.unsupportedNode("transpose (matrix operation)")
        }
    }

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    static func legacyEvalFunction(_ name: String, args: [Double]) throws -> Double {
        switch (name, args.count) {
        // Trigonometric
        case ("sin", 1): return sin(args[0])
        case ("cos", 1): return cos(args[0])
        case ("tan", 1): return tan(args[0])
        case ("asin", 1), ("arcsin", 1): return asin(args[0])
        case ("acos", 1), ("arccos", 1): return acos(args[0])
        case ("atan", 1), ("arctan", 1): return atan(args[0])
        case ("atan2", 2): return atan2(args[0], args[1])
        // Hyperbolic
        case ("sinh", 1): return sinh(args[0])
        case ("cosh", 1): return cosh(args[0])
        case ("tanh", 1): return tanh(args[0])
        case ("asinh", 1), ("arcsinh", 1): return asinh(args[0])
        case ("acosh", 1), ("arccosh", 1): return acosh(args[0])
        case ("atanh", 1), ("arctanh", 1): return atanh(args[0])
        // Exponential and logarithmic (SciPy convention: log = natural log)
        case ("exp", 1): return exp(args[0])
        case ("log", 1), ("ln", 1): return log(args[0])
        case ("log10", 1): return log10(args[0])
        case ("log2", 1), ("lg", 1): return log2(args[0])
        // Power and roots
        case ("sqrt", 1): return sqrt(args[0])
        case ("cbrt", 1): return cbrt(args[0])
        case ("pow", 2): return pow(args[0], args[1])
        case ("hypot", 2): return hypot(args[0], args[1])
        // Absolute value, sign, rounding
        case ("abs", 1): return abs(args[0])
        case ("sign", 1), ("sgn", 1): return args[0] > 0 ? 1.0 : (args[0] < 0 ? -1.0 : 0.0)
        case ("floor", 1): return floor(args[0])
        case ("ceil", 1): return ceil(args[0])
        case ("round", 1): return Foundation.round(args[0])
        case ("trunc", 1): return Foundation.trunc(args[0])
        // Min/max and interpolation
        case ("min", 2): return Swift.min(args[0], args[1])
        case ("max", 2): return Swift.max(args[0], args[1])
        case ("min", _) where args.count >= 2: return args.min()!
        case ("max", _) where args.count >= 2: return args.max()!
        case ("clamp", 3): return Swift.min(Swift.max(args[0], args[1]), args[2])
        case ("lerp", 3): return args[0] + (args[1] - args[0]) * args[2]
        // Angle conversion
        case ("rad", 1): return args[0] * .pi / 180.0
        case ("deg", 1): return args[0] * 180.0 / .pi
        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    static func legacyResolveComplexConstant(_ c: MathLexConstant) throws -> Complex {
        switch c {
        case .pi: return Complex(.pi)
        case .e: return Complex(M_E)
        case .i: return Complex(re: 0, im: 1)
        case .infinity: return Complex(.infinity)
        case .negInfinity: return Complex(-.infinity)
        case .nan: return Complex(.nan)
        case .j, .k:
            throw MathExprError.unsupportedNode("quaternion basis requires quaternion arithmetic")
        }
    }

    static func legacyEvalComplexBinary(
        _ op: BinaryOp, _ l: Complex, _ r: Complex
    ) throws -> Complex {
        switch op {
        case .add: return l + r
        case .sub: return l - r
        case .mul: return l * r
        case .div:
            if r.re == 0 && r.im == 0 { throw MathExprError.divisionByZero }
            return l / r
        case .pow:
            if l.re == 0 && l.im == 0 { return Complex(0) }
            return (r * l.log).exp
        case .mod:
            throw MathExprError.unsupportedNode("modulo over complex numbers")
        case .plusMinus, .minusPlus:
            throw MathExprError.unsupportedNode("BinaryOp(\(op))")
        }
    }

    static func legacyEvalComplexUnary(_ op: UnaryOp, _ v: Complex) throws -> Complex {
        switch op {
        case .neg: return Complex(re: -v.re, im: -v.im)
        case .pos: return v
        case .factorial:
            guard v.im == 0, v.re >= 0 else {
                throw MathExprError.invalidArguments("factorial requires non-negative real")
            }
            return Complex(tgamma(v.re + 1))
        case .transpose:
            throw MathExprError.unsupportedNode("transpose (matrix operation)")
        }
    }

    static func legacyEvalComplexFunction(
        _ name: String, args: [Complex]
    ) throws -> Complex {
        guard args.count == 1 else {
            // Multi-arg functions: fall back to real if all args are real
            if args.allSatisfy({ $0.im == 0 }) {
                let reals = args.map(\.re)
                let result = try legacyEvalFunction(name, args: reals)
                return Complex(result)
            }
            throw MathExprError.invalidArguments(
                "\(name) with \(args.count) args not supported for complex evaluation")
        }
        let z = args[0]
        switch name {
        case "exp": return z.exp
        case "log", "ln": return z.log
        case "sqrt": return z.sqrt
        case "sin": return z.sin
        case "cos": return z.cos
        case "tan": return z.tan
        case "sinh": return z.sinh
        case "cosh": return z.cosh
        case "tanh": return z.tanh
        case "abs": return Complex(z.abs)
        case "conj": return z.conj
        case "real", "re": return Complex(z.re)
        case "imag", "im": return Complex(z.im)
        case "arg", "phase": return Complex(z.arg)
        default:
            // Fall back to real evaluation for purely real arguments
            if z.im == 0 {
                let result = try legacyEvalFunction(name, args: [z.re])
                return Complex(result)
            }
            throw MathExprError.invalidArguments(
                "function '\(name)' not supported for complex arguments")
        }
    }
}
