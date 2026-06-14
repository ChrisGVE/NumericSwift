//
//  NumericDispatch.swift
//  NumericSwift
//
//  Central dispatch surface for the unified numeric pipeline.
//
//  This file contains the three public entry points and the §4.3a 1×1-matrix
//  coercion helper. Operator sub-dispatchers live in:
//    • NumericDispatch+BinaryOps.swift   (applyAddSub / applyMul / applyDiv /
//                                         applyPow / applyMod)
//    • NumericDispatch+UnaryFunctions.swift (applyNeg / applyFactorial /
//                                            applyTranspose + function routers)
//    • NumericDispatch+EvalStubs.swift   (EVAL-cell stubs, Tasks 10-12 fill in)
//
//  Routing discriminant (9.2): `NumericValue.Kind` from NumericValue+Accessors.swift
//  is the routing key for all dispatch switches. No competing kind enum is defined
//  here — the existing Kind enum serves the §15 truth table directly.
//
//  Error-boundary model (§AC2.2):
//    • Group-A operators (add/sub/hadamard/elementDiv/dot/div(m,scalar)) use
//      LinAlg `precondition` internally. The dispatch layer PRE-VALIDATES shapes/
//      divisors and throws `LinAlg.LinAlgError.dimensionMismatch` or
//      `MathExprError.divisionByZero` BEFORE the call to prevent traps.
//    • Group-B named functions (trace/det/inv/expm/logm/sqrtm/cdet/cinv) already
//      throw `LinAlgError.notSquare` internally. The dispatch layer calls with
//      `try` and PROPAGATES their errors — no redundant pre-guard.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - NumericDispatch

/// Central dispatch surface for the unified numeric pipeline.
///
/// Routes `(operator/function, NumericValue kinds)` → typed result following the
/// §15 truth table. Three public entry points cover all dispatch needs:
///
/// - ``applyBinary(_:lhs:rhs:)`` — binary arithmetic/algebraic operators
/// - ``applyUnary(_:operand:)``  — unary prefix/postfix operators
/// - ``applyFunction(_:args:)``  — named function calls
///
/// ## Handler seams
///
/// Every truth-table cell marked **EVAL** in §15 is currently a stub in
/// `NumericDispatch+EvalStubs.swift`. Tasks 10–16 replace those stubs with real
/// implementations by providing `extension NumericDispatch` in separate files.
/// The stubs throw `MathExprError.unsupportedNode("not yet implemented: … (Task N)")`.
///
/// ## Error boundary
///
/// - Group-A operators: dispatch pre-validates and throws before LinAlg preconditions.
/// - Group-B functions: thrown `LinAlgError` propagates from the named function itself.
///
/// ## Thread safety
///
/// `NumericDispatch` is a pure `enum` namespace with no stored state. All methods
/// are re-entrant; multiple threads may call them concurrently.
public enum NumericDispatch {

    // MARK: - Binary dispatch

    /// Route a binary operator over two `NumericValue` operands.
    ///
    /// Dispatches according to the §15 truth table. Group-A operators pre-validate
    /// operand shapes/divisors and throw before invoking `LinAlg`; Group-B named
    /// functions propagate `LinAlgError.notSquare` directly.
    ///
    /// - Parameters:
    ///   - op:  The binary operator.
    ///   - lhs: Left-hand operand.
    ///   - rhs: Right-hand operand.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError` or `LinAlgError` for invalid combinations.
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
                "plusMinus/minusPlus are display-only operators without numeric semantics")
        }
    }

    // MARK: - Unary dispatch

    /// Route a unary operator over a `NumericValue` operand.
    ///
    /// - Parameters:
    ///   - op:      The unary operator.
    ///   - operand: The operand.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError` for invalid combinations.
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
    /// Name validation follows AC2.3a (three distinct error cases):
    ///   1. Unknown name → `MathExprError.unknownFunction(name)`
    ///   2. Known name, wrong arity → `MathExprError.invalidArguments("… expects N arg(s)")`
    ///   3. Known name, right arity, unsupported kind →
    ///      `MathExprError.invalidArguments("…")`
    ///
    /// - Parameters:
    ///   - name: The function name (case-sensitive).
    ///   - args: The evaluated argument list.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError` or `LinAlgError` as described above.
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
        case "dotProduct", "hadamard", "elementDiv":
            return try applyMultiArgFunction(name, args: args)
        case "crossProduct":
            throw MathExprError.unsupportedNode(
                "crossProduct not yet implemented (deferred §14)")
        case "min":
            return try applyMinMax(name: "min", args: args)
        case "max":
            return try applyMinMax(name: "max", args: args)
        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    // MARK: - §4.3a coercion helper

    /// Collapse a 1×1 `.matrix` result to `.scalar`.
    ///
    /// Applied only at the `dot`/vec·vec result site per §4.3a. A user-constructed
    /// 1×1 matrix is never coerced implicitly; coercion is local to this call.
    ///
    /// A 1×1 `.complexMatrix` is NOT coerced here (complex vec·vec uses a
    /// separate path in ``evalComplexMatrixDotProduct``).
    static func coerce1x1(_ value: NumericValue) -> NumericValue {
        if case .matrix(let m) = value, m.rows == 1, m.cols == 1 {
            return .scalar(m[0, 0])
        }
        return value
    }
}
