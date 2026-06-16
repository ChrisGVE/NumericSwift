//
//  NumericDispatch.swift
//  NumericSwift
//
//  Central dispatch surface for the unified numeric pipeline.
//
//  This file contains the three public entry points. Operator sub-dispatchers
//  and lattice helpers live in:
//    • NumericDispatch+CoercionLattice.swift  (§15 coercion lattice: coerce1x1,
//                                              coerce1x1Complex, promoteToComplexMatrix,
//                                              promoteScalarToComplex — single documented
//                                              §4.3a collapse sites and promotion helpers)
//    • NumericDispatch+BinaryOps.swift   (applyAddSub / applyMul / applyDiv /
//                                         applyPow / applyMod)
//    • NumericDispatch+UnaryFunctions.swift (applyNeg / applyFactorial /
//                                            applyTranspose + function routers)
//    • NumericDispatch+ComplexMatrix{Arithmetic,Functions,Helpers}.swift and
//      NumericDispatch+MatrixPower.swift   (the matrix/complex-matrix EVAL cells)
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
/// - ``applyBinary(_:lhs:rhs:complexMode:)`` — binary arithmetic/algebraic operators
/// - ``applyUnary(_:operand:)``  — unary prefix/postfix operators
/// - ``applyFunction(_:args:complexMode:)``  — named function calls
///
/// ## Handler seams
///
/// Every truth-table cell marked **EVAL** in §15 is implemented across the
/// `extension NumericDispatch` files (`+BinaryOps`, `+ComplexMatrixArithmetic`,
/// `+ComplexMatrixFunctions`, `+MatrixPower`). The only remaining deferral is
/// `complexMatrix^scalar` (`evalComplexMatrixPow`, §14), which throws
/// `MathExprError.unsupportedNode`.
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
    /// Dispatches according to the §15 truth table. The error model follows the
    /// **two-mechanism Group-A / Group-B split** (§4.5/AC2.2):
    ///
    /// - **Group-A operators** (`+`, `-`, `*`, `/`, `hadamard`, `elementDiv`, `dotProduct`)
    ///   use `precondition` internally in `LinAlg`. The dispatcher **pre-validates** operand
    ///   shapes and divisors, throwing `MathExprError.shapeMismatch` or `.divisionByZero`
    ///   **before** any `LinAlg` precondition can fire, so a shape mismatch is always a
    ///   recoverable error, never a process trap.
    /// - **Group-B named functions** (`inv`, `det`, `trace`, `expm`, `logm`, `sqrtm`,
    ///   `cdet`, `cinv`) already throw `LinAlgError.notSquare` themselves; the dispatcher
    ///   calls them with `try` and propagates that error directly.
    ///
    /// **1×1 coercion (§4.3a):** a `matrix*matrix` or `dotProduct` result that is 1×1
    /// (i.e. vec·vec) is automatically collapsed to `.scalar`. The complex analogue
    /// collapses a 1×1 `complexMatrix` result from `CM*CM` or `dotProduct(CM,CM)` to
    /// `.complex`. This does *not* apply to user-constructed 1×1 matrices, nor to 1×1
    /// results of any other operation (add, hadamard, transpose, inv, …).
    /// See `NumericDispatch+CoercionLattice.swift` for the full lattice specification.
    ///
    /// - Parameters:
    ///   - op:  The binary operator.
    ///   - lhs: Left-hand operand.
    ///   - rhs: Right-hand operand.
    ///   - complexMode: When `true`, the `^` operator promotes a negative-real
    ///     scalar base with a non-integer exponent to the complex principal value
    ///     instead of producing NaN (GitHub issue #1). Defaults to `false`, which
    ///     leaves the real-domain `pow` behaviour unchanged. Only `^` consults
    ///     this flag; all other operators ignore it.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError.shapeMismatch` for Group-A shape mismatches;
    ///           `MathExprError.divisionByZero` for scalar or matrix ÷ 0;
    ///           `MathExprError.invalidArguments` for undefined kind combinations;
    ///           `LinAlgError.invalidParameter` when the result shape exceeds the soft cap;
    ///           `LinAlgError.notSquare` propagated from Group-B named functions.
    public static func applyBinary(
        _ op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue,
        complexMode: Bool = false
    ) throws -> NumericValue {
        switch op {
        case .add, .sub:
            return try applyAddSub(op, lhs: lhs, rhs: rhs)
        case .mul:
            return try applyMul(lhs: lhs, rhs: rhs)
        case .div:
            return try applyDiv(lhs: lhs, rhs: rhs)
        case .pow:
            return try applyPow(lhs: lhs, rhs: rhs, complexMode: complexMode)
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
    /// ## Soft-cap enforcement (AC3.6 / CONS-07)
    ///
    /// Unary operators that allocate a result matrix pre-check the result shape
    /// via `LinAlg.checkSoftCap(rows:cols:)` before any allocation:
    ///
    /// - `.neg` on a real or complex matrix: same-shape result.
    /// - `.transpose` on a real or complex matrix: transposed shape (`cols × rows`).
    ///
    /// When the result would exceed `LinAlg.maxEvaluatorMatrixElements` the check
    /// throws `LinAlgError.invalidParameter` (CONS-07 — **never** `MathExprError`).
    ///
    /// > Note: This is a **per-result** guard.  It bounds the element count of a
    /// > single output matrix, not the cumulative working set of a chained expression
    /// > (MF-5 / §5 deferral).
    ///
    /// - Parameters:
    ///   - op:      The unary operator.
    ///   - operand: The operand.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError` for invalid kind/argument combinations;
    ///           `LinAlgError.invalidParameter` when the result shape exceeds
    ///           the evaluator soft cap (CONS-07).
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
    /// Dispatch uses the unified `functionRegistry` (see
    /// `NumericDispatch+FunctionRegistry.swift`). Three distinct error paths follow
    /// AC2.3a:
    ///   1. Unknown name → `MathExprError.unknownFunction(name)`
    ///   2. Known name, wrong arity → `MathExprError.invalidArguments("… expects N arg(s)")`
    ///   3. Known name, right arity, unsupported operand kind →
    ///      `MathExprError.invalidArguments("…")`
    ///
    /// Group-B functions (trace/det/inv/expm/logm/sqrtm/cdet/cinv) propagate
    /// `LinAlgError.notSquare` unmodified when the input matrix is non-square.
    ///
    /// ## Soft-cap enforcement (AC3.6 / CONS-07)
    ///
    /// Functions that allocate a result matrix pre-check the result shape via
    /// `LinAlg.checkSoftCap(rows:cols:)` before any LAPACK call or allocation.
    /// Affected matrix-producing functions: `inv`, `exp` (expm), `log` (logm),
    /// `sqrt` (sqrtm), `cinv`, `hadamard`, `elementDiv`, `dotProduct`.
    /// When the result shape would exceed `LinAlg.maxEvaluatorMatrixElements`
    /// the check throws `LinAlgError.invalidParameter` (CONS-07).
    ///
    /// > Note: This is a **per-result** guard.  It bounds the element count of a
    /// > single output matrix, not the cumulative working set of a chained
    /// > expression (MF-5 / §5 deferral).
    ///
    /// ## Complex-context promotion (`complexMode`)
    ///
    /// When `complexMode` is `true`, a negative-real scalar argument to `sqrt`,
    /// `log`, or `ln` is promoted to a `.complex` value before dispatch, so the
    /// function returns the complex principal value instead of NaN (GitHub
    /// issue #1). This is the function-side half of the legacy `evaluateComplex`
    /// complex-context behaviour; the `^` operator half lives in `applyPow`.
    /// The set is intentionally narrow — exactly the names whose legacy complex
    /// path was complex-native (`z.sqrt` / `z.log`). Names that the legacy
    /// complex evaluator routed through the *real* fallback (`log10`, `log2`,
    /// `cbrt`, inverse trig, the 2-arg `pow` function, …) are NOT promoted, so
    /// they keep returning NaN for negative reals exactly as before.
    ///
    /// - Parameters:
    ///   - name: The function name (case-sensitive).
    ///   - args: The evaluated argument list.
    ///   - complexMode: Promote negative-real `sqrt`/`log`/`ln` arguments to
    ///     `.complex`. Defaults to `false`.
    /// - Returns: The result as a `NumericValue`.
    /// - Throws: `MathExprError.unknownFunction` for unrecognised names;
    ///           `MathExprError.invalidArguments` for arity or kind mismatches;
    ///           `LinAlgError.notSquare` propagated from Group-B functions;
    ///           `LinAlgError.invalidParameter` when the result shape exceeds
    ///           the evaluator soft cap (CONS-07).
    public static func applyFunction(
        _ name: String,
        args: [NumericValue],
        complexMode: Bool = false
    ) throws -> NumericValue {
        // Step 1: Name lookup — AC2.3a error 1.
        guard let descriptor = functionRegistry[name] else {
            throw MathExprError.unknownFunction(name)
        }

        // Step 2: Arity check — AC2.3a error 2.
        guard args.count >= descriptor.arityMin && args.count <= descriptor.arityMax else {
            let arityDesc = descriptor.arityMin == descriptor.arityMax
                ? "\(descriptor.arityMin)"
                : "\(descriptor.arityMin)–\(descriptor.arityMax)"
            throw MathExprError.invalidArguments(
                "\(name) requires \(arityDesc) argument(s), got \(args.count)")
        }

        // Step 2.5: Complex-context promotion (issue #1) — narrow to the three
        // complex-native single-argument functions whose negative-real input the
        // real path sends to NaN.
        let dispatchArgs = complexMode
            ? promoteNegativeRealForComplexMode(name, args: args)
            : args

        // Step 3: Invoke handler — kind validation and Group-B error propagation
        // are performed inside the handler (AC2.3a error 3 + Group-B contract).
        return try descriptor.handler(name, dispatchArgs)
    }

    /// Promote a negative-real scalar argument of `sqrt`/`log`/`ln` to `.complex`
    /// so the complex-native function path is taken (issue #1, complex mode).
    ///
    /// Only the listed names and only a `.scalar(x)` with `x < 0` are promoted;
    /// every other argument is returned unchanged. `x == 0` is left as a scalar
    /// because `sqrt(0)`/`log(0)` agree on the real and complex paths.
    private static func promoteNegativeRealForComplexMode(
        _ name: String,
        args: [NumericValue]
    ) -> [NumericValue] {
        guard name == "sqrt" || name == "log" || name == "ln",
              args.count == 1,
              case .scalar(let x) = args[0],
              x < 0 else {
            return args
        }
        return [.complex(Complex(x))]
    }

}
