//
//  UnifiedEvaluator.swift
//  NumericSwift
//
//  The unified numeric evaluation front door for the NumericSwift pipeline.
//
//  This file contains:
//    - `MathExpr.evaluateUnified` — the public entry point, delegates to
//      `UnifiedEvaluatorCore.eval` (defined in UnifiedEvaluatorCore.swift).
//    - `MathExpr.extractDouble` / `MathExpr.extractComplex` — result
//      extraction helpers that bridge `NumericValue` back to the typed public
//      `evaluate`/`evaluateComplex` wrappers in `MathExpr.swift`.
//
//  ## Related files
//
//    UnifiedEvaluatorCore.swift   — recursive eval switch + leaf handlers
//    UnifiedEvaluatorMatrix.swift — vector/matrix literal nodes + linalg nodes
//    MathExpr.swift               — public evaluate/evaluateComplex wrappers
//
//  ## Thread safety
//
//  `MathExpr.evaluateUnified` is a static method with no stored state; it is
//  re-entrant. `NumericDispatch` and `UnifiedEvaluatorCore` are also stateless
//  (pure enum namespaces).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Unified evaluator public entry point

extension MathExpr {

    /// Evaluate a `MathLexExpression` AST over a `NumericValue` variable
    /// binding dictionary, returning a `NumericValue`.
    ///
    /// This is the unified evaluation front door introduced in Phase 3 of the
    /// unified-numeric-pipeline. It handles all numeric node kinds the AST
    /// can produce — scalars, complex numbers, matrices, and the linear-algebra
    /// AST nodes (`.dotProduct`, `.determinant`, `.matrixInverse`, `.trace`,
    /// `.conjugateTranspose`, `.rank`) — in one recursive pass.
    ///
    /// ## Variable binding
    ///
    /// Variables in the expression are resolved from `values`. A key missing
    /// from `values` throws `MathExprError.undefinedVariable`. On the default
    /// build (no MathLex Rust backend) the parser never emits `.vector` /
    /// `.matrix` literal nodes, so matrices must be supplied via `values`.
    /// On the mathlex build both paths are supported simultaneously.
    ///
    /// ## Fallback-parser bracket limitation
    ///
    /// The default-build pure-Swift parser has **no bracket tokenizer**.
    /// Expressions such as `[1,2,3]` or `[[1,2],[3,4]]` cannot be parsed and
    /// will throw `MathExprError.parseError` from `MathExpr.parse(_:)`.
    /// This is intentional — on the default build, matrix values enter the
    /// pipeline through the `values:` binding dictionary. Bracket-literal
    /// parsing is only available with the opt-in mathlex Rust backend
    /// (`NUMERICSWIFT_INCLUDE_MATHLEX=1`).
    ///
    /// **Imaginary literals are not affected** — expressions such as `2*i`,
    /// `3.5*i`, or `1 + 2*i` parse and evaluate correctly on the default build.
    ///
    /// ## Operator semantics
    ///
    /// - `*` between two matrices is **matrix multiplication** (matmul), not
    ///   element-wise. Element-wise multiplication is the `hadamard` named function.
    /// - `dot(u, v)` for two column vectors returns a `.scalar` after the
    ///   1×1 → scalar coercion (§4.3a).
    ///
    /// ## Error surface
    ///
    /// Throws `MathExprError` for evaluation failures:
    ///   - `.undefinedVariable(name)` when a `values` key is absent.
    ///   - `.unknownFunction(name)` for an unrecognised function name.
    ///   - `.divisionByZero` for scalar/matrix division by zero.
    ///   - `.invalidArguments(_)` for arity or kind mismatches.
    ///   - `.unsupportedNode(_)` for calculus/logic/set-theory AST nodes.
    ///   - `.nonFiniteFloat` for a `.float(nil)` NaN-sentinel in the AST.
    ///   - `.shapeMismatch(_)` for Group-A shape violations.
    ///
    /// `LinAlgError.notSquare` propagates unchanged from Group-B functions.
    ///
    /// ## Complex-context promotion (`complexMode`)
    ///
    /// When `complexMode` is `true`, the negative-real domain functions that the
    /// real path sends to NaN are instead promoted to their complex principal
    /// value: `sqrt`/`log`/`ln` of a negative-real scalar, and the `^` operator
    /// with a negative-real base and a non-integer exponent. The public
    /// `evaluateComplex` wrapper sets this `true` so it regains the legacy
    /// complex-context behaviour (GitHub issue #1); the real `evaluate` wrapper
    /// leaves it `false`, preserving the frozen real NaN contract. The
    /// `pow(x, y)` *function* never promotes (legacy routed it through the real
    /// fallback) — only the `^` *operator* does.
    ///
    /// - Parameters:
    ///   - ast: The decoded `MathLexExpression` AST to evaluate.
    ///   - values: Variable bindings — any `NumericValue` kind is accepted.
    ///   - complexMode: Promote negative-real `sqrt`/`log`/`ln`/`^` to the
    ///     complex principal value instead of NaN. Defaults to `false`.
    /// - Returns: The evaluated result as a `NumericValue`.
    /// - Throws: `MathExprError` or `LinAlgError` as described above.
    public static func evaluateUnified(
        _ ast: MathLexExpression,
        values: [String: NumericValue] = [:],
        complexMode: Bool = false
    ) throws -> NumericValue {
        try UnifiedEvaluatorCore.eval(ast, values: values, complexMode: complexMode)
    }
}

// MARK: - Result extraction helpers for public wrapper handoff

extension MathExpr {

    /// Extract a `Double` from a `NumericValue`, for bridging the unified
    /// evaluator back to the public `evaluate(_:variables:) -> Double` API.
    ///
    /// Scalars are returned directly. Complex values with zero imaginary part
    /// are coerced to `Double` (their real component). All other kinds
    /// (matrix, complexMatrix, complex-with-nonzero-imag) throw
    /// `MathExprError.invalidArguments` so the Phase 4 wrapper can surface a
    /// clean error rather than silently truncating.
    ///
    /// - Parameter value: The `NumericValue` to extract from.
    /// - Returns: The `Double` payload.
    /// - Throws: `MathExprError.invalidArguments` if `value` is not
    ///   representable as a real scalar.
    static func extractDouble(_ value: NumericValue) throws -> Double {
        switch value {
        case .scalar(let x):
            return x
        case .complex(let z):
            guard z.im == 0 else {
                throw MathExprError.invalidArguments(
                    "result is complex (\(z)) — use evaluateUnified for complex results")
            }
            return z.re
        case .matrix, .complexMatrix:
            throw MathExprError.invalidArguments(
                "result is a \(value.typeAndShapeDescription) — use evaluateUnified for matrix results")
        }
    }

    /// Extract a `Complex` from a `NumericValue`, for bridging the unified
    /// evaluator back to the public `evaluateComplex(_:variables:complexVariables:)` API.
    ///
    /// Scalars are promoted to `Complex` (imaginary part = 0). Complex values
    /// are returned directly. Matrix kinds throw `MathExprError.invalidArguments`.
    ///
    /// - Parameter value: The `NumericValue` to extract from.
    /// - Returns: The `Complex` payload (possibly promoted from scalar).
    /// - Throws: `MathExprError.invalidArguments` if `value` is a matrix.
    static func extractComplex(_ value: NumericValue) throws -> Complex {
        switch value {
        case .scalar(let x):
            return Complex(x)
        case .complex(let z):
            return z
        case .matrix, .complexMatrix:
            throw MathExprError.invalidArguments(
                "result is a \(value.typeAndShapeDescription) — use evaluateUnified for matrix results")
        }
    }
}
