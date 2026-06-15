//
//  NumericDispatch+FunctionDispatchers.swift
//  NumericSwift
//
//  Named-function sub-dispatchers for the unified numeric pipeline.
//
//  Covers:
//    • applyTrigFunction / evalScalarTrig / evalComplexTrig
//    • applyExpLogSqrt (exp / log / sqrt — scalar + complex + matrix)
//    • applyAbsInvDetTrace (abs / inv / det / trace — multi-kind)
//    • applyTransposeFunction / applyMultiArgFunction / applyMinMax
//    • applyCdetCinv (cdet / cinv — complexMatrix only)
//
//  Unary-operator handlers (applyNeg / applyFactorial / applyTransposeUnary)
//  live in NumericDispatch+UnaryFunctions.swift.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

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
        case "sin":              return Foundation.sin(x)
        case "cos":              return Foundation.cos(x)
        case "tan":              return Foundation.tan(x)
        case "asin", "arcsin":   return Foundation.asin(x)
        case "acos", "arccos":   return Foundation.acos(x)
        case "atan", "arctan":   return Foundation.atan(x)
        case "sinh":             return Foundation.sinh(x)
        case "cosh":             return Foundation.cosh(x)
        case "tanh":             return Foundation.tanh(x)
        case "asinh", "arcsinh": return Foundation.asinh(x)
        case "acosh", "arccosh": return Foundation.acosh(x)
        case "atanh", "arctanh": return Foundation.atanh(x)
        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    static func evalComplexTrig(_ name: String, _ z: Complex) throws -> Complex {
        switch name {
        case "sin":              return z.sin
        case "cos":              return z.cos
        case "tan":              return z.tan
        case "asin", "arcsin":   return z.asin
        case "acos", "arccos":   return z.acos
        case "atan", "arctan":   return z.atan
        case "sinh":             return z.sinh
        case "cosh":             return z.cosh
        case "tanh":             return z.tanh
        case "asinh", "arcsinh": return z.asinh
        case "acosh", "arccosh": return z.acosh
        case "atanh", "arctanh": return z.atanh
        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    /// Route `exp`, `log`, or `sqrt` to the correct scalar, complex, or matrix handler.
    ///
    /// Matrix dispatch follows the **Group-B try-and-propagate** model:
    /// the function calls the throwing `LinAlg` operation inside `try` and lets
    /// `LinAlgError.notSquare` propagate unmodified to the caller.
    /// A `nil` return from `logm`/`sqrtm` (non-diagonalizable or negative-eigenvalue
    /// matrix) is converted to `MathExprError.invalidArguments` with a diagnostic
    /// message that states the mathematical reason.
    ///
    /// - Parameters:
    ///   - name: One of `"exp"`, `"log"`, `"sqrt"` (case-sensitive).
    ///   - args: Exactly one `NumericValue` argument.
    /// - Returns: Scalar, complex, or matrix result per the §15 truth table.
    /// - Throws: `MathExprError.invalidArguments` for unsupported kind combinations
    ///           or when `logm`/`sqrtm` returns `nil`;
    ///           `LinAlgError.notSquare` propagated from `expm`/`logm`/`sqrtm`
    ///           when the matrix is not square.
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
            // Soft-cap: expm returns a same-shape matrix; check before LAPACK call.
            // Group-B: expm already throws .notSquare; propagate.
            let em = arg.asMatrix!
            try LinAlg.checkSoftCap(rows: em.rows, cols: em.cols)
            return .matrix(try LinAlg.expm(em))
        case ("exp", .complexMatrix):
            throw MathExprError.invalidArguments(
                "exp(complexMatrix) is not supported; expm is defined for real matrices only")

        case ("log", .scalar):
            return .scalar(Foundation.log(arg.asScalar!))
        case ("log", .complex):
            return .complex(arg.asComplex!.log)
        case ("log", .matrix):
            // Soft-cap: logm returns a same-shape matrix; check before LAPACK call.
            // Group-B: logm throws .notSquare; nil → non-diagonalizable.
            let lm = arg.asMatrix!
            try LinAlg.checkSoftCap(rows: lm.rows, cols: lm.cols)
            guard let result = try LinAlg.logm(lm) else {
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
            // Soft-cap: sqrtm returns a same-shape matrix; check before LAPACK call.
            // Group-B: sqrtm throws .notSquare; nil → negative eigenvalues.
            let sm = arg.asMatrix!
            try LinAlg.checkSoftCap(rows: sm.rows, cols: sm.cols)
            guard let result = try LinAlg.sqrtm(sm) else {
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

    /// Route `abs`, `inv`, `det`, or `trace` to the correct scalar, complex, or matrix handler.
    ///
    /// Matrix dispatch follows the **Group-B try-and-propagate** model:
    ///
    /// - `trace(M)` and `det(M)` call the throwing `LinAlg` operation directly;
    ///   `LinAlgError.notSquare` propagates to the caller without pre-validation.
    /// - `inv(M)` calls `LinAlg.inv` (which throws `.notSquare` for non-square inputs
    ///   and returns `nil` for singular square matrices).  A `nil` result is converted
    ///   to `MathExprError.invalidArguments("inverse of singular matrix")`.
    /// - `abs(M)` returns the Frobenius norm (MF-2 specification; **not** spectral norm).
    ///
    /// - Parameters:
    ///   - name: One of `"abs"`, `"inv"`, `"det"`, `"trace"` (case-sensitive).
    ///   - args: Exactly one `NumericValue` argument.
    /// - Returns: Scalar or matrix result per the §15 truth table.
    /// - Throws: `MathExprError.invalidArguments` for unsupported kind combinations
    ///           or when `inv` receives a singular matrix;
    ///           `LinAlgError.notSquare` propagated from `trace`/`det`/`inv`
    ///           when the matrix is not square.
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
            // Soft-cap: inv returns an n×n matrix (same shape as square input).
            // Group-B: inv throws .notSquare; nil → singular.
            let im = arg.asMatrix!
            try LinAlg.checkSoftCap(rows: im.rows, cols: im.cols)
            guard let result = try LinAlg.inv(im) else {
                throw MathExprError.invalidArguments("inverse of singular matrix")
            }
            return .matrix(result)
        case ("inv", .complexMatrix):
            // Group-B: cinv throws .notSquare; nil → singular.
            // Soft-cap pre-check: cinv returns an n×n complex matrix.
            let cm = arg.asComplexMatrix!
            try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
            guard let result = try LinAlg.cinv(cm) else {
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
            // Group-B: cdet throws .notSquare; nil → LAPACK failure (MF-9).
            // (0,0) is a valid result for exactly-singular matrix (DOM-01) — NOT an error.
            guard let tuple = try LinAlg.cdet(arg.asComplexMatrix!) else {
                throw LinAlg.LinAlgError.invalidParameter(
                    "cdet: LAPACK failed (info < 0)")
            }
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
        case "elementDiv":
            return try applyElementDiv(lhs: args[0], rhs: args[1])
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

    /// Route `cdet` or `cinv` to the complex-matrix determinant / inverse (Group-B).
    ///
    /// Both functions require exactly one `.complexMatrix` argument.  Scalars,
    /// complex scalars, and real matrices are rejected with `MathExprError.invalidArguments`.
    ///
    /// ## `cdet` semantics (MF-9, DOM-01)
    ///
    /// Delegates to `LinAlg.cdet(_:)` which internally runs `zgetrf_`:
    ///
    /// - **Non-square input**: `LinAlg.cdet` throws `LinAlgError.notSquare`, which
    ///   propagates unmodified to the caller (Group-B contract).
    /// - **Exactly singular (`zgetrf_` info > 0)**: `LinAlg.cdet` returns `(0, 0)`.
    ///   This is a **valid** determinant value, not an error — a singular matrix
    ///   has determinant zero.  The adapter wraps it as `.complex(0+0i)`.
    /// - **LAPACK illegal argument (`zgetrf_` info < 0)**: `LinAlg.cdet` returns `nil`.
    ///   This path is not reachable from well-formed public input but is preserved
    ///   by the adapter which converts `nil` → `LinAlgError.invalidParameter`.
    ///
    /// Result type: `.complex` unconditionally (never auto-collapsed to `.scalar`
    /// even when the imaginary part is zero).  This is consistent with how the
    /// dispatch table treats all complex-matrix → determinant results.
    ///
    /// ## `cinv` semantics (SEC-N03)
    ///
    /// Delegates to `LinAlg.cinv(_:)` which runs `zgetrf_` + `zgetri_`:
    ///
    /// - **Non-square input**: `LinAlg.cinv` throws `LinAlgError.notSquare`, propagated.
    /// - **Singular or LAPACK failure (info ≠ 0 at either stage)**: `LinAlg.cinv`
    ///   returns `nil`.  Unlike `cdet`, `cinv` collapses both `info > 0` (singular)
    ///   and `info < 0` (illegal argument) into a single `nil`; the adapter converts
    ///   this to `MathExprError.invalidArguments("inverse of singular complex matrix")`.
    /// - **Soft cap**: `cinv` returns an n×n complex matrix.  A pre-check via
    ///   `LinAlg.checkSoftCap(rows:cols:)` guards against over-allocation before
    ///   calling the LAPACK routine.
    ///
    /// - Parameters:
    ///   - name: `"cdet"` or `"cinv"` (case-sensitive).
    ///   - args: Exactly one `.complexMatrix` argument.
    /// - Returns: `.complex` for `cdet`; `.complexMatrix` for `cinv`.
    /// - Throws: `MathExprError.invalidArguments` for wrong arity or non-`complexMatrix` kind;
    ///           `LinAlgError.notSquare` propagated from `LinAlg.cdet`/`cinv` for non-square input;
    ///           `LinAlgError.invalidParameter` when `cdet` returns `nil` (LAPACK info < 0);
    ///           `LinAlgError.invalidParameter` from `checkSoftCap` when the result exceeds the
    ///           soft matrix-size cap (cinv only).
    static func applyCdetCinv(
        _ name: String,
        args: [NumericValue]
    ) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 1 argument, got \(args.count)")
        }
        let arg = args[0]
        guard case .complexMatrix = arg.kind else {
            throw MathExprError.invalidArguments(
                "\(name) requires a complex matrix argument, got \(arg.kind)")
        }
        let cm = arg.asComplexMatrix!
        switch name {
        case "cdet":
            // Group-B: notSquare propagated from LinAlg.cdet; no pre-guard.
            // cdet returns a scalar complex value, so no soft-cap applies.
            guard let tuple = try LinAlg.cdet(cm) else {
                // info < 0: LAPACK illegal argument — not reachable from valid public
                // input, but must not be silently swallowed (MF-9).
                throw LinAlg.LinAlgError.invalidParameter(
                    "cdet: LAPACK zgetrf_ returned info < 0 (illegal argument)")
            }
            // (0, 0) is the correct determinant of a singular matrix (DOM-01) — value, not error.
            return .complex(Complex(re: tuple.re, im: tuple.im))
        case "cinv":
            // Soft-cap pre-check: cinv allocates an n×n complex matrix.
            try LinAlg.checkSoftCap(rows: cm.rows, cols: cm.cols)
            // Group-B: notSquare propagated from LinAlg.cinv; no pre-guard.
            guard let result = try LinAlg.cinv(cm) else {
                // nil collapses both singular (info > 0) and illegal-argument (info < 0).
                throw MathExprError.invalidArguments("inverse of singular complex matrix")
            }
            return .complexMatrix(result)
        default:
            throw MathExprError.unknownFunction(name)
        }
    }
}
