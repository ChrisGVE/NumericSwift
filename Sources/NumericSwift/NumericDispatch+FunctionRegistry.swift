//
//  NumericDispatch+FunctionRegistry.swift
//  NumericSwift
//
//  Unified function registry for the `applyFunction(_:args:)` dispatch surface.
//
//  ## Design
//
//  The registry is a single static dictionary `[String: FunctionDescriptor]` that
//  maps every supported function name to a descriptor containing:
//
//    • `arityMin` / `arityMax` — accepted argument count range (most functions
//      have min == max; "min" and "max" accept 1 or 2 scalars).
//    • `handler` — `(String, [NumericValue]) throws → NumericValue`.
//
//  Functions are classified Group-A / Group-B for the two-mechanism error model
//  (see the "Function corpus" tables below). The classification governs how each
//  handler is written — Group-A pre-validates so its `LinAlg` preconditions are
//  unreachable for well-formed inputs (the evaluator soft cap is the one
//  exception, surfaced as a thrown `LinAlgError.invalidParameter`); Group-B calls
//  `LinAlg` directly and propagates `LinAlgError` unmodified. It is a property of
//  each handler's implementation, not a runtime dispatch input.
//
//  `applyFunction` performs three validation steps before invoking the handler:
//    1. Name lookup → `.unknownFunction` if absent.
//    2. Arity check → `.invalidArguments` if outside [arityMin, arityMax].
//    3. Operand-kind check (via the handler or descriptor's `allowedKinds`) →
//       `.invalidArguments` if unsupported.
//  These correspond to AC2.3a errors (unknown / wrong arity / wrong kind).
//
//  ## Function corpus
//
//  ### Group-A scalar/complex functions (scalar + complex only; matrices rejected)
//
//  | Name(s)                    | Args | Notes                              |
//  |----------------------------|------|------------------------------------|
//  | sin, cos, tan              |  1   | trig                               |
//  | asin/arcsin, acos/arccos,  |  1   | inverse trig (aliases supported)   |
//  | atan/arctan                |      |                                    |
//  | sinh, cosh, tanh           |  1   | hyperbolic                         |
//  | asinh/arcsinh, acosh/      |  1   | inverse hyperbolic                 |
//  | arccosh, atanh/arctanh     |      |                                    |
//  | atan2                      |  2   | scalar only                        |
//  | cbrt                       |  1   | cube root, scalar only             |
//  | pow                        |  2   | scalar power, scalar only          |
//  | hypot                      |  2   | scalar only                        |
//  | floor, ceil, round, trunc  |  1   | scalar only                        |
//  | sign, sgn                  |  1   | scalar only                        |
//  | clamp                      |  3   | scalar only                        |
//  | lerp                       |  3   | scalar only                        |
//  | rad, deg                   |  1   | angle conversion, scalar only      |
//  | min, max                   | 1–2  | scalar only                        |
//  | conj                       |  1   | complex only                       |
//  | real/re, imag/im           |  1   | complex only                       |
//  | arg/phase                  |  1   | complex only                       |
//
//  ### Group-A scalar/complex/matrix functions
//
//  | Name                       | Args | Notes                              |
//  |----------------------------|------|------------------------------------|
//  | exp                        |  1   | exp(Matrix) → expm (Group-B)       |
//  | log, ln                    |  1   | log(Matrix) → logm (Group-B)       |
//  | log10                      |  1   | scalar only                        |
//  | log2, lg                   |  1   | scalar only                        |
//  | sqrt                       |  1   | sqrt(Matrix) → sqrtm (Group-B)     |
//  | abs                        |  1   | |scalar|, |z|, Frobenius(M/CM)     |
//  | transpose                  |  1   | all kinds                          |
//
//  ### Group-B matrix/complex-matrix functions (LinAlgError.notSquare propagates)
//
//  | Name      | Args | Input kinds             | Output            |
//  |-----------|------|-------------------------|-------------------|
//  | inv       |  1   | matrix / complexMatrix  | same kind         |
//  | det       |  1   | matrix / complexMatrix  | scalar / complex  |
//  | trace     |  1   | matrix / complexMatrix  | scalar / complex  |
//  | expm via exp | 1 | matrix only             | matrix            |
//  | logm via log | 1 | matrix only             | matrix (may fail) |
//  | sqrtm via sqrt|1 | matrix only             | matrix (may fail) |
//  | cdet      |  1   | complexMatrix only      | complex           |
//  | cinv      |  1   | complexMatrix only      | complexMatrix     |
//
//  ### Group-A 2-arg matrix functions
//
//  | Name        | Args | Notes                                |
//  |-------------|------|--------------------------------------|
//  | dotProduct  |  2   | matrix/complexMatrix bilinear dot    |
//  | hadamard    |  2   | element-wise product                 |
//  | elementDiv  |  2   | element-wise division                |
//
//  ### Deferred
//
//  | Name          | Status                                           |
//  |---------------|--------------------------------------------------|
//  | crossProduct  | Deferred §14 — throws .unsupportedNode           |
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - FunctionDescriptor

/// A registry entry describing one named function in the dispatch table.
///
/// The handler receives the validated (name, args) pair and performs the
/// actual computation. Arity and kind validation happen BEFORE the handler
/// is invoked (see `applyFunction(_:args:)`).
///
/// Handler receives: (`name: String`, `args: [NumericValue]`) throws → `NumericValue`.
/// Passing `name` lets shared handlers (e.g. `applyTrigFunction`) discriminate
/// between related function names without requiring separate closures.
struct FunctionDescriptor: Sendable {
    /// Minimum number of accepted arguments (≥ 0).
    let arityMin: Int
    /// Maximum number of accepted arguments. Equal to `arityMin` for fixed-arity functions.
    let arityMax: Int
    /// The handler invoked after name/arity/kind validation passes.
    ///
    /// Declared `@Sendable` so the compiler rejects any future handler that
    /// captures non-`Sendable` reference state — `functionRegistry` is a shared
    /// `static let` read concurrently, and all current handlers are stateless
    /// (they reference only other stateless `enum`-namespace statics).
    let handler: @Sendable (String, [NumericValue]) throws -> NumericValue
}

// MARK: - Registry

extension NumericDispatch {

    // MARK: Scalar-only helpers (Group-A)

    /// Evaluate a scalar-only 1-arg function. Rejects non-scalar input with `.invalidArguments`.
    private static func scalarOnly1(
        _ name: String,
        args: [NumericValue],
        fn: (Double) -> Double
    ) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 1 argument, got \(args.count)")
        }
        guard case .scalar(let x) = args[0] else {
            throw MathExprError.invalidArguments(
                "\(name) is only defined for scalars, got \(args[0].kind)")
        }
        return .scalar(fn(x))
    }

    /// Evaluate a scalar-only 2-arg function. Rejects non-scalar inputs with `.invalidArguments`.
    private static func scalarOnly2(
        _ name: String,
        args: [NumericValue],
        fn: (Double, Double) -> Double
    ) throws -> NumericValue {
        guard args.count == 2 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 2 arguments, got \(args.count)")
        }
        guard case .scalar(let a) = args[0], case .scalar(let b) = args[1] else {
            throw MathExprError.invalidArguments(
                "\(name) requires 2 scalar arguments")
        }
        return .scalar(fn(a, b))
    }

    /// Evaluate a scalar-only 3-arg function.
    private static func scalarOnly3(
        _ name: String,
        args: [NumericValue],
        fn: (Double, Double, Double) -> Double
    ) throws -> NumericValue {
        guard args.count == 3 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 3 arguments, got \(args.count)")
        }
        guard case .scalar(let a) = args[0],
              case .scalar(let b) = args[1],
              case .scalar(let c) = args[2] else {
            throw MathExprError.invalidArguments(
                "\(name) requires 3 scalar arguments")
        }
        return .scalar(fn(a, b, c))
    }

    // MARK: Complex-only helpers (Group-A)

    /// Evaluate a complex-only 1-arg function.
    /// Purely real arguments (im == 0) are accepted and return `.complex` as the complex path does.
    private static func complexOnly1(
        _ name: String,
        args: [NumericValue],
        fn: (Complex) -> Complex
    ) throws -> NumericValue {
        guard args.count == 1 else {
            throw MathExprError.invalidArguments(
                "\(name) requires exactly 1 argument, got \(args.count)")
        }
        switch args[0] {
        case .complex(let z):
            return .complex(fn(z))
        case .scalar(let x):
            // Promote scalar to complex (matches legacy MathExpr.evaluateComplex behaviour).
            return .complex(fn(Complex(x)))
        default:
            throw MathExprError.invalidArguments(
                "\(name) is only defined for complex (or real) scalars, got \(args[0].kind)")
        }
    }

    // MARK: Registry table

    /// The canonical function registry.
    ///
    /// Keys are exact function names (case-sensitive). Aliases (e.g. "arcsin", "ln")
    /// have their own entries pointing to the same handler.
    ///
    /// Computed once at first access via a `static let`.
    static let functionRegistry: [String: FunctionDescriptor] = buildRegistry()

    /// Build the registry by merging the four logical sub-sections.
    private static func buildRegistry() -> [String: FunctionDescriptor] {
        var r: [String: FunctionDescriptor] = [:]
        r.merge(registryTrigAndScalar()) { _, new in new }
        r.merge(registryExpLogSqrtAbs()) { _, new in new }
        r.merge(registryMatrixFunctions()) { _, new in new }
        r.merge(registryComplexFunctions()) { _, new in new }
        return r
    }

    /// Trig, hyperbolic trig, and scalar-only utility functions (atan2, pow, rounding, etc.).
    ///
    /// Delegates to `registryTrig` (sin/cos/tan, inverse, hyperbolic) and
    /// `registryScalarMath` (atan2, pow, sign, floor/ceil/round, clamp, lerp, etc.).
    private static func registryTrigAndScalar() -> [String: FunctionDescriptor] {
        var r = registryTrig()
        r.merge(registryScalarMath()) { _, new in new }
        return r
    }

    /// Trigonometric and hyperbolic functions (all delegating to `applyTrigFunction`).
    private static func registryTrig() -> [String: FunctionDescriptor] {
        var r: [String: FunctionDescriptor] = [:]
        let trigNames: [String] = [
            "sin", "cos", "tan",
            "asin", "arcsin", "acos", "arccos", "atan", "arctan",
            "sinh", "cosh", "tanh",
            "asinh", "arcsinh", "acosh", "arccosh", "atanh", "arctanh",
        ]
        for name in trigNames {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
                try applyTrigFunction(n, args: args)
            }
        }
        return r
    }

    /// Scalar-only utility math: atan2, pow, hypot, sign, rounding, clamp, lerp, angle, min/max.
    private static func registryScalarMath() -> [String: FunctionDescriptor] {
        var r: [String: FunctionDescriptor] = [:]
        r["atan2"] = FunctionDescriptor(arityMin: 2, arityMax: 2) { n, args in
            try scalarOnly2(n, args: args) { Foundation.atan2($0, $1) }
        }
        r["cbrt"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try scalarOnly1(n, args: args) { Darwin.cbrt($0) }
        }
        r["pow"] = FunctionDescriptor(arityMin: 2, arityMax: 2) { n, args in
            try scalarOnly2(n, args: args) { Foundation.pow($0, $1) }
        }
        r["hypot"] = FunctionDescriptor(arityMin: 2, arityMax: 2) { n, args in
            try scalarOnly2(n, args: args) { Foundation.hypot($0, $1) }
        }
        for name in ["sign", "sgn"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { _, args in
                try scalarOnly1("sign", args: args) { x in x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0) }
            }
        }
        r["floor"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try scalarOnly1(n, args: args) { Foundation.floor($0) }
        }
        r["ceil"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try scalarOnly1(n, args: args) { Foundation.ceil($0) }
        }
        r["round"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try scalarOnly1(n, args: args) { Foundation.round($0) }
        }
        r["trunc"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try scalarOnly1(n, args: args) { Foundation.trunc($0) }
        }
        r["clamp"] = FunctionDescriptor(arityMin: 3, arityMax: 3) { n, args in
            try scalarOnly3(n, args: args) { x, lo, hi in Swift.min(Swift.max(x, lo), hi) }
        }
        r["lerp"] = FunctionDescriptor(arityMin: 3, arityMax: 3) { n, args in
            try scalarOnly3(n, args: args) { a, b, t in a + (b - a) * t }
        }
        r["rad"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try scalarOnly1(n, args: args) { $0 * .pi / 180.0 }
        }
        r["deg"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try scalarOnly1(n, args: args) { $0 * 180.0 / .pi }
        }
        for name in ["min", "max"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 2) { n, args in
                try applyMinMax(name: n, args: args)
            }
        }
        return r
    }

    /// exp / log / ln / log10 / log2 / lg / sqrt / abs / transpose (scalar+complex+matrix).
    private static func registryExpLogSqrtAbs() -> [String: FunctionDescriptor] {
        var r: [String: FunctionDescriptor] = [:]
        r["exp"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try applyExpLogSqrt(n, args: args)
        }
        for name in ["log", "ln"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { _, args in
                try applyExpLogSqrt("log", args: args)
            }
        }
        r["log10"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try scalarOnly1(n, args: args) { Foundation.log10($0) }
        }
        for name in ["log2", "lg"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { _, args in
                try scalarOnly1("log2", args: args) { Foundation.log2($0) }
            }
        }
        r["sqrt"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try applyExpLogSqrt(n, args: args)
        }
        r["abs"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try applyAbsInvDetTrace(n, args: args)
        }
        r["transpose"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { _, args in
            try applyTransposeFunction(args: args)
        }
        return r
    }

    /// Matrix Group-B functions (inv/det/trace/cdet/cinv) and multi-arg matrix functions.
    private static func registryMatrixFunctions() -> [String: FunctionDescriptor] {
        var r: [String: FunctionDescriptor] = [:]
        for name in ["inv", "det", "trace"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
                try applyAbsInvDetTrace(n, args: args)
            }
        }
        for name in ["cdet", "cinv"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
                try applyCdetCinv(n, args: args)
            }
        }
        for name in ["dotProduct", "hadamard", "elementDiv"] {
            r[name] = FunctionDescriptor(arityMin: 2, arityMax: 2) { n, args in
                try applyMultiArgFunction(n, args: args)
            }
        }
        r["crossProduct"] = FunctionDescriptor(arityMin: 2, arityMax: 2) { n, _ in
            throw MathExprError.unsupportedNode("\(n) not yet implemented (deferred §14)")
        }
        return r
    }

    /// Complex-only functions (conj / real / re / imag / im / arg / phase).
    private static func registryComplexFunctions() -> [String: FunctionDescriptor] {
        var r: [String: FunctionDescriptor] = [:]
        r["conj"] = FunctionDescriptor(arityMin: 1, arityMax: 1) { n, args in
            try complexOnly1(n, args: args) { $0.conj }
        }
        for name in ["real", "re"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { _, args in
                try complexOnly1("real", args: args) { Complex($0.re) }
            }
        }
        for name in ["imag", "im"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { _, args in
                try complexOnly1("imag", args: args) { Complex($0.im) }
            }
        }
        for name in ["arg", "phase"] {
            r[name] = FunctionDescriptor(arityMin: 1, arityMax: 1) { _, args in
                try complexOnly1("arg", args: args) { Complex($0.arg) }
            }
        }
        return r
    }
}
