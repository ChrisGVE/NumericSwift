//
//  UnifiedEvaluator.swift
//  NumericSwift
//
//  The unified numeric evaluation front door for the NumericSwift pipeline.
//
//  This is Phase 3 of the unified-numeric-pipeline tag. It is the **single**
//  recursive evaluator that traverses a `MathLexExpression` AST and produces
//  a `NumericValue`. It replaces the three legacy evaluators *internally* —
//  the existing public scalar/complex APIs (`MathExpr.evaluate`,
//  `MathExpr.evaluateComplex`) are not touched here; they are refactored to
//  delegate here in Phase 4.
//
//  ## Variable binding model
//
//  Variables are resolved from a `[String: NumericValue]` dictionary called
//  `values`. A missing key throws `MathExprError.undefinedVariable`. On
//  default builds (no mathlex Rust crate), matrix *literal* AST nodes
//  (`.vector`/`.matrix`) are never emitted by the parser — matrices flow in
//  through the `values` dict instead. The evaluator handles both paths:
//  - AST `.vector`/`.matrix` nodes (mathlex build): evaluated into
//    `NumericValue.matrix` via `UnifiedEvaluatorMatrix.swift`.
//  - Matrix `NumericValue` entries in `values` (default build): returned
//    directly by the variable-resolution arm.
//
//  ## AST node coverage
//
//  Handles: integer, float, variable, constant, rational, complex (AST
//  constructor), quaternion (partially — see note), binary, unary, function,
//  vector, matrix, dotProduct, determinant, matrixInverse, trace,
//  conjugateTranspose, rank.
//
//  Unsupported (deliberately throws `.unsupportedNode`): derivative,
//  partialDerivative, integral, multipleIntegral, closedIntegral, limit,
//  sum, product, equation, inequality, forAll, exists, logical, markedVector,
//  crossProduct, outerProduct, gradient, divergence, curl, laplacian, nabla,
//  numberSetExpr, setOperation, setRelationExpr, setBuilder, emptySet,
//  powerSet, tensor, kroneckerDelta, leviCivita, functionSignature,
//  composition, differential, wedgeProduct, relation.
//
//  ## Error model
//
//  Preserves the full `MathExprError` surface:
//    parseError         — not emitted here (parser duty)
//    undefinedVariable  — emitted on missing `values` key
//    unknownFunction    — emitted by `NumericDispatch.applyFunction`
//    divisionByZero     — emitted by `NumericDispatch.applyBinary`
//    invalidArguments   — emitted by dispatcher on arity/kind mismatch
//    unsupportedNode    — emitted for out-of-scope AST nodes
//    nonFiniteFloat     — emitted for `.float(nil)` (NaN-sentinel)
//    shapeMismatch      — emitted by Group-A dispatcher pre-checks
//
//  `LinAlgError` from Group-B functions propagates unchanged.
//
//  ## Soft-cap enforcement (§4.8 / CONS-07)
//
//  Matrix literal construction (`.vector`/`.matrix` nodes) pre-validates the
//  result shape via `LinAlg.checkSoftCap(rows:cols:)` before allocating.
//
//  ## Thread safety
//
//  `MathExpr.evaluateUnified` is a static method with no stored state; it is
//  re-entrant. `NumericDispatch` is also stateless (pure enum namespace).
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Unified evaluator public entry point

extension MathExpr {

    // MARK: Public API

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
    /// ## Operator semantics
    ///
    /// - `*` between two matrices is **matrix multiplication** (matmul), not
    ///   element-wise. This matches NumericSwift's shipped `LinAlg` operator
    ///   and scipy.linalg conventions. Element-wise multiplication is the
    ///   `hadamard` named function.
    /// - `dot(u, v)` for two column vectors returns a `.scalar` after the
    ///   1×1 → scalar coercion (§4.3a).
    /// - `*` for scalar/complex follows the existing scalar/complex rules.
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
    /// `LinAlgError.notSquare` propagates unchanged from Group-B functions
    /// (`det`, `inv`, `trace`, `expm`, `logm`, `sqrtm`, `cdet`, `cinv`).
    ///
    /// - Parameters:
    ///   - ast: The decoded `MathLexExpression` AST to evaluate.
    ///   - values: Variable bindings — any `NumericValue` kind is accepted.
    /// - Returns: The evaluated result as a `NumericValue`.
    /// - Throws: `MathExprError` or `LinAlgError` as described above.
    public static func evaluateUnified(
        _ ast: MathLexExpression,
        values: [String: NumericValue] = [:]
    ) throws -> NumericValue {
        try UnifiedEvaluatorCore.eval(ast, values: values)
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

// MARK: - Core recursive evaluator (internal)

/// Internal namespace for the unified evaluation pass.
///
/// `UnifiedEvaluatorCore` is a pure-function namespace (enum with no cases)
/// containing the recursive `eval` function and all leaf-node handlers. The
/// matrix-literal and linear-algebra node handlers live in
/// `UnifiedEvaluatorMatrix.swift` to stay within the per-file line budget.
enum UnifiedEvaluatorCore {

    // MARK: Main recursive entry

    /// Recursively evaluate a `MathLexExpression` node.
    ///
    /// The switch is exhaustive over the full `MathLexExpression` enum so the
    /// compiler enforces coverage. New AST cases added to `MathLexAST.swift`
    /// produce a compile error here rather than silently falling through.
    ///
    /// - Parameters:
    ///   - ast: The node to evaluate.
    ///   - values: Variable bindings shared across the entire traversal.
    /// - Returns: A `NumericValue` for this subtree.
    /// - Throws: `MathExprError` or `LinAlgError`.
    static func eval(_ ast: MathLexExpression, values: [String: NumericValue]) throws -> NumericValue {
        switch ast {

        // MARK: Literal leaf nodes (subtasks 18.2, 18.13)
        case .integer(let n):
            return .scalar(Double(n))

        case .float(let v):
            guard let v else { throw MathExprError.nonFiniteFloat }
            return .scalar(v)

        // MARK: Variable resolution (subtask 18.3)
        case .variable(let name):
            guard let value = values[name] else {
                throw MathExprError.undefinedVariable(name)
            }
            return value

        // MARK: Named mathematical constants (subtask 18.2)
        case .constant(let c):
            return try evalConstant(c)

        // MARK: Rational number constructor (subtask 18.4)
        case .rational(let num, let den):
            let n = try eval(num, values: values)
            let d = try eval(den, values: values)
            return try NumericDispatch.applyBinary(.div, lhs: n, rhs: d)

        // MARK: Complex number constructor (subtask 18.4)
        case .complex(let re, let im):
            let r = try eval(re, values: values)
            let i = try eval(im, values: values)
            return try buildComplexFromParts(re: r, im: i)

        // MARK: Binary operators (subtask 18.5)
        case .binary(let op, let left, let right):
            let lhs = try eval(left, values: values)
            let rhs = try eval(right, values: values)
            return try NumericDispatch.applyBinary(op, lhs: lhs, rhs: rhs)

        // MARK: Unary operators (subtask 18.6)
        case .unary(let op, let operand):
            let val = try eval(operand, values: values)
            return try NumericDispatch.applyUnary(op, operand: val)

        // MARK: Function calls (subtask 18.7)
        case .function(let name, let args):
            let argVals = try args.map { try eval($0, values: values) }
            return try NumericDispatch.applyFunction(name, args: argVals)

        // MARK: Matrix/vector literal nodes (subtasks 18.8, 18.14)
        case .vector(let elements):
            return try UnifiedEvaluatorMatrix.evalVector(elements, values: values)

        case .matrix(let rows):
            return try UnifiedEvaluatorMatrix.evalMatrix(rows, values: values)

        // MARK: Linear-algebra AST nodes (subtask 18.9)
        case .dotProduct(let left, let right):
            let l = try eval(left, values: values)
            let r = try eval(right, values: values)
            return try NumericDispatch.applyFunction("dotProduct", args: [l, r])

        case .determinant(let matrix):
            let m = try eval(matrix, values: values)
            return try NumericDispatch.applyFunction("det", args: [m])

        case .matrixInverse(let matrix):
            let m = try eval(matrix, values: values)
            return try NumericDispatch.applyFunction("inv", args: [m])

        case .trace(let matrix):
            let m = try eval(matrix, values: values)
            return try NumericDispatch.applyFunction("trace", args: [m])

        case .conjugateTranspose(let matrix):
            // Hermitian adjoint: transpose then conjugate each element.
            // For real matrices this equals the regular transpose.
            let m = try eval(matrix, values: values)
            return try UnifiedEvaluatorMatrix.evalConjugateTranspose(m)

        case .rank(let matrix):
            let m = try eval(matrix, values: values)
            return try UnifiedEvaluatorMatrix.evalRank(m)

        // MARK: Quaternion (subtask 18.4, partial)
        case .quaternion:
            throw MathExprError.unsupportedNode(
                "quaternion arithmetic is not yet supported (deferred to v-next §14)")

        // MARK: Unsupported nodes — all remaining cases (subtask 18.10)
        case .derivative, .partialDerivative, .integral, .multipleIntegral,
             .closedIntegral, .limit, .sum, .product:
            throw MathExprError.unsupportedNode(nodeLabel(ast))

        case .equation, .inequality, .forAll, .exists, .logical:
            throw MathExprError.unsupportedNode(nodeLabel(ast))

        case .markedVector, .crossProduct, .outerProduct, .gradient, .divergence,
             .curl, .laplacian, .nabla:
            throw MathExprError.unsupportedNode(nodeLabel(ast))

        case .numberSetExpr, .setOperation, .setRelationExpr, .setBuilder,
             .emptySet, .powerSet:
            throw MathExprError.unsupportedNode(nodeLabel(ast))

        case .tensor, .kroneckerDelta, .leviCivita:
            throw MathExprError.unsupportedNode(nodeLabel(ast))

        case .functionSignature, .composition, .differential, .wedgeProduct, .relation:
            throw MathExprError.unsupportedNode(nodeLabel(ast))
        }
    }

    // MARK: Constant resolution (subtask 18.2)

    /// Resolve a `MathLexConstant` to a `NumericValue`.
    ///
    /// Named constants that are inherently complex (`.i`, `.j`, `.k`) resolve
    /// to `.complex` values. The quaternion constants `.j` and `.k` are
    /// partially supported — they return complex imaginary unit representations
    /// — because true quaternion arithmetic is out of scope (§14 deferral).
    private static func evalConstant(_ c: MathLexConstant) throws -> NumericValue {
        switch c {
        case .pi:         return .scalar(Double.pi)
        case .e:          return .scalar(M_E)
        case .i:          return .complex(Complex(re: 0, im: 1))
        case .j:          return .complex(Complex(re: 0, im: 1))  // quaternion j treated as i for now
        case .k:
            throw MathExprError.unsupportedNode(
                "quaternion constant k requires quaternion arithmetic (deferred §14)")
        case .infinity:   return .scalar(Double.infinity)
        case .negInfinity: return .scalar(-Double.infinity)
        case .nan:        return .scalar(Double.nan)
        }
    }

    // MARK: Complex constructor helper (subtask 18.4)

    /// Build a `NumericValue` from the evaluated real and imaginary parts of a
    /// `.complex(re:im:)` AST node.
    ///
    /// Both parts are evaluated independently and may each be scalar or complex.
    /// The resulting value is always a `.complex` NumericValue with the
    /// real part taken from `re` and the imaginary part taken from `im`.
    ///
    /// If either part is itself complex (e.g. `(a+bi) + (c+di)*i`), the
    /// standard complex arithmetic applies: only the `.re` component of each
    /// part contributes, matching legacy `evaluateComplex` behaviour.
    private static func buildComplexFromParts(
        re: NumericValue,
        im: NumericValue
    ) throws -> NumericValue {
        let rVal: Double
        let iVal: Double
        switch re {
        case .scalar(let x): rVal = x
        case .complex(let z): rVal = z.re
        default:
            throw MathExprError.invalidArguments(
                "complex() constructor: real part must be scalar, got \(re.typeAndShapeDescription)")
        }
        switch im {
        case .scalar(let x): iVal = x
        case .complex(let z): iVal = z.re
        default:
            throw MathExprError.invalidArguments(
                "complex() constructor: imaginary part must be scalar, got \(im.typeAndShapeDescription)")
        }
        return .complex(Complex(re: rVal, im: iVal))
    }

    // MARK: AST node label (subtask 18.10)

    /// Return a human-readable label for unsupported AST nodes.
    ///
    /// Used as the message payload of `MathExprError.unsupportedNode`.
    static func nodeLabel(_ expr: MathLexExpression) -> String {
        switch expr {
        case .integer: return "Integer"
        case .float: return "Float"
        case .variable: return "Variable"
        case .constant: return "Constant"
        case .rational: return "Rational"
        case .complex: return "Complex"
        case .quaternion: return "Quaternion"
        case .binary: return "Binary"
        case .unary: return "Unary"
        case .function: return "Function"
        case .derivative: return "Derivative"
        case .partialDerivative: return "PartialDerivative"
        case .integral: return "Integral"
        case .multipleIntegral: return "MultipleIntegral"
        case .closedIntegral: return "ClosedIntegral"
        case .limit: return "Limit"
        case .sum: return "Sum"
        case .product: return "Product"
        case .vector: return "Vector"
        case .matrix: return "Matrix"
        case .equation: return "Equation"
        case .inequality: return "Inequality"
        case .forAll: return "ForAll"
        case .exists: return "Exists"
        case .logical: return "Logical"
        case .markedVector: return "MarkedVector"
        case .dotProduct: return "DotProduct"
        case .crossProduct: return "CrossProduct"
        case .outerProduct: return "OuterProduct"
        case .gradient: return "Gradient"
        case .divergence: return "Divergence"
        case .curl: return "Curl"
        case .laplacian: return "Laplacian"
        case .nabla: return "Nabla"
        case .determinant: return "Determinant"
        case .trace: return "Trace"
        case .rank: return "Rank"
        case .conjugateTranspose: return "ConjugateTranspose"
        case .matrixInverse: return "MatrixInverse"
        case .numberSetExpr: return "NumberSetExpr"
        case .setOperation: return "SetOperation"
        case .setRelationExpr: return "SetRelationExpr"
        case .setBuilder: return "SetBuilder"
        case .emptySet: return "EmptySet"
        case .powerSet: return "PowerSet"
        case .tensor: return "Tensor"
        case .kroneckerDelta: return "KroneckerDelta"
        case .leviCivita: return "LeviCivita"
        case .functionSignature: return "FunctionSignature"
        case .composition: return "Composition"
        case .differential: return "Differential"
        case .wedgeProduct: return "WedgeProduct"
        case .relation: return "Relation"
        }
    }
}
