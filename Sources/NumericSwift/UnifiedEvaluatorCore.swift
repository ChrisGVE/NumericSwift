//
//  UnifiedEvaluatorCore.swift
//  NumericSwift
//
//  Internal recursive evaluation engine for the unified numeric pipeline.
//
//  `UnifiedEvaluatorCore` is a pure-function namespace (enum with no cases)
//  containing the recursive `eval` function and all leaf-node handlers. The
//  matrix-literal and linear-algebra node handlers live in
//  `UnifiedEvaluatorMatrix.swift` to stay within the per-file line budget.
//
//  The public front door (`MathExpr.evaluateUnified`) and the result
//  extraction helpers (`extractDouble`, `extractComplex`) live in
//  `UnifiedEvaluator.swift`.
//
//  ## AST node coverage
//
//  Handles: integer, float, variable, constant, rational, complex (AST
//  constructor), quaternion (partially), binary, unary, function,
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
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Core recursive evaluator (internal)

/// Internal namespace for the unified evaluation pass.
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
    ///   - complexMode: When `true`, the negative-real domain functions that the
    ///     real path sends to NaN are promoted to the complex principal value:
    ///     `sqrt`/`log`/`ln` of a negative-real scalar, and the `^` operator with
    ///     a negative-real base and a non-integer exponent. This restores the
    ///     legacy `evaluateComplex` complex-context behaviour that the unified
    ///     front door otherwise collapses (GitHub issue #1). The flag propagates
    ///     through every recursive subexpression so nested promotions work
    ///     (`sqrt(-1) + sqrt(-2)`). The real `evaluate` wrapper leaves it `false`,
    ///     preserving the frozen real NaN contract. The `pow(x, y)` *function*
    ///     intentionally does NOT promote (legacy routed it through the real
    ///     fallback), only the `^` *operator* does.
    /// - Returns: A `NumericValue` for this subtree.
    /// - Throws: `MathExprError` or `LinAlgError`.
    // swiftlint:disable:next function_body_length cyclomatic_complexity
    static func eval(
        _ ast: MathLexExpression,
        values: [String: NumericValue],
        complexMode: Bool = false
    ) throws -> NumericValue {
        switch ast {

        // MARK: Literal leaf nodes
        case .integer(let n):
            return .scalar(Double(n))

        case .float(let v):
            guard let v else { throw MathExprError.nonFiniteFloat }
            return .scalar(v)

        // MARK: Variable resolution
        case .variable(let name):
            guard let value = values[name] else {
                throw MathExprError.undefinedVariable(name)
            }
            return value

        // MARK: Named mathematical constants
        case .constant(let c):
            return try evalConstant(c)

        // MARK: Rational number constructor
        case .rational(let num, let den):
            let n = try eval(num, values: values, complexMode: complexMode)
            let d = try eval(den, values: values, complexMode: complexMode)
            return try NumericDispatch.applyBinary(.div, lhs: n, rhs: d, complexMode: complexMode)

        // MARK: Complex number constructor
        case .complex(let re, let im):
            let r = try eval(re, values: values, complexMode: complexMode)
            let i = try eval(im, values: values, complexMode: complexMode)
            return try buildComplexFromParts(re: r, im: i)

        // MARK: Binary operators
        case .binary(let op, let left, let right):
            let lhs = try eval(left, values: values, complexMode: complexMode)
            let rhs = try eval(right, values: values, complexMode: complexMode)
            return try NumericDispatch.applyBinary(op, lhs: lhs, rhs: rhs, complexMode: complexMode)

        // MARK: Unary operators
        case .unary(let op, let operand):
            let val = try eval(operand, values: values, complexMode: complexMode)
            return try NumericDispatch.applyUnary(op, operand: val)

        // MARK: Function calls
        case .function(let name, let args):
            let argVals = try args.map { try eval($0, values: values, complexMode: complexMode) }
            return try NumericDispatch.applyFunction(name, args: argVals, complexMode: complexMode)

        // MARK: Matrix/vector literal nodes
        case .vector(let elements):
            return try UnifiedEvaluatorMatrix.evalVector(elements, values: values)

        case .matrix(let rows):
            return try UnifiedEvaluatorMatrix.evalMatrix(rows, values: values)

        // MARK: Linear-algebra AST nodes
        case .dotProduct(let left, let right):
            let l = try eval(left, values: values, complexMode: complexMode)
            let r = try eval(right, values: values, complexMode: complexMode)
            return try NumericDispatch.applyFunction("dotProduct", args: [l, r], complexMode: complexMode)

        case .determinant(let matrix):
            let m = try eval(matrix, values: values, complexMode: complexMode)
            return try NumericDispatch.applyFunction("det", args: [m], complexMode: complexMode)

        case .matrixInverse(let matrix):
            let m = try eval(matrix, values: values, complexMode: complexMode)
            return try NumericDispatch.applyFunction("inv", args: [m], complexMode: complexMode)

        case .trace(let matrix):
            let m = try eval(matrix, values: values, complexMode: complexMode)
            return try NumericDispatch.applyFunction("trace", args: [m], complexMode: complexMode)

        case .conjugateTranspose(let matrix):
            // Hermitian adjoint: transpose then conjugate each element.
            // For real matrices this equals the regular transpose.
            let m = try eval(matrix, values: values, complexMode: complexMode)
            return try UnifiedEvaluatorMatrix.evalConjugateTranspose(m)

        case .rank(let matrix):
            let m = try eval(matrix, values: values, complexMode: complexMode)
            return try UnifiedEvaluatorMatrix.evalRank(m)

        // MARK: Quaternion (partial)
        case .quaternion:
            throw MathExprError.unsupportedNode(
                "quaternion arithmetic is not yet supported (deferred to v-next §14)")

        // MARK: Unsupported nodes — all remaining cases
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

    // MARK: Constant resolution

    /// Resolve a `MathLexConstant` to a `NumericValue`.
    ///
    /// Named constants that are inherently complex (`.i`, `.j`) resolve to
    /// `.complex` values. The quaternion constant `.k` is unsupported (§14).
    private static func evalConstant(_ c: MathLexConstant) throws -> NumericValue {
        switch c {
        case .pi:         return .scalar(Double.pi)
        case .e:          return .scalar(M_E)
        case .i:          return .complex(Complex(re: 0, im: 1))
        case .j:          return .complex(Complex(re: 0, im: 1))  // quaternion j treated as i for now
        case .k:
            throw MathExprError.unsupportedNode(
                "quaternion constant k requires quaternion arithmetic (deferred §14)")
        case .infinity:    return .scalar(Double.infinity)
        case .negInfinity: return .scalar(-Double.infinity)
        case .nan:         return .scalar(Double.nan)
        }
    }

    // MARK: Complex constructor helper

    /// Build a `NumericValue` from the evaluated real and imaginary parts of a
    /// `.complex(re:im:)` AST node.
    ///
    /// Both parts are evaluated independently and may each be scalar or complex.
    /// Only the `.re` component of each part contributes, matching legacy
    /// `evaluateComplex` behaviour.
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

    // MARK: AST node label

    /// Return a human-readable label for unsupported AST nodes.
    ///
    /// Used as the message payload of `MathExprError.unsupportedNode`.
    // swiftlint:disable:next cyclomatic_complexity function_body_length
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
