//
//  MathExprAST.swift
//  NumericSwift
//
//  AST utility functions for MathExpr: variable discovery, substitution,
//  AST-to-string serialization, and the node-label helper used by evaluators.
//
//  These are pure tree-walk operations — they do not invoke any evaluation
//  path. The `nodeLabel` function is `internal` so the test target and
//  other evaluator files can reference it.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

// MARK: - Variable discovery

extension MathExpr {

    /// Find all variable names in a parsed expression string.
    ///
    /// - Parameter expression: The expression string to analyze
    /// - Returns: Set of variable names
    /// - Throws: `MathExprError.parseError` if parsing fails
    public static func findVariables(in expression: String) throws -> Set<String> {
        #if NUMERICSWIFT_MATHLEX
          let mlExpr: MathExpression
          do {
              mlExpr = try MathExpression.parse(expression)
          } catch {
              throw MathExprError.parseError(error.localizedDescription)
          }
          return mlExpr.variables
        #else
          return try MathExprFallbackParser.findVariables(in: expression)
        #endif
    }

    /// Find all variable names in an AST by walking the tree.
    ///
    /// - Parameter ast: The AST to analyze
    /// - Returns: Set of variable names
    public static func findVariables(in ast: MathLexExpression) -> Set<String> {
        var result: Set<String> = []
        collectVariables(ast, into: &result)
        return result
    }

    private static func collectVariables(
        _ ast: MathLexExpression, into result: inout Set<String>
    ) {
        switch ast {
        case .variable(let name):
            result.insert(name)
        case .binary(_, let left, let right):
            collectVariables(left, into: &result)
            collectVariables(right, into: &result)
        case .unary(_, let operand):
            collectVariables(operand, into: &result)
        case .function(_, let args):
            for arg in args { collectVariables(arg, into: &result) }
        case .rational(let num, let den):
            collectVariables(num, into: &result)
            collectVariables(den, into: &result)
        case .complex(let re, let im):
            collectVariables(re, into: &result)
            collectVariables(im, into: &result)
        default:
            break
        }
    }
}

// MARK: - toString

extension MathExpr {

    /// Convert an expression string to its normalized plain-text representation.
    ///
    /// - Parameter expression: The expression string
    /// - Returns: Normalized plain-text representation
    /// - Throws: `MathExprError.parseError` if parsing fails
    public static func toString(_ expression: String) throws -> String {
        #if NUMERICSWIFT_MATHLEX
          let mlExpr: MathExpression
          do {
              mlExpr = try MathExpression.parse(expression)
          } catch {
              throw MathExprError.parseError(error.localizedDescription)
          }
          return mlExpr.description
        #else
          // Round-trip via AST: parse then reconstruct a string from the AST.
          let ast = try MathExprFallbackParser.parseExpression(expression)
          return astToString(ast)
        #endif
    }
}

// MARK: - AST → String (fallback mode only)

extension MathExpr {

    #if !NUMERICSWIFT_MATHLEX
      /// Reconstruct a plain-text expression string from a MathLexExpression AST.
      ///
      /// Covers the arithmetic/function/variable/constant subset produced by the
      /// fallback parser. Complex nodes and calculus nodes are represented as
      /// `<NodeType>` placeholders.
      static func astToString(_ expr: MathLexExpression) -> String {
          switch expr {
          case .integer(let n):
              return String(n)
          case .float(let v):
              guard let v else { return "nan" }
              if v.isNaN { return "nan" }
              if v.isInfinite { return v > 0 ? "inf" : "-inf" }
              if v == Foundation.floor(v) && !v.isInfinite { return String(Int64(v)) }
              return String(v)
          case .variable(let name):
              return name
          case .constant(let c):
              return astConstantToString(c)
          case .binary(let op, let left, let right):
              return astBinaryToString(op, left, right)
          case .unary(let op, let operand):
              return astUnaryToString(op, operand)
          case .function(let name, let args):
              let argList = args.map { astToString($0) }.joined(separator: ", ")
              return "\(name)(\(argList))"
          case .rational(let num, let den):
              return "(\(astToString(num)) / \(astToString(den)))"
          case .complex(let re, let im):
              return "(\(astToString(re)) + \(astToString(im))*i)"
          default:
              return "<\(nodeLabel(expr))>"
          }
      }

      private static func astConstantToString(_ c: MathLexConstant) -> String {
          switch c {
          case .pi: return "pi"
          case .e: return "e"
          case .i: return "i"
          case .j: return "j"
          case .k: return "k"
          case .infinity: return "inf"
          case .negInfinity: return "-inf"
          case .nan: return "nan"
          }
      }

      private static func astBinaryToString(
          _ op: BinaryOp, _ left: MathLexExpression, _ right: MathLexExpression
      ) -> String {
          let opStr: String
          switch op {
          case .add: opStr = "+"
          case .sub: opStr = "-"
          case .mul: opStr = "*"
          case .div: opStr = "/"
          case .pow: opStr = "^"
          case .mod: opStr = "%"
          case .plusMinus: opStr = "±"
          case .minusPlus: opStr = "∓"
          }
          return "(\(astToString(left)) \(opStr) \(astToString(right)))"
      }

      private static func astUnaryToString(_ op: UnaryOp, _ operand: MathLexExpression) -> String {
          switch op {
          case .neg: return "-\(astToString(operand))"
          case .pos: return astToString(operand)
          case .factorial: return "\(astToString(operand))!"
          case .transpose: return "\(astToString(operand))ᵀ"
          }
      }
    #endif
}

// MARK: - Substitution

extension MathExpr {

    /// Substitute variables in an AST with replacement expressions.
    ///
    /// - Parameters:
    ///   - ast: The AST to transform
    ///   - substitutions: Dictionary of variable name to replacement AST
    /// - Returns: New AST with substitutions applied
    public static func substitute(
        _ ast: MathLexExpression, with substitutions: [String: MathLexExpression]
    ) -> MathLexExpression {
        switch ast {
        case .variable(let name):
            return substitutions[name] ?? ast
        case .binary(let op, let left, let right):
            return .binary(
                op: op,
                left: substitute(left, with: substitutions),
                right: substitute(right, with: substitutions))
        case .unary(let op, let operand):
            return .unary(op: op, operand: substitute(operand, with: substitutions))
        case .function(let name, let args):
            return .function(name: name, args: args.map { substitute($0, with: substitutions) })
        case .rational(let num, let den):
            return .rational(
                numerator: substitute(num, with: substitutions),
                denominator: substitute(den, with: substitutions))
        case .complex(let re, let im):
            return .complex(
                real: substitute(re, with: substitutions),
                imaginary: substitute(im, with: substitutions))
        default:
            return ast
        }
    }
}

// MARK: - Node label helper

extension MathExpr {

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
