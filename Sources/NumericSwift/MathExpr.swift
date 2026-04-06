//
//  MathExpr.swift
//  NumericSwift
//
//  Mathematical expression parser and evaluator.
//
//  Parsing is delegated to the mathlex crate via its Swift FFI. This module
//  provides the evaluation layer that walks the decoded AST and computes
//  numeric results (Double or Complex).
//

import Foundation
import MathLex

// MARK: - Errors

/// Errors for mathematical expression parsing and evaluation.
public enum MathExprError: Error, Equatable {
  case parseError(String)
  case undefinedVariable(String)
  case unknownFunction(String)
  case divisionByZero
  case invalidArguments(String)
  case unsupportedNode(String)
  case nonFiniteFloat

  public var description: String {
    switch self {
    case .parseError(let msg):
      return "parse error: \(msg)"
    case .undefinedVariable(let name):
      return "undefined variable '\(name)'"
    case .unknownFunction(let name):
      return "unknown function '\(name)'"
    case .divisionByZero:
      return "division by zero"
    case .invalidArguments(let msg):
      return "invalid arguments: \(msg)"
    case .unsupportedNode(let name):
      return "unsupported AST node: \(name)"
    case .nonFiniteFloat:
      return "non-finite float in AST"
    }
  }
}

// MARK: - Math Expression Parser and Evaluator

/// Mathematical expression parser and evaluator.
///
/// Parsing is handled by the mathlex crate. Evaluation walks the decoded
/// `MathLexExpression` AST to compute `Double` or `Complex` results.
public struct MathExpr {

  // MARK: - Parsing

  /// Parse a plain-text mathematical expression into an AST.
  ///
  /// Uses the mathlex parser via JSON serialization to produce a decoded
  /// `MathLexExpression` tree.
  ///
  /// - Parameter expression: The expression string to parse
  /// - Returns: The decoded AST
  /// - Throws: `MathExprError.parseError` if the input is invalid
  public static func parse(_ expression: String) throws -> MathLexExpression {
    let mlExpr: MathExpression
    do {
      mlExpr = try MathExpression.parse(expression)
    } catch {
      throw MathExprError.parseError(error.localizedDescription)
    }
    let json: String
    do {
      json = try mlExpr.toJSON()
    } catch {
      throw MathExprError.parseError("JSON serialization failed: \(error.localizedDescription)")
    }
    guard let data = json.data(using: .utf8) else {
      throw MathExprError.parseError("JSON encoding failed")
    }
    do {
      return try JSONDecoder().decode(MathLexExpression.self, from: data)
    } catch {
      throw MathExprError.parseError("AST decode failed: \(error.localizedDescription)")
    }
  }

  /// Parse a LaTeX mathematical expression into an AST.
  ///
  /// - Parameter latex: The LaTeX expression string to parse
  /// - Returns: The decoded AST
  /// - Throws: `MathExprError.parseError` if the input is invalid
  public static func parseLatex(_ latex: String) throws -> MathLexExpression {
    let mlExpr: MathExpression
    do {
      mlExpr = try MathExpression.parseLatex(latex)
    } catch {
      throw MathExprError.parseError(error.localizedDescription)
    }
    let json: String
    do {
      json = try mlExpr.toJSON()
    } catch {
      throw MathExprError.parseError("JSON serialization failed: \(error.localizedDescription)")
    }
    guard let data = json.data(using: .utf8) else {
      throw MathExprError.parseError("JSON encoding failed")
    }
    do {
      return try JSONDecoder().decode(MathLexExpression.self, from: data)
    } catch {
      throw MathExprError.parseError("AST decode failed: \(error.localizedDescription)")
    }
  }

  // MARK: - Real Evaluation

  /// Evaluate an AST with given variable bindings.
  ///
  /// - Parameters:
  ///   - ast: The AST to evaluate
  ///   - variables: Dictionary of variable name to value
  /// - Returns: The computed result
  /// - Throws: `MathExprError` if evaluation fails
  public static func evaluate(
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
      return try resolveConstant(c)
    case .rational(let num, let den):
      return try evaluate(num, variables: variables) / evaluate(den, variables: variables)
    case .binary(let op, let left, let right):
      return try evalBinary(op, left, right, variables: variables)
    case .unary(let op, let operand):
      return try evalUnary(op, operand, variables: variables)
    case .function(let name, let args):
      let vals = try args.map { try evaluate($0, variables: variables) }
      return try evalFunction(name, args: vals)
    default:
      throw MathExprError.unsupportedNode(nodeLabel(ast))
    }
  }

  /// Parse and evaluate an expression string.
  ///
  /// - Parameters:
  ///   - expression: The expression string to evaluate
  ///   - variables: Dictionary of variable name to value
  /// - Returns: The computed result
  /// - Throws: `MathExprError` if parsing or evaluation fails
  public static func eval(_ expression: String, variables: [String: Double] = [:]) throws -> Double
  {
    let ast = try parse(expression)
    return try evaluate(ast, variables: variables)
  }

  // MARK: - Constant Resolution

  private static func resolveConstant(_ c: MathLexConstant) throws -> Double {
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

  // MARK: - Binary Operators

  private static func evalBinary(
    _ op: BinaryOp,
    _ left: MathLexExpression,
    _ right: MathLexExpression,
    variables: [String: Double]
  ) throws -> Double {
    let l = try evaluate(left, variables: variables)
    let r = try evaluate(right, variables: variables)
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

  // MARK: - Unary Operators

  private static func evalUnary(
    _ op: UnaryOp,
    _ operand: MathLexExpression,
    variables: [String: Double]
  ) throws -> Double {
    let v = try evaluate(operand, variables: variables)
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

  // MARK: - Function Evaluation

  // swiftlint:disable:next cyclomatic_complexity function_body_length
  private static func evalFunction(_ name: String, args: [Double]) throws -> Double {
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
    case ("trunc", 1): return trunc(args[0])
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

  // MARK: - Complex Evaluation

  /// Evaluate an AST to a complex number.
  ///
  /// Supports the same operators and functions as the real evaluator, plus
  /// imaginary number handling via the `i` constant.
  public static func evaluateComplex(
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
      return try resolveComplexConstant(c)
    case .rational(let num, let den):
      let n = try evaluateComplex(num, variables: variables, complexVariables: complexVariables)
      let d = try evaluateComplex(den, variables: variables, complexVariables: complexVariables)
      return n / d
    case .complex(let re, let im):
      let r = try evaluateComplex(re, variables: variables, complexVariables: complexVariables)
      let i = try evaluateComplex(im, variables: variables, complexVariables: complexVariables)
      return Complex(re: r.re, im: i.re)
    case .binary(let op, let left, let right):
      let l = try evaluateComplex(left, variables: variables, complexVariables: complexVariables)
      let r = try evaluateComplex(right, variables: variables, complexVariables: complexVariables)
      return try evalComplexBinary(op, l, r)
    case .unary(let op, let operand):
      let v = try evaluateComplex(
        operand, variables: variables, complexVariables: complexVariables)
      return try evalComplexUnary(op, v)
    case .function(let name, let args):
      let vals = try args.map {
        try evaluateComplex($0, variables: variables, complexVariables: complexVariables)
      }
      return try evalComplexFunction(name, args: vals)
    default:
      throw MathExprError.unsupportedNode(nodeLabel(ast))
    }
  }

  private static func resolveComplexConstant(_ c: MathLexConstant) throws -> Complex {
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

  private static func evalComplexBinary(
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

  private static func evalComplexUnary(_ op: UnaryOp, _ v: Complex) throws -> Complex {
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

  private static func evalComplexFunction(_ name: String, args: [Complex]) throws -> Complex {
    guard args.count == 1 else {
      // Multi-arg functions: fall back to real if all args are real
      if args.allSatisfy({ $0.im == 0 }) {
        let reals = args.map(\.re)
        let result = try evalFunction(name, args: reals)
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
        let result = try evalFunction(name, args: [z.re])
        return Complex(result)
      }
      throw MathExprError.invalidArguments(
        "function '\(name)' not supported for complex arguments")
    }
  }

  // MARK: - Utilities

  /// Find all variable names in a parsed expression string.
  ///
  /// Delegates directly to mathlex's `variables` property.
  ///
  /// - Parameter expression: The expression string to analyze
  /// - Returns: Set of variable names
  /// - Throws: `MathExprError.parseError` if parsing fails
  public static func findVariables(in expression: String) throws -> Set<String> {
    let mlExpr: MathExpression
    do {
      mlExpr = try MathExpression.parse(expression)
    } catch {
      throw MathExprError.parseError(error.localizedDescription)
    }
    return mlExpr.variables
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

  private static func collectVariables(_ ast: MathLexExpression, into result: inout Set<String>) {
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

  /// Convert an expression string to its plain-text representation via mathlex.
  ///
  /// - Parameter expression: The expression string
  /// - Returns: Normalized plain-text representation
  /// - Throws: `MathExprError.parseError` if parsing fails
  public static func toString(_ expression: String) throws -> String {
    let mlExpr: MathExpression
    do {
      mlExpr = try MathExpression.parse(expression)
    } catch {
      throw MathExprError.parseError(error.localizedDescription)
    }
    return mlExpr.description
  }

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

  // MARK: - Node Label Helper

  private static func nodeLabel(_ expr: MathLexExpression) -> String {
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
