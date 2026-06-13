//
//  MathExprFallbackParser.swift
//  NumericSwift
//
//  Pure-Swift tokenizer and shunting-yard parser that produces MathLexExpression
//  nodes. Compiled only when the MathLex Rust crate is NOT included
//  (NUMERICSWIFT_MATHLEX flag absent). Grammar covered:
//    - Integer and floating-point literals (including scientific notation)
//    - Imaginary literals: 2i, 5.0i
//    - Named constants: pi, e, inf, nan, i (imaginary unit when standalone)
//    - Variables (any identifier not in constants/functions)
//    - Binary operators: + - * / % ^ with standard precedence; ^ right-associative
//    - Unary minus (emits `.unary(.neg, x)`, MathLex shape) / unary plus (no-op)
//    - Function calls: name(arg, ...) for all functions supported by evalFunction
//    - Parentheses
//
//  LaTeX parsing is NOT supported — MathExpr.parseLatex() requires the MathLex backend.
//

#if !NUMERICSWIFT_MATHLEX

import Foundation

// MARK: - Internal token type (fallback only)

private enum FBToken: Equatable {
  case integer(Int64)
  case float(Double)
  case imaginary(Double)   // coefficient × i
  case constant(MathLexConstant)
  case variable(String)
  case function(String)
  case op(String)
  case lparen
  case rparen
  case comma
}

// MARK: - Fallback parser namespace

enum MathExprFallbackParser {

  // MARK: Known identifiers

  private static let knownFunctions: Set<String> = [
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "arcsinh", "arccosh", "arctanh",
    "exp", "log", "ln", "log10", "log2", "lg",
    "sqrt", "cbrt", "pow", "hypot",
    "abs", "sign", "sgn", "floor", "ceil", "round", "trunc",
    "min", "max", "clamp", "lerp",
    "rad", "deg",
  ]

  private static let constantMap: [String: MathLexConstant] = [
    "pi": .pi,
    "e": .e,
    "inf": .infinity,
    "infinity": .infinity,
    "nan": .nan,
    "i": .i,
  ]

  // MARK: - Tokenize

  fileprivate static func tokenize(_ expression: String) throws -> [FBToken] {  // swiftlint:disable:this function_body_length
    var tokens: [FBToken] = []
    var idx = expression.startIndex

    while idx < expression.endIndex {
      let ch = expression[idx]

      if ch.isWhitespace {
        idx = expression.index(after: idx)
        continue
      }

      // Numbers
      if ch.isNumber || (ch == "." && idx < expression.endIndex) {
        let (tok, next) = try parseNumber(expression, startingAt: idx)
        tokens.append(tok)
        idx = next
        continue
      }

      // Operators and punctuation
      switch ch {
      case "+", "-", "*", "/", "^", "%":
        tokens.append(.op(String(ch)))
        idx = expression.index(after: idx)
        continue
      case "(":
        tokens.append(.lparen)
        idx = expression.index(after: idx)
        continue
      case ")":
        tokens.append(.rparen)
        idx = expression.index(after: idx)
        continue
      case ",":
        tokens.append(.comma)
        idx = expression.index(after: idx)
        continue
      default:
        break
      }

      // Identifiers
      if ch.isLetter || ch == "_" {
        let (ident, next) = parseIdentifier(expression, startingAt: idx)
        idx = next

        if let c = constantMap[ident] {
          // "i" followed directly by digits is impossible here (digits are parsed first),
          // but if we see bare "i" treat as imaginary constant
          tokens.append(.constant(c))
        } else if knownFunctions.contains(ident) {
          tokens.append(.function(ident))
        } else {
          tokens.append(.variable(ident))
        }
        continue
      }

      let offset = expression.distance(from: expression.startIndex, to: idx)
      throw MathExprError.parseError("unexpected character '\(ch)' at position \(offset)")
    }

    return tokens
  }

  // MARK: - Parse a number token

  private static func parseNumber(
    _ expr: String, startingAt start: String.Index
  ) throws -> (FBToken, String.Index) {
    var idx = start
    var raw = ""
    var hasDecimal = false
    var hasExp = false

    while idx < expr.endIndex {
      let c = expr[idx]
      if c.isNumber {
        raw.append(c)
        idx = expr.index(after: idx)
      } else if c == "." && !hasDecimal && !hasExp {
        raw.append(c)
        hasDecimal = true
        idx = expr.index(after: idx)
      } else if (c == "e" || c == "E") && !hasExp && !raw.isEmpty {
        raw.append(c)
        hasExp = true
        idx = expr.index(after: idx)
        if idx < expr.endIndex {
          let nc = expr[idx]
          if nc == "+" || nc == "-" {
            raw.append(nc)
            idx = expr.index(after: idx)
          }
        }
      } else {
        break
      }
    }

    // Check for imaginary suffix  e.g. "2i" or "3.5i"
    let hasImaginarySuffix: Bool
    if idx < expr.endIndex && expr[idx] == "i" {
      let after = expr.index(after: idx)
      hasImaginarySuffix = after >= expr.endIndex || !expr[after].isLetter
    } else {
      hasImaginarySuffix = false
    }

    if hasDecimal || hasExp {
      guard let v = Double(raw) else {
        throw MathExprError.parseError("invalid number '\(raw)'")
      }
      if hasImaginarySuffix {
        return (.imaginary(v), expr.index(after: idx))
      }
      return (.float(v), idx)
    } else {
      // Integer path — try Int64 first for exact representation
      if let n = Int64(raw) {
        if hasImaginarySuffix {
          return (.imaginary(Double(n)), expr.index(after: idx))
        }
        return (.integer(n), idx)
      }
      // Fallback to Double for very large integer literals
      guard let v = Double(raw) else {
        throw MathExprError.parseError("invalid number '\(raw)'")
      }
      if hasImaginarySuffix {
        return (.imaginary(v), expr.index(after: idx))
      }
      return (.float(v), idx)
    }
  }

  // MARK: - Parse an identifier

  private static func parseIdentifier(
    _ expr: String, startingAt start: String.Index
  ) -> (String, String.Index) {
    var idx = start
    var ident = ""
    while idx < expr.endIndex {
      let c = expr[idx]
      if c.isLetter || c.isNumber || c == "_" {
        ident.append(c)
        idx = expr.index(after: idx)
      } else {
        break
      }
    }
    return (ident, idx)
  }

  // MARK: - Shunting-yard → MathLexExpression

  private static let precedence: [String: Int] = [
    "+": 1, "-": 1,
    "*": 2, "/": 2, "%": 2,
    "u-": 2,  // unary minus: prefix, binds below `^` so `-x^2` == `-(x^2)`
    "^": 3,
  ]
  // `^` is right-associative; `u-` is a right-associative prefix operator so a
  // run of unary minuses (`--x`) nests outermost-last.
  private static let rightAssociative: Set<String> = ["^", "u-"]

  // swiftlint:disable:next cyclomatic_complexity function_body_length
  fileprivate static func parse(_ tokens: [FBToken]) throws -> MathLexExpression {
    var output: [MathLexExpression] = []
    var operators: [FBToken] = []
    var argCounts: [Int] = []
    var prev: FBToken? = nil

    for token in tokens {
      switch token {
      case .integer(let n):
        output.append(.integer(n))

      case .float(let v):
        output.append(.float(v))

      case .imaginary(let coeff):
        // coeff*i — store as complex(real: 0, imaginary: float(coeff))
        output.append(.complex(real: .integer(0), imaginary: .float(coeff)))

      case .constant(let c):
        output.append(.constant(c))

      case .variable(let name):
        output.append(.variable(name))

      case .function:
        operators.append(token)
        argCounts.append(1)

      case .comma:
        while let top = operators.last {
          if case .lparen = top { break }
          try popOp(&operators, &output)
        }
        if !argCounts.isEmpty {
          argCounts[argCounts.count - 1] += 1
        }

      case .op(let opStrRaw):
        // Determine unary minus/plus
        var isUnary = false
        if opStrRaw == "-" || opStrRaw == "+" {
          if prev == nil {
            isUnary = true
          } else if let p = prev {
            switch p {
            case .lparen, .op, .comma: isUnary = true
            default: break
            }
          }
        }

        var opStr = opStrRaw
        if isUnary {
          if opStrRaw == "+" {
            // unary "+" is a no-op
            prev = token
            continue
          }
          // unary "-" → dedicated prefix operator emitting `.unary(.neg, x)`,
          // matching the MathLex AST shape (not `.binary(.sub, 0, x)`).
          opStr = "u-"
        }

        let p1 = precedence[opStr] ?? 0
        while let top = operators.last {
          guard case .op(let topOp) = top else { break }
          let p2 = precedence[topOp] ?? 0
          let shouldPop =
            rightAssociative.contains(opStr) ? p1 < p2 : p1 <= p2
          if !shouldPop { break }
          try popOp(&operators, &output)
        }
        operators.append(.op(opStr))

      case .lparen:
        // If previous token was a variable, convert it to a function call
        if case .variable = prev {
          if let last = output.last, case .variable(let name) = last {
            output.removeLast()
            operators.append(.function(name))
            argCounts.append(1)
          }
        }
        operators.append(token)

      case .rparen:
        while let top = operators.last {
          if case .lparen = top { break }
          try popOp(&operators, &output)
        }
        guard !operators.isEmpty, case .lparen = operators.last else {
          throw MathExprError.parseError("unmatched parenthesis")
        }
        operators.removeLast()  // discard lparen

        // Check if a function is waiting
        if let top = operators.last, case .function(let name) = top {
          operators.removeLast()
          let nArgs = argCounts.isEmpty ? 1 : argCounts.removeLast()
          guard output.count >= nArgs else {
            throw MathExprError.parseError("not enough arguments for '\(name)'")
          }
          let args = Array(output[(output.count - nArgs)...])
          output.removeLast(nArgs)
          output.append(.function(name: name, args: args))
        }
      }
      prev = token
    }

    while !operators.isEmpty {
      guard case .op = operators.last else {
        throw MathExprError.parseError("unmatched parenthesis")
      }
      try popOp(&operators, &output)
    }

    guard let result = output.first, output.count == 1 else {
      if output.isEmpty {
        throw MathExprError.parseError("empty expression")
      }
      throw MathExprError.parseError("malformed expression")
    }
    return result
  }

  // MARK: - Pop operator helper

  fileprivate static func popOp(
    _ operators: inout [FBToken],
    _ output: inout [MathLexExpression]
  ) throws {
    guard let top = operators.popLast(), case .op(let opStr) = top else {
      throw MathExprError.parseError("internal parser error: expected operator")
    }

    // Unary minus is a prefix operator: pop a single operand.
    if opStr == "u-" {
      guard let operand = output.popLast() else {
        throw MathExprError.parseError("malformed expression near unary '-'")
      }
      output.append(.unary(op: .neg, operand: operand))
      return
    }

    guard output.count >= 2 else {
      throw MathExprError.parseError("malformed expression near '\(opStr)'")
    }
    let right = output.removeLast()
    let left = output.removeLast()
    let bop: BinaryOp
    switch opStr {
    case "+": bop = .add
    case "-": bop = .sub
    case "*": bop = .mul
    case "/": bop = .div
    case "%": bop = .mod
    case "^": bop = .pow
    default:
      throw MathExprError.parseError("unknown operator '\(opStr)'")
    }
    output.append(.binary(op: bop, left: left, right: right))
  }

  // MARK: - Public entry point

  /// Parse an expression string into a MathLexExpression AST.
  static func parseExpression(_ expression: String) throws -> MathLexExpression {
    let tokens = try tokenize(expression)
    return try parse(tokens)
  }

  // MARK: - Find variables (AST walk, no mathlex needed)

  static func findVariables(in expression: String) throws -> Set<String> {
    let ast = try parseExpression(expression)
    return MathExpr.findVariables(in: ast)
  }
}

#endif  // !NUMERICSWIFT_MATHLEX
