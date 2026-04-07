//
//  MathExprSimplify.swift
//  NumericSwift
//
//  Symbolic simplification engine for MathLexExpression ASTs.
//
//  Simplification is bottom-up: children are simplified first, then rules
//  are applied to the result. Splitting rules into SimplifyRules.swift keeps
//  each file within the 400-line limit.
//

import Foundation
import MathLex

// MARK: - Simplification Entry Point

extension MathExpr {

  /// Simplify a mathematical expression AST symbolically.
  ///
  /// Applies constant folding, identity rules, double-negation elimination,
  /// basic like-term combining, and power rules — all bottom-up.
  ///
  /// - Parameter expr: The AST to simplify
  /// - Returns: A simplified AST that is mathematically equivalent
  public static func simplify(_ expr: MathLexExpression) -> MathLexExpression {
    // Simplify children first (bottom-up), then apply rules
    let simplified = simplifyChildren(expr)
    return applyRules(simplified)
  }

  // MARK: - Bottom-up Child Simplification

  // swiftlint:disable:next cyclomatic_complexity
  private static func simplifyChildren(
    _ expr: MathLexExpression
  ) -> MathLexExpression {
    switch expr {
    case .binary(let op, let left, let right):
      return .binary(op: op, left: simplify(left), right: simplify(right))
    case .unary(let op, let operand):
      return .unary(op: op, operand: simplify(operand))
    case .function(let name, let args):
      return .function(name: name, args: args.map { simplify($0) })
    case .rational(let num, let den):
      return .rational(numerator: simplify(num), denominator: simplify(den))
    case .complex(let re, let im):
      return .complex(real: simplify(re), imaginary: simplify(im))
    case .equation(let left, let right):
      return .equation(left: simplify(left), right: simplify(right))
    case .sum(let index, let lower, let upper, let body):
      return .sum(
        index: index, lower: simplify(lower),
        upper: simplify(upper), body: simplify(body))
    case .product(let index, let lower, let upper, let body):
      return .product(
        index: index, lower: simplify(lower),
        upper: simplify(upper), body: simplify(body))
    default:
      return expr
    }
  }

  // MARK: - Rule Dispatcher

  private static func applyRules(_ expr: MathLexExpression) -> MathLexExpression {
    switch expr {
    case .binary(let op, let left, let right):
      return MathExprSimplifyRules.applyBinaryRules(op: op, left: left, right: right)
    case .unary(let op, let operand):
      return MathExprSimplifyRules.applyUnaryRules(op: op, operand: operand)
    case .function(let name, let args):
      return MathExprSimplifyRules.applyFunctionRules(name: name, args: args)
    default:
      return expr
    }
  }
}

// MARK: - Numeric Helpers (internal, shared with rules)

/// Extract a numeric value from an integer or float literal node.
func numericValue(_ expr: MathLexExpression) -> Double? {
  switch expr {
  case .integer(let n): return Double(n)
  case .float(let v): return v
  default: return nil
  }
}

/// Wrap a Double as the most compact literal (integer if exact, float otherwise).
func numericLiteral(_ value: Double) -> MathLexExpression {
  let rounded = value.rounded()
  if value == rounded && !value.isInfinite && !value.isNaN
    && rounded >= Double(Int64.min) && rounded <= Double(Int64.max)
  {
    return .integer(Int64(rounded))
  }
  return .float(value)
}
