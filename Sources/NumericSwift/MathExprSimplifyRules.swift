//
//  MathExprSimplifyRules.swift
//  NumericSwift
//
//  Individual simplification rules applied by MathExprSimplify.
//  Each rule category is a separate static method for clarity.
//

import Foundation
import MathLex

// MARK: - Rule Engine

/// Namespace for individual simplification rule groups.
enum MathExprSimplifyRules {

  // MARK: - Binary Rules

  // swiftlint:disable:next cyclomatic_complexity function_body_length
  static func applyBinaryRules(
    op: BinaryOp, left: MathLexExpression, right: MathLexExpression
  ) -> MathLexExpression {
    // Constant folding: both sides are numeric literals
    if let lv = numericValue(left), let rv = numericValue(right) {
      if let folded = foldBinary(op: op, lv: lv, rv: rv) {
        return folded
      }
    }
    switch op {
    case .add:
      return applyAddRules(left: left, right: right)
    case .sub:
      return applySubRules(left: left, right: right)
    case .mul:
      return applyMulRules(left: left, right: right)
    case .div:
      return applyDivRules(left: left, right: right)
    case .pow:
      return applyPowRules(left: left, right: right)
    default:
      return .binary(op: op, left: left, right: right)
    }
  }

  // MARK: - Constant Folding

  private static func foldBinary(op: BinaryOp, lv: Double, rv: Double) -> MathLexExpression? {
    switch op {
    case .add: return numericLiteral(lv + rv)
    case .sub: return numericLiteral(lv - rv)
    case .mul: return numericLiteral(lv * rv)
    case .div:
      guard rv != 0 else { return nil }
      return numericLiteral(lv / rv)
    case .pow: return numericLiteral(pow(lv, rv))
    case .mod:
      guard rv != 0 else { return nil }
      return numericLiteral(lv.truncatingRemainder(dividingBy: rv))
    default: return nil
    }
  }

  // MARK: - Add Rules

  private static func applyAddRules(
    left: MathLexExpression, right: MathLexExpression
  ) -> MathLexExpression {
    // x + 0 → x
    if numericValue(right) == 0 { return left }
    // 0 + x → x
    if numericValue(left) == 0 { return right }
    // x + x → 2 * x
    if structurallyEqual(left, right) {
      return .binary(op: .mul, left: .integer(2), right: left)
    }
    // a*x + b*x → (a+b)*x  and  x + b*x → (1+b)*x
    if let (coeff, base) = extractLinearTerm(left),
      let (coeffR, baseR) = extractLinearTerm(right),
      structurallyEqual(base, baseR)
    {
      let sum = numericLiteral(coeff + coeffR)
      return .binary(op: .mul, left: sum, right: base)
    }
    return .binary(op: .add, left: left, right: right)
  }

  // MARK: - Sub Rules

  private static func applySubRules(
    left: MathLexExpression, right: MathLexExpression
  ) -> MathLexExpression {
    // x - 0 → x
    if numericValue(right) == 0 { return left }
    // x - x → 0
    if structurallyEqual(left, right) { return .integer(0) }
    // a*x - b*x → (a-b)*x
    if let (coeff, base) = extractLinearTerm(left),
      let (coeffR, baseR) = extractLinearTerm(right),
      structurallyEqual(base, baseR)
    {
      let diff = numericLiteral(coeff - coeffR)
      if numericValue(diff) == 0 { return .integer(0) }
      return .binary(op: .mul, left: diff, right: base)
    }
    return .binary(op: .sub, left: left, right: right)
  }

  // MARK: - Mul Rules

  private static func applyMulRules(
    left: MathLexExpression, right: MathLexExpression
  ) -> MathLexExpression {
    // x * 0 → 0, 0 * x → 0
    if numericValue(left) == 0 || numericValue(right) == 0 { return .integer(0) }
    // x * 1 → x
    if numericValue(right) == 1 { return left }
    // 1 * x → x
    if numericValue(left) == 1 { return right }
    // x^a * x^b → x^(a+b)
    if let (base1, exp1) = extractPowerParts(left),
      let (base2, exp2) = extractPowerParts(right),
      structurallyEqual(base1, base2),
      let e1 = numericValue(exp1), let e2 = numericValue(exp2)
    {
      return .binary(op: .pow, left: base1, right: numericLiteral(e1 + e2))
    }
    return .binary(op: .mul, left: left, right: right)
  }

  // MARK: - Div Rules

  private static func applyDivRules(
    left: MathLexExpression, right: MathLexExpression
  ) -> MathLexExpression {
    // x / 1 → x
    if numericValue(right) == 1 { return left }
    return .binary(op: .div, left: left, right: right)
  }

  // MARK: - Pow Rules

  // swiftlint:disable:next cyclomatic_complexity
  private static func applyPowRules(
    left: MathLexExpression, right: MathLexExpression
  ) -> MathLexExpression {
    let rv = numericValue(right)
    let lv = numericValue(left)
    // x ^ 0 → 1 (treat as generally valid; edge case x=0 not handled symbolically)
    if rv == 0 { return .integer(1) }
    // x ^ 1 → x
    if rv == 1 { return left }
    // 1 ^ x → 1
    if lv == 1 { return .integer(1) }
    // 0 ^ x → 0 (for positive x; we apply conservatively when x is a positive literal)
    if lv == 0, let expVal = rv, expVal > 0 { return .integer(0) }
    // (x^a)^b → x^(a*b) when a,b are numeric
    if case .binary(let innerOp, let innerBase, let innerExp) = left,
      innerOp == .pow,
      let a = numericValue(innerExp),
      let b = rv
    {
      return .binary(op: .pow, left: innerBase, right: numericLiteral(a * b))
    }
    return .binary(op: .pow, left: left, right: right)
  }

  // MARK: - Unary Rules

  static func applyUnaryRules(op: UnaryOp, operand: MathLexExpression) -> MathLexExpression {
    guard op == .neg else {
      return .unary(op: op, operand: operand)
    }
    // -(0) → 0
    if numericValue(operand) == 0 { return .integer(0) }
    // -(-x) → x
    if case .unary(let innerOp, let inner) = operand, innerOp == .neg {
      return inner
    }
    // -(numeric literal) → fold immediately
    if let v = numericValue(operand) { return numericLiteral(-v) }
    return .unary(op: .neg, operand: operand)
  }

  // MARK: - Function Rules (constant folding for known values)

  static func applyFunctionRules(name: String, args: [MathLexExpression]) -> MathLexExpression {
    guard args.count == 1, let v = numericValue(args[0]) else {
      return .function(name: name, args: args)
    }
    if let result = foldKnownFunction(name: name, arg: v) {
      return numericLiteral(result)
    }
    return .function(name: name, args: args)
  }

  private static func foldKnownFunction(name: String, arg: Double) -> Double? {
    switch name {
    case "sin": return sin(arg)
    case "cos": return cos(arg)
    case "tan": return tan(arg)
    case "exp": return exp(arg)
    case "log", "ln": return arg > 0 ? log(arg) : nil
    case "log10": return arg > 0 ? log10(arg) : nil
    case "log2": return arg > 0 ? log2(arg) : nil
    case "sqrt": return arg >= 0 ? sqrt(arg) : nil
    case "abs": return abs(arg)
    case "floor": return floor(arg)
    case "ceil": return ceil(arg)
    case "round": return Foundation.round(arg)
    default: return nil
    }
  }

  // MARK: - Structural Equality

  // swiftlint:disable:next cyclomatic_complexity
  static func structurallyEqual(
    _ a: MathLexExpression, _ b: MathLexExpression
  ) -> Bool {
    switch (a, b) {
    case (.integer(let x), .integer(let y)): return x == y
    case (.float(let x), .float(let y)): return x == y
    case (.variable(let x), .variable(let y)): return x == y
    case (.constant(let x), .constant(let y)): return x == y
    case (.binary(let op1, let l1, let r1), .binary(let op2, let l2, let r2)):
      return op1 == op2 && structurallyEqual(l1, l2) && structurallyEqual(r1, r2)
    case (.unary(let op1, let o1), .unary(let op2, let o2)):
      return op1 == op2 && structurallyEqual(o1, o2)
    case (.function(let n1, let a1), .function(let n2, let a2)):
      guard n1 == n2, a1.count == a2.count else { return false }
      return zip(a1, a2).allSatisfy { structurallyEqual($0, $1) }
    default: return false
    }
  }

  // MARK: - Pattern Helpers

  /// Extract coefficient and base from a term: returns (coeff, base).
  /// Handles: numeric literal → (value, literal), c*x → (c, x), x → (1.0, x).
  static func extractLinearTerm(
    _ expr: MathLexExpression
  ) -> (Double, MathLexExpression)? {
    // Pure numeric: coefficient IS the literal, no variable base to combine
    if numericValue(expr) != nil { return nil }
    // c * x  or  x * c
    if case .binary(let op, let left, let right) = expr, op == .mul {
      if let c = numericValue(left) { return (c, right) }
      if let c = numericValue(right) { return (c, left) }
    }
    // x alone has implicit coefficient 1
    return (1.0, expr)
  }

  /// Extract (base, exponent) from x^n or x (treated as x^1).
  static func extractPowerParts(
    _ expr: MathLexExpression
  ) -> (MathLexExpression, MathLexExpression)? {
    if case .binary(let op, let base, let exp) = expr, op == .pow {
      return (base, exp)
    }
    // x treated as x^1 for combining x^a * x
    return (expr, .integer(1))
  }
}
