//
//  MathExprTests.swift
//  NumericSwift
//

import XCTest

@testable import NumericSwift

final class MathExprTests: XCTestCase {

  // MARK: - Parser Tests

  func testParseNumber() throws {
    let ast = try MathExpr.parse("42")
    if case .integer(let n) = ast {
      XCTAssertEqual(n, 42)
    } else {
      XCTFail("Expected integer, got \(ast)")
    }
  }

  func testParseFloat() throws {
    let ast = try MathExpr.parse("3.14")
    if case .float(let v) = ast {
      XCTAssertEqual(v!, 3.14, accuracy: 1e-10)
    } else {
      XCTFail("Expected float, got \(ast)")
    }
  }

  func testParseAddition() throws {
    let ast = try MathExpr.parse("1 + 2")
    if case .binary(let op, let left, let right) = ast {
      XCTAssertEqual(op, .add)
      if case .integer(1) = left {} else { XCTFail("Expected integer(1)") }
      if case .integer(2) = right {} else { XCTFail("Expected integer(2)") }
    } else {
      XCTFail("Expected binary, got \(ast)")
    }
  }

  func testParsePrecedence() throws {
    // 1 + 2 * 3 should be 1 + (2 * 3)
    let result = try MathExpr.eval("1 + 2 * 3")
    XCTAssertEqual(result, 7, accuracy: 1e-10)
  }

  func testParseParentheses() throws {
    let result = try MathExpr.eval("(1 + 2) * 3")
    XCTAssertEqual(result, 9, accuracy: 1e-10)
  }

  func testParseRightAssociativePower() throws {
    // 2 ^ 3 ^ 2 should be 2 ^ (3 ^ 2) = 512
    let result = try MathExpr.eval("2 ^ 3 ^ 2")
    XCTAssertEqual(result, 512, accuracy: 1e-10)
  }

  func testParseUnaryMinus() throws {
    let result = try MathExpr.eval("-5")
    XCTAssertEqual(result, -5, accuracy: 1e-10)
  }

  func testParseFunctionCall() throws {
    let ast = try MathExpr.parse("sin(0)")
    if case .function(let name, let args) = ast {
      XCTAssertEqual(name, "sin")
      XCTAssertEqual(args.count, 1)
    } else {
      XCTFail("Expected function, got \(ast)")
    }
  }

  func testParseMultiArgFunction() throws {
    let ast = try MathExpr.parse("atan2(1, 0)")
    if case .function(let name, let args) = ast {
      XCTAssertEqual(name, "atan2")
      XCTAssertEqual(args.count, 2)
    } else {
      XCTFail("Expected function, got \(ast)")
    }
  }

  func testParseNestedFunctions() throws {
    let ast = try MathExpr.parse("sin(cos(0))")
    if case .function(let name, let args) = ast {
      XCTAssertEqual(name, "sin")
      XCTAssertEqual(args.count, 1)
      if case .function(let inner, _) = args[0] {
        XCTAssertEqual(inner, "cos")
      } else {
        XCTFail("Expected nested function")
      }
    } else {
      XCTFail("Expected function, got \(ast)")
    }
  }

  // MARK: - Evaluator Tests

  func testEvalNumber() throws {
    let result = try MathExpr.eval("42")
    XCTAssertEqual(result, 42, accuracy: 1e-10)
  }

  func testEvalArithmetic() throws {
    XCTAssertEqual(try MathExpr.eval("2 + 3"), 5, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("10 - 4"), 6, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("3 * 4"), 12, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("15 / 3"), 5, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("2 ^ 10"), 1024, accuracy: 1e-10)
  }

  func testEvalPrecedence() throws {
    XCTAssertEqual(try MathExpr.eval("1 + 2 * 3"), 7, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("(1 + 2) * 3"), 9, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("2 + 3 ^ 2"), 11, accuracy: 1e-10)
  }

  func testEvalConstants() throws {
    XCTAssertEqual(try MathExpr.eval("pi"), .pi, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("e"), M_E, accuracy: 1e-10)
    XCTAssertTrue((try MathExpr.eval("inf")).isInfinite)
    XCTAssertTrue((try MathExpr.eval("nan")).isNaN)
  }

  func testEvalVariables() throws {
    XCTAssertEqual(try MathExpr.eval("x", variables: ["x": 5]), 5, accuracy: 1e-10)
    XCTAssertEqual(
      try MathExpr.eval("x + y", variables: ["x": 3, "y": 4]), 7, accuracy: 1e-10)
    XCTAssertEqual(
      try MathExpr.eval("x^2 + 2*x + 1", variables: ["x": 3]), 16, accuracy: 1e-10)
  }

  func testEvalUndefinedVariable() {
    XCTAssertThrowsError(try MathExpr.eval("x")) { error in
      guard let mathError = error as? MathExprError,
        case .undefinedVariable(let name) = mathError
      else {
        XCTFail("Expected undefinedVariable error")
        return
      }
      XCTAssertEqual(name, "x")
    }
  }

  // MARK: - Function Tests

  func testEvalTrigonometric() throws {
    XCTAssertEqual(try MathExpr.eval("sin(0)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("cos(0)"), 1, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("tan(0)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("sin(pi/2)"), 1, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("cos(pi)"), -1, accuracy: 1e-10)
  }

  func testEvalInverseTrigonometric() throws {
    XCTAssertEqual(try MathExpr.eval("asin(0)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("acos(1)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("atan(0)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("atan2(1, 0)"), .pi / 2, accuracy: 1e-10)
  }

  func testEvalHyperbolic() throws {
    XCTAssertEqual(try MathExpr.eval("sinh(0)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("cosh(0)"), 1, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("tanh(0)"), 0, accuracy: 1e-10)
  }

  func testEvalInverseHyperbolic() throws {
    XCTAssertEqual(try MathExpr.eval("asinh(0)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("acosh(1)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("atanh(0)"), 0, accuracy: 1e-10)
  }

  func testEvalExponentialAndLog() throws {
    XCTAssertEqual(try MathExpr.eval("exp(0)"), 1, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("exp(1)"), M_E, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("log(e)"), 1, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("ln(e)"), 1, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("log10(100)"), 2, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("log2(8)"), 3, accuracy: 1e-10)
  }

  func testEvalPowerAndRoots() throws {
    XCTAssertEqual(try MathExpr.eval("sqrt(16)"), 4, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("cbrt(27)"), 3, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("cbrt(-8)"), -2, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("pow(2, 10)"), 1024, accuracy: 1e-10)
  }

  func testEvalAbsAndSign() throws {
    XCTAssertEqual(try MathExpr.eval("abs(-5)"), 5, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("abs(5)"), 5, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("sign(-5)"), -1, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("sign(5)"), 1, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("sign(0)"), 0, accuracy: 1e-10)
  }

  func testEvalRounding() throws {
    XCTAssertEqual(try MathExpr.eval("floor(3.7)"), 3, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("ceil(3.2)"), 4, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("round(3.5)"), 4, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("round(3.4)"), 3, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("trunc(3.9)"), 3, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("trunc(-3.9)"), -3, accuracy: 1e-10)
  }

  func testEvalMinMaxClampLerp() throws {
    XCTAssertEqual(try MathExpr.eval("min(3, 5)"), 3, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("max(3, 5)"), 5, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("clamp(10, 0, 5)"), 5, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("clamp(-3, 0, 5)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("clamp(3, 0, 5)"), 3, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("lerp(0, 10, 0.5)"), 5, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("lerp(0, 10, 0)"), 0, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("lerp(0, 10, 1)"), 10, accuracy: 1e-10)
  }

  func testEvalAngleConversion() throws {
    XCTAssertEqual(try MathExpr.eval("rad(180)"), .pi, accuracy: 1e-10)
    XCTAssertEqual(try MathExpr.eval("deg(pi)"), 180, accuracy: 1e-10)
  }

  func testEvalDivisionByZero() {
    XCTAssertThrowsError(try MathExpr.eval("1/0")) { error in
      guard let mathError = error as? MathExprError,
        case .divisionByZero = mathError
      else {
        XCTFail("Expected divisionByZero error")
        return
      }
    }
  }

  func testEvalUnknownFunction() {
    XCTAssertThrowsError(try MathExpr.eval("unknown(5)")) { error in
      guard let mathError = error as? MathExprError else {
        XCTFail("Expected MathExprError, got \(error)")
        return
      }
      // mathlex may reject at parse time or we reject at eval time
      switch mathError {
      case .unknownFunction(let name):
        XCTAssertEqual(name, "unknown")
      case .parseError:
        // mathlex rejected the unknown function at parse time — also valid
        break
      default:
        XCTFail("Expected unknownFunction or parseError, got \(mathError)")
      }
    }
  }

  // MARK: - Find Variables Tests

  func testFindVariablesSimple() throws {
    let vars = try MathExpr.findVariables(in: "x + y")
    XCTAssertEqual(vars, ["x", "y"])
  }

  func testFindVariablesWithConstants() throws {
    let vars = try MathExpr.findVariables(in: "x + pi")
    XCTAssertEqual(vars, ["x"])  // pi is a constant, not a variable
  }

  func testFindVariablesInFunction() throws {
    let vars = try MathExpr.findVariables(in: "sin(x) + cos(y)")
    XCTAssertEqual(vars, ["x", "y"])
  }

  func testFindVariablesDuplicates() throws {
    let vars = try MathExpr.findVariables(in: "x + x * x")
    XCTAssertEqual(vars, ["x"])
  }

  // MARK: - Substitution Tests

  func testSubstituteVariable() throws {
    let ast = try MathExpr.parse("x")
    let result = MathExpr.substitute(ast, with: ["x": .integer(5)])
    if case .integer(5) = result {
    } else {
      XCTFail("Expected integer(5), got \(result)")
    }
  }

  func testSubstituteInExpression() throws {
    let ast = try MathExpr.parse("x + y")
    let result = MathExpr.substitute(ast, with: ["x": .integer(3)])
    // Should substitute x but keep y
    let vars = MathExpr.findVariables(in: result)
    XCTAssertTrue(vars.contains("y"))
    XCTAssertFalse(vars.contains("x"))
  }

  // MARK: - Complex Expression Tests

  func testComplexExpression1() throws {
    // Quadratic formula style: (-b + sqrt(b^2 - 4*a*c)) / (2*a)
    let result = try MathExpr.eval(
      "(-b + sqrt(b^2 - 4*a*c)) / (2*a)",
      variables: ["a": 1, "b": -5, "c": 6])
    XCTAssertEqual(result, 3, accuracy: 1e-10)
  }

  func testComplexExpression2() throws {
    // sin^2 + cos^2 = 1
    let result = try MathExpr.eval("sin(pi/4)^2 + cos(pi/4)^2")
    XCTAssertEqual(result, 1, accuracy: 1e-10)
  }

  func testComplexExpression3() throws {
    // exp(log(10)) = 10
    let result = try MathExpr.eval("exp(log(10))")
    XCTAssertEqual(result, 10, accuracy: 1e-10)
  }

  func testComplexExpression4() throws {
    // (2 + 3) * 4 - 10 / 2 + 3^2 = 24
    let result = try MathExpr.eval("(2 + 3) * 4 - 10 / 2 + 3^2")
    XCTAssertEqual(result, 24, accuracy: 1e-10)
  }

  // MARK: - Complex Number Evaluation Tests

  func testEvalComplexImaginaryUnit() throws {
    let ast = try MathExpr.parse("2 * i")
    let result = try MathExpr.evaluateComplex(ast)
    XCTAssertEqual(result.re, 0, accuracy: 1e-10)
    XCTAssertEqual(result.im, 2, accuracy: 1e-10)
  }

  func testEvalComplexAddition() throws {
    let ast = try MathExpr.parse("3 + 2 * i")
    let result = try MathExpr.evaluateComplex(ast)
    XCTAssertEqual(result.re, 3, accuracy: 1e-10)
    XCTAssertEqual(result.im, 2, accuracy: 1e-10)
  }

  func testEvalComplexWithVariables() throws {
    let ast = try MathExpr.parse("z + 1")
    let result = try MathExpr.evaluateComplex(
      ast, complexVariables: ["z": Complex(re: 2, im: 3)])
    XCTAssertEqual(result.re, 3, accuracy: 1e-10)
    XCTAssertEqual(result.im, 3, accuracy: 1e-10)
  }

  func testEvalComplexMultiplication() throws {
    // (1 + i) * (1 - i) = 1 - i^2 = 2
    let ast = try MathExpr.parse("(1 + i) * (1 - i)")
    let result = try MathExpr.evaluateComplex(ast)
    XCTAssertEqual(result.re, 2, accuracy: 1e-10)
    XCTAssertEqual(result.im, 0, accuracy: 1e-10)
  }

  func testEvalComplexExp() throws {
    // exp(i*pi) = -1 (Euler's identity)
    let ast = try MathExpr.parse("exp(i * pi)")
    let result = try MathExpr.evaluateComplex(ast)
    XCTAssertEqual(result.re, -1, accuracy: 1e-10)
    XCTAssertEqual(result.im, 0, accuracy: 1e-10)
  }

  // MARK: - LaTeX Parsing Tests

  func testParseLatexFraction() throws {
    let result = try MathExpr.evaluate(try MathExpr.parseLatex(#"\frac{1}{2}"#))
    XCTAssertEqual(result, 0.5, accuracy: 1e-10)
  }

  func testParseLatexSqrt() throws {
    let result = try MathExpr.evaluate(try MathExpr.parseLatex(#"\sqrt{16}"#))
    XCTAssertEqual(result, 4, accuracy: 1e-10)
  }

  func testParseLatexTrig() throws {
    let result = try MathExpr.evaluate(try MathExpr.parseLatex(#"\sin(0)"#))
    XCTAssertEqual(result, 0, accuracy: 1e-10)
  }
}
