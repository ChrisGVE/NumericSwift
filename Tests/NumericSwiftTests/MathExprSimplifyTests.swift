//
//  MathExprSimplifyTests.swift
//  NumericSwift
//

import XCTest

@testable import NumericSwift

// MARK: - Structural Assertion Helpers

/// Assert two AST nodes are structurally equal using the rules engine helper.
private func assertAST(
  _ actual: MathLexExpression, equals expected: MathLexExpression,
  _ message: String = "", file: StaticString = #filePath, line: UInt = #line
) {
  XCTAssertTrue(
    MathExprSimplifyRules.structurallyEqual(actual, expected),
    message.isEmpty ? "Expected \(expected), got \(actual)" : message,
    file: file, line: line
  )
}

// MARK: - Simplification Tests

final class MathExprSimplifyTests: XCTestCase {

  // MARK: - Constant Folding: Arithmetic

  func testFoldIntegerAddition() throws {
    let ast = try MathExpr.parse("2 + 3")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(5))
  }

  func testFoldIntegerSubtraction() throws {
    let ast = try MathExpr.parse("10 - 4")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(6))
  }

  func testFoldIntegerMultiplication() throws {
    let ast = try MathExpr.parse("3 * 7")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(21))
  }

  func testFoldIntegerDivisionExact() throws {
    let ast = try MathExpr.parse("6 / 2")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(3))
  }

  func testFoldIntegerDivisionFloat() throws {
    let ast = try MathExpr.parse("1 / 4")
    let result = MathExpr.simplify(ast)
    if case .float(let v) = result {
      XCTAssertEqual(v!, 0.25, accuracy: 1e-12)
    } else {
      XCTFail("Expected float(0.25), got \(result)")
    }
  }

  func testFoldIntegerPower() throws {
    let ast = try MathExpr.parse("2 ^ 8")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(256))
  }

  func testFoldNestedConstants() throws {
    // (2 + 3) * 4 → 5 * 4 → 20
    let ast = try MathExpr.parse("(2 + 3) * 4")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(20))
  }

  // MARK: - Constant Folding: Functions

  func testFoldSinZero() throws {
    let ast = try MathExpr.parse("sin(0)")
    let result = MathExpr.simplify(ast)
    if case .integer(let n) = result {
      XCTAssertEqual(n, 0)
    } else if case .float(let v) = result {
      XCTAssertEqual(v!, 0.0, accuracy: 1e-12)
    } else {
      XCTFail("Expected 0, got \(result)")
    }
  }

  func testFoldCosZero() throws {
    let ast = try MathExpr.parse("cos(0)")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(1))
  }

  func testFoldSqrtFour() throws {
    let ast = try MathExpr.parse("sqrt(4)")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(2))
  }

  func testFoldAbsNegative() throws {
    let ast = try MathExpr.parse("abs(-3)")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(3))
  }

  // MARK: - Identity Rules: Addition

  func testAddZeroRight() throws {
    let ast = try MathExpr.parse("x + 0")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .variable("x"))
  }

  func testAddZeroLeft() throws {
    let ast = try MathExpr.parse("0 + x")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .variable("x"))
  }

  // MARK: - Identity Rules: Subtraction

  func testSubtractZero() throws {
    let ast = try MathExpr.parse("x - 0")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .variable("x"))
  }

  // MARK: - Identity Rules: Multiplication

  func testMulOneRight() throws {
    let ast = try MathExpr.parse("x * 1")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .variable("x"))
  }

  func testMulOneLeft() throws {
    let ast = try MathExpr.parse("1 * x")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .variable("x"))
  }

  func testMulZeroRight() throws {
    let ast = try MathExpr.parse("x * 0")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(0))
  }

  func testMulZeroLeft() throws {
    let ast = try MathExpr.parse("0 * x")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(0))
  }

  // MARK: - Identity Rules: Division

  func testDivByOne() throws {
    let ast = try MathExpr.parse("x / 1")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .variable("x"))
  }

  // MARK: - Identity Rules: Power

  func testPowZero() throws {
    let ast = try MathExpr.parse("x ^ 0")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(1))
  }

  func testPowOne() throws {
    let ast = try MathExpr.parse("x ^ 1")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .variable("x"))
  }

  func testOnePowX() throws {
    let ast = try MathExpr.parse("1 ^ x")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(1))
  }

  func testZeroPowPositive() throws {
    let ast = try MathExpr.parse("0 ^ 3")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(0))
  }

  // MARK: - Double Negation

  func testDoubleNegation() throws {
    let ast = try MathExpr.parse("-(-x)")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .variable("x"))
  }

  func testNegationOfZero() throws {
    let ast = try MathExpr.parse("-(0)")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(0))
  }

  // MARK: - Like Term Combining

  func testXPlusX() throws {
    // x + x → 2 * x
    let ast = try MathExpr.parse("x + x")
    let result = MathExpr.simplify(ast)
    let expected = MathLexExpression.binary(op: .mul, left: .integer(2), right: .variable("x"))
    assertAST(result, equals: expected)
  }

  func testXMinusX() throws {
    let ast = try MathExpr.parse("x - x")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(0))
  }

  func testCoeffCombiningAdd() throws {
    // 2*x + 3*x → 5*x
    let ast = try MathExpr.parse("2*x + 3*x")
    let result = MathExpr.simplify(ast)
    let expected = MathLexExpression.binary(op: .mul, left: .integer(5), right: .variable("x"))
    assertAST(result, equals: expected)
  }

  func testCoeffCombiningSub() throws {
    // 5*x - 2*x → 3*x
    let ast = try MathExpr.parse("5*x - 2*x")
    let result = MathExpr.simplify(ast)
    let expected = MathLexExpression.binary(op: .mul, left: .integer(3), right: .variable("x"))
    assertAST(result, equals: expected)
  }

  func testCoeffSubToZero() throws {
    // 3*x - 3*x → 0
    let ast = try MathExpr.parse("3*x - 3*x")
    let result = MathExpr.simplify(ast)
    assertAST(result, equals: .integer(0))
  }

  // MARK: - Power Simplification

  func testPowerOfPower() throws {
    // (x^2)^3 → x^6
    let ast = try MathExpr.parse("(x^2)^3")
    let result = MathExpr.simplify(ast)
    let expected = MathLexExpression.binary(op: .pow, left: .variable("x"), right: .integer(6))
    assertAST(result, equals: expected)
  }

  func testPowerMulSameBase() throws {
    // x^2 * x^3 → x^5
    let ast = try MathExpr.parse("x^2 * x^3")
    let result = MathExpr.simplify(ast)
    let expected = MathLexExpression.binary(op: .pow, left: .variable("x"), right: .integer(5))
    assertAST(result, equals: expected)
  }

  func testPowerMulWithImplicitOne() throws {
    // x * x^2 → x^3 (x treated as x^1)
    let ast = try MathExpr.parse("x * x^2")
    let result = MathExpr.simplify(ast)
    let expected = MathLexExpression.binary(op: .pow, left: .variable("x"), right: .integer(3))
    assertAST(result, equals: expected)
  }

  // MARK: - Numerical Equivalence (evaluate simplified vs original)

  func testSimplifiedEvaluatesCorrectly() throws {
    let expressions = [
      "2*x + 3*x",
      "x * 1",
      "x + 0",
      "x ^ 1",
      "(x^2)^3",
    ]
    let xValue = 3.0
    for expr in expressions {
      let original = try MathExpr.parse(expr)
      let simplified = MathExpr.simplify(original)
      let originalVal = try MathExpr.evaluate(original, variables: ["x": xValue])
      let simplifiedVal = try MathExpr.evaluate(simplified, variables: ["x": xValue])
      XCTAssertEqual(
        originalVal, simplifiedVal, accuracy: 1e-10,
        "Mismatch for '\(expr)': original=\(originalVal), simplified=\(simplifiedVal)"
      )
    }
  }
}
