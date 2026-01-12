//
//  MathExprTests.swift
//  NumericSwift
//

import XCTest
@testable import NumericSwift

final class MathExprTests: XCTestCase {

    // MARK: - Tokenizer Tests

    func testTokenizeNumber() throws {
        let tokens = try MathExpr.tokenize("42")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens[0], .number(42))
    }

    func testTokenizeDecimalNumber() throws {
        let tokens = try MathExpr.tokenize("3.14159")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens[0], .number(3.14159))
    }

    func testTokenizeScientificNotation() throws {
        let tokens = try MathExpr.tokenize("1.5e10")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens[0], .number(1.5e10))
    }

    func testTokenizeNegativeExponent() throws {
        let tokens = try MathExpr.tokenize("2.5E-3")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens[0], .number(2.5e-3))
    }

    func testTokenizeImaginary() throws {
        let tokens = try MathExpr.tokenize("5i")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens[0], .imaginary(5))
    }

    func testTokenizeOperators() throws {
        let tokens = try MathExpr.tokenize("+ - * / ^")
        XCTAssertEqual(tokens.count, 5)
        XCTAssertEqual(tokens[0], .operator("+"))
        XCTAssertEqual(tokens[1], .operator("-"))
        XCTAssertEqual(tokens[2], .operator("*"))
        XCTAssertEqual(tokens[3], .operator("/"))
        XCTAssertEqual(tokens[4], .operator("^"))
    }

    func testTokenizeParentheses() throws {
        let tokens = try MathExpr.tokenize("(x)")
        XCTAssertEqual(tokens.count, 3)
        XCTAssertEqual(tokens[0], .lparen)
        XCTAssertEqual(tokens[1], .variable("x"))
        XCTAssertEqual(tokens[2], .rparen)
    }

    func testTokenizeFunction() throws {
        let tokens = try MathExpr.tokenize("sin(x)")
        XCTAssertEqual(tokens.count, 4)
        XCTAssertEqual(tokens[0], .function("sin"))
        XCTAssertEqual(tokens[1], .lparen)
        XCTAssertEqual(tokens[2], .variable("x"))
        XCTAssertEqual(tokens[3], .rparen)
    }

    func testTokenizeConstant() throws {
        let tokens = try MathExpr.tokenize("pi")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens[0], .constant("pi"))
    }

    func testTokenizeVariable() throws {
        let tokens = try MathExpr.tokenize("myVar_123")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens[0], .variable("myVar_123"))
    }

    func testTokenizeComma() throws {
        let tokens = try MathExpr.tokenize("atan2(y, x)")
        XCTAssertEqual(tokens.count, 6)
        XCTAssertEqual(tokens[0], .function("atan2"))
        XCTAssertEqual(tokens[3], .comma)
    }

    func testTokenizeComplexExpression() throws {
        let tokens = try MathExpr.tokenize("sin(x) + 2*pi")
        XCTAssertEqual(tokens.count, 8)
        XCTAssertEqual(tokens[0], .function("sin"))
        XCTAssertEqual(tokens[1], .lparen)
        XCTAssertEqual(tokens[2], .variable("x"))
        XCTAssertEqual(tokens[3], .rparen)
        XCTAssertEqual(tokens[4], .operator("+"))
        XCTAssertEqual(tokens[5], .number(2))
        XCTAssertEqual(tokens[6], .operator("*"))
        XCTAssertEqual(tokens[7], .constant("pi"))
    }

    func testTokenizeWhitespaceHandling() throws {
        let tokens1 = try MathExpr.tokenize("1+2")
        let tokens2 = try MathExpr.tokenize("1 + 2")
        let tokens3 = try MathExpr.tokenize("  1  +  2  ")
        XCTAssertEqual(tokens1, tokens2)
        XCTAssertEqual(tokens2, tokens3)
    }

    func testTokenizeUnexpectedCharacter() {
        XCTAssertThrowsError(try MathExpr.tokenize("1 @ 2")) { error in
            guard let mathError = error as? MathExprError else {
                XCTFail("Expected MathExprError")
                return
            }
            if case .unexpectedCharacter(let char, _) = mathError {
                XCTAssertEqual(char, "@")
            } else {
                XCTFail("Expected unexpectedCharacter error")
            }
        }
    }

    // MARK: - Parser Tests

    func testParseNumber() throws {
        let tokens = try MathExpr.tokenize("42")
        let ast = try MathExpr.parse(tokens)
        XCTAssertEqual(ast, .number(42))
    }

    func testParseAddition() throws {
        let tokens = try MathExpr.tokenize("1 + 2")
        let ast = try MathExpr.parse(tokens)
        XCTAssertEqual(ast, .binary(op: "+", left: .number(1), right: .number(2)))
    }

    func testParsePrecedence() throws {
        // 1 + 2 * 3 should be 1 + (2 * 3)
        let tokens = try MathExpr.tokenize("1 + 2 * 3")
        let ast = try MathExpr.parse(tokens)
        XCTAssertEqual(ast, .binary(op: "+",
                                    left: .number(1),
                                    right: .binary(op: "*", left: .number(2), right: .number(3))))
    }

    func testParseParentheses() throws {
        // (1 + 2) * 3
        let tokens = try MathExpr.tokenize("(1 + 2) * 3")
        let ast = try MathExpr.parse(tokens)
        XCTAssertEqual(ast, .binary(op: "*",
                                    left: .binary(op: "+", left: .number(1), right: .number(2)),
                                    right: .number(3)))
    }

    func testParseRightAssociativePower() throws {
        // 2 ^ 3 ^ 2 should be 2 ^ (3 ^ 2) = 2 ^ 9 = 512
        let tokens = try MathExpr.tokenize("2 ^ 3 ^ 2")
        let ast = try MathExpr.parse(tokens)
        let result = try MathExpr.evaluate(ast)
        XCTAssertEqual(result, 512, accuracy: 1e-10)
    }

    func testParseUnaryMinus() throws {
        let tokens = try MathExpr.tokenize("-5")
        let ast = try MathExpr.parse(tokens)
        let result = try MathExpr.evaluate(ast)
        XCTAssertEqual(result, -5, accuracy: 1e-10)
    }

    func testParseFunctionCall() throws {
        let tokens = try MathExpr.tokenize("sin(0)")
        let ast = try MathExpr.parse(tokens)
        XCTAssertEqual(ast, .call(name: "sin", args: [.number(0)]))
    }

    func testParseMultiArgFunction() throws {
        let tokens = try MathExpr.tokenize("atan2(1, 0)")
        let ast = try MathExpr.parse(tokens)
        XCTAssertEqual(ast, .call(name: "atan2", args: [.number(1), .number(0)]))
    }

    func testParseNestedFunctions() throws {
        let tokens = try MathExpr.tokenize("sin(cos(0))")
        let ast = try MathExpr.parse(tokens)
        XCTAssertEqual(ast, .call(name: "sin", args: [.call(name: "cos", args: [.number(0)])]))
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
        XCTAssertEqual(try MathExpr.eval("e"), Darwin.M_E, accuracy: 1e-10)
        XCTAssertTrue((try MathExpr.eval("inf")).isInfinite)
        XCTAssertTrue((try MathExpr.eval("nan")).isNaN)
    }

    func testEvalVariables() throws {
        XCTAssertEqual(try MathExpr.eval("x", variables: ["x": 5]), 5, accuracy: 1e-10)
        XCTAssertEqual(try MathExpr.eval("x + y", variables: ["x": 3, "y": 4]), 7, accuracy: 1e-10)
        XCTAssertEqual(try MathExpr.eval("x^2 + 2*x + 1", variables: ["x": 3]), 16, accuracy: 1e-10)
    }

    func testEvalUndefinedVariable() {
        XCTAssertThrowsError(try MathExpr.eval("x")) { error in
            guard let mathError = error as? MathExprError,
                  case .undefinedVariable(let name) = mathError else {
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
        XCTAssertEqual(try MathExpr.eval("atan2(1, 0)"), .pi/2, accuracy: 1e-10)
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
        XCTAssertEqual(try MathExpr.eval("exp(1)"), Darwin.M_E, accuracy: 1e-10)
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
                  case .divisionByZero = mathError else {
                XCTFail("Expected divisionByZero error")
                return
            }
        }
    }

    func testEvalUnknownFunction() {
        XCTAssertThrowsError(try MathExpr.eval("unknown(5)")) { error in
            guard let mathError = error as? MathExprError,
                  case .unknownFunction(let name) = mathError else {
                XCTFail("Expected unknownFunction error")
                return
            }
            XCTAssertEqual(name, "unknown")
        }
    }

    // MARK: - toString Tests

    func testToStringNumber() throws {
        XCTAssertEqual(MathExpr.toString(.number(42)), "42")
        XCTAssertEqual(MathExpr.toString(.number(3.14)), "3.14")
        XCTAssertEqual(MathExpr.toString(.number(.nan)), "nan")
        XCTAssertEqual(MathExpr.toString(.number(.infinity)), "inf")
        XCTAssertEqual(MathExpr.toString(.number(-.infinity)), "-inf")
    }

    func testToStringImaginary() throws {
        XCTAssertEqual(MathExpr.toString(.imaginary(1)), "i")
        XCTAssertEqual(MathExpr.toString(.imaginary(-1)), "-i")
        XCTAssertEqual(MathExpr.toString(.imaginary(5)), "5.0i")
    }

    func testToStringVariable() throws {
        XCTAssertEqual(MathExpr.toString(.variable("x")), "x")
    }

    func testToStringConstant() throws {
        XCTAssertEqual(MathExpr.toString(.constant("pi")), "pi")
    }

    func testToStringBinary() throws {
        let ast: MathExprAST = .binary(op: "+", left: .number(1), right: .number(2))
        XCTAssertEqual(MathExpr.toString(ast), "1 + 2")
    }

    func testToStringCall() throws {
        let ast: MathExprAST = .call(name: "sin", args: [.variable("x")])
        XCTAssertEqual(MathExpr.toString(ast), "sin(x)")
    }

    func testToStringRoundTrip() throws {
        let expressions = [
            "1 + 2",
            "x * y",
            "sin(x)",
            "atan2(y, x)",
            "pi"
        ]

        for expr in expressions {
            let tokens = try MathExpr.tokenize(expr)
            let ast = try MathExpr.parse(tokens)
            let str = MathExpr.toString(ast)
            let tokens2 = try MathExpr.tokenize(str)
            let ast2 = try MathExpr.parse(tokens2)
            XCTAssertEqual(ast, ast2, "Round trip failed for: \(expr)")
        }
    }

    // MARK: - Substitution Tests

    func testSubstituteVariable() throws {
        let ast: MathExprAST = .variable("x")
        let result = MathExpr.substitute(ast, with: ["x": .number(5)])
        XCTAssertEqual(result, .number(5))
    }

    func testSubstituteInExpression() throws {
        let tokens = try MathExpr.tokenize("x + y")
        let ast = try MathExpr.parse(tokens)
        let result = MathExpr.substitute(ast, with: ["x": .number(3)])
        let expected: MathExprAST = .binary(op: "+", left: .number(3), right: .variable("y"))
        XCTAssertEqual(result, expected)
    }

    func testSubstituteWithExpression() throws {
        let tokens = try MathExpr.tokenize("x + 1")
        let ast = try MathExpr.parse(tokens)
        let replacement: MathExprAST = .binary(op: "*", left: .number(2), right: .variable("y"))
        let result = MathExpr.substitute(ast, with: ["x": replacement])
        // (2 * y) + 1
        let expected: MathExprAST = .binary(op: "+",
                                            left: .binary(op: "*", left: .number(2), right: .variable("y")),
                                            right: .number(1))
        XCTAssertEqual(result, expected)
    }

    // MARK: - Find Variables Tests

    func testFindVariablesSimple() throws {
        let tokens = try MathExpr.tokenize("x + y")
        let ast = try MathExpr.parse(tokens)
        let vars = MathExpr.findVariables(in: ast)
        XCTAssertEqual(vars, ["x", "y"])
    }

    func testFindVariablesWithConstants() throws {
        let tokens = try MathExpr.tokenize("x + pi")
        let ast = try MathExpr.parse(tokens)
        let vars = MathExpr.findVariables(in: ast)
        XCTAssertEqual(vars, ["x"])  // pi is a constant, not a variable
    }

    func testFindVariablesInFunction() throws {
        let tokens = try MathExpr.tokenize("sin(x) + cos(y)")
        let ast = try MathExpr.parse(tokens)
        let vars = MathExpr.findVariables(in: ast)
        XCTAssertEqual(vars, ["x", "y"])
    }

    func testFindVariablesDuplicates() throws {
        let tokens = try MathExpr.tokenize("x + x * x")
        let ast = try MathExpr.parse(tokens)
        let vars = MathExpr.findVariables(in: ast)
        XCTAssertEqual(vars, ["x"])  // Should only list x once
    }

    // MARK: - Complex Expression Tests

    func testComplexExpression1() throws {
        // Quadratic formula style: (-b + sqrt(b^2 - 4*a*c)) / (2*a)
        let result = try MathExpr.eval("(-b + sqrt(b^2 - 4*a*c)) / (2*a)",
                                       variables: ["a": 1, "b": -5, "c": 6])
        XCTAssertEqual(result, 3, accuracy: 1e-10)
    }

    func testComplexExpression2() throws {
        // Compound expression with multiple functions
        let result = try MathExpr.eval("sin(pi/4)^2 + cos(pi/4)^2")
        XCTAssertEqual(result, 1, accuracy: 1e-10)
    }

    func testComplexExpression3() throws {
        // Nested function calls
        let result = try MathExpr.eval("exp(log(10))")
        XCTAssertEqual(result, 10, accuracy: 1e-10)
    }

    func testComplexExpression4() throws {
        // Expression with all basic operations
        let result = try MathExpr.eval("(2 + 3) * 4 - 10 / 2 + 3^2")
        // (5) * 4 - 5 + 9 = 20 - 5 + 9 = 24
        XCTAssertEqual(result, 24, accuracy: 1e-10)
    }
}
