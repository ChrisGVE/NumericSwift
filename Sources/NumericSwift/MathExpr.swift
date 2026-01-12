//
//  MathExpr.swift
//  NumericSwift
//
//  Mathematical expression tokenizer and parser
//
//  Provides tokenization and parsing of mathematical expressions.
//  Supports standard math operators, functions, constants, and variables.
//

import Foundation

// MARK: - Token Types

/// Token types for mathematical expression parsing
public enum MathExprToken: Equatable, Sendable {
    case number(Double)
    case imaginary(Double)  // e.g., 2i, 5i (coefficient of imaginary unit)
    case `operator`(String)
    case function(String)
    case lparen
    case rparen
    case comma
    case variable(String)
    case constant(String)

    /// String description for debugging
    public var description: String {
        switch self {
        case .number(let n):
            return "number(\(n))"
        case .imaginary(let n):
            return "imaginary(\(n))"
        case .operator(let op):
            return "operator(\(op))"
        case .function(let name):
            return "function(\(name))"
        case .lparen:
            return "lparen"
        case .rparen:
            return "rparen"
        case .comma:
            return "comma"
        case .variable(let name):
            return "variable(\(name))"
        case .constant(let name):
            return "constant(\(name))"
        }
    }
}

// MARK: - Errors

/// Errors for mathematical expression parsing
public enum MathExprError: Error, Equatable {
    case unexpectedCharacter(Character, at: Int)
    case invalidNumber(String)
    case unexpectedEnd
    case unmatchedParenthesis
    case undefinedVariable(String)
    case unknownFunction(String)
    case divisionByZero
    case invalidArguments(String)

    public var description: String {
        switch self {
        case .unexpectedCharacter(let char, let pos):
            return "unexpected character '\(char)' at position \(pos)"
        case .invalidNumber(let str):
            return "invalid number '\(str)'"
        case .unexpectedEnd:
            return "unexpected end of expression"
        case .unmatchedParenthesis:
            return "unmatched parenthesis"
        case .undefinedVariable(let name):
            return "undefined variable '\(name)'"
        case .unknownFunction(let name):
            return "unknown function '\(name)'"
        case .divisionByZero:
            return "division by zero"
        case .invalidArguments(let msg):
            return "invalid arguments: \(msg)"
        }
    }
}

// MARK: - AST Node Types

/// AST node for parsed mathematical expressions
public indirect enum MathExprAST: Equatable, Sendable {
    case number(Double)
    case imaginary(Double)
    case constant(String)
    case variable(String)
    case unary(op: String, operand: MathExprAST)
    case binary(op: String, left: MathExprAST, right: MathExprAST)
    case call(name: String, args: [MathExprAST])
}

// MARK: - Math Expression Tokenizer and Parser

/// Mathematical expression tokenizer, parser, and evaluator
public struct MathExpr {

    // MARK: - Known Functions and Constants

    /// Known mathematical functions
    public static let knownFunctions: Set<String> = [
        // Trigonometric
        "sin", "cos", "tan",
        "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh",
        // Exponential and logarithmic
        "exp", "log", "log10", "log2", "ln",
        // Power and roots
        "sqrt", "cbrt", "pow",
        // Absolute value and sign
        "abs", "sign", "floor", "ceil", "round", "trunc",
        // Min/max and interpolation
        "min", "max", "clamp", "lerp",
        // Other
        "rad", "deg"
    ]

    /// Known mathematical constants
    public static let knownConstants: Set<String> = [
        "pi", "e", "inf", "nan"
    ]

    /// Constant values
    public static let constantValues: [String: Double] = [
        "pi": .pi,
        "e": Darwin.M_E,
        "inf": .infinity,
        "nan": .nan
    ]

    // MARK: - Tokenizer

    /// Tokenize a mathematical expression string.
    ///
    /// - Parameter expression: The expression to tokenize
    /// - Returns: Array of tokens
    /// - Throws: MathExprError if expression contains invalid characters
    public static func tokenize(_ expression: String) throws -> [MathExprToken] {
        var tokens: [MathExprToken] = []
        var index = expression.startIndex

        while index < expression.endIndex {
            let char = expression[index]

            // Skip whitespace
            if char.isWhitespace {
                index = expression.index(after: index)
                continue
            }

            // Numbers (including decimals, scientific notation, and imaginary suffix)
            if char.isNumber || (char == "." && index < expression.endIndex) {
                let (number, endIndex) = try parseNumber(expression, startingAt: index)
                // Check for imaginary suffix 'i' (no space between number and i)
                if endIndex < expression.endIndex && expression[endIndex] == "i" {
                    let nextIdx = expression.index(after: endIndex)
                    // Ensure 'i' is not followed by more identifier characters (like 'in', 'if')
                    if nextIdx >= expression.endIndex || !expression[nextIdx].isLetter {
                        tokens.append(.imaginary(number))
                        index = nextIdx
                        continue
                    }
                }
                tokens.append(.number(number))
                index = endIndex
                continue
            }

            // Operators
            if "+-*/^=".contains(char) {
                tokens.append(.operator(String(char)))
                index = expression.index(after: index)
                continue
            }

            // Parentheses
            if char == "(" {
                tokens.append(.lparen)
                index = expression.index(after: index)
                continue
            }

            if char == ")" {
                tokens.append(.rparen)
                index = expression.index(after: index)
                continue
            }

            // Comma (for multi-argument functions)
            if char == "," {
                tokens.append(.comma)
                index = expression.index(after: index)
                continue
            }

            // Identifiers (functions, constants, variables)
            if char.isLetter || char == "_" {
                let (identifier, endIndex) = parseIdentifier(expression, startingAt: index)
                index = endIndex

                if knownFunctions.contains(identifier) {
                    tokens.append(.function(identifier))
                } else if knownConstants.contains(identifier) {
                    tokens.append(.constant(identifier))
                } else {
                    tokens.append(.variable(identifier))
                }
                continue
            }

            // Unknown character
            throw MathExprError.unexpectedCharacter(char, at: expression.distance(from: expression.startIndex, to: index))
        }

        return tokens
    }

    /// Parse a number from the expression.
    private static func parseNumber(_ expression: String, startingAt start: String.Index) throws -> (Double, String.Index) {
        var index = start
        var numberString = ""
        var hasDecimal = false
        var hasExponent = false

        while index < expression.endIndex {
            let char = expression[index]

            if char.isNumber {
                numberString.append(char)
                index = expression.index(after: index)
            } else if char == "." && !hasDecimal && !hasExponent {
                numberString.append(char)
                hasDecimal = true
                index = expression.index(after: index)
            } else if (char == "e" || char == "E") && !hasExponent && !numberString.isEmpty {
                numberString.append(char)
                hasExponent = true
                index = expression.index(after: index)
                // Check for optional sign after exponent
                if index < expression.endIndex {
                    let nextChar = expression[index]
                    if nextChar == "+" || nextChar == "-" {
                        numberString.append(nextChar)
                        index = expression.index(after: index)
                    }
                }
            } else {
                break
            }
        }

        guard let number = Double(numberString) else {
            throw MathExprError.invalidNumber(numberString)
        }

        return (number, index)
    }

    /// Parse an identifier from the expression.
    private static func parseIdentifier(_ expression: String, startingAt start: String.Index) -> (String, String.Index) {
        var index = start
        var identifier = ""

        while index < expression.endIndex {
            let char = expression[index]
            if char.isLetter || char.isNumber || char == "_" {
                identifier.append(char)
                index = expression.index(after: index)
            } else {
                break
            }
        }

        return (identifier, index)
    }

    // MARK: - Parser

    /// Operator precedence
    private static let precedence: [String: Int] = [
        "+": 1, "-": 1,
        "*": 2, "/": 2,
        "^": 3
    ]

    /// Right-associative operators
    private static let rightAssociative: Set<String> = ["^"]

    /// Parse tokens into an AST using shunting-yard algorithm.
    ///
    /// - Parameter tokens: Array of tokens to parse
    /// - Returns: AST root node
    /// - Throws: MathExprError if parsing fails
    public static func parse(_ tokens: [MathExprToken]) throws -> MathExprAST {
        var output: [MathExprAST] = []
        var operators: [MathExprToken] = []
        var argCounts: [Int] = []

        var prevToken: MathExprToken? = nil

        for token in tokens {
            switch token {
            case .number(let n):
                output.append(.number(n))

            case .imaginary(let n):
                output.append(.imaginary(n))

            case .constant(let name):
                output.append(.constant(name))

            case .variable(let name):
                output.append(.variable(name))

            case .function:
                operators.append(token)
                argCounts.append(1)

            case .comma:
                while !operators.isEmpty {
                    if case .lparen = operators.last! { break }
                    try popOperator(&operators, &output)
                }
                if !argCounts.isEmpty {
                    argCounts[argCounts.count - 1] += 1
                }

            case .operator(let op):
                // Handle unary minus
                if op == "-" {
                    var isUnary = prevToken == nil
                    if let prev = prevToken {
                        switch prev {
                        case .lparen, .operator, .comma:
                            isUnary = true
                        default:
                            break
                        }
                    }
                    if isUnary {
                        output.append(.number(0))
                    }
                }

                let op1Prec = precedence[op] ?? 0
                while !operators.isEmpty {
                    guard case .operator(let topOp) = operators.last! else { break }
                    let op2Prec = precedence[topOp] ?? 0
                    if rightAssociative.contains(op) {
                        if op1Prec >= op2Prec { break }
                    } else {
                        if op1Prec > op2Prec { break }
                    }
                    try popOperator(&operators, &output)
                }
                operators.append(token)

            case .lparen:
                // Check if previous token was a variable - treat as function call
                if case .variable = prevToken {
                    // Pop the variable from output and convert to function call
                    if let lastAST = output.last, case .variable(let name) = lastAST {
                        output.removeLast()
                        operators.append(.function(name))
                        argCounts.append(1)
                    }
                }
                operators.append(token)

            case .rparen:
                while !operators.isEmpty {
                    if case .lparen = operators.last! { break }
                    try popOperator(&operators, &output)
                }
                if !operators.isEmpty {
                    operators.removeLast() // Remove lparen
                }
                // Handle function call
                if !operators.isEmpty, case .function(let name) = operators.last! {
                    operators.removeLast()
                    let numArgs = argCounts.isEmpty ? 1 : argCounts.removeLast()
                    var args: [MathExprAST] = []
                    for _ in 0..<numArgs {
                        if !output.isEmpty {
                            args.insert(output.removeLast(), at: 0)
                        }
                    }
                    output.append(.call(name: name, args: args))
                }
            }
            prevToken = token
        }

        while !operators.isEmpty {
            try popOperator(&operators, &output)
        }

        guard let result = output.first else {
            throw MathExprError.unexpectedEnd
        }

        return result
    }

    private static func popOperator(_ operators: inout [MathExprToken], _ output: inout [MathExprAST]) throws {
        guard let op = operators.popLast() else {
            throw MathExprError.unexpectedEnd
        }

        guard case .operator(let opStr) = op else {
            throw MathExprError.unmatchedParenthesis
        }

        guard output.count >= 2 else {
            throw MathExprError.unexpectedEnd
        }

        let right = output.removeLast()
        let left = output.removeLast()
        output.append(.binary(op: opStr, left: left, right: right))
    }

    // MARK: - Evaluator

    /// Evaluate an AST with given variable bindings.
    ///
    /// - Parameters:
    ///   - ast: The AST to evaluate
    ///   - variables: Dictionary of variable name to value
    /// - Returns: The computed result
    /// - Throws: MathExprError if evaluation fails
    public static func evaluate(_ ast: MathExprAST, variables: [String: Double] = [:]) throws -> Double {
        switch ast {
        case .number(let n):
            return n

        case .imaginary:
            // For real evaluation, imaginary numbers are not supported
            throw MathExprError.invalidArguments("imaginary numbers require complex evaluation")

        case .constant(let name):
            guard let value = constantValues[name] else {
                throw MathExprError.undefinedVariable(name)
            }
            return value

        case .variable(let name):
            guard let value = variables[name] else {
                throw MathExprError.undefinedVariable(name)
            }
            return value

        case .unary(let op, let operand):
            let val = try evaluate(operand, variables: variables)
            switch op {
            case "-":
                return -val
            default:
                throw MathExprError.invalidArguments("unknown unary operator: \(op)")
            }

        case .binary(let op, let left, let right):
            let leftVal = try evaluate(left, variables: variables)
            let rightVal = try evaluate(right, variables: variables)
            switch op {
            case "+":
                return leftVal + rightVal
            case "-":
                return leftVal - rightVal
            case "*":
                return leftVal * rightVal
            case "/":
                if rightVal == 0 {
                    throw MathExprError.divisionByZero
                }
                return leftVal / rightVal
            case "^":
                return pow(leftVal, rightVal)
            default:
                throw MathExprError.invalidArguments("unknown operator: \(op)")
            }

        case .call(let name, let args):
            let argValues = try args.map { try evaluate($0, variables: variables) }
            return try evaluateFunction(name, args: argValues)
        }
    }

    /// Evaluate a function call.
    private static func evaluateFunction(_ name: String, args: [Double]) throws -> Double {
        switch name {
        // Trigonometric
        case "sin":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return sin(args[0])
        case "cos":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return cos(args[0])
        case "tan":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return tan(args[0])
        case "asin":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return asin(args[0])
        case "acos":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return acos(args[0])
        case "atan":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return atan(args[0])
        case "atan2":
            guard args.count == 2 else { throw MathExprError.invalidArguments("\(name) requires 2 arguments") }
            return atan2(args[0], args[1])

        // Hyperbolic
        case "sinh":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return sinh(args[0])
        case "cosh":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return cosh(args[0])
        case "tanh":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return tanh(args[0])
        case "asinh":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return asinh(args[0])
        case "acosh":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return acosh(args[0])
        case "atanh":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return atanh(args[0])

        // Exponential and logarithmic
        case "exp":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return exp(args[0])
        case "log", "ln":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return log(args[0])
        case "log10":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return log10(args[0])
        case "log2":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return log2(args[0])

        // Power and roots
        case "sqrt":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return sqrt(args[0])
        case "cbrt":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return args[0] >= 0 ? pow(args[0], 1.0/3.0) : -pow(-args[0], 1.0/3.0)
        case "pow":
            guard args.count == 2 else { throw MathExprError.invalidArguments("\(name) requires 2 arguments") }
            return pow(args[0], args[1])

        // Absolute value and sign
        case "abs":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return abs(args[0])
        case "sign":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return args[0] > 0 ? 1.0 : (args[0] < 0 ? -1.0 : 0.0)
        case "floor":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return floor(args[0])
        case "ceil":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return ceil(args[0])
        case "round":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return round(args[0])
        case "trunc":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return trunc(args[0])

        // Min/max and interpolation
        case "min":
            guard args.count >= 2 else { throw MathExprError.invalidArguments("\(name) requires at least 2 arguments") }
            return args.min()!
        case "max":
            guard args.count >= 2 else { throw MathExprError.invalidArguments("\(name) requires at least 2 arguments") }
            return args.max()!
        case "clamp":
            guard args.count == 3 else { throw MathExprError.invalidArguments("\(name) requires 3 arguments") }
            return Swift.min(Swift.max(args[0], args[1]), args[2])
        case "lerp":
            guard args.count == 3 else { throw MathExprError.invalidArguments("\(name) requires 3 arguments") }
            return args[0] + (args[1] - args[0]) * args[2]

        // Angle conversion
        case "rad":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return args[0] * .pi / 180.0
        case "deg":
            guard args.count == 1 else { throw MathExprError.invalidArguments("\(name) requires 1 argument") }
            return args[0] * 180.0 / .pi

        default:
            throw MathExprError.unknownFunction(name)
        }
    }

    // MARK: - Convenience Methods

    /// Parse and evaluate an expression string.
    ///
    /// - Parameters:
    ///   - expression: The expression string to evaluate
    ///   - variables: Dictionary of variable name to value
    /// - Returns: The computed result
    /// - Throws: MathExprError if parsing or evaluation fails
    public static func eval(_ expression: String, variables: [String: Double] = [:]) throws -> Double {
        let tokens = try tokenize(expression)
        let ast = try parse(tokens)
        return try evaluate(ast, variables: variables)
    }

    /// Convert AST back to expression string.
    ///
    /// - Parameter ast: The AST to convert
    /// - Returns: Expression string representation
    public static func toString(_ ast: MathExprAST) -> String {
        switch ast {
        case .number(let n):
            if n.isNaN { return "nan" }
            if n.isInfinite { return n > 0 ? "inf" : "-inf" }
            if n == floor(n) { return String(Int(n)) }
            return String(n)

        case .imaginary(let n):
            if n == 1 { return "i" }
            if n == -1 { return "-i" }
            return "\(n)i"

        case .constant(let name):
            return name

        case .variable(let name):
            return name

        case .unary(let op, let operand):
            let operandStr = toString(operand)
            if case .binary = operand {
                return "\(op)(\(operandStr))"
            }
            return "\(op)\(operandStr)"

        case .binary(let op, let left, let right):
            let leftStr = toString(left)
            let rightStr = toString(right)

            let opPrec = precedence[op] ?? 0
            var leftResult = leftStr
            var rightResult = rightStr

            if case .binary(let leftOp, _, _) = left {
                if (precedence[leftOp] ?? 0) < opPrec {
                    leftResult = "(\(leftStr))"
                }
            }

            if case .binary(let rightOp, _, _) = right {
                if (precedence[rightOp] ?? 0) <= opPrec {
                    rightResult = "(\(rightStr))"
                }
            }

            return "\(leftResult) \(op) \(rightResult)"

        case .call(let name, let args):
            let argStrs = args.map { toString($0) }
            return "\(name)(\(argStrs.joined(separator: ", ")))"
        }
    }

    /// Substitute variables in AST with values or expressions.
    ///
    /// - Parameters:
    ///   - ast: The AST to transform
    ///   - substitutions: Dictionary of variable name to replacement AST
    /// - Returns: New AST with substitutions applied
    public static func substitute(_ ast: MathExprAST, with substitutions: [String: MathExprAST]) -> MathExprAST {
        switch ast {
        case .number, .imaginary, .constant:
            return ast

        case .variable(let name):
            return substitutions[name] ?? ast

        case .unary(let op, let operand):
            return .unary(op: op, operand: substitute(operand, with: substitutions))

        case .binary(let op, let left, let right):
            return .binary(op: op,
                          left: substitute(left, with: substitutions),
                          right: substitute(right, with: substitutions))

        case .call(let name, let args):
            return .call(name: name, args: args.map { substitute($0, with: substitutions) })
        }
    }

    /// Find all variable names in an AST.
    ///
    /// - Parameter ast: The AST to analyze
    /// - Returns: Set of variable names
    public static func findVariables(in ast: MathExprAST) -> Set<String> {
        var result: Set<String> = []
        collectVariables(ast, into: &result)
        return result
    }

    private static func collectVariables(_ ast: MathExprAST, into result: inout Set<String>) {
        switch ast {
        case .variable(let name):
            result.insert(name)
        case .unary(_, let operand):
            collectVariables(operand, into: &result)
        case .binary(_, let left, let right):
            collectVariables(left, into: &result)
            collectVariables(right, into: &result)
        case .call(_, let args):
            for arg in args {
                collectVariables(arg, into: &result)
            }
        default:
            break
        }
    }
}
