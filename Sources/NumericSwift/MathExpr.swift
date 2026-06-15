//
//  MathExpr.swift
//  NumericSwift
//
//  Mathematical expression parser and evaluation public API.
//
//  Two parse backends are supported:
//    • MathLex (Rust crate, opt-in via NUMERICSWIFT_INCLUDE_MATHLEX=1):
//        full grammar including LaTeX; compile flag NUMERICSWIFT_MATHLEX
//    • Pure-Swift fallback (default): hand-rolled tokenizer + shunting-yard
//        in MathExprFallbackParser.swift; covers standard arithmetic/function
//        grammar. LaTeX is NOT available in the fallback mode.
//
//  The evaluation layer (evaluate, evaluateComplex, etc.) is shared and
//  independent of the parse backend.
//
//  File layout:
//    MathExpr.swift              — this file: error type, parse, public wrappers
//    MathExprLegacy.swift        — legacy scalar + complex oracle evaluators
//    MathExprAST.swift           — AST utilities: variables, substitute, toString, nodeLabel
//    UnifiedEvaluator.swift      — evaluateUnified front door + extract helpers
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

#if NUMERICSWIFT_MATHLEX
  import MathLex
#endif

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
    /// Matrix shape incompatibility detected by the Group-A dispatcher pre-check.
    ///
    /// Thrown before any `LinAlg` precondition can fire, so callers always receive
    /// a recoverable error rather than a process trap. Carries a human-readable
    /// message naming the operator and the two incompatible shapes.
    case shapeMismatch(String)

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
        case .shapeMismatch(let msg):
            return "shape mismatch: \(msg)"
        }
    }
}

// MARK: - Math Expression Parser and Evaluator

/// Mathematical expression parser and evaluator.
///
/// Parsing is handled by the mathlex crate when compiled with
/// `NUMERICSWIFT_INCLUDE_MATHLEX=1`, otherwise uses the built-in pure-Swift
/// shunting-yard parser (standard arithmetic grammar, no LaTeX).
/// Evaluation walks the decoded `MathLexExpression` AST to compute
/// `Double`, `Complex`, or `NumericValue` results.
public struct MathExpr {

    // MARK: - Parsing

    /// Parse a plain-text mathematical expression into an AST.
    ///
    /// When compiled with `NUMERICSWIFT_INCLUDE_MATHLEX=1`, delegates to the
    /// mathlex Rust crate for full grammar support. Otherwise uses the built-in
    /// pure-Swift shunting-yard parser (standard arithmetic grammar, no LaTeX).
    ///
    /// - Parameter expression: The expression string to parse
    /// - Returns: The decoded AST
    /// - Throws: `MathExprError.parseError` if the input is invalid
    public static func parse(_ expression: String) throws -> MathLexExpression {
        #if NUMERICSWIFT_MATHLEX
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
              throw MathExprError.parseError(
                  "JSON serialization failed: \(error.localizedDescription)")
          }
          guard let data = json.data(using: .utf8) else {
              throw MathExprError.parseError("JSON encoding failed")
          }
          do {
              return try JSONDecoder().decode(MathLexExpression.self, from: data)
          } catch {
              throw MathExprError.parseError("AST decode failed: \(error.localizedDescription)")
          }
        #else
          return try MathExprFallbackParser.parseExpression(expression)
        #endif
    }

    /// Parse a LaTeX mathematical expression into an AST.
    ///
    /// - Parameter latex: The LaTeX expression string to parse
    /// - Returns: The decoded AST
    /// - Throws: `MathExprError.parseError` if the input is invalid
    /// - Note: LaTeX parsing requires the MathLex backend
    ///   (`NUMERICSWIFT_INCLUDE_MATHLEX=1`). Without it this method always
    ///   throws `.parseError("LaTeX parsing requires the MathLex backend")`.
    public static func parseLatex(_ latex: String) throws -> MathLexExpression {
        #if NUMERICSWIFT_MATHLEX
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
              throw MathExprError.parseError(
                  "JSON serialization failed: \(error.localizedDescription)")
          }
          guard let data = json.data(using: .utf8) else {
              throw MathExprError.parseError("JSON encoding failed")
          }
          do {
              return try JSONDecoder().decode(MathLexExpression.self, from: data)
          } catch {
              throw MathExprError.parseError("AST decode failed: \(error.localizedDescription)")
          }
        #else
          _ = latex
          throw MathExprError.parseError(
              "LaTeX parsing requires the MathLex backend (NUMERICSWIFT_INCLUDE_MATHLEX=1)")
        #endif
    }

    // MARK: - Real Evaluation (public — delegates to unified evaluator)

    /// Evaluate an AST with given variable bindings.
    ///
    /// Delegates to `MathExpr.evaluateUnified` and unwraps the result to
    /// `Double`. The unified evaluator is the single recursive pass introduced
    /// in Phase 3; this wrapper maintains the original public signature for
    /// full backward compatibility with existing callers (e.g. LuaSwift).
    ///
    /// The legacy scalar implementation is retained internally as
    /// `legacyScalarEvaluate` so the frozen-snapshot oracle can regenerate
    /// from the true legacy path without going through the unified evaluator
    /// (preventing the vacuous-gate bug). See `MathExprLegacy.swift`.
    ///
    /// - Parameters:
    ///   - ast: The AST to evaluate
    ///   - variables: Dictionary of variable name to value
    /// - Returns: The computed result
    /// - Throws: `MathExprError` if evaluation fails
    public static func evaluate(
        _ ast: MathLexExpression, variables: [String: Double] = [:]
    ) throws -> Double {
        // Bridge [String: Double] → [String: NumericValue] for the unified front door.
        let values = variables.mapValues { NumericValue.scalar($0) }
        let result = try evaluateUnified(ast, values: values)
        return try extractDouble(result)
    }

    /// Parse and evaluate an expression string.
    ///
    /// - Parameters:
    ///   - expression: The expression string to evaluate
    ///   - variables: Dictionary of variable name to value
    /// - Returns: The computed result
    /// - Throws: `MathExprError` if parsing or evaluation fails
    public static func eval(_ expression: String, variables: [String: Double] = [:]) throws -> Double {
        let ast = try parse(expression)
        return try evaluate(ast, variables: variables)
    }

    // MARK: - Complex Evaluation (public — delegates to unified evaluator)

    /// Evaluate an AST to a complex number.
    ///
    /// Delegates to `MathExpr.evaluateUnified` and unwraps the result to
    /// `Complex`. Supports the same operators and functions as the real
    /// evaluator, plus imaginary number handling via the `i` constant.
    ///
    /// The legacy complex implementation is retained internally as
    /// `legacyComplexEvaluate` for the snapshot oracle — see
    /// `legacyScalarEvaluate` for the rationale. See `MathExprLegacy.swift`.
    public static func evaluateComplex(
        _ ast: MathLexExpression,
        variables: [String: Double] = [:],
        complexVariables: [String: Complex] = [:]
    ) throws -> Complex {
        // Bridge both variable dictionaries to [String: NumericValue].
        // Complex variables take precedence over real ones for the same key.
        var values: [String: NumericValue] = variables.mapValues { .scalar($0) }
        for (key, z) in complexVariables {
            values[key] = .complex(z)
        }
        let result = try evaluateUnified(ast, values: values)
        return try extractComplex(result)
    }
}
