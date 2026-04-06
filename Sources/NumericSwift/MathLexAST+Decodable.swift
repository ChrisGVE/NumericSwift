//
//  MathLexAST+Decodable.swift
//  NumericSwift
//
//  JSONDecoder conformance for MathLexExpression.
//

import Foundation

// MARK: - CodingKey for MathLexExpression

/// Coding keys for all MathLexExpression variants.
enum MathLexTopKey: String, CodingKey {
  case Integer, Float, Variable, Constant
  case Rational, Complex, Quaternion
  case Binary, Unary, Function
  case Derivative, PartialDerivative, Integral, MultipleIntegral
  case ClosedIntegral, Limit, Sum, Product
  case Vector, Matrix
  case Equation, Inequality
  case ForAll, Exists, Logical
  case MarkedVector, DotProduct, CrossProduct, OuterProduct
  case Gradient, Divergence, Curl, Laplacian, Nabla
  case Determinant, Trace, Rank, ConjugateTranspose, MatrixInverse
  case NumberSetExpr, SetOperation, SetRelationExpr, SetBuilder
  case EmptySet, PowerSet
  case Tensor, KroneckerDelta, LeviCivita
  case FunctionSignature, Composition, Differential, WedgeProduct
  case Relation
}

// MARK: - Decodable

extension MathLexExpression {

  public init(from decoder: Decoder) throws {
    // Unit variants arrive as bare JSON strings.
    if let str = try? decoder.singleValueContainer().decode(String.self) {
      switch str {
      case "Nabla":
        self = .nabla
        return
      case "EmptySet":
        self = .emptySet
        return
      default: break
      }
    }
    let c = try decoder.container(keyedBy: MathLexTopKey.self)
    if let result = try Self.decodeLiterals(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeArithmetic(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeCalculus(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeCollectionsAndRelations(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeLinearAlgebra(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeSetTheory(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeTensorsAndForms(from: c) {
      self = result
      return
    }
    throw DecodingError.dataCorrupted(
      DecodingError.Context(
        codingPath: decoder.codingPath,
        debugDescription: "Unknown MathLexExpression variant"
      ))
  }

  // MARK: Literals

  static func decodeLiterals(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if let v = try c.decodeIfPresent(Int64.self, forKey: .Integer) {
      return .integer(v)
    }
    if c.contains(.Float) {
      let v = try c.decodeIfPresent(Double.self, forKey: .Float)
      return .float(v)
    }
    if let v = try c.decodeIfPresent(String.self, forKey: .Variable) {
      return .variable(v)
    }
    if let v = try c.decodeIfPresent(MathLexConstant.self, forKey: .Constant) {
      return .constant(v)
    }
    if let v = try c.decodeIfPresent(NumberSet.self, forKey: .NumberSetExpr) {
      return .numberSetExpr(v)
    }
    return nil
  }

  // MARK: Arithmetic

  static func decodeArithmetic(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.Rational) {
      struct P: Decodable { let numerator, denominator: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Rational)
      return .rational(numerator: p.numerator, denominator: p.denominator)
    }
    if c.contains(.Complex) {
      struct P: Decodable { let real, imaginary: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Complex)
      return .complex(real: p.real, imaginary: p.imaginary)
    }
    if c.contains(.Quaternion) {
      struct P: Decodable { let real, i, j, k: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Quaternion)
      return .quaternion(real: p.real, i: p.i, j: p.j, k: p.k)
    }
    if c.contains(.Binary) {
      struct P: Decodable {
        let op: BinaryOp
        let left, right: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Binary)
      return .binary(op: p.op, left: p.left, right: p.right)
    }
    if c.contains(.Unary) {
      struct P: Decodable {
        let op: UnaryOp
        let operand: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Unary)
      return .unary(op: p.op, operand: p.operand)
    }
    if c.contains(.Function) {
      struct P: Decodable {
        let name: String
        let args: [MathLexExpression]
      }
      let p = try c.decode(P.self, forKey: .Function)
      return .function(name: p.name, args: p.args)
    }
    return nil
  }

  // MARK: Calculus

  static func decodeCalculus(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.Derivative) {
      struct P: Decodable {
        let expr: MathLexExpression
        let `var`: String
        let order: UInt32
      }
      let p = try c.decode(P.self, forKey: .Derivative)
      return .derivative(expr: p.expr, variable: p.var, order: p.order)
    }
    if c.contains(.PartialDerivative) {
      struct P: Decodable {
        let expr: MathLexExpression
        let `var`: String
        let order: UInt32
      }
      let p = try c.decode(P.self, forKey: .PartialDerivative)
      return .partialDerivative(expr: p.expr, variable: p.var, order: p.order)
    }
    if c.contains(.Integral) {
      struct P: Decodable {
        let integrand: MathLexExpression
        let `var`: String
        let bounds: IntegralBounds?
      }
      let p = try c.decode(P.self, forKey: .Integral)
      return .integral(integrand: p.integrand, variable: p.var, bounds: p.bounds)
    }
    if c.contains(.MultipleIntegral) {
      struct P: Decodable {
        let dimension: UInt8
        let integrand: MathLexExpression
        let bounds: MultipleBounds?
        let vars: [String]
      }
      let p = try c.decode(P.self, forKey: .MultipleIntegral)
      return .multipleIntegral(
        dimension: p.dimension, integrand: p.integrand, bounds: p.bounds, vars: p.vars)
    }
    if c.contains(.ClosedIntegral) {
      struct P: Decodable {
        let dimension: UInt8
        let integrand: MathLexExpression
        let surface: String?
        let `var`: String
      }
      let p = try c.decode(P.self, forKey: .ClosedIntegral)
      return .closedIntegral(
        dimension: p.dimension, integrand: p.integrand, surface: p.surface, variable: p.var)
    }
    if c.contains(.Limit) {
      struct P: Decodable {
        let expr: MathLexExpression
        let `var`: String
        let to: MathLexExpression
        let direction: LimitDirection
      }
      let p = try c.decode(P.self, forKey: .Limit)
      return .limit(expr: p.expr, variable: p.var, to: p.to, direction: p.direction)
    }
    if c.contains(.Sum) {
      struct P: Decodable {
        let index: String
        let lower, upper, body: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Sum)
      return .sum(index: p.index, lower: p.lower, upper: p.upper, body: p.body)
    }
    if c.contains(.Product) {
      struct P: Decodable {
        let index: String
        let lower, upper, body: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Product)
      return .product(index: p.index, lower: p.lower, upper: p.upper, body: p.body)
    }
    return nil
  }
}
