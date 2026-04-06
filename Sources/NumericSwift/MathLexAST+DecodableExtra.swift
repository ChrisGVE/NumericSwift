//
//  MathLexAST+DecodableExtra.swift
//  NumericSwift
//
//  Collections, linear algebra, set theory, and tensor decode helpers.
//

import Foundation

extension MathLexExpression {

  // MARK: Collections, equations, and logic

  static func decodeCollectionsAndRelations(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.Vector) {
      return .vector(try c.decode([MathLexExpression].self, forKey: .Vector))
    }
    if c.contains(.Matrix) {
      return .matrix(try c.decode([[MathLexExpression]].self, forKey: .Matrix))
    }
    if c.contains(.Equation) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Equation)
      return .equation(left: p.left, right: p.right)
    }
    if c.contains(.Inequality) {
      struct P: Decodable {
        let op: InequalityOp
        let left, right: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Inequality)
      return .inequality(op: p.op, left: p.left, right: p.right)
    }
    if c.contains(.ForAll) {
      struct P: Decodable {
        let variable: String
        let domain: MathLexExpression?
        let body: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .ForAll)
      return .forAll(variable: p.variable, domain: p.domain, body: p.body)
    }
    if c.contains(.Exists) {
      struct P: Decodable {
        let variable: String
        let domain: MathLexExpression?
        let body: MathLexExpression
        let unique: Bool
      }
      let p = try c.decode(P.self, forKey: .Exists)
      return .exists(variable: p.variable, domain: p.domain, body: p.body, unique: p.unique)
    }
    if c.contains(.Logical) {
      struct P: Decodable {
        let op: LogicalOp
        let operands: [MathLexExpression]
      }
      let p = try c.decode(P.self, forKey: .Logical)
      return .logical(op: p.op, operands: p.operands)
    }
    return nil
  }

  // MARK: Linear algebra

  static func decodeLinearAlgebra(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.MarkedVector) {
      struct P: Decodable {
        let name: String
        let notation: VectorNotation
      }
      let p = try c.decode(P.self, forKey: .MarkedVector)
      return .markedVector(name: p.name, notation: p.notation)
    }
    if c.contains(.DotProduct) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .DotProduct)
      return .dotProduct(left: p.left, right: p.right)
    }
    if c.contains(.CrossProduct) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .CrossProduct)
      return .crossProduct(left: p.left, right: p.right)
    }
    if c.contains(.OuterProduct) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .OuterProduct)
      return .outerProduct(left: p.left, right: p.right)
    }
    if c.contains(.Gradient) {
      struct P: Decodable { let expr: MathLexExpression }
      return .gradient(expr: try c.decode(P.self, forKey: .Gradient).expr)
    }
    if c.contains(.Divergence) {
      struct P: Decodable { let field: MathLexExpression }
      return .divergence(field: try c.decode(P.self, forKey: .Divergence).field)
    }
    if c.contains(.Curl) {
      struct P: Decodable { let field: MathLexExpression }
      return .curl(field: try c.decode(P.self, forKey: .Curl).field)
    }
    if c.contains(.Laplacian) {
      struct P: Decodable { let expr: MathLexExpression }
      return .laplacian(expr: try c.decode(P.self, forKey: .Laplacian).expr)
    }
    if c.contains(.Determinant) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .determinant(matrix: try c.decode(P.self, forKey: .Determinant).matrix)
    }
    if c.contains(.Trace) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .trace(matrix: try c.decode(P.self, forKey: .Trace).matrix)
    }
    if c.contains(.Rank) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .rank(matrix: try c.decode(P.self, forKey: .Rank).matrix)
    }
    if c.contains(.ConjugateTranspose) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .conjugateTranspose(
        matrix: try c.decode(P.self, forKey: .ConjugateTranspose).matrix)
    }
    if c.contains(.MatrixInverse) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .matrixInverse(matrix: try c.decode(P.self, forKey: .MatrixInverse).matrix)
    }
    return nil
  }

  // MARK: Set theory

  static func decodeSetTheory(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.SetOperation) {
      struct P: Decodable {
        let op: SetOp
        let left, right: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .SetOperation)
      return .setOperation(op: p.op, left: p.left, right: p.right)
    }
    if c.contains(.SetRelationExpr) {
      struct P: Decodable {
        let relation: SetRelation
        let element, set: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .SetRelationExpr)
      return .setRelationExpr(relation: p.relation, element: p.element, set: p.set)
    }
    if c.contains(.SetBuilder) {
      struct P: Decodable {
        let variable: String
        let domain: MathLexExpression?
        let predicate: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .SetBuilder)
      return .setBuilder(variable: p.variable, domain: p.domain, predicate: p.predicate)
    }
    if c.contains(.PowerSet) {
      struct P: Decodable { let set: MathLexExpression }
      return .powerSet(set: try c.decode(P.self, forKey: .PowerSet).set)
    }
    return nil
  }

  // MARK: Tensors, differential forms, and function algebra

  static func decodeTensorsAndForms(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.Tensor) {
      struct P: Decodable {
        let name: String
        let indices: [TensorIndex]
      }
      let p = try c.decode(P.self, forKey: .Tensor)
      return .tensor(name: p.name, indices: p.indices)
    }
    if c.contains(.KroneckerDelta) {
      struct P: Decodable { let indices: [TensorIndex] }
      return .kroneckerDelta(indices: try c.decode(P.self, forKey: .KroneckerDelta).indices)
    }
    if c.contains(.LeviCivita) {
      struct P: Decodable { let indices: [TensorIndex] }
      return .leviCivita(indices: try c.decode(P.self, forKey: .LeviCivita).indices)
    }
    if c.contains(.FunctionSignature) {
      struct P: Decodable {
        let name: String
        let domain, codomain: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .FunctionSignature)
      return .functionSignature(name: p.name, domain: p.domain, codomain: p.codomain)
    }
    if c.contains(.Composition) {
      struct P: Decodable { let outer, inner: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Composition)
      return .composition(outer: p.outer, inner: p.inner)
    }
    if c.contains(.Differential) {
      struct P: Decodable { let `var`: String }
      return .differential(variable: try c.decode(P.self, forKey: .Differential).var)
    }
    if c.contains(.WedgeProduct) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .WedgeProduct)
      return .wedgeProduct(left: p.left, right: p.right)
    }
    if c.contains(.Relation) {
      struct P: Decodable {
        let op: RelationOp
        let left, right: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Relation)
      return .relation(op: p.op, left: p.left, right: p.right)
    }
    return nil
  }
}
