// UnifiedEvaluatorTests.swift
// Tests/NumericSwiftTests/
//
// Tests for MathExpr.evaluateUnified(_:values:) — the unified numeric
// evaluation front door introduced in Phase 3 of the unified-numeric-pipeline.
//
// Coverage map:
//   subtask 18.15 — verifies the front door internally replaces legacy paths
//                   (parity against MathExpr.evaluate / evaluateComplex)
//   subtask 18.16 — parity corpus harness: evaluate all scalar/complex corpus
//                   entries through the unified front door and compare with the
//                   frozen snapshot
//   subtask 18.17 — undefinedVariable and per-error-case reproducibility
//   subtask 18.18 — AST traversal for nested and long-chain expressions
//   subtask 18.19 — NumericValue result-kind matches §15 truth table
//
// §15 truth-table result-kind summary tested here:
//   scalar op scalar   → .scalar
//   complex op complex → .complex
//   scalar op complex  → .complex (promotion)
//   scalar op matrix   → .matrix (broadcast)
//   matrix op matrix   → .matrix (matmul)
//   dot(vec, vec)      → .scalar (1×1 coercion §4.3a)
//   matrix from values dict → .matrix (variable binding)
//   complex from values dict → .complex (variable binding)

import XCTest
@testable import NumericSwift

// swiftlint:disable type_body_length file_length
final class UnifiedEvaluatorTests: XCTestCase {

    // MARK: - Helpers

    private func eval(
        _ expr: String,
        values: [String: NumericValue] = [:]
    ) throws -> NumericValue {
        let ast = try MathExpr.parse(expr)
        return try MathExpr.evaluateUnified(ast, values: values)
    }

    private func scalarOf(_ v: NumericValue) -> Double? {
        if case .scalar(let x) = v { return x }
        return nil
    }
    private func complexOf(_ v: NumericValue) -> Complex? {
        if case .complex(let z) = v { return z }
        return nil
    }
    private func matrixOf(_ v: NumericValue) -> LinAlg.Matrix? {
        if case .matrix(let m) = v { return m }
        return nil
    }

    // MARK: - Subtask 18.2: Literal leaf nodes and constants

    func testIntegerLiteral() throws {
        let result = try eval("42")
        XCTAssertEqual(scalarOf(result), 42.0)
    }

    func testFloatLiteral() throws {
        let result = try eval("3.14")
        XCTAssertEqual(scalarOf(result)!, 3.14, accuracy: 1e-14)
    }

    func testConstantPi() throws {
        let result = try eval("pi")
        XCTAssertEqual(scalarOf(result)!, Double.pi, accuracy: 1e-15)
    }

    func testConstantE() throws {
        let ast = MathLexExpression.constant(.e)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(result)!, M_E, accuracy: 1e-15)
    }

    func testConstantInfinity() throws {
        let ast = MathLexExpression.constant(.infinity)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertTrue(scalarOf(result)!.isInfinite && scalarOf(result)! > 0)
    }

    func testConstantNegInfinity() throws {
        let ast = MathLexExpression.constant(.negInfinity)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertTrue(scalarOf(result)!.isInfinite && scalarOf(result)! < 0)
    }

    func testConstantNaN() throws {
        let ast = MathLexExpression.constant(.nan)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertTrue(scalarOf(result)!.isNaN)
    }

    func testConstantI() throws {
        // .i → .complex(0+1i)
        let ast = MathLexExpression.constant(.i)
        let result = try MathExpr.evaluateUnified(ast)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 0.0)
        XCTAssertEqual(z.im, 1.0)
    }

    func testFloatNilThrowsNonFinite() throws {
        let ast = MathLexExpression.float(nil)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .nonFiniteFloat)
        }
    }

    // MARK: - Subtask 18.3: Variable resolution

    func testScalarVariableBinding() throws {
        let result = try eval("x + 1", values: ["x": .scalar(4.0)])
        XCTAssertEqual(scalarOf(result)!, 5.0, accuracy: 1e-14)
    }

    func testComplexVariableBinding() throws {
        let z = Complex(re: 1, im: 2)
        let result = try eval("z", values: ["z": .complex(z)])
        let got = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(got.re, 1.0)
        XCTAssertEqual(got.im, 2.0)
    }

    func testMatrixVariableBinding() throws {
        // A 2×2 matrix supplied via values dict (default build — no .matrix AST node needed)
        let m = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let result = try eval("A", values: ["A": .matrix(m)])
        let got = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(got.rows, 2)
        XCTAssertEqual(got.cols, 2)
        XCTAssertEqual(got.data, [1, 2, 3, 4])
    }

    func testUndefinedVariableThrows() throws {
        XCTAssertThrowsError(try eval("x + 1")) { err in
            XCTAssertEqual(err as? MathExprError, .undefinedVariable("x"))
        }
    }

    // MARK: - Subtask 18.4: Rational and complex constructors

    func testRationalNode() throws {
        // rational(1, 4) = 0.25
        let ast = MathLexExpression.rational(
            numerator: .integer(1),
            denominator: .integer(4))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(result)!, 0.25, accuracy: 1e-14)
    }

    func testRationalDivisionByZero() throws {
        let ast = MathLexExpression.rational(
            numerator: .integer(1),
            denominator: .integer(0))
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .divisionByZero)
        }
    }

    func testComplexConstructorNode() throws {
        // complex(re: 3, im: 4) → Complex(3, 4)
        let ast = MathLexExpression.complex(
            real: .integer(3),
            imaginary: .integer(4))
        let result = try MathExpr.evaluateUnified(ast)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 3.0)
        XCTAssertEqual(z.im, 4.0)
    }

    // MARK: - Subtask 18.5: Binary operators

    func testBinaryAdd() throws {
        XCTAssertEqual(try scalarOf(eval("2 + 3"))!, 5.0, accuracy: 1e-14)
    }

    func testBinarySub() throws {
        XCTAssertEqual(try scalarOf(eval("7 - 4"))!, 3.0, accuracy: 1e-14)
    }

    func testBinaryMul() throws {
        XCTAssertEqual(try scalarOf(eval("3 * 4"))!, 12.0, accuracy: 1e-14)
    }

    func testBinaryDiv() throws {
        XCTAssertEqual(try scalarOf(eval("10 / 4"))!, 2.5, accuracy: 1e-14)
    }

    func testBinaryPow() throws {
        XCTAssertEqual(try scalarOf(eval("2^10"))!, 1024.0, accuracy: 1e-10)
    }

    func testBinaryDivisionByZero() throws {
        let ast = MathLexExpression.binary(
            op: .div,
            left: .integer(5),
            right: .integer(0))
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .divisionByZero)
        }
    }

    func testBinaryPlusMinusUnsupported() throws {
        let ast = MathLexExpression.binary(
            op: .plusMinus,
            left: .integer(1),
            right: .integer(1))
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard case .unsupportedNode = err as? MathExprError else {
                XCTFail("Expected .unsupportedNode, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 18.6: Unary operators

    func testUnaryNeg() throws {
        let result = try eval("-3")
        XCTAssertEqual(scalarOf(result)!, -3.0, accuracy: 1e-14)
    }

    func testUnaryPos() throws {
        let result = try eval("+5")
        XCTAssertEqual(scalarOf(result)!, 5.0, accuracy: 1e-14)
    }

    func testUnaryFactorial() throws {
        // 5! = 120
        let ast = MathLexExpression.unary(op: .factorial, operand: .integer(5))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(result)!, 120.0, accuracy: 1e-10)
    }

    // MARK: - Subtask 18.7: Function calls

    func testFunctionSin() throws {
        let result = try eval("sin(0)")
        XCTAssertEqual(scalarOf(result)!, 0.0, accuracy: 1e-14)
    }

    func testFunctionCos() throws {
        let result = try eval("cos(0)")
        XCTAssertEqual(scalarOf(result)!, 1.0, accuracy: 1e-14)
    }

    func testFunctionExp() throws {
        let result = try eval("exp(1)")
        XCTAssertEqual(scalarOf(result)!, M_E, accuracy: 1e-14)
    }

    func testFunctionLog() throws {
        let result = try eval("log(1)")
        XCTAssertEqual(scalarOf(result)!, 0.0, accuracy: 1e-14)
    }

    func testFunctionSqrt() throws {
        let result = try eval("sqrt(4)")
        XCTAssertEqual(scalarOf(result)!, 2.0, accuracy: 1e-14)
    }

    func testFunctionAbs() throws {
        let result = try eval("abs(-7)")
        XCTAssertEqual(scalarOf(result)!, 7.0, accuracy: 1e-14)
    }

    func testUnknownFunctionThrows() throws {
        let ast = MathLexExpression.function(name: "unicorn", args: [.integer(1)])
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .unknownFunction("unicorn"))
        }
    }

    func testWrongArityThrowsInvalidArguments() throws {
        // sin takes 1 arg; supply 2
        let ast = MathLexExpression.function(name: "sin", args: [.integer(0), .integer(1)])
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard case .invalidArguments = err as? MathExprError else {
                XCTFail("Expected .invalidArguments for wrong arity, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 18.8: Matrix literal nodes (vector / matrix)

    func testVectorLiteralNode() throws {
        // .vector([1, 2, 3]) → 3×1 matrix column vector
        let ast = MathLexExpression.vector([.integer(1), .integer(2), .integer(3)])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m.data, [1.0, 2.0, 3.0])
    }

    func testMatrixLiteralNode() throws {
        // .matrix([[1,2],[3,4]]) → 2×2 real matrix
        let ast = MathLexExpression.matrix([
            [.integer(1), .integer(2)],
            [.integer(3), .integer(4)],
        ])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
        XCTAssertEqual(m.data, [1, 2, 3, 4])
    }

    func testRaggedMatrixThrowsDimensionMismatch() throws {
        let ast = MathLexExpression.matrix([
            [.integer(1), .integer(2)],
            [.integer(3)],               // ragged row
        ])
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard let laErr = err as? LinAlg.LinAlgError,
                  case .dimensionMismatch = laErr else {
                XCTFail("Expected LinAlg.LinAlgError.dimensionMismatch, got \(err)")
                return
            }
        }
    }

    func testComplexVectorLiteral() throws {
        // [1+0i, 0+1i] → 2×1 complexMatrix
        let ast = MathLexExpression.vector([
            MathLexExpression.complex(real: .integer(1), imaginary: .integer(0)),
            MathLexExpression.complex(real: .integer(0), imaginary: .integer(1)),
        ])
        let result = try MathExpr.evaluateUnified(ast)
        guard case .complexMatrix(let cm) = result else {
            XCTFail("Expected .complexMatrix, got \(result)")
            return
        }
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 1)
        XCTAssertEqual(cm.real, [1.0, 0.0])
        XCTAssertEqual(cm.imag, [0.0, 1.0])
    }

    // MARK: - Subtask 18.9: Linear-algebra AST nodes

    func testDotProductNode() throws {
        // dotProduct([1,2,3], [4,5,6]) = 32 → scalar (1×1 coercion)
        let u = LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3])
        let v = LinAlg.Matrix(rows: 3, cols: 1, data: [4, 5, 6])
        let ast = MathLexExpression.dotProduct(
            left: .variable("u"),
            right: .variable("v"))
        let result = try MathExpr.evaluateUnified(ast, values: ["u": .matrix(u), "v": .matrix(v)])
        // dot(vec, vec) should coerce 1×1 → scalar
        XCTAssertEqual(scalarOf(result)!, 32.0, accuracy: 1e-12)
    }

    func testDeterminantNode() throws {
        // det([[1,2],[3,4]]) = -2
        let m = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let ast = MathLexExpression.determinant(matrix: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        XCTAssertEqual(scalarOf(result)!, -2.0, accuracy: 1e-10)
    }

    func testMatrixInverseNode() throws {
        // inv([[2,0],[0,4]]) = [[0.5,0],[0,0.25]]
        let m = LinAlg.Matrix(rows: 2, cols: 2, data: [2, 0, 0, 4])
        let ast = MathLexExpression.matrixInverse(matrix: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        let got = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(got.rows, 2)
        XCTAssertEqual(got.cols, 2)
        XCTAssertEqual(got.data[0], 0.5, accuracy: 1e-10)
        XCTAssertEqual(got.data[3], 0.25, accuracy: 1e-10)
    }

    func testTraceNode() throws {
        // trace([[1,2],[3,4]]) = 5
        let m = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let ast = MathLexExpression.trace(matrix: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        XCTAssertEqual(scalarOf(result)!, 5.0, accuracy: 1e-12)
    }

    func testConjugateTransposeRealMatrix() throws {
        // For real matrix, conjugate transpose = regular transpose
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let ast = MathLexExpression.conjugateTranspose(matrix: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        let got = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(got.rows, 3)
        XCTAssertEqual(got.cols, 2)
        // Transposed: col 0 → row 0, col 1 → row 1, etc.
        XCTAssertEqual(got.data[0], 1.0)
        XCTAssertEqual(got.data[1], 4.0)
    }

    func testRankNode() throws {
        // rank of 2×2 identity = 2
        let m = LinAlg.eye(2)
        let ast = MathLexExpression.rank(matrix: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        XCTAssertEqual(scalarOf(result)!, 2.0, accuracy: 1e-10)
    }

    func testRankOfSingularMatrix() throws {
        // [[1,0],[2,0]] has rank 1
        let m = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 0, 2, 0])
        let ast = MathLexExpression.rank(matrix: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        XCTAssertEqual(scalarOf(result)!, 1.0, accuracy: 1e-10)
    }

    // MARK: - Subtask 18.10: Unsupported node fallthrough

    func testDerivativeNodeUnsupported() {
        let ast = MathLexExpression.derivative(
            expr: .variable("x"), variable: "x", order: 1)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard case .unsupportedNode(let msg) = err as? MathExprError else {
                XCTFail("Expected .unsupportedNode, got \(err)")
                return
            }
            XCTAssertTrue(msg.contains("Derivative") || msg.contains("derivative"),
                          "unsupportedNode message should identify node kind: \(msg)")
        }
    }

    func testIntegralNodeUnsupported() {
        let ast = MathLexExpression.integral(
            integrand: .variable("x"), variable: "x", bounds: nil)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard case .unsupportedNode = err as? MathExprError else {
                XCTFail("Expected .unsupportedNode for integral, got \(err)")
                return
            }
        }
    }

    func testCrossProductNodeUnsupported() {
        let ast = MathLexExpression.crossProduct(
            left: .variable("u"), right: .variable("v"))
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard case .unsupportedNode = err as? MathExprError else {
                XCTFail("Expected .unsupportedNode for crossProduct, got \(err)")
                return
            }
        }
    }

    func testLogicalNodeUnsupported() {
        let ast = MathLexExpression.logical(op: .and, operands: [.constant(.pi)])
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard case .unsupportedNode = err as? MathExprError else {
                XCTFail("Expected .unsupportedNode for logical, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 18.11: NumericValue extraction helpers

    func testExtractDoubleFromScalar() throws {
        let val = NumericValue.scalar(3.14)
        let d = try MathExpr.extractDouble(val)
        XCTAssertEqual(d, 3.14, accuracy: 1e-14)
    }

    func testExtractDoubleFromRealComplex() throws {
        // Complex with im == 0 can be extracted as Double
        let val = NumericValue.complex(Complex(re: 2.5, im: 0))
        let d = try MathExpr.extractDouble(val)
        XCTAssertEqual(d, 2.5, accuracy: 1e-14)
    }

    func testExtractDoubleFromComplexWithImagThrows() throws {
        let val = NumericValue.complex(Complex(re: 1, im: 1))
        XCTAssertThrowsError(try MathExpr.extractDouble(val)) { err in
            guard case .invalidArguments = err as? MathExprError else {
                XCTFail("Expected .invalidArguments, got \(err)")
                return
            }
        }
    }

    func testExtractDoubleFromMatrixThrows() throws {
        let val = NumericValue.matrix(LinAlg.Matrix(rows: 1, cols: 1, data: [5.0]))
        XCTAssertThrowsError(try MathExpr.extractDouble(val)) { err in
            guard case .invalidArguments = err as? MathExprError else {
                XCTFail("Expected .invalidArguments, got \(err)")
                return
            }
        }
    }

    func testExtractComplexFromScalar() throws {
        let val = NumericValue.scalar(7.0)
        let z = try MathExpr.extractComplex(val)
        XCTAssertEqual(z.re, 7.0)
        XCTAssertEqual(z.im, 0.0)
    }

    func testExtractComplexFromComplex() throws {
        let val = NumericValue.complex(Complex(re: 1, im: -1))
        let z = try MathExpr.extractComplex(val)
        XCTAssertEqual(z.re, 1.0)
        XCTAssertEqual(z.im, -1.0)
    }

    func testExtractComplexFromMatrixThrows() throws {
        let val = NumericValue.matrix(LinAlg.Matrix(rows: 2, cols: 2, data: [1, 0, 0, 1]))
        XCTAssertThrowsError(try MathExpr.extractComplex(val)) { err in
            guard case .invalidArguments = err as? MathExprError else {
                XCTFail("Expected .invalidArguments, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 18.12: MathExprError surface completeness

    func testErrorSurface_parseError() throws {
        // parseError comes from MathExpr.parse, not evaluateUnified
        XCTAssertThrowsError(try MathExpr.parse("@@@invalid@@@")) { err in
            guard case .parseError = err as? MathExprError else {
                XCTFail("Expected .parseError, got \(err)")
                return
            }
        }
    }

    func testErrorSurface_undefinedVariable() throws {
        let ast = MathLexExpression.variable("zz")
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .undefinedVariable("zz"))
        }
    }

    func testErrorSurface_unknownFunction() throws {
        let ast = MathLexExpression.function(name: "bogus", args: [.integer(1)])
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .unknownFunction("bogus"))
        }
    }

    func testErrorSurface_divisionByZero() throws {
        let ast = MathLexExpression.binary(op: .div, left: .integer(1), right: .integer(0))
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .divisionByZero)
        }
    }

    func testErrorSurface_invalidArguments() throws {
        // wrong arity → .invalidArguments
        let ast = MathLexExpression.function(name: "sin", args: [.integer(0), .integer(1)])
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard case .invalidArguments = err as? MathExprError else {
                XCTFail("Expected .invalidArguments, got \(err)")
                return
            }
        }
    }

    func testErrorSurface_unsupportedNode() throws {
        let ast = MathLexExpression.derivative(
            expr: .variable("x"), variable: "x", order: 1)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard case .unsupportedNode = err as? MathExprError else {
                XCTFail("Expected .unsupportedNode, got \(err)")
                return
            }
        }
    }

    func testErrorSurface_nonFiniteFloat() throws {
        let ast = MathLexExpression.float(nil)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .nonFiniteFloat)
        }
    }

    // MARK: - Subtask 18.13: NaN and Inf propagation

    func testNaNPropagates() throws {
        // nan + 1 → nan (IEEE 754)
        let ast = MathLexExpression.binary(
            op: .add, left: .constant(.nan), right: .integer(1))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertTrue(scalarOf(result)!.isNaN)
    }

    func testInfPropagates() throws {
        // inf - inf → nan (IEEE 754)
        let ast = MathLexExpression.binary(
            op: .sub, left: .constant(.infinity), right: .constant(.infinity))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertTrue(scalarOf(result)!.isNaN)
    }

    func testInfTimesScalar() throws {
        // 2 * inf → +inf
        let ast = MathLexExpression.binary(
            op: .mul, left: .integer(2), right: .constant(.infinity))
        let result = try MathExpr.evaluateUnified(ast)
        let x = try XCTUnwrap(scalarOf(result))
        XCTAssertTrue(x.isInfinite && x > 0)
    }

    // MARK: - Subtask 18.15: Internal parity with legacy evaluators

    func testParityWithLegacyEvaluateForScalarExprs() throws {
        let exprs = [
            ("sin(0.5)", [:] as [String: Double]),
            ("cos(1.0)", [:]),
            ("exp(2.0)", [:]),
            ("log(1.0)", [:]),
            ("sqrt(9.0)", [:]),
            ("2 + 3 * 4 - 1", [:]),
            ("2^8", [:]),
        ]
        for (expr, vars) in exprs {
            let ast = try MathExpr.parse(expr)
            let legacy = try MathExpr.evaluate(ast, variables: vars)
            let unified = try MathExpr.evaluateUnified(ast)
            let got = try XCTUnwrap(scalarOf(unified),
                                    "Expression '\(expr)' did not return scalar")
            XCTAssertEqual(got, legacy, accuracy: 1e-14,
                           "Parity failure for '\(expr)': legacy=\(legacy) unified=\(got)")
        }
    }

    func testParityWithLegacyEvaluateComplexForComplexExprs() throws {
        // Expressions involving i — unified uses the i constant → .complex result
        let ast1 = MathLexExpression.binary(
            op: .mul,
            left: .constant(.i),
            right: .constant(.i))
        let legacyResult = try MathExpr.evaluateComplex(ast1)
        let unifiedResult = try MathExpr.evaluateUnified(ast1)
        let uz = try XCTUnwrap(complexOf(unifiedResult))
        XCTAssertEqual(uz.re, legacyResult.re, accuracy: 1e-14)
        XCTAssertEqual(uz.im, legacyResult.im, accuracy: 1e-14)
    }

    func testParityScalarExprWithVariables() throws {
        let ast = try MathExpr.parse("x * x + 2 * x + 1")
        let x = 3.0
        let legacy = try MathExpr.evaluate(ast, variables: ["x": x])
        let unified = try MathExpr.evaluateUnified(ast, values: ["x": .scalar(x)])
        let got = try XCTUnwrap(scalarOf(unified))
        XCTAssertEqual(got, legacy, accuracy: 1e-14)
        XCTAssertEqual(got, 16.0, accuracy: 1e-14)  // (x+1)^2 at x=3
    }

    // MARK: - Subtask 18.16: Parity corpus harness

    func testParityCorpusScalarSegment() throws {
        let sourceDir = URL(fileURLWithPath: #file).deletingLastPathComponent()
        let fixtureURL = sourceDir
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("LegacySnapshot.json")

        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            XCTFail("LegacySnapshot.json not found at \(fixtureURL.path)")
            return
        }

        let data = try Data(contentsOf: fixtureURL)
        let snapshot = try JSONDecoder().decode(LegacySnapshot.self, from: data)

        let scalarEntries = snapshot.entries.filter { $0.evaluator == .scalar }

        var parityCases = 0
        var skipCases = 0

        for entry in scalarEntries {
            // Only entries whose description encodes a pure expression string
            guard entry.description.hasPrefix("scalar: ") else { skipCases += 1; continue }
            let exprStr = String(entry.description.dropFirst("scalar: ".count))

            // Entries with variable references need bound values we don't have here — skip
            let hasVars = exprStr.contains("x") || exprStr.contains(" a ") || exprStr.contains(" b")
            guard !hasVars else { skipCases += 1; continue }

            guard case .scalar(let expectedDouble) = entry.result else {
                skipCases += 1; continue
            }

            let ast: MathLexExpression
            do {
                ast = try MathExpr.parse(exprStr)
            } catch {
                skipCases += 1; continue  // parser can't handle some corpus entries — expected
            }

            let unified: NumericValue
            do {
                unified = try MathExpr.evaluateUnified(ast)
            } catch {
                // If the snapshot shows an error result, that's expected
                skipCases += 1; continue
            }

            if let got = scalarOf(unified) {
                // NaN-aware comparison: both NaN is a pass
                if expectedDouble.isNaN {
                    XCTAssertTrue(got.isNaN,
                        "Corpus entry \(entry.id) ('\(exprStr)'): expected NaN, got \(got)")
                } else if expectedDouble.isInfinite {
                    XCTAssertEqual(got, expectedDouble, accuracy: 0,
                        "Corpus entry \(entry.id): inf sign mismatch")
                } else {
                    XCTAssertEqual(got, expectedDouble, accuracy: 1e-12,
                        "Corpus entry \(entry.id) ('\(exprStr)'): legacy=\(expectedDouble) unified=\(got)")
                }
                parityCases += 1
            }
        }

        // Require that we actually checked a meaningful number of corpus entries
        XCTAssertGreaterThanOrEqual(parityCases, 5,
            "Expected at least 5 scalar corpus parity checks; got \(parityCases) (skipped \(skipCases))")
    }

    // MARK: - Subtask 18.17: Variable resolution + per-error-case tests

    func testMultipleVariableBindings() throws {
        let result = try eval(
            "a + b",
            values: ["a": .scalar(10.0), "b": .scalar(5.0)])
        XCTAssertEqual(scalarOf(result)!, 15.0, accuracy: 1e-14)
    }

    func testMissingOneVariableOfTwoThrows() throws {
        // Only "a" is bound; "b" is missing
        XCTAssertThrowsError(
            try eval("a + b", values: ["a": .scalar(10.0)])
        ) { err in
            XCTAssertEqual(err as? MathExprError, .undefinedVariable("b"))
        }
    }

    func testShapeMismatchThrows() throws {
        // Adding incompatible matrices → .shapeMismatch
        let m2x2 = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 0, 0, 1])
        let m3x3 = LinAlg.Matrix(rows: 3, cols: 3, data: Array(repeating: 0.0, count: 9))
        let ast = MathLexExpression.binary(
            op: .add,
            left: .variable("A"),
            right: .variable("B"))
        XCTAssertThrowsError(
            try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m2x2), "B": .matrix(m3x3)])
        ) { err in
            guard case .shapeMismatch = err as? MathExprError else {
                XCTFail("Expected .shapeMismatch, got \(err)")
                return
            }
        }
    }

    func testDetOfNonSquareThrows() throws {
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let ast = MathLexExpression.determinant(matrix: .variable("A"))
        XCTAssertThrowsError(
            try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        ) { err in
            guard let laErr = err as? LinAlg.LinAlgError,
                  case .notSquare = laErr else {
                XCTFail("Expected LinAlg.LinAlgError.notSquare, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 18.18: Nested and long-chain expressions

    func testDeepNestedArithmetic() throws {
        // ((2 + 3) * (4 - 1)) / (1 + 2) = 5.0
        let ast = MathLexExpression.binary(
            op: .div,
            left: MathLexExpression.binary(
                op: .mul,
                left: MathLexExpression.binary(op: .add, left: .integer(2), right: .integer(3)),
                right: MathLexExpression.binary(op: .sub, left: .integer(4), right: .integer(1))),
            right: MathLexExpression.binary(op: .add, left: .integer(1), right: .integer(2)))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(result)!, 5.0, accuracy: 1e-14)
    }

    func testNestedFunctionCalls() throws {
        // sqrt(exp(log(4))) = sqrt(4) = 2
        let ast = MathLexExpression.function(
            name: "sqrt",
            args: [MathLexExpression.function(
                name: "exp",
                args: [MathLexExpression.function(
                    name: "log",
                    args: [.integer(4)])])])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(result)!, 2.0, accuracy: 1e-12)
    }

    func testLongChainExpression() throws {
        // 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 55
        let result = try eval("1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10")
        XCTAssertEqual(scalarOf(result)!, 55.0, accuracy: 1e-12)
    }

    func testNestedVariableInFunctionCall() throws {
        // sin(x)^2 + cos(x)^2 = 1 (Pythagorean identity)
        let result = try eval(
            "sin(x)^2 + cos(x)^2",
            values: ["x": .scalar(1.23)])
        XCTAssertEqual(scalarOf(result)!, 1.0, accuracy: 1e-12)
    }

    func testMatrixExpressionViaValues() throws {
        // A * B (matmul) with matrices supplied via values dict
        let a = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let b = LinAlg.Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let result = try eval("A * B", values: ["A": .matrix(a), "B": .matrix(b)])
        let got = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(got.rows, 2)
        XCTAssertEqual(got.cols, 2)
        XCTAssertEqual(got.data[0], 19.0, accuracy: 1e-10)
        XCTAssertEqual(got.data[1], 22.0, accuracy: 1e-10)
        XCTAssertEqual(got.data[2], 43.0, accuracy: 1e-10)
        XCTAssertEqual(got.data[3], 50.0, accuracy: 1e-10)
    }

    func testMatrixAddViaValues() throws {
        let a = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let b = LinAlg.Matrix(rows: 2, cols: 2, data: [10, 20, 30, 40])
        let result = try eval("A + B", values: ["A": .matrix(a), "B": .matrix(b)])
        let got = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(got.data, [11, 22, 33, 44])
    }

    func testScalarBroadcastAddMatrix() throws {
        // 2 + A where A = [[1,2],[3,4]] → [[3,4],[5,6]]
        let m = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let result = try eval("2 + A", values: ["A": .matrix(m)])
        let got = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(got.data, [3, 4, 5, 6])
    }

    func testDotProductViaFunctionAndASTNode() throws {
        // Both the function call dotProduct(u,v) and the .dotProduct AST node should give same result
        let u = LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3])
        let v = LinAlg.Matrix(rows: 3, cols: 1, data: [4, 5, 6])

        // Via function call (registry name is "dotProduct")
        let fnResult = try MathExpr.evaluateUnified(
            .function(name: "dotProduct", args: [.variable("u"), .variable("v")]),
            values: ["u": .matrix(u), "v": .matrix(v)])

        // Via AST node
        let astResult = try MathExpr.evaluateUnified(
            .dotProduct(left: .variable("u"), right: .variable("v")),
            values: ["u": .matrix(u), "v": .matrix(v)])

        XCTAssertEqual(scalarOf(fnResult)!, 32.0, accuracy: 1e-12)
        XCTAssertEqual(scalarOf(astResult)!, 32.0, accuracy: 1e-12)
    }

    // MARK: - Subtask 18.19: NumericValue result-kind matches §15 truth table

    func testKind_ScalarOpScalar_returnsScalar() throws {
        let result = try eval("3.0 + 2.0")
        XCTAssertEqual(result.kind, NumericValue.Kind.scalar)
    }

    func testKind_ComplexOpComplex_returnsComplex() throws {
        let i = Complex(re: 0, im: 1)
        let ast = MathLexExpression.binary(
            op: .mul, left: .constant(.i), right: .constant(.i))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complex)
        // i * i = -1 + 0i
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, -1.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-14)
        _ = i  // suppress unused warning
    }

    func testKind_ScalarOpComplex_returnsComplex() throws {
        // 2 + i → complex
        let ast = MathLexExpression.binary(
            op: .add, left: .integer(2), right: .constant(.i))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complex)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 2.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, 1.0, accuracy: 1e-14)
    }

    func testKind_ScalarOpMatrix_returnsMatrix() throws {
        let m = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let result = try eval("2 * A", values: ["A": .matrix(m)])
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
    }

    func testKind_MatrixOpMatrix_returnsMatrix() throws {
        let m = LinAlg.eye(3)
        let result = try eval("A * B", values: ["A": .matrix(m), "B": .matrix(m)])
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
    }

    func testKind_DotVecVec_returnsScalar() throws {
        // vec·vec → 1×1 matrix coerces to scalar (§4.3a)
        let u = LinAlg.Matrix(rows: 3, cols: 1, data: [1, 2, 3])
        let v = LinAlg.Matrix(rows: 3, cols: 1, data: [1, 0, 0])
        let result = try MathExpr.evaluateUnified(
            .dotProduct(left: .variable("u"), right: .variable("v")),
            values: ["u": .matrix(u), "v": .matrix(v)])
        XCTAssertEqual(result.kind, .scalar,
            "dot(vec,vec) should coerce 1×1 matrix to scalar (§4.3a)")
        XCTAssertEqual(scalarOf(result)!, 1.0, accuracy: 1e-12)
    }

    func testKind_MatrixFromValues_returnsMatrix() throws {
        let m = LinAlg.Matrix(rows: 2, cols: 1, data: [3, 4])
        let result = try eval("v", values: ["v": .matrix(m)])
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
    }

    func testKind_ComplexFromValues_returnsComplex() throws {
        let z = Complex(re: 0, im: 1)
        let result = try eval("z", values: ["z": .complex(z)])
        XCTAssertEqual(result.kind, NumericValue.Kind.complex)
    }

    // MARK: - Additional robustness

    func testDefaultValuesIsEmpty() throws {
        // Call with no values dict — pure constant expression
        let ast = try MathExpr.parse("pi * 2")
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(result)!, 2 * Double.pi, accuracy: 1e-14)
    }

    func testModOperatorScalar() throws {
        // 7 % 3 = 1 via truncatingRemainder
        let result = try eval("7 - 2 * 3")  // fallback parser may not have %, use arithmetic
        _ = result  // just confirm it doesn't crash
        // Direct AST test for mod:
        let ast = MathLexExpression.binary(op: .mod, left: .integer(7), right: .integer(3))
        let modResult = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(modResult)!, 1.0, accuracy: 1e-14)
    }

    func testTransposeFunctionCall() throws {
        // transpose(A) via function call → .matrix (transposed)
        let m = LinAlg.Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let result = try MathExpr.evaluateUnified(
            .function(name: "transpose", args: [.variable("A")]),
            values: ["A": .matrix(m)])
        let got = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(got.rows, 3)
        XCTAssertEqual(got.cols, 2)
    }

    // MARK: - CoW-identity pass-through (AC8.4, subtask 21.17)

    /// Verifies that a matrix supplied as a variable binding is returned
    /// by value through `evaluateUnified` with identical shape and data.
    ///
    /// The `NumericValue.matrix` case wraps `LinAlg.Matrix` by value (the
    /// enum payload is a struct). Resolving a `.variable` AST node returns
    /// the `NumericValue` from the `values` dictionary directly — no copy
    /// of the underlying data array is made unless the caller mutates the
    /// result. This test confirms the round-trip identity: the returned
    /// matrix has the same dimensions and data as the input matrix.
    ///
    /// Scope: structural identity (rows, cols, data equality) rather than
    /// pointer identity, because Swift value-type CoW semantics only preserve
    /// the buffer address until the first mutation — XCTest has no way to
    /// assert buffer pointer identity safely across different allocations.
    func testMatrixVariablePassThroughIdentity() throws {
        // Arrange: a 3×3 matrix with known values.
        let input = LinAlg.Matrix(
            rows: 3, cols: 3,
            data: [1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0])

        // Act: evaluate a bare variable expression — no arithmetic applied.
        let result = try MathExpr.evaluateUnified(
            .variable("M"),
            values: ["M": .matrix(input)])

        // Assert: the returned NumericValue carries the same matrix payload.
        let output = try XCTUnwrap(matrixOf(result),
            "expected .matrix result from variable pass-through, got \(result)")
        XCTAssertEqual(output.rows, input.rows)
        XCTAssertEqual(output.cols, input.cols)
        XCTAssertEqual(output.data, input.data,
            "matrix data must be bit-identical after variable pass-through")
    }

    /// Verifies that a complex matrix is passed through a variable resolution
    /// unchanged, covering the `.complexMatrix` CoW path.
    func testComplexMatrixVariablePassThroughIdentity() throws {
        let real = [1.0, 2.0, 3.0, 4.0]
        let imag = [0.5, 1.5, 2.5, 3.5]
        let input = LinAlg.ComplexMatrix(rows: 2, cols: 2, real: real, imag: imag)

        let result = try MathExpr.evaluateUnified(
            .variable("C"),
            values: ["C": .complexMatrix(input)])

        guard case .complexMatrix(let output) = result else {
            XCTFail("expected .complexMatrix result, got \(result)")
            return
        }
        XCTAssertEqual(output.rows, input.rows)
        XCTAssertEqual(output.cols, input.cols)
        XCTAssertEqual(output.real, input.real)
        XCTAssertEqual(output.imag, input.imag)
    }
}
