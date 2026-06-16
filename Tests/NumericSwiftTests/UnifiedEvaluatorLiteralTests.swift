// UnifiedEvaluatorLiteralTests.swift
// Tests/NumericSwiftTests/
//
// Task 19 — Phase 3: Handle matrix/vector/complex literal AST nodes.
//
// Coverage map (subtasks 19.1–19.19):
//   19.1  — NumericValue mapping table verified by tests below
//   19.2  — literal-node dispatch to UnifiedEvaluatorMatrix confirmed
//   19.3  — .constant node evaluation (pi/e/i/j/inf/negInf/nan)
//   19.4  — .complex(real:imaginary:) literal node → NumericValue.complex
//   19.5  — homogeneity / promotion scan (real/complex element detection)
//   19.6  — real .vector → NumericValue.matrix (column vector)
//   19.7  — complex .vector → NumericValue.complexMatrix
//   19.8  — ragged matrix rows → LinAlgError.dimensionMismatch
//   19.9  — real .matrix → NumericValue.matrix
//   19.10 — complex .matrix → NumericValue.complexMatrix
//   19.11 — nested literals (complex elements inside .vector inside .matrix)
//   19.12 — fallback-parser bracket limitation documented (see PARSER-SCOPE NOTE)
//   19.13 — nodeLabel coverage (no unsupportedNode for handled node kinds)
//   19.14 — conditional-compilation note: tests use direct AST construction;
//            mathlex gating not needed on the default build
//   19.15 — scalar/complex literal node direct-AST tests
//   19.16 — real vector/matrix literal direct-AST tests
//   19.17 — complex vector/matrix literal direct-AST tests
//   19.18 — default-build parseError + imaginary-literal regression
//   19.19 — frozen-snapshot parity for literal-node corpus
//
// ## PARSER-SCOPE NOTE (§4.9 / ARCH-01)
//
// On the default build (no `NUMERICSWIFT_INCLUDE_MATHLEX=1`) the pure-Swift
// fallback parser has NO bracket tokenizer and cannot parse `[1,2,3]` or
// `[[1,2],[3,4]]`. Attempting to do so results in `MathExprError.parseError`
// from the parser — NOT from the evaluator. This is by design: the evaluator
// handles `.vector` / `.matrix` AST nodes correctly; the nodes simply cannot
// be produced by the default parser.
//
// Imaginary literals such as `2i` ARE supported by the fallback tokenizer
// (MathExpr.parse handles the `i` suffix) and continue to work unchanged.
//
// Test approach: construct `.vector` / `.matrix` AST nodes directly in Swift
// and feed them to `MathExpr.evaluateUnified` — this verifies the evaluator
// logic independently of which parser is active. The parse-level limitation
// is covered by `testDefaultBuildBracketLiteralYieldsParseError` below.
//
// Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// swiftlint:disable type_body_length
final class UnifiedEvaluatorLiteralTests: XCTestCase {

    // MARK: - Helpers

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

    private func complexMatrixOf(_ v: NumericValue) -> LinAlg.ComplexMatrix? {
        if case .complexMatrix(let cm) = v { return cm }
        return nil
    }

    // MARK: - Subtask 19.3: .constant node evaluation

    func testConstantPiEvaluatesToScalar() throws {
        let ast = MathLexExpression.constant(.pi)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(result)!, Double.pi, accuracy: 1e-15)
    }

    func testConstantEEvaluatesToScalar() throws {
        let ast = MathLexExpression.constant(.e)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(scalarOf(result)!, M_E, accuracy: 1e-15)
    }

    func testConstantIEvaluatesToComplex() throws {
        // i → .complex(0+1i)
        let ast = MathLexExpression.constant(.i)
        let result = try MathExpr.evaluateUnified(ast)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 0.0, accuracy: 1e-15)
        XCTAssertEqual(z.im, 1.0, accuracy: 1e-15)
        XCTAssertEqual(result.kind, NumericValue.Kind.complex)
    }

    func testConstantJThrows() {
        // `.j` is a quaternion basis element distinct from the imaginary unit,
        // not representable in complex arithmetic. It throws `.unsupportedNode`
        // exactly like `.k` and the legacy complex oracle (CR-D4). It previously
        // aliased to `.complex(0+1i)`, silently wrong for quaternion expressions.
        let ast = MathLexExpression.constant(.j)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { error in
            guard case MathExprError.unsupportedNode(let msg) = error else {
                return XCTFail("expected .unsupportedNode, got \(error)")
            }
            XCTAssertEqual(msg, "quaternion basis requires quaternion arithmetic")
        }
    }

    func testConstantKThrows() {
        // `.k` shares the same quaternion-basis error as `.j`.
        let ast = MathLexExpression.constant(.k)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { error in
            guard case MathExprError.unsupportedNode(let msg) = error else {
                return XCTFail("expected .unsupportedNode, got \(error)")
            }
            XCTAssertEqual(msg, "quaternion basis requires quaternion arithmetic")
        }
    }

    func testConstantInfinityPositive() throws {
        let ast = MathLexExpression.constant(.infinity)
        let result = try MathExpr.evaluateUnified(ast)
        let x = try XCTUnwrap(scalarOf(result))
        XCTAssertTrue(x.isInfinite && x > 0)
    }

    func testConstantNegInfinity() throws {
        let ast = MathLexExpression.constant(.negInfinity)
        let result = try MathExpr.evaluateUnified(ast)
        let x = try XCTUnwrap(scalarOf(result))
        XCTAssertTrue(x.isInfinite && x < 0)
    }

    func testConstantNaN() throws {
        let ast = MathLexExpression.constant(.nan)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertTrue(scalarOf(result)!.isNaN)
    }

    // MARK: - Subtask 19.4: .complex(real:imaginary:) literal node

    func testComplexLiteralNode_integerParts() throws {
        // .complex(re: 3, im: 4) → .complex(Complex(3+4i))
        let ast = MathLexExpression.complex(
            real: .integer(3),
            imaginary: .integer(4))
        let result = try MathExpr.evaluateUnified(ast)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 3.0, accuracy: 1e-15)
        XCTAssertEqual(z.im, 4.0, accuracy: 1e-15)
    }

    func testComplexLiteralNode_floatParts() throws {
        let ast = MathLexExpression.complex(
            real: .float(1.5),
            imaginary: .float(-2.5))
        let result = try MathExpr.evaluateUnified(ast)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 1.5, accuracy: 1e-14)
        XCTAssertEqual(z.im, -2.5, accuracy: 1e-14)
    }

    func testComplexLiteralNode_zeroImaginary() throws {
        // re=5, im=0 → .complex(5+0i) — stays complex, not collapsed to scalar
        let ast = MathLexExpression.complex(
            real: .integer(5),
            imaginary: .integer(0))
        let result = try MathExpr.evaluateUnified(ast)
        // The .complex constructor always yields .complex regardless of imaginary value
        XCTAssertEqual(result.kind, NumericValue.Kind.complex)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 5.0, accuracy: 1e-15)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-15)
    }

    func testComplexLiteralNode_expressionParts() throws {
        // re = pi, im = 1/2 → .complex(π + 0.5i)
        let ast = MathLexExpression.complex(
            real: .constant(.pi),
            imaginary: .rational(numerator: .integer(1), denominator: .integer(2)))
        let result = try MathExpr.evaluateUnified(ast)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, Double.pi, accuracy: 1e-14)
        XCTAssertEqual(z.im, 0.5, accuracy: 1e-14)
    }

    func testComplexLiteralNode_matrixPartThrows() throws {
        // Supplying a matrix as either part → .invalidArguments
        let mat = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 0, 0, 1])
        let ast = MathLexExpression.complex(
            real: .variable("M"),
            imaginary: .integer(1))
        XCTAssertThrowsError(
            try MathExpr.evaluateUnified(ast, values: ["M": .matrix(mat)])
        ) { err in
            guard case .invalidArguments = err as? MathExprError else {
                XCTFail("Expected .invalidArguments, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 19.6: Real .vector literal → NumericValue.matrix

    func testRealVectorLiteral_singleElement() throws {
        // A 1-element vector must stay .matrix(1×1), not collapse to .scalar
        let ast = MathLexExpression.vector([.float(7.0)])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m.data[0], 7.0, accuracy: 1e-15)
    }

    func testRealVectorLiteral_threeElements() throws {
        let ast = MathLexExpression.vector([.integer(1), .integer(2), .integer(3)])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 1, "Vector must be a column vector (cols == 1)")
        XCTAssertEqual(m.data, [1, 2, 3])
    }

    func testRealVectorLiteral_expressionElements() throws {
        // Elements may themselves be sub-expressions
        let ast = MathLexExpression.vector([
            .binary(op: .add, left: .integer(1), right: .integer(1)),
            .constant(.pi),
        ])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m.data[0], 2.0, accuracy: 1e-15)
        XCTAssertEqual(m.data[1], Double.pi, accuracy: 1e-15)
    }

    func testEmptyVectorLiteral() throws {
        // An empty vector → .matrix(0×1)
        let ast = MathLexExpression.vector([])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 0)
        XCTAssertEqual(m.cols, 1)
        XCTAssertTrue(m.data.isEmpty)
    }

    func testVectorLiteral_matrixElementThrows() throws {
        // A vector element that evaluates to a matrix is invalid
        let mat = LinAlg.Matrix(rows: 2, cols: 1, data: [1, 2])
        let ast = MathLexExpression.vector([
            .integer(1),
            .variable("v"),   // binds to a matrix
        ])
        XCTAssertThrowsError(
            try MathExpr.evaluateUnified(ast, values: ["v": .matrix(mat)])
        ) { err in
            guard case .invalidArguments = err as? MathExprError else {
                XCTFail("Expected .invalidArguments for matrix-valued element, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 19.7: Complex .vector literal → NumericValue.complexMatrix

    func testComplexVectorLiteral_allComplexElements() throws {
        // [1+2i, 3+4i] → .complexMatrix(2×1)
        let ast = MathLexExpression.vector([
            MathLexExpression.complex(real: .integer(1), imaginary: .integer(2)),
            MathLexExpression.complex(real: .integer(3), imaginary: .integer(4)),
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix)
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 1)
        XCTAssertEqual(cm.real, [1.0, 3.0])
        XCTAssertEqual(cm.imag, [2.0, 4.0])
    }

    func testComplexVectorLiteral_mixedRealAndComplex() throws {
        // [1, 0+1i] — one real element, one complex → promotes all to .complexMatrix
        let ast = MathLexExpression.vector([
            .integer(1),
            MathLexExpression.complex(real: .integer(0), imaginary: .integer(1)),
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix,
            "A single complex element must promote the whole vector to complexMatrix")
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 1)
        XCTAssertEqual(cm.real[0], 1.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[0], 0.0, accuracy: 1e-15)  // promoted real: imag = 0
        XCTAssertEqual(cm.real[1], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[1], 1.0, accuracy: 1e-15)
    }

    func testComplexVectorLiteral_constantIasElement() throws {
        // [i, 1] — .constant(.i) as element → promotes to complexMatrix
        let ast = MathLexExpression.vector([
            .constant(.i),
            .integer(1),
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix)
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.real[0], 0.0, accuracy: 1e-15)  // i: re=0
        XCTAssertEqual(cm.imag[0], 1.0, accuracy: 1e-15)  // i: im=1
        XCTAssertEqual(cm.real[1], 1.0, accuracy: 1e-15)  // 1: re=1
        XCTAssertEqual(cm.imag[1], 0.0, accuracy: 1e-15)  // 1: im=0
    }

    // MARK: - Subtask 19.8: Ragged matrix rows

    func testRaggedMatrix_secondRowShorter() throws {
        let ast = MathLexExpression.matrix([
            [.integer(1), .integer(2)],
            [.integer(3)],
        ])
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard let laErr = err as? LinAlg.LinAlgError,
                  case .dimensionMismatch = laErr else {
                XCTFail("Expected LinAlg.LinAlgError.dimensionMismatch, got \(err)")
                return
            }
        }
    }

    func testRaggedMatrix_secondRowLonger() throws {
        let ast = MathLexExpression.matrix([
            [.integer(1)],
            [.integer(2), .integer(3)],
        ])
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            guard let laErr = err as? LinAlg.LinAlgError,
                  case .dimensionMismatch = laErr else {
                XCTFail("Expected LinAlg.LinAlgError.dimensionMismatch, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 19.9: Real .matrix literal → NumericValue.matrix

    func testRealMatrixLiteral_2x2() throws {
        let ast = MathLexExpression.matrix([
            [.integer(1), .integer(2)],
            [.integer(3), .integer(4)],
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.cols, 2)
        XCTAssertEqual(m.data, [1, 2, 3, 4])
    }

    func testRealMatrixLiteral_1x1() throws {
        // A 1×1 matrix stays .matrix, not collapsed to .scalar
        let ast = MathLexExpression.matrix([[.float(42.0)]])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m.data[0], 42.0, accuracy: 1e-15)
    }

    func testRealMatrixLiteral_singleRow() throws {
        // 1×3 row vector expressed as a matrix
        let ast = MathLexExpression.matrix([[.integer(10), .integer(20), .integer(30)]])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 3)
        XCTAssertEqual(m.data, [10, 20, 30])
    }

    func testEmptyMatrixLiteral() throws {
        // An empty matrix (no rows) → .matrix(0×0)
        let ast = MathLexExpression.matrix([])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 0)
        XCTAssertEqual(m.cols, 0)
        XCTAssertTrue(m.data.isEmpty)
    }

    func testRealMatrixLiteral_subExpressionElements() throws {
        // Elements may be sub-expressions that evaluate to real scalars
        let ast = MathLexExpression.matrix([
            [.binary(op: .add, left: .integer(1), right: .integer(1)), .constant(.pi)],
        ])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 1)
        XCTAssertEqual(m.cols, 2)
        XCTAssertEqual(m.data[0], 2.0, accuracy: 1e-15)
        XCTAssertEqual(m.data[1], Double.pi, accuracy: 1e-14)
    }

    // MARK: - Subtask 19.10: Complex .matrix literal → NumericValue.complexMatrix

    func testComplexMatrixLiteral_allComplexElements() throws {
        // [[1+2i, 3+4i]] → .complexMatrix(1×2)
        let ast = MathLexExpression.matrix([[
            MathLexExpression.complex(real: .integer(1), imaginary: .integer(2)),
            MathLexExpression.complex(real: .integer(3), imaginary: .integer(4)),
        ]])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix)
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 1)
        XCTAssertEqual(cm.cols, 2)
        XCTAssertEqual(cm.real, [1.0, 3.0])
        XCTAssertEqual(cm.imag, [2.0, 4.0])
    }

    func testComplexMatrixLiteral_mixedRealAndComplex() throws {
        // [[1, 2+3i], [4, 5]] — one complex element in row 0 → whole matrix promoted
        let ast = MathLexExpression.matrix([
            [.integer(1), MathLexExpression.complex(real: .integer(2), imaginary: .integer(3))],
            [.integer(4), .integer(5)],
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix,
            "A single complex element must promote the whole matrix to complexMatrix")
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 2)
        // Row 0: [1+0i, 2+3i]
        XCTAssertEqual(cm.real[0], 1.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[0], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.real[1], 2.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[1], 3.0, accuracy: 1e-15)
        // Row 1: [4+0i, 5+0i]
        XCTAssertEqual(cm.real[2], 4.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[2], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.real[3], 5.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[3], 0.0, accuracy: 1e-15)
    }

    func testComplexMatrixLiteral_matrixElementThrows() throws {
        // A matrix-valued element inside a matrix literal is invalid
        let innerMat = LinAlg.Matrix(rows: 2, cols: 2, data: [1, 0, 0, 1])
        let ast = MathLexExpression.matrix([
            [.integer(1), .variable("M")],
        ])
        XCTAssertThrowsError(
            try MathExpr.evaluateUnified(ast, values: ["M": .matrix(innerMat)])
        ) { err in
            guard case .invalidArguments = err as? MathExprError else {
                XCTFail("Expected .invalidArguments for matrix-valued matrix element, got \(err)")
                return
            }
        }
    }

    // MARK: - Subtask 19.11: Nested literals

    func testNestedVector_insideVectorExpression() throws {
        // .vector containing a .constant(.i) together with constants
        // This is "nested" in the sense that element evaluation recurses
        let ast = MathLexExpression.vector([
            .binary(op: .mul, left: .integer(2), right: .constant(.i)),
            .binary(op: .mul, left: .integer(3), right: .constant(.i)),
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix)
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.real[0], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[0], 2.0, accuracy: 1e-15)
        XCTAssertEqual(cm.real[1], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[1], 3.0, accuracy: 1e-15)
    }

    func testMatrixWithComplexElementsFromExpression() throws {
        // [[2*i, 0], [0, -1*i]] → .complexMatrix(2×2)
        let ast = MathLexExpression.matrix([
            [
                .binary(op: .mul, left: .integer(2), right: .constant(.i)),
                .integer(0),
            ],
            [
                .integer(0),
                .binary(op: .mul, left: .unary(op: .neg, operand: .integer(1)), right: .constant(.i)),
            ],
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix)
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 2)
        XCTAssertEqual(cm.imag[0], 2.0, accuracy: 1e-15)   // [0,0]: 2i
        XCTAssertEqual(cm.real[1], 0.0, accuracy: 1e-15)   // [0,1]: 0
        XCTAssertEqual(cm.real[2], 0.0, accuracy: 1e-15)   // [1,0]: 0
        XCTAssertEqual(cm.imag[3], -1.0, accuracy: 1e-15)  // [1,1]: -i
    }

    // MARK: - Subtask 19.13: nodeLabel covers literal kinds (no unsupportedNode fallthrough)

    func testNodeLabelForVector() throws {
        // A .vector node must evaluate (not throw .unsupportedNode)
        let ast = MathLexExpression.vector([.integer(1)])
        // Just confirming it doesn't throw unsupportedNode
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
    }

    func testNodeLabelForMatrix() throws {
        let ast = MathLexExpression.matrix([[.integer(1)]])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.matrix)
    }

    func testNodeLabelForComplexConstructor() throws {
        let ast = MathLexExpression.complex(real: .integer(1), imaginary: .integer(0))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complex)
    }

    // MARK: - Subtask 19.15: Scalar/complex literal node direct-AST tests

    func testIntegerLiteralNodeYieldsScalar() throws {
        let ast = MathLexExpression.integer(42)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.scalar)
        XCTAssertEqual(scalarOf(result)!, 42.0, accuracy: 1e-15)
    }

    func testFloatLiteralNodeYieldsScalar() throws {
        let ast = MathLexExpression.float(2.718)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.scalar)
        XCTAssertEqual(scalarOf(result)!, 2.718, accuracy: 1e-14)
    }

    func testFloatNilLiteralThrowsNonFinite() throws {
        let ast = MathLexExpression.float(nil)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { err in
            XCTAssertEqual(err as? MathExprError, .nonFiniteFloat)
        }
    }

    func testComplexConstructorYieldsComplex() throws {
        let ast = MathLexExpression.complex(real: .float(1.0), imaginary: .float(2.0))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complex)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 1.0, accuracy: 1e-15)
        XCTAssertEqual(z.im, 2.0, accuracy: 1e-15)
    }

    func testRationalNodeYieldsScalar() throws {
        let ast = MathLexExpression.rational(numerator: .integer(1), denominator: .integer(3))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.scalar)
        XCTAssertEqual(scalarOf(result)!, 1.0 / 3.0, accuracy: 1e-15)
    }

    // MARK: - Subtask 19.16: Real vector/matrix literal direct-AST tests

    func testRealVectorDirectAST_values() throws {
        let ast = MathLexExpression.vector([.float(10.0), .float(20.0), .float(30.0)])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m.data[0], 10.0, accuracy: 1e-15)
        XCTAssertEqual(m.data[1], 20.0, accuracy: 1e-15)
        XCTAssertEqual(m.data[2], 30.0, accuracy: 1e-15)
    }

    func testRealMatrixDirectAST_3x3() throws {
        // 3×3 identity-like matrix
        let ast = MathLexExpression.matrix([
            [.integer(1), .integer(0), .integer(0)],
            [.integer(0), .integer(1), .integer(0)],
            [.integer(0), .integer(0), .integer(1)],
        ])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 3)
        // Diagonal: [0,0]=1, [1,1]=1, [2,2]=1
        XCTAssertEqual(m.data[0], 1.0, accuracy: 1e-15)
        XCTAssertEqual(m.data[4], 1.0, accuracy: 1e-15)
        XCTAssertEqual(m.data[8], 1.0, accuracy: 1e-15)
        // Off-diagonal samples: [0,1]=0, [1,0]=0
        XCTAssertEqual(m.data[1], 0.0, accuracy: 1e-15)
        XCTAssertEqual(m.data[3], 0.0, accuracy: 1e-15)
    }

    // MARK: - Subtask 19.17: Complex vector/matrix literal direct-AST tests

    func testComplexVectorDirectAST_unitCircle() throws {
        // [1+0i, 0+1i, -1+0i, 0-1i] — four points on unit circle
        let ast = MathLexExpression.vector([
            MathLexExpression.complex(real: .integer(1), imaginary: .integer(0)),
            MathLexExpression.complex(real: .integer(0), imaginary: .integer(1)),
            MathLexExpression.complex(real: .unary(op: .neg, operand: .integer(1)), imaginary: .integer(0)),
            MathLexExpression.complex(real: .integer(0), imaginary: .unary(op: .neg, operand: .integer(1))),
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix)
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 4)
        XCTAssertEqual(cm.cols, 1)
        XCTAssertEqual(cm.real, [1, 0, -1, 0], accuracy: 1e-15)
        XCTAssertEqual(cm.imag, [0, 1, 0, -1], accuracy: 1e-15)
    }

    func testComplexMatrixDirectAST_2x2() throws {
        // [[i, 1], [0, -i]] → .complexMatrix(2×2)
        let ast = MathLexExpression.matrix([
            [
                .constant(.i),
                .integer(1),
            ],
            [
                .integer(0),
                .unary(op: .neg, operand: .constant(.i)),
            ],
        ])
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complexMatrix)
        let cm = try XCTUnwrap(complexMatrixOf(result))
        XCTAssertEqual(cm.rows, 2)
        XCTAssertEqual(cm.cols, 2)
        // Row 0: [0+1i, 1+0i]
        XCTAssertEqual(cm.real[0], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[0], 1.0, accuracy: 1e-15)
        XCTAssertEqual(cm.real[1], 1.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[1], 0.0, accuracy: 1e-15)
        // Row 1: [0+0i, 0-1i]
        XCTAssertEqual(cm.real[2], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[2], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.real[3], 0.0, accuracy: 1e-15)
        XCTAssertEqual(cm.imag[3], -1.0, accuracy: 1e-15)
    }

    // MARK: - Subtask 19.18: Default-build parseError + imaginary-literal regression

    /// Bracket literals cannot be parsed on the default build.
    ///
    /// The pure-Swift fallback parser has no bracket tokenizer, so `[1,2,3]`
    /// and `[[1,2],[3,4]]` yield `MathExprError.parseError`. This is correct
    /// and expected — matrices flow in via the `values:` dictionary on the
    /// default build, or via mathlex AST nodes when mathlex is enabled.
    func testDefaultBuildBracketLiteralYieldsParseError() {
        let bracketExpressions = ["[1,2,3]", "[[1,2],[3,4]]", "[i, 0]"]
        for expr in bracketExpressions {
            XCTAssertThrowsError(
                try MathExpr.parse(expr),
                "Expected parseError for bracket expression '\(expr)' on default build"
            ) { err in
                guard case .parseError = err as? MathExprError else {
                    XCTFail("Expected .parseError for '\(expr)', got \(err)")
                    return
                }
            }
        }
    }

    /// Imaginary literals ARE supported by the fallback tokenizer.
    ///
    /// Expressions using the `i` suffix (e.g., `2i`, `3.5i`) or the constant
    /// `i` must parse and evaluate correctly on the default build. This is a
    /// regression guard: the bracket limitation must not affect imaginary
    /// literal support.
    func testImaginaryLiteralRegressionDefaultBuild() throws {
        // "2*i" parses as: multiply(2, constant(i)) → .complex(0+2i)
        let ast = try MathExpr.parse("2*i")
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertEqual(result.kind, NumericValue.Kind.complex)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, 0.0, accuracy: 1e-15)
        XCTAssertEqual(z.im, 2.0, accuracy: 1e-14)
    }

    func testImaginaryLiteralInExpression() throws {
        // (1 + i) * (1 - i) = 1 - i^2 = 1 - (-1) = 2
        let ast = try MathExpr.parse("(1 + i) * (1 - i)")
        let result = try MathExpr.evaluateUnified(ast)
        // Result may be complex with im≈0 or scalar depending on dispatch
        switch result {
        case .scalar(let x):
            XCTAssertEqual(x, 2.0, accuracy: 1e-12)
        case .complex(let z):
            XCTAssertEqual(z.re, 2.0, accuracy: 1e-12)
            XCTAssertEqual(z.im, 0.0, accuracy: 1e-12)
        default:
            XCTFail("Expected scalar or complex, got \(result)")
        }
    }

    // MARK: - Subtask 19.19: Frozen-snapshot parity for literal-node corpus
    //
    // Since the LegacySnapshot.json only contains scalar and complex entries
    // (pre-unified pipeline, no vector/matrix literals were parseable), the
    // parity test here verifies that the existing snapshot entries continue
    // to evaluate identically through the evaluator — confirming literal-node
    // handling didn't regress any scalar path.

    func testSnapshotParityUnchanged_scalarExpressions() throws {
        // Representative frozen values from LegacySnapshot.json §spot-checks
        // These constants are cross-referenced with SciPy in ParityCorpusTests.
        // Verified here through evaluateUnified to confirm no regression.
        let cases: [(MathLexExpression, Double)] = [
            (.float(0.0), 0.0),
            (.float(1.0), 1.0),
            (.constant(.pi), Double.pi),
            (.constant(.e), M_E),
            (.binary(op: .add, left: .integer(1), right: .integer(1)), 2.0),
            (.binary(op: .mul, left: .integer(3), right: .integer(4)), 12.0),
            (.function(name: "sin", args: [.float(0.0)]), 0.0),
            (.function(name: "cos", args: [.float(0.0)]), 1.0),
            (.function(name: "exp", args: [.integer(0)]), 1.0),
        ]
        for (ast, expected) in cases {
            let result = try MathExpr.evaluateUnified(ast)
            let got = try XCTUnwrap(scalarOf(result),
                "Expected scalar for AST \(ast)")
            if expected.isNaN {
                XCTAssertTrue(got.isNaN, "Expected NaN for \(ast)")
            } else {
                XCTAssertEqual(got, expected, accuracy: 1e-12,
                    "Snapshot parity failed for \(ast): expected \(expected) got \(got)")
            }
        }
    }

    func testLiteralNodeSnapshotValues_complexConstant() throws {
        // i^2 = -1 — snapshot value for complex constant evaluation
        let ast = MathLexExpression.binary(
            op: .mul, left: .constant(.i), right: .constant(.i))
        let result = try MathExpr.evaluateUnified(ast)
        let z = try XCTUnwrap(complexOf(result))
        XCTAssertEqual(z.re, -1.0, accuracy: 1e-14)
        XCTAssertEqual(z.im, 0.0, accuracy: 1e-14)
    }

    func testLiteralNodeSnapshotValues_vectorColumn() throws {
        // A canonical 3-element column vector: values match direct construction
        let ast = MathLexExpression.vector([.float(1.0), .float(0.0), .float(0.0)])
        let result = try MathExpr.evaluateUnified(ast)
        let m = try XCTUnwrap(matrixOf(result))
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 1)
        XCTAssertEqual(m.data, [1.0, 0.0, 0.0])
    }
}

// MARK: - XCTAssertEqual overload for [Double] with accuracy

private func XCTAssertEqual(
    _ expression1: [Double],
    _ expression2: [Double],
    accuracy: Double,
    file: StaticString = #file,
    line: UInt = #line
) {
    guard expression1.count == expression2.count else {
        XCTFail("Array length mismatch: \(expression1.count) vs \(expression2.count)",
                file: file, line: line)
        return
    }
    for (i, (a, b)) in zip(expression1, expression2).enumerated() {
        XCTAssertEqual(a, b, accuracy: accuracy,
            "Element \(i): \(a) ≠ \(b) (within \(accuracy))",
            file: file, line: line)
    }
}
