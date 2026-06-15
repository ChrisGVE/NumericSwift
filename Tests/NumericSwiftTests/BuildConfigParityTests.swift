// BuildConfigParityTests.swift
// Tests/NumericSwiftTests/
//
// Build-config parity tests for Task #23 (Phase 3, unified-pipeline tag).
//
// PURPOSE
// -------
// Verify that the dispatch core and unified evaluator produce IDENTICAL
// NumericValue results for the shared corpus (scalar / complex /
// explicitly-constructed-matrix) regardless of the build configuration:
//   • Default build:  NUMERICSWIFT_MATHLEX absent  (this file's CI lane)
//   • Mathlex build:  NUMERICSWIFT_MATHLEX present  (see CI requirement below)
//
// APPROACH
// --------
// The shared corpus is driven through evaluateUnified using AST nodes
// constructed directly in Swift — never via MathExpr.parse() — so the
// parse backend (mathlex Rust crate vs. pure-Swift fallback) is not
// involved.  Matrix and vector values enter via the `values:` binding
// dictionary, which is the only mechanism available on the default build
// and is equally valid on the mathlex build.  The dispatch core and
// unified evaluator contain NO #if NUMERICSWIFT_MATHLEX guards — their
// logic is identical in both configs.
//
// WHAT IS NOT TESTED HERE (mathlex-only)
// ----------------------------------------
// Bracket-literal parsing ([1,2,3] / [[a,b],[c,d]]) is only available
// when the mathlex Rust backend is compiled in.  The default build emits
// MathExprError.parseError for such expressions; this is intentional and
// documented in Task 19.  Those cases are not part of the shared corpus
// and are not exercised here.
//
// The `testMathlexOnlyLiteralParseFails` test below confirms that the
// default build correctly rejects bracket literals with a .parseError,
// pinning the documented skip behaviour.  Under the mathlex build, that
// test is excluded and the corresponding bracket-literal cases must be
// verified in the CI job described below.
//
// CI REQUIREMENT — mathlex-enabled parity verification
// -----------------------------------------------------
// A CI environment that has a `../mathlex` checkout MUST run an
// additional job to verify that the mathlex-on build is parity-compliant:
//
//   Environment:  NUMERICSWIFT_INCLUDE_MATHLEX=1 swift test \
//                   --filter BuildConfigParityTests
//
//   The same filter runs this file.  In a mathlex build:
//     • All shared-corpus tests below MUST pass with identical assertions.
//     • The `testMathlexOnlyLiteralParseFails` guard is excluded (#if).
//     • Additional bracket-literal cases SHOULD be added (same file or a
//       companion MathLexBracketLiteralTests.swift) to cover:
//         - Vector literal: "[1.0, 2.0, 3.0]" → .matrix(3×1)
//         - Real matrix literal: "[[1.0,2.0],[3.0,4.0]]" → .matrix(2×2)
//         - Complex vector: "[1+2i, 3+4i]" → .complexMatrix(2×1)
//         - Complex matrix: "[[1+i,2],[0,1-i]]" → .complexMatrix(2×2)
//         - Matrix arithmetic via bracket literals: "[[1,0],[0,1]] * x"
//           where x is bound via values: (matrix path)
//       Each case must assert the same NumericValue via isApproximatelyEqual
//       (tolerance 1e-10) or isExactlyEqual for integer-valued entries.
//       The assertion value must match the corresponding non-literal path
//       (i.e. the result obtained by binding the same matrix via values:).
//
// EQUALITY CONVENTIONS
// --------------------
// Scalar entries:          isExactlyEqual (bit-identical per IEEE 754)
// Complex entries:         isApproximatelyEqual (tolerance 1e-10)
// Real-matrix entries:     isApproximatelyEqual (tolerance 1e-10)
// Complex-matrix entries:  isApproximatelyEqual (tolerance 1e-10)
// Error entries:           XCTAssertThrowsError — error category is pinned
//
// FILE DISCIPLINE
// ---------------
// This file does NOT modify any other test file or any Sources/ file.
// Other agents are concurrently authoring Tasks 26 and 27 test files.
//
// swiftlint:disable type_body_length file_length function_body_length

import XCTest
@testable import NumericSwift

// MARK: - BuildConfigParityTests

final class BuildConfigParityTests: XCTestCase {

    // MARK: - Helpers

    /// Assert that a NumericValue result is exactly equal to an expected value.
    ///
    /// Uses `isExactlyEqual`, which applies IEEE 754 equality (NaN non-reflexive,
    /// +0.0 == -0.0).  Appropriate for integer-valued scalars and edge-value cases.
    private func assertExactly(
        _ result: NumericValue,
        equals expected: NumericValue,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertTrue(
            result.isExactlyEqual(to: expected),
            message.isEmpty
                ? "Expected \(expected), got \(result)"
                : "\(message): expected \(expected), got \(result)",
            file: file, line: line)
    }

    /// Assert that a NumericValue result is approximately equal to an expected value.
    ///
    /// Uses `isApproximatelyEqual(tolerance:)`.  Appropriate for transcendental
    /// functions, matrix decompositions, and complex arithmetic.
    private func assertApprox(
        _ result: NumericValue,
        equals expected: NumericValue,
        tolerance: Double = 1e-10,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertTrue(
            result.isApproximatelyEqual(to: expected, tolerance: tolerance),
            message.isEmpty
                ? "Expected \(expected) (±\(tolerance)), got \(result)"
                : "\(message): expected \(expected) (±\(tolerance)), got \(result)",
            file: file, line: line)
    }

    // MARK: - Scalar corpus (shared — both build configs)
    //
    // AST nodes are constructed directly.  No MathExpr.parse() call, so the
    // build-config parse backend does not affect these tests.

    /// Integer literal node → scalar(n)
    func testSharedScalar_IntegerLiteral() throws {
        let ast = MathLexExpression.integer(7)
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(7.0), "integer(7)")
    }

    /// Float literal node → scalar(v)
    func testSharedScalar_FloatLiteral() throws {
        let ast = MathLexExpression.float(3.14)
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(3.14), "float(3.14)")
    }

    /// Pi constant → scalar(π)
    func testSharedScalar_ConstantPi() throws {
        let ast = MathLexExpression.constant(.pi)
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(.pi), "constant(.pi)")
    }

    /// E constant → scalar(e)
    func testSharedScalar_ConstantE() throws {
        let ast = MathLexExpression.constant(.e)
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(M_E), "constant(.e)")
    }

    /// Addition: 3 + 4 = 7
    func testSharedScalar_Addition() throws {
        let ast = MathLexExpression.binary(
            op: .add,
            left: .integer(3),
            right: .integer(4))
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(7.0), "3 + 4")
    }

    /// Subtraction: 10 − 3 = 7
    func testSharedScalar_Subtraction() throws {
        let ast = MathLexExpression.binary(
            op: .sub,
            left: .integer(10),
            right: .integer(3))
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(7.0), "10 - 3")
    }

    /// Multiplication: 6 × 7 = 42
    func testSharedScalar_Multiplication() throws {
        let ast = MathLexExpression.binary(
            op: .mul,
            left: .integer(6),
            right: .integer(7))
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(42.0), "6 * 7")
    }

    /// Division: 1 / 4 = 0.25
    func testSharedScalar_Division() throws {
        let ast = MathLexExpression.binary(
            op: .div,
            left: .integer(1),
            right: .integer(4))
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(0.25), "1 / 4")
    }

    /// Power: 2 ^ 10 = 1024
    func testSharedScalar_Power() throws {
        let ast = MathLexExpression.binary(
            op: .pow,
            left: .integer(2),
            right: .integer(10))
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(1024.0), "2 ^ 10")
    }

    /// Modulo: 17 % 5 = 2
    func testSharedScalar_Modulo() throws {
        let ast = MathLexExpression.binary(
            op: .mod,
            left: .integer(17),
            right: .integer(5))
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(2.0), "17 mod 5")
    }

    /// Unary negation: −42
    func testSharedScalar_UnaryNeg() throws {
        let ast = MathLexExpression.unary(op: .neg, operand: .integer(42))
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(-42.0), "neg(42)")
    }

    /// Unary positive: +42 (no-op)
    func testSharedScalar_UnaryPos() throws {
        let ast = MathLexExpression.unary(op: .pos, operand: .integer(42))
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(42.0), "pos(42)")
    }

    /// sin(π/6) = 0.5
    func testSharedScalar_FunctionSin() throws {
        let ast = MathLexExpression.function(
            name: "sin",
            args: [
                .binary(op: .div, left: .constant(.pi), right: .integer(6))
            ])
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(0.5), "sin(π/6)")
    }

    /// cos(0) = 1
    func testSharedScalar_FunctionCos() throws {
        let ast = MathLexExpression.function(name: "cos", args: [.integer(0)])
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(1.0), "cos(0)")
    }

    /// sqrt(2)
    func testSharedScalar_FunctionSqrt() throws {
        let ast = MathLexExpression.function(name: "sqrt", args: [.integer(2)])
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(sqrt(2.0)), "sqrt(2)")
    }

    /// exp(1) = e
    func testSharedScalar_FunctionExp() throws {
        let ast = MathLexExpression.function(name: "exp", args: [.integer(1)])
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(M_E), "exp(1)")
    }

    /// log(e) = 1
    func testSharedScalar_FunctionLog() throws {
        let ast = MathLexExpression.function(
            name: "log", args: [.constant(.e)])
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(1.0), "log(e)")
    }

    /// abs(-5) = 5
    func testSharedScalar_FunctionAbs() throws {
        let ast = MathLexExpression.function(
            name: "abs",
            args: [.unary(op: .neg, operand: .integer(5))])
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(5.0), "abs(-5)")
    }

    /// Variable binding — scalar variable
    func testSharedScalar_VariableBinding() throws {
        let ast = MathLexExpression.binary(
            op: .add,
            left: .variable("x"),
            right: .integer(1))
        let result = try MathExpr.evaluateUnified(ast, values: ["x": .scalar(41.0)])
        assertExactly(result, equals: .scalar(42.0), "x + 1 where x=41")
    }

    /// Rational node: 1/3
    func testSharedScalar_RationalNode() throws {
        let ast = MathLexExpression.rational(
            numerator: .integer(1),
            denominator: .integer(3))
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .scalar(1.0 / 3.0), "rational(1,3)")
    }

    /// Division by zero → divisionByZero error
    func testSharedScalar_DivisionByZero() throws {
        let ast = MathLexExpression.binary(
            op: .div,
            left: .integer(1),
            right: .integer(0))
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { error in
            guard let me = error as? MathExprError, case .divisionByZero = me else {
                XCTFail("Expected .divisionByZero, got \(error)")
                return
            }
        }
    }

    /// Undefined variable → undefinedVariable error
    func testSharedScalar_UndefinedVariable() throws {
        let ast = MathLexExpression.variable("z_undefined")
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { error in
            guard let me = error as? MathExprError,
                  case .undefinedVariable(let name) = me else {
                XCTFail("Expected .undefinedVariable, got \(error)")
                return
            }
            XCTAssertEqual(name, "z_undefined")
        }
    }

    /// float(nil) sentinel → nonFiniteFloat error
    func testSharedScalar_NonFiniteFloatSentinel() throws {
        let ast = MathLexExpression.float(nil)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { error in
            guard let me = error as? MathExprError, case .nonFiniteFloat = me else {
                XCTFail("Expected .nonFiniteFloat, got \(error)")
                return
            }
        }
    }

    /// IEEE-754 edge — +infinity constant
    func testSharedScalar_IEEE754_PosInfinity() throws {
        let ast = MathLexExpression.constant(.infinity)
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(.infinity), "+infinity constant")
    }

    /// IEEE-754 edge — −infinity constant
    func testSharedScalar_IEEE754_NegInfinity() throws {
        let ast = MathLexExpression.constant(.negInfinity)
        let result = try MathExpr.evaluateUnified(ast)
        assertExactly(result, equals: .scalar(-.infinity), "-infinity constant")
    }

    // MARK: - Complex corpus (shared — both build configs)

    /// Imaginary constant i → complex(0+1i)
    func testSharedComplex_ConstantI() throws {
        let ast = MathLexExpression.constant(.i)
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .complex(Complex(re: 0, im: 1)), "constant .i")
    }

    /// Complex constructor node: real=1, imaginary=2 → complex(1+2i)
    func testSharedComplex_ComplexNode() throws {
        let ast = MathLexExpression.complex(real: .integer(1), imaginary: .integer(2))
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .complex(Complex(re: 1, im: 2)), "complex(1, 2)")
    }

    /// Complex addition: (1+2i) + (3+4i) = 4+6i
    func testSharedComplex_Addition() throws {
        let lhs = MathLexExpression.complex(real: .integer(1), imaginary: .integer(2))
        let rhs = MathLexExpression.complex(real: .integer(3), imaginary: .integer(4))
        let ast = MathLexExpression.binary(op: .add, left: lhs, right: rhs)
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(result, equals: .complex(Complex(re: 4, im: 6)), "(1+2i)+(3+4i)")
    }

    /// Complex multiplication: i × i = −1
    func testSharedComplex_MultiplicationI_squared() throws {
        let ast = MathLexExpression.binary(
            op: .mul,
            left: .constant(.i),
            right: .constant(.i))
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(
            result, equals: .complex(Complex(re: -1, im: 0)),
            "i*i = -1")
    }

    /// Euler identity: exp(i·π) = −1 + 0i
    func testSharedComplex_EulerIdentity() throws {
        // Bind 'i' as complex variable via values:
        let ipi = MathLexExpression.binary(
            op: .mul,
            left: .constant(.i),
            right: .constant(.pi))
        let ast = MathLexExpression.function(name: "exp", args: [ipi])
        let result = try MathExpr.evaluateUnified(ast)
        assertApprox(
            result, equals: .complex(Complex(re: -1, im: 0)),
            "exp(i*π)")
    }

    /// Complex variable binding
    func testSharedComplex_VariableBinding() throws {
        let ast = MathLexExpression.binary(
            op: .add,
            left: .variable("z"),
            right: .integer(1))
        let z = Complex(re: 3.0, im: -2.0)
        let result = try MathExpr.evaluateUnified(ast, values: ["z": .complex(z)])
        assertApprox(
            result, equals: .complex(Complex(re: 4.0, im: -2.0)),
            "z + 1 where z=3-2i")
    }

    /// sqrt of negative scalar via complex promotion — sqrt(-1) = i
    func testSharedComplex_SqrtNegativeViaComplexBinding() throws {
        // Feed complex value: sqrt(z) where z = -1+0i
        let ast = MathLexExpression.function(name: "sqrt", args: [.variable("z")])
        let z = Complex(re: -1, im: 0)
        let result = try MathExpr.evaluateUnified(ast, values: ["z": .complex(z)])
        assertApprox(
            result, equals: .complex(Complex(re: 0, im: 1)),
            "sqrt(-1+0i)")
    }

    /// Division by zero on complex → divisionByZero
    func testSharedComplex_DivisionByZero() throws {
        let zero = MathLexExpression.complex(real: .integer(0), imaginary: .integer(0))
        let ast = MathLexExpression.binary(
            op: .div,
            left: .complex(real: .integer(1), imaginary: .integer(0)),
            right: zero)
        XCTAssertThrowsError(try MathExpr.evaluateUnified(ast)) { error in
            guard let me = error as? MathExprError, case .divisionByZero = me else {
                XCTFail("Expected .divisionByZero, got \(error)")
                return
            }
        }
    }

    // MARK: - Explicitly-constructed matrix corpus (shared — both build configs)
    //
    // Matrices are bound via the `values:` dictionary.  No bracket-literal parsing
    // is used, so these tests are config-independent.

    /// Matrix variable binding — pass through unchanged
    func testSharedMatrix_VariableBinding() throws {
        let m = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let ast = MathLexExpression.variable("A")
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        assertApprox(result, equals: .matrix(m), "matrix variable passthrough")
    }

    /// Scalar × matrix broadcast: 2 × A
    func testSharedMatrix_ScalarMatrixMultiplication() throws {
        let m = LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])  // 2×2 identity
        let ast = MathLexExpression.binary(
            op: .mul,
            left: .integer(2),
            right: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(m)])
        let expected = LinAlg.Matrix([[2.0, 0.0], [0.0, 2.0]])
        assertApprox(result, equals: .matrix(expected), "2 * I_2")
    }

    /// Matrix + Matrix: element-wise addition
    func testSharedMatrix_MatrixAddition() throws {
        let a = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let b = LinAlg.Matrix([[5.0, 6.0], [7.0, 8.0]])
        let ast = MathLexExpression.binary(
            op: .add,
            left: .variable("A"),
            right: .variable("B"))
        let result = try MathExpr.evaluateUnified(
            ast, values: ["A": .matrix(a), "B": .matrix(b)])
        let expected = LinAlg.Matrix([[6.0, 8.0], [10.0, 12.0]])
        assertApprox(result, equals: .matrix(expected), "A + B")
    }

    /// Matrix × Matrix: matmul (NOT element-wise)
    func testSharedMatrix_MatrixMultiplication() throws {
        // [1 2; 3 4] × [5 6; 7 8] = [19 22; 43 50]
        let a = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let b = LinAlg.Matrix([[5.0, 6.0], [7.0, 8.0]])
        let ast = MathLexExpression.binary(
            op: .mul,
            left: .variable("A"),
            right: .variable("B"))
        let result = try MathExpr.evaluateUnified(
            ast, values: ["A": .matrix(a), "B": .matrix(b)])
        let expected = LinAlg.Matrix([[19.0, 22.0], [43.0, 50.0]])
        assertApprox(result, equals: .matrix(expected), "A * B (matmul)")
    }

    /// det(A) for 2×2 identity → scalar(1)
    func testSharedMatrix_DeterminantIdentity() throws {
        let identity = LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])
        let ast = MathLexExpression.determinant(matrix: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(identity)])
        assertApprox(result, equals: .scalar(1.0), "det(I_2)")
    }

    /// trace(A) for 2×2 diagonal matrix → scalar(sum of diagonal)
    func testSharedMatrix_Trace() throws {
        let diag = LinAlg.Matrix([[3.0, 0.0], [0.0, 7.0]])
        let ast = MathLexExpression.trace(matrix: .variable("A"))
        let result = try MathExpr.evaluateUnified(ast, values: ["A": .matrix(diag)])
        assertApprox(result, equals: .scalar(10.0), "trace([[3,0],[0,7]])")
    }

    /// dot(u, v) for two column vectors → scalar (1×1 coercion via §4.3a)
    func testSharedMatrix_DotProductColumnVectors() throws {
        // u = [1; 2; 3], v = [4; 5; 6] → dot = 1×4 + 2×5 + 3×6 = 32
        let u = LinAlg.Matrix([[1.0], [2.0], [3.0]])
        let v = LinAlg.Matrix([[4.0], [5.0], [6.0]])
        let ast = MathLexExpression.dotProduct(
            left: .variable("u"),
            right: .variable("v"))
        let result = try MathExpr.evaluateUnified(
            ast, values: ["u": .matrix(u), "v": .matrix(v)])
        assertApprox(result, equals: .scalar(32.0), "dot(u,v)")
    }

    /// 1×1 matrix → scalar coercion via §4.3a
    func testSharedMatrix_OneByOneCoercion() throws {
        // dot([5],[5]) should collapse to scalar(25), not matrix(1×1)
        let u = LinAlg.Matrix([[5.0]])
        let v = LinAlg.Matrix([[5.0]])
        let ast = MathLexExpression.dotProduct(
            left: .variable("u"),
            right: .variable("v"))
        let result = try MathExpr.evaluateUnified(
            ast, values: ["u": .matrix(u), "v": .matrix(v)])
        // Result must be .scalar — not .matrix(1×1)
        guard case .scalar(let x) = result else {
            XCTFail("1×1 dot should coerce to .scalar, got \(result)")
            return
        }
        XCTAssertEqual(x, 25.0, accuracy: 1e-10)
    }

    // MARK: - Complex-matrix corpus (shared — both build configs)

    /// ComplexMatrix variable binding — pass through unchanged
    func testSharedComplexMatrix_VariableBinding() throws {
        let base = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let cm = LinAlg.ComplexMatrix(base)
        let ast = MathLexExpression.variable("CM")
        let result = try MathExpr.evaluateUnified(ast, values: ["CM": .complexMatrix(cm)])
        assertApprox(result, equals: .complexMatrix(cm), "complexMatrix variable passthrough")
    }

    /// Scalar × ComplexMatrix broadcast
    func testSharedComplexMatrix_ScalarBroadcast() throws {
        let base = LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])
        let cm = LinAlg.ComplexMatrix(base)
        let ast = MathLexExpression.binary(
            op: .mul,
            left: .integer(3),
            right: .variable("CM"))
        let result = try MathExpr.evaluateUnified(ast, values: ["CM": .complexMatrix(cm)])
        // Expect complexMatrix with real part scaled by 3
        guard case .complexMatrix(let resultCM) = result else {
            XCTFail("Expected .complexMatrix, got \(result)")
            return
        }
        XCTAssertEqual(resultCM.rows, 2)
        XCTAssertEqual(resultCM.cols, 2)
        // Diagonal elements should be 3.0; off-diagonal 0.0
        let expectedReal = [3.0, 0.0, 0.0, 3.0]
        for (idx, (actual, exp)) in zip(resultCM.real, expectedReal).enumerated() {
            XCTAssertEqual(actual, exp, accuracy: 1e-10,
                           "real[\(idx)]: expected \(exp) got \(actual)")
        }
    }

    // MARK: - Dispatch-core no-drift assertions
    //
    // These tests directly probe the dispatch core without going through parse(),
    // confirming that the dispatch routing produces identical NumericValue kinds
    // and values in both build configs.

    /// Scalar + Scalar → Scalar kind (dispatch routes correctly)
    func testDispatchCore_ScalarPlusScalar_Kind() throws {
        let ast = MathLexExpression.binary(op: .add, left: .integer(1), right: .integer(2))
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertTrue(result.isScalar, "scalar+scalar must produce scalar, got \(result)")
    }

    /// Complex + Complex → Complex kind
    func testDispatchCore_ComplexPlusComplex_Kind() throws {
        let lhs = MathLexExpression.complex(real: .integer(1), imaginary: .integer(0))
        let rhs = MathLexExpression.complex(real: .integer(0), imaginary: .integer(1))
        let ast = MathLexExpression.binary(op: .add, left: lhs, right: rhs)
        let result = try MathExpr.evaluateUnified(ast)
        XCTAssertTrue(result.isComplex, "complex+complex must produce complex, got \(result)")
    }

    /// Matrix + Matrix → Matrix kind
    func testDispatchCore_MatrixPlusMatrix_Kind() throws {
        let a = LinAlg.Matrix([[1.0, 2.0], [3.0, 4.0]])
        let b = LinAlg.Matrix([[0.0, 0.0], [0.0, 0.0]])
        let ast = MathLexExpression.binary(op: .add, left: .variable("A"), right: .variable("B"))
        let result = try MathExpr.evaluateUnified(
            ast, values: ["A": .matrix(a), "B": .matrix(b)])
        XCTAssertTrue(result.isMatrix, "matrix+matrix must produce matrix, got \(result)")
    }

    /// Scalar * Matrix → Matrix kind (scalar broadcast)
    func testDispatchCore_ScalarTimesMatrix_Kind() throws {
        let m = LinAlg.Matrix([[1.0, 0.0], [0.0, 1.0]])
        let ast = MathLexExpression.binary(op: .mul, left: .integer(5), right: .variable("M"))
        let result = try MathExpr.evaluateUnified(ast, values: ["M": .matrix(m)])
        XCTAssertTrue(result.isMatrix, "scalar*matrix must produce matrix, got \(result)")
    }

    /// NaN non-reflexivity — a value containing NaN is never isExactlyEqual to itself
    func testDispatchCore_NaN_NonReflexive() throws {
        let nanValue = NumericValue.scalar(Double.nan)
        XCTAssertFalse(nanValue.isExactlyEqual(to: nanValue),
                       "NaN must not be exactly equal to itself")
    }

    /// isApproximatelyEqual also rejects NaN
    func testDispatchCore_NaN_NotApproximatelyEqual() throws {
        let nanValue = NumericValue.scalar(Double.nan)
        XCTAssertFalse(nanValue.isApproximatelyEqual(to: nanValue, tolerance: 1e-10),
                       "NaN must not be approximately equal to itself")
    }

    /// Cross-kind never equal: scalar vs complex
    func testDispatchCore_CrossKind_ScalarVsComplex_NeverEqual() {
        let s = NumericValue.scalar(1.0)
        let c = NumericValue.complex(Complex(re: 1.0, im: 0.0))
        XCTAssertFalse(s.isExactlyEqual(to: c), "scalar(1) != complex(1+0i): different kinds")
        XCTAssertFalse(s.isApproximatelyEqual(to: c), "scalar(1) ≉ complex(1+0i): different kinds")
    }

    /// Cross-kind never equal: matrix vs scalar
    func testDispatchCore_CrossKind_MatrixVsScalar_NeverEqual() {
        let s = NumericValue.scalar(1.0)
        let m = NumericValue.matrix(LinAlg.Matrix([[1.0]]))
        XCTAssertFalse(s.isExactlyEqual(to: m), "scalar ≠ 1×1 matrix: different kinds")
    }

    // MARK: - NaN / ±inf parity across configs

    /// +inf propagates through binary add
    func testSharedScalar_InfinityPropagation() throws {
        let ast = MathLexExpression.binary(
            op: .add,
            left: .constant(.infinity),
            right: .integer(1))
        let result = try MathExpr.evaluateUnified(ast)
        guard case .scalar(let x) = result else {
            XCTFail("Expected scalar, got \(result)")
            return
        }
        XCTAssertEqual(x, .infinity)
    }

    /// −inf constant round-trips
    func testSharedScalar_NegInfinityRoundTrip() throws {
        let ast = MathLexExpression.constant(.negInfinity)
        let result = try MathExpr.evaluateUnified(ast)
        guard case .scalar(let x) = result else {
            XCTFail("Expected scalar, got \(result)")
            return
        }
        XCTAssertEqual(x, -.infinity)
    }

    // MARK: - Mathlex-only literal parse guard (default build only)
    //
    // On the default build (NUMERICSWIFT_MATHLEX absent), bracket-literal
    // expressions must throw MathExprError.parseError. This pins the
    // documented skip behaviour described in Task 19 and in the file header.
    //
    // Under the mathlex build this block is excluded — the mathlex CI job
    // must instead assert that these expressions PARSE AND EVALUATE CORRECTLY
    // to matching NumericValue results (see CI REQUIREMENT in file header).

    #if !NUMERICSWIFT_MATHLEX

    /// "[1, 2, 3]" is not parseable on the default build → .parseError
    func testDefaultBuild_VectorLiteral_ParseError() {
        XCTAssertThrowsError(try MathExpr.parse("[1, 2, 3]")) { error in
            guard let me = error as? MathExprError, case .parseError = me else {
                XCTFail("Expected .parseError, got \(error)")
                return
            }
        }
    }

    /// "[[1,2],[3,4]]" is not parseable on the default build → .parseError
    func testDefaultBuild_MatrixLiteral_ParseError() {
        XCTAssertThrowsError(try MathExpr.parse("[[1,2],[3,4]]")) { error in
            guard let me = error as? MathExprError, case .parseError = me else {
                XCTFail("Expected .parseError, got \(error)")
                return
            }
        }
    }

    #endif  // !NUMERICSWIFT_MATHLEX
}
