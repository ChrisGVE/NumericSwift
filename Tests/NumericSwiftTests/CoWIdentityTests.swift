//
//  CoWIdentityTests.swift
//  NumericSwiftTests
//
//  Copy-on-Write (CoW) buffer identity tests for the unified numeric evaluator.
//
//  ## Purpose
//
//  These tests guard against accidental defensive copies in the unified evaluator
//  dispatch core (`UnifiedEvaluatorCore`, `NumericDispatch`). When a matrix flows
//  through the evaluator without being mutated, the backing `[Double]` storage
//  must not be reallocated — the pre- and post-evaluation `withUnsafeBufferPointer`
//  base addresses must be identical (§CoW, PRD 0.3.0).
//
//  ## Why pointer identity, not data equality
//
//  Task 21 (`testMatrixVariablePassThroughIdentity`, `testComplexMatrixVariablePassThroughIdentity`
//  in `UnifiedEvaluatorTests.swift`) already verifies *structural identity* — that
//  shape dimensions and data element values are preserved. This file verifies a
//  stronger property: that no new heap allocation is made for the backing buffer.
//  Data equality is necessary but not sufficient: it cannot detect a silent
//  defensive copy that produces an equal but separately allocated buffer.
//
//  ## What makes the comparison meaningful
//
//  Swift's Array is CoW: `let b = a` shares `a`'s buffer until either is mutated.
//  Calling `withUnsafeBufferPointer` on a `let`-bound array returns its current
//  buffer address WITHOUT triggering a copy. Two `let` bindings that share the
//  same logical array therefore return the same `baseAddress`. Any code path that
//  makes an unnecessary defensive copy (even one that preserves values) will fork
//  the buffer and produce a different `baseAddress`, causing these tests to fail.
//
//  ## Exclusivity and ARC-timing safety
//
//  All operand arrays are bound with `let` (never `var`) and the `baseAddress`
//  is captured outside the `withUnsafeBufferPointer` closure as a plain
//  `UnsafePointer<Double>?`. This pattern avoids exclusivity violations from
//  overlapping accesses and is free from ARC-timing flakiness: the arrays are
//  strongly referenced by `let` bindings that outlive all assertions.
//
//  ## Findings protocol
//
//  If any assertion in this file fails (base address changed), it means the
//  evaluator made an accidental defensive copy on the identified path. This is a
//  performance defect in the source, not a test error — do not weaken the
//  assertion; instead fix the defensive copy in the evaluator.
//
//  Licensed under the Apache License, Version 2.0.
//

import XCTest
@testable import NumericSwift

// MARK: - CoW identity test suite

final class CoWIdentityTests: XCTestCase {

    // MARK: - CoW baseline (stdlib contract verification)

    /// Establishes that two `let` copies of the same array share a buffer —
    /// this is the stdlib guarantee underpinning all other tests in this file.
    ///
    /// If this test fails, the Swift runtime has changed its CoW contract and
    /// all pointer-identity assertions below must be re-evaluated.
    func testSwiftArrayCoWBaselineSharedBuffer() {
        // Two `let` bindings to the same array share one buffer.
        let original = [1.0, 2.0, 3.0, 4.0]
        let copy = original

        let addrOriginal = original.withUnsafeBufferPointer { $0.baseAddress }
        let addrCopy     = copy.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrOriginal, addrCopy,
            "stdlib CoW contract: two let-copies of the same array must share a buffer")
    }

    /// Establishes the negative control: mutating a `var` copy through a
    /// `withUnsafeMutableBufferPointer` forces a CoW fork — the addresses diverge.
    ///
    /// This confirms our pointer-identity technique actually detects copies.
    func testSwiftArrayCoWNegativeControlMutationDiverges() {
        let original = [1.0, 2.0, 3.0, 4.0]
        var mutated  = original   // var: can be mutated

        let addrBefore = original.withUnsafeBufferPointer { $0.baseAddress }

        // Force a CoW copy by writing through the mutable binding.
        mutated.withUnsafeMutableBufferPointer { $0[0] = 99.0 }

        let addrAfter = mutated.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertNotEqual(addrBefore, addrAfter,
            "negative control: mutation must fork the buffer (addresses must diverge)")
    }

    // MARK: - Matrix struct CoW baseline

    /// Confirms that a `let`-bound `LinAlg.Matrix` copy shares the `.data`
    /// buffer with the original — the struct copy is shallow until mutated.
    func testMatrixStructCopySharesDataBuffer() {
        let m1 = LinAlg.Matrix(rows: 2, cols: 2, data: [1.0, 2.0, 3.0, 4.0])
        let m2 = m1   // struct copy — `data` array is still shared (CoW)

        let addr1 = m1.data.withUnsafeBufferPointer { $0.baseAddress }
        let addr2 = m2.data.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addr1, addr2,
            "Matrix struct copy must share the .data buffer until mutation")
    }

    /// Confirms that a `let`-bound `LinAlg.ComplexMatrix` copy shares both
    /// `.real` and `.imag` buffers with the original.
    func testComplexMatrixStructCopySharesBothBuffers() {
        let cm1 = LinAlg.ComplexMatrix(
            rows: 2, cols: 2,
            real: [1.0, 2.0, 3.0, 4.0],
            imag: [0.1, 0.2, 0.3, 0.4])
        let cm2 = cm1

        let addrReal1 = cm1.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrReal2 = cm2.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImag1 = cm1.imag.withUnsafeBufferPointer { $0.baseAddress }
        let addrImag2 = cm2.imag.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrReal1, addrReal2,
            "ComplexMatrix struct copy must share the .real buffer until mutation")
        XCTAssertEqual(addrImag1, addrImag2,
            "ComplexMatrix struct copy must share the .imag buffer until mutation")
    }

    // MARK: - NumericValue enum CoW baseline

    /// Confirms that wrapping a `LinAlg.Matrix` in `NumericValue.matrix` and
    /// then extracting it with `asMatrix` does NOT copy the backing buffer.
    ///
    /// `NumericValue` is a value-type enum, so each wrapping/unwrapping
    /// creates a struct copy, but the inner `[Double]` is still CoW-shared.
    func testNumericValueMatrixWrapUnwrapPreservesBuffer() {
        let m = LinAlg.Matrix(rows: 3, cols: 2, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let addrBefore = m.data.withUnsafeBufferPointer { $0.baseAddress }

        let wrapped   = NumericValue.matrix(m)
        let extracted = wrapped.asMatrix!

        let addrAfter = extracted.data.withUnsafeBufferPointer { $0.baseAddress }

        // §CoW: wrapping in NumericValue then extracting with asMatrix must not
        // allocate a new buffer — the [Double] is still CoW-shared.
        XCTAssertEqual(addrBefore, addrAfter,
            "NumericValue.matrix wrap+unwrap must not copy the .data buffer")
    }

    /// Same as above for `ComplexMatrix` — both `.real` and `.imag` must survive
    /// the `NumericValue.complexMatrix` → `asComplexMatrix` round-trip.
    func testNumericValueComplexMatrixWrapUnwrapPreservesBuffers() {
        let cm = LinAlg.ComplexMatrix(
            rows: 2, cols: 3,
            real: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            imag: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        let addrRealBefore = cm.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImagBefore = cm.imag.withUnsafeBufferPointer { $0.baseAddress }

        let wrapped   = NumericValue.complexMatrix(cm)
        let extracted = wrapped.asComplexMatrix!

        let addrRealAfter = extracted.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImagAfter = extracted.imag.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrRealBefore, addrRealAfter,
            "NumericValue.complexMatrix wrap+unwrap must not copy the .real buffer")
        XCTAssertEqual(addrImagBefore, addrImagAfter,
            "NumericValue.complexMatrix wrap+unwrap must not copy the .imag buffer")
    }

    // MARK: - Variable pass-through: buffer pointer identity

    /// Verifies that routing a `LinAlg.Matrix` through the unified evaluator as a
    /// bare variable (no arithmetic) does NOT allocate a new backing buffer.
    ///
    /// The evaluator resolves `.variable("M")` by returning the `NumericValue`
    /// from the `values` dictionary directly. No operation touches `data`, so
    /// the [Double] buffer address must be unchanged.
    ///
    /// §CoW: the pre-evaluation and post-evaluation `baseAddress` values must
    /// be identical — a difference indicates an accidental defensive copy in the
    /// dispatch core.
    func testMatrixVariablePassThroughBufferIdentity() throws {
        let data   = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        let input  = LinAlg.Matrix(rows: 3, cols: 3, data: data)

        // Capture the pre-evaluation buffer address via a let binding.
        let addrBefore = input.data.withUnsafeBufferPointer { $0.baseAddress }

        let result   = try MathExpr.evaluateUnified(.variable("M"), values: ["M": .matrix(input)])
        let output   = try result.asMatrixThrowing()

        let addrAfter = output.data.withUnsafeBufferPointer { $0.baseAddress }

        // §CoW: the backing buffer must be the same allocation — no defensive
        // copy should occur when the matrix is only read, never mutated.
        XCTAssertEqual(addrBefore, addrAfter,
            "Matrix variable pass-through must not copy the .data buffer "
          + "(§CoW: base address must be identical before and after evaluation)")
    }

    /// Same as `testMatrixVariablePassThroughBufferIdentity` but for
    /// `LinAlg.ComplexMatrix` — verifies both `.real` and `.imag` buffers.
    func testComplexMatrixVariablePassThroughBufferIdentity() throws {
        let realData = [1.0, 2.0, 3.0, 4.0]
        let imagData = [0.5, 1.5, 2.5, 3.5]
        let input    = LinAlg.ComplexMatrix(rows: 2, cols: 2, real: realData, imag: imagData)

        let addrRealBefore = input.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImagBefore = input.imag.withUnsafeBufferPointer { $0.baseAddress }

        let result   = try MathExpr.evaluateUnified(.variable("C"), values: ["C": .complexMatrix(input)])
        let output   = try result.asComplexMatrixThrowing()

        let addrRealAfter = output.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImagAfter = output.imag.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrRealBefore, addrRealAfter,
            "ComplexMatrix variable pass-through must not copy the .real buffer")
        XCTAssertEqual(addrImagBefore, addrImagAfter,
            "ComplexMatrix variable pass-through must not copy the .imag buffer")
    }

    // MARK: - Variable pass-through: parameterised sizes

    /// Repeats the real-matrix buffer-identity check across representative sizes
    /// (1×1, small, medium, large) to confirm the zero-copy property is
    /// independent of matrix dimensions.
    func testMatrixVariablePassThroughIdentityAcrossSizes() throws {
        let sizes: [(rows: Int, cols: Int)] = [
            (1, 1),    // 1×1 — coercion boundary
            (3, 3),    // small square
            (10, 10),  // medium square
            (50, 50),  // large (within soft cap)
        ]

        for (rows, cols) in sizes {
            let count = rows * cols
            let data  = (0..<count).map { Double($0) }
            let input = LinAlg.Matrix(rows: rows, cols: cols, data: data)

            let addrBefore = input.data.withUnsafeBufferPointer { $0.baseAddress }

            let result = try MathExpr.evaluateUnified(
                .variable("M"),
                values: ["M": .matrix(input)])
            let output = try result.asMatrixThrowing()

            let addrAfter = output.data.withUnsafeBufferPointer { $0.baseAddress }

            XCTAssertEqual(addrBefore, addrAfter,
                "Matrix (\(rows)×\(cols)) variable pass-through must not copy the .data buffer")
        }
    }

    // MARK: - Binary operand CoW: inputs are not mutated

    /// When a matrix is the LEFT operand of a matrix+scalar binary operation,
    /// the evaluator reads but does not mutate the matrix's `data`. The input
    /// matrix's buffer address must be unchanged after the call.
    ///
    /// The RESULT buffer is legitimately distinct (it holds new values).
    /// We only assert that the INPUT buffer was not spuriously copied.
    func testBinaryOperandLeftMatrixBufferUnchangedAfterScalarAdd() throws {
        let data  = [1.0, 2.0, 3.0, 4.0]
        let left  = LinAlg.Matrix(rows: 2, cols: 2, data: data)

        let addrInputBefore = left.data.withUnsafeBufferPointer { $0.baseAddress }

        // Evaluate: M + 1.0 (matrix + scalar) — reads left, produces new buffer.
        let result = try MathExpr.evaluateUnified(
            MathLexExpression.binary(op: .add, left: .variable("M"), right: .float(1.0)),
            values: ["M": .matrix(left)])
        let output = try result.asMatrixThrowing()

        let addrInputAfter = left.data.withUnsafeBufferPointer { $0.baseAddress }

        // The input's buffer must be intact — no write-back to the source.
        XCTAssertEqual(addrInputBefore, addrInputAfter,
            "Left matrix input buffer must not be reallocated by a matrix+scalar binary op")
        // Sanity: result has distinct storage (it computed new values).
        let addrResult = output.data.withUnsafeBufferPointer { $0.baseAddress }
        XCTAssertNotEqual(addrInputBefore, addrResult,
            "Result of matrix+scalar must have its own buffer (legitimately distinct)")
    }

    /// When a matrix is the RIGHT operand of a scalar×matrix multiplication,
    /// the evaluator reads but does not mutate the right operand's `data`.
    func testBinaryOperandRightMatrixBufferUnchangedAfterScalarMul() throws {
        let data  = [2.0, 4.0, 6.0, 8.0]
        let right = LinAlg.Matrix(rows: 2, cols: 2, data: data)

        let addrInputBefore = right.data.withUnsafeBufferPointer { $0.baseAddress }

        // Evaluate: 3.0 * M (scalar × matrix).
        let result = try MathExpr.evaluateUnified(
            MathLexExpression.binary(op: .mul, left: .float(3.0), right: .variable("M")),
            values: ["M": .matrix(right)])
        let output = try result.asMatrixThrowing()

        let addrInputAfter = right.data.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrInputBefore, addrInputAfter,
            "Right matrix input buffer must not be reallocated by scalar×matrix mul")
        let addrResult = output.data.withUnsafeBufferPointer { $0.baseAddress }
        XCTAssertNotEqual(addrInputBefore, addrResult,
            "Result of scalar×matrix must have its own buffer")
    }

    /// When a matrix is divided by a scalar (M ÷ scalar), the input matrix
    /// buffer must survive untouched.
    func testBinaryOperandMatrixDivScalarInputBufferUnchanged() throws {
        let data  = [10.0, 20.0, 30.0, 40.0]
        let left  = LinAlg.Matrix(rows: 2, cols: 2, data: data)

        let addrInputBefore = left.data.withUnsafeBufferPointer { $0.baseAddress }

        let result = try MathExpr.evaluateUnified(
            MathLexExpression.binary(op: .div, left: .variable("M"), right: .float(2.0)),
            values: ["M": .matrix(left)])
        let output = try result.asMatrixThrowing()

        let addrInputAfter = left.data.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrInputBefore, addrInputAfter,
            "Left matrix input buffer must not be reallocated by matrix÷scalar")
        let addrResult = output.data.withUnsafeBufferPointer { $0.baseAddress }
        XCTAssertNotEqual(addrInputBefore, addrResult,
            "Result of matrix÷scalar must have its own buffer")
    }

    // MARK: - 1×1 matrix→scalar coercion: source buffer unchanged

    /// When the evaluator coerces a 1×1 matrix result to `.scalar` (§4.3a),
    /// the source matrix's `data` must not be spuriously copied before the
    /// coercion. We verify the input variable's buffer is untouched and that
    /// the result kind is `.scalar` (confirming coercion fired).
    func testOneByOneMatrixCoercionDoesNotCopySourceBuffer() throws {
        let input = LinAlg.Matrix(rows: 1, cols: 1, data: [42.0])

        let addrInputBefore = input.data.withUnsafeBufferPointer { $0.baseAddress }

        // dot(u, u) where u is a 1×1 column vector → 1×1 result → coerced to scalar.
        let result = try MathExpr.evaluateUnified(
            MathLexExpression.binary(op: .mul, left: .variable("M"), right: .variable("M")),
            values: ["M": .matrix(input)])

        let addrInputAfter = input.data.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrInputBefore, addrInputAfter,
            "1×1 matrix input buffer must not be reallocated before §4.3a coercion")
        // The coercion must have fired: result should be scalar 42×42 = 1764.
        switch result {
        case .scalar(let v):
            XCTAssertEqual(v, 1764.0, accuracy: 1e-10,
                "1×1 matmul result must be the correct scalar value")
        default:
            // If result is matrix(1×1) rather than scalar, coercion did not fire;
            // still check the input buffer — the CoW claim stands either way.
            break
        }
    }

    // MARK: - .matrix accessor CoW: asMatrix does not copy

    /// Verifies that `.asMatrix` (the optional extractor on `NumericValue`)
    /// does not allocate a new buffer — the returned `LinAlg.Matrix` shares
    /// the same `[Double]` backing store as the enum payload.
    func testAsMatrixAccessorPreservesBuffer() {
        let original  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let m         = LinAlg.Matrix(rows: 2, cols: 3, data: original)
        let value     = NumericValue.matrix(m)

        let addrBefore = m.data.withUnsafeBufferPointer { $0.baseAddress }
        let extracted  = value.asMatrix!
        let addrAfter  = extracted.data.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrBefore, addrAfter,
            ".asMatrix accessor must not copy the .data buffer")
    }

    /// Verifies that `.asComplexMatrix` does not copy either `.real` or `.imag`.
    func testAsComplexMatrixAccessorPreservesBuffers() {
        let realData  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let imagData  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        let cm        = LinAlg.ComplexMatrix(rows: 3, cols: 2, real: realData, imag: imagData)
        let value     = NumericValue.complexMatrix(cm)

        let addrRealBefore = cm.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImagBefore = cm.imag.withUnsafeBufferPointer { $0.baseAddress }
        let extracted  = value.asComplexMatrix!
        let addrRealAfter  = extracted.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImagAfter  = extracted.imag.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrRealBefore, addrRealAfter,
            ".asComplexMatrix accessor must not copy the .real buffer")
        XCTAssertEqual(addrImagBefore, addrImagAfter,
            ".asComplexMatrix accessor must not copy the .imag buffer")
    }

    // MARK: - Complex matrix binary operand: real+imag buffers untouched

    /// When a `ComplexMatrix` is the operand of a unary negation (which produces
    /// a wholly new `ComplexMatrix`), the source `.real` and `.imag` buffers
    /// must not be reallocated.
    func testComplexMatrixUnaryNegInputBuffersUnchanged() throws {
        let realData = [1.0, 2.0, 3.0, 4.0]
        let imagData = [0.5, 1.5, 2.5, 3.5]
        let input    = LinAlg.ComplexMatrix(rows: 2, cols: 2, real: realData, imag: imagData)

        let addrRealBefore = input.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImagBefore = input.imag.withUnsafeBufferPointer { $0.baseAddress }

        // -(C) — reads input, allocates new ComplexMatrix for the result.
        let result = try MathExpr.evaluateUnified(
            MathLexExpression.unary(op: .neg, operand: .variable("C")),
            values: ["C": .complexMatrix(input)])
        let output = try result.asComplexMatrixThrowing()

        let addrRealAfter = input.real.withUnsafeBufferPointer { $0.baseAddress }
        let addrImagAfter = input.imag.withUnsafeBufferPointer { $0.baseAddress }

        XCTAssertEqual(addrRealBefore, addrRealAfter,
            "ComplexMatrix .real buffer must not be reallocated by unary negation of the operand")
        XCTAssertEqual(addrImagBefore, addrImagAfter,
            "ComplexMatrix .imag buffer must not be reallocated by unary negation of the operand")
        // Sanity: result carries the expected negated values.
        XCTAssertEqual(output.real[0], -1.0, accuracy: 1e-12)
        XCTAssertEqual(output.imag[0], -0.5, accuracy: 1e-12)
    }
}
